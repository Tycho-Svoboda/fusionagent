"""Full e2e benchmark: Analyze → Research → Generate → Benchmark on H100.

Runs the complete FusionAgent pipeline on a LLaMA-style TransformerBlock,
generates real Triton kernels via OpenAI, benchmarks them, and then runs
a dummy training workload comparing fused vs unfused forward passes.
"""

import os
import sys
import time
import math
import textwrap
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from fusionagent.graph.analyzer import GraphAnalyzer
from fusionagent.research.retriever import ResearchRetriever
from fusionagent.generator.codegen import KernelGenerator
from fusionagent.harness.benchmark import BenchmarkHarness
from fusionagent.types import FusionCandidate


# ---------------------------------------------------------------------------
# Model definition — LLaMA-style TransformerBlock
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU_MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, n_heads: int = 8, mlp_mult: int = 4):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU_MLP(dim, dim * mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniTransformer(nn.Module):
    """Small transformer for the dummy training workload."""
    def __init__(self, vocab_size: int = 1024, dim: int = 512, n_heads: int = 8,
                 n_layers: int = 4, max_seq: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Part 1: Full pipeline benchmark per candidate
# ---------------------------------------------------------------------------

def run_pipeline_benchmark():
    print("=" * 70)
    print("PART 1: FusionAgent E2E Pipeline — Per-Candidate Kernel Benchmarks")
    print("=" * 70)

    device = "cuda:0"
    dim, n_heads = 512, 8

    # --- Task 1: Graph Analysis ---
    print("\n[Task 1] GraphAnalyzer: analyzing TransformerBlock...")
    block = TransformerBlock(dim=dim, n_heads=n_heads).eval()
    sample = torch.randn(1, 128, dim)
    analyzer = GraphAnalyzer(block, sample_input=sample)
    candidates = analyzer.analyze()
    print(f"  Found {len(candidates)} fusion candidates:")
    for i, c in enumerate(candidates):
        print(f"    {i+1}. ops={c.ops}  memory_bound={c.memory_bound}  "
              f"overhead={c.launch_overhead_us:.1f}μs")

    # --- Task 2: Research Retrieval ---
    print("\n[Task 2] ResearchRetriever: gathering prior work...")
    retriever = ResearchRetriever()
    contexts = {}
    for i, c in enumerate(candidates):
        try:
            ctx = retriever.retrieve(c)
            contexts[i] = ctx
            print(f"  Candidate {i+1}: novelty={ctx.novelty_score:.2f}, "
                  f"{len(ctx.prior_implementations)} priors, "
                  f"{len(ctx.known_pitfalls)} pitfalls")
        except Exception as e:
            print(f"  Candidate {i+1}: retrieval failed ({e}), continuing...")
            contexts[i] = None

    # --- Task 3: Kernel Generation + Task 4: Benchmark ---
    print("\n[Tasks 3-4] KernelGenerator + BenchmarkHarness:")
    generator = KernelGenerator()
    harness = BenchmarkHarness(device=device)

    results = []
    for i, c in enumerate(candidates):
        print(f"\n  --- Candidate {i+1}: {c.ops} ---")

        # Generate
        print(f"  Generating Triton kernel via LLM...")
        try:
            code = generator.generate(c, context=contexts.get(i))
        except Exception as e:
            print(f"  Generation failed: {e}")
            results.append(None)
            continue

        if "KernelGenerator failed" in code:
            print(f"  LLM returned invalid code, skipping benchmark.")
            results.append(None)
            continue

        print(f"  Generated {len(code.splitlines())} lines of kernel code.")

        # Benchmark
        print(f"  Benchmarking on {device}...")
        try:
            result = harness.evaluate(code, c)
            results.append(result)
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            results.append(None)
            continue

        if result.passed_correctness:
            print(f"  ✓ Correct  |  fused={result.latency_us:.1f}μs  "
                  f"baseline={result.baseline_latency_us:.1f}μs  "
                  f"speedup={result.speedup:.2f}x")
        else:
            print(f"  ✗ Failed: {result.compile_error}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'#':<4} {'Ops':<35} {'Correct':<9} {'Fused(μs)':<11} "
          f"{'Base(μs)':<11} {'Speedup':<9}")
    print("-" * 70)
    for i, (c, r) in enumerate(zip(candidates, results)):
        ops_str = " → ".join(c.ops)[:33]
        if r is None:
            print(f"{i+1:<4} {ops_str:<35} {'SKIP':<9}")
        elif r.passed_correctness:
            print(f"{i+1:<4} {ops_str:<35} {'✓':<9} {r.latency_us:<11.1f} "
                  f"{r.baseline_latency_us:<11.1f} {r.speedup:<9.2f}x")
        else:
            err = (r.compile_error or "unknown")[:30]
            print(f"{i+1:<4} {ops_str:<35} {'✗':<9} {err}")

    return candidates, results


# ---------------------------------------------------------------------------
# Part 2: Dummy training workload — fused vs unfused comparison
# ---------------------------------------------------------------------------

def run_training_benchmark():
    print("\n\n" + "=" * 70)
    print("PART 2: Training Workload — Fused vs Unfused Forward Pass")
    print("=" * 70)

    device = "cuda:0"
    vocab_size, dim, n_heads, n_layers = 1024, 512, 8, 4
    batch_size, seq_len = 8, 256
    n_steps = 50

    print(f"\nModel: MiniTransformer ({n_layers}L, dim={dim}, heads={n_heads})")
    print(f"Training: {n_steps} steps, batch={batch_size}, seq={seq_len}")

    model = MiniTransformer(
        vocab_size=vocab_size, dim=dim, n_heads=n_heads, n_layers=n_layers
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count * 4 / 1e6:.1f} MB fp32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    def make_batch():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return x, y

    # --- Baseline training (unfused, standard PyTorch) ---
    print(f"\n[Baseline] Standard PyTorch eager mode...")
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    baseline_losses = []
    for step in range(n_steps):
        x, y = make_batch()
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        baseline_losses.append(loss.item())

    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - t0
    baseline_throughput = (n_steps * batch_size * seq_len) / baseline_time

    print(f"  Time: {baseline_time:.2f}s  |  {baseline_time/n_steps*1000:.1f}ms/step  "
          f"|  {baseline_throughput:.0f} tok/s")
    print(f"  Loss: {baseline_losses[0]:.3f} → {baseline_losses[-1]:.3f}")

    # --- torch.compile training ---
    print(f"\n[torch.compile] Compiled mode...")
    model_compiled = MiniTransformer(
        vocab_size=vocab_size, dim=dim, n_heads=n_heads, n_layers=n_layers
    ).to(device)
    model_compiled.load_state_dict(model.state_dict())  # same init
    model_compiled = torch.compile(model_compiled)
    optimizer2 = torch.optim.AdamW(model_compiled.parameters(), lr=3e-4)

    # Warmup compile
    print("  Warming up torch.compile...")
    x, y = make_batch()
    logits = model_compiled(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()
    torch.cuda.synchronize()

    model_compiled.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    compiled_losses = []
    for step in range(n_steps):
        x, y = make_batch()
        logits = model_compiled(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        compiled_losses.append(loss.item())

    torch.cuda.synchronize()
    compiled_time = time.perf_counter() - t0
    compiled_throughput = (n_steps * batch_size * seq_len) / compiled_time

    print(f"  Time: {compiled_time:.2f}s  |  {compiled_time/n_steps*1000:.1f}ms/step  "
          f"|  {compiled_throughput:.0f} tok/s")
    print(f"  Loss: {compiled_losses[0]:.3f} → {compiled_losses[-1]:.3f}")

    # --- Summary ---
    speedup = baseline_time / compiled_time if compiled_time > 0 else 0
    print(f"\n{'='*70}")
    print(f"TRAINING BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Mode':<25} {'Time(s)':<10} {'ms/step':<10} {'tok/s':<12} {'Speedup':<10}")
    print(f"  {'-'*65}")
    print(f"  {'PyTorch Eager':<25} {baseline_time:<10.2f} "
          f"{baseline_time/n_steps*1000:<10.1f} {baseline_throughput:<12.0f} {'1.00x':<10}")
    print(f"  {'torch.compile':<25} {compiled_time:<10.2f} "
          f"{compiled_time/n_steps*1000:<10.1f} {compiled_throughput:<12.0f} "
          f"{speedup:.2f}x")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  GPU: {gpu_name}")
    print(f"  Peak memory: {gpu_mem:.1f} GB")

    return baseline_time, compiled_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        import triton
        print(f"Triton {triton.__version__}")
    except ImportError:
        print("Triton: not installed")

    candidates, results = run_pipeline_benchmark()
    baseline_time, compiled_time = run_training_benchmark()
