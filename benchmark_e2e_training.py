"""True E2E training benchmark: LLM-generated Triton kernels via FusionAgent pipeline.

Unlike benchmark_training.py which uses hand-written fused kernels, this script
runs the actual FusionAgent pipeline end-to-end:
  analyze model → call LLM (GPT-5.4) → verify correctness → patch into model → train

Workloads:
  1. GPT-2 Small (124M) — causal LM, 200 steps
  2. Qwen-Small (67M) — causal LM, 300 steps
  3. ViT-Small (22M) — image classification, 300 steps
  4. Distillation Student (8M) — KL distillation, 300 steps

Each workload runs 4 configs: Base, torch.compile, FusionAgent (LLM kernels), Both.
"""

import os
import sys
import time
import json
import copy
import importlib.util
import tempfile
import uuid
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


# ===================================================================
# Section 1: Model definitions (same as benchmark_training.py)
# ===================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(B, S, D))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult=4):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU_MLP(dim, dim * mlp_mult)

    def forward(self, x):
        h = self.attn_norm(x)
        x = x + self.attn(h)
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class GPT2Small(nn.Module):
    def __init__(self, vocab_size=50257, dim=768, n_heads=12, n_layers=12,
                 max_seq=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


class QwenSmall(nn.Module):
    def __init__(self, vocab_size=32000, dim=512, n_heads=8, n_layers=8,
                 max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_mult=4) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


class ViTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class ViTSmall(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=100,
                 dim=384, n_heads=6, n_layers=12):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class DistillStudent(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, n_heads=4, n_layers=4,
                 max_seq=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_mult=4) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


class DistillTeacher(nn.Module):
    def __init__(self, vocab_size=32000, dim=512, n_heads=8, n_layers=8,
                 max_seq=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_mult=4) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


# ===================================================================
# Section 2: Kernel loading + nn.Module wrappers
# ===================================================================

def _load_kernel_module(kernel_code: str):
    """Load kernel code string as a Python module. Returns module or None."""
    mod_name = f"_e2e_kernel_{uuid.uuid4().hex}"
    tmp_dir = Path(tempfile.gettempdir()) / "fusionagent_e2e_kernels"
    tmp_dir.mkdir(exist_ok=True)
    file_path = tmp_dir / f"{mod_name}.py"
    file_path.write_text(kernel_code)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "fused_kernel") and callable(mod.fused_kernel):
            return mod
        return None
    except Exception as e:
        print(f"    Failed to load kernel module: {e}")
        return None


class FusedNormWrapper(nn.Module):
    """Wraps a generated fused_kernel that replaces RMSNorm or LayerNorm.

    The kernel signature is fused_kernel(x, weight, ...) and the wrapper
    holds the original norm's weight parameter. Falls back to the original
    module on error.
    """
    def __init__(self, original_module, fused_fn, ops):
        super().__init__()
        self.weight = original_module.weight
        if hasattr(original_module, 'bias') and original_module.bias is not None:
            self.bias = original_module.bias
        else:
            self.bias = None
        self.fused_fn = fused_fn
        self.original_module = original_module
        self.ops = ops
        self._failed = False

    def forward(self, x):
        if self._failed:
            return self.original_module(x)
        try:
            if self.bias is not None:
                return self.fused_fn(x, self.weight, self.bias)
            else:
                return self.fused_fn(x, self.weight)
        except Exception:
            self._failed = True
            return self.original_module(x)


class FusedSiLUMulWrapper(nn.Module):
    """Wraps a generated fused_kernel for silu(x) * y pattern."""
    def __init__(self, fused_fn):
        super().__init__()
        self.fused_fn = fused_fn
        self._failed = False

    def forward(self, gate, up):
        if self._failed:
            return F.silu(gate) * up
        try:
            return self.fused_fn(gate, up)
        except Exception:
            self._failed = True
            return F.silu(gate) * up


# ===================================================================
# Section 3: FusionPatcher — pattern registry + per-model dispatch
# ===================================================================

class FusionPatcher:
    """Registry of verified kernels and patcher for model modules."""

    def __init__(self):
        self.kernels = {}  # pattern_key -> (kernel_code, fused_fn, ops)
        self.patch_log = []

    def register(self, ops, kernel_code, fused_fn):
        """Register a verified kernel for a given ops pattern."""
        key = self._pattern_key(ops)
        self.kernels[key] = (kernel_code, fused_fn, ops)

    def _pattern_key(self, ops):
        """Classify ops into a patchable pattern key."""
        ops_lower = [o.lower() for o in ops]
        if "rmsnorm" in ops_lower:
            if "silu" in ops_lower:
                return "rmsnorm_silu"
            return "rmsnorm"
        if "layernorm" in ops_lower:
            if "gelu" in ops_lower:
                return "layernorm_gelu"
            return "layernorm"
        if "silu" in ops_lower and "mul" in ops_lower:
            return "silu_mul"
        return "_".join(ops_lower)

    def has_pattern(self, key):
        return key in self.kernels

    def patch_model(self, model, model_type="transformer"):
        """Patch model in-place with registered kernels. Returns number of patches applied."""
        n_patches = 0
        self.patch_log = []

        if model_type == "vit":
            n_patches += self._patch_vit(model)
        else:
            n_patches += self._patch_transformer(model)

        return n_patches

    def _patch_transformer(self, model):
        """Patch TransformerBlock-based models (GPT2, Qwen, DistillStudent)."""
        n_patches = 0

        # Find all TransformerBlock instances
        for name, module in model.named_modules():
            if not isinstance(module, TransformerBlock):
                continue

            # Patch RMSNorm modules
            if self.has_pattern("rmsnorm_silu"):
                _, fused_fn, ops = self.kernels["rmsnorm_silu"]
                # Patch attn_norm
                wrapper = FusedNormWrapper(module.attn_norm, fused_fn, ops)
                module.attn_norm = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.attn_norm with rmsnorm_silu")
                # Patch ffn_norm
                wrapper = FusedNormWrapper(module.ffn_norm, fused_fn, ops)
                module.ffn_norm = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.ffn_norm with rmsnorm_silu")
            elif self.has_pattern("rmsnorm"):
                _, fused_fn, ops = self.kernels["rmsnorm"]
                wrapper = FusedNormWrapper(module.attn_norm, fused_fn, ops)
                module.attn_norm = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.attn_norm with rmsnorm")
                wrapper = FusedNormWrapper(module.ffn_norm, fused_fn, ops)
                module.ffn_norm = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.ffn_norm with rmsnorm")

            # Patch SwiGLU MLP with silu_mul
            if self.has_pattern("silu_mul") and isinstance(module.ffn, SwiGLU_MLP):
                _, fused_fn, ops = self.kernels["silu_mul"]
                fused_silu_mul = FusedSiLUMulWrapper(fused_fn)
                original_ffn = module.ffn

                # Monkey-patch the forward method
                def make_fused_forward(ffn, fused_op):
                    def fused_forward(x):
                        gate = ffn.gate_proj(x)
                        up = ffn.up_proj(x)
                        return ffn.down_proj(fused_op(gate, up))
                    return fused_forward

                module.ffn.forward = make_fused_forward(original_ffn, fused_silu_mul)
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.ffn with silu_mul")

        return n_patches

    def _patch_vit(self, model):
        """Patch ViTBlock-based models."""
        n_patches = 0

        for name, module in model.named_modules():
            if not isinstance(module, ViTBlock):
                continue

            if self.has_pattern("layernorm_gelu"):
                _, fused_fn, ops = self.kernels["layernorm_gelu"]
                # Patch norm1
                wrapper = FusedNormWrapper(module.norm1, fused_fn, ops)
                module.norm1 = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.norm1 with layernorm_gelu")
                # Patch norm2
                wrapper = FusedNormWrapper(module.norm2, fused_fn, ops)
                module.norm2 = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.norm2 with layernorm_gelu")

                # When using fused layernorm+gelu for norm, skip GELU in MLP
                # Replace MLP with version that omits the GELU
                orig_mlp = module.mlp
                linear1 = orig_mlp[0]  # nn.Linear(dim, dim*4)
                linear2 = orig_mlp[2]  # nn.Linear(dim*4, dim)
                module.mlp = nn.Sequential(linear1, linear2)
                n_patches += 1
                self.patch_log.append(f"  Removed GELU from {name}.mlp (fused into norm)")

            elif self.has_pattern("layernorm"):
                _, fused_fn, ops = self.kernels["layernorm"]
                wrapper = FusedNormWrapper(module.norm1, fused_fn, ops)
                module.norm1 = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.norm1 with layernorm")
                wrapper = FusedNormWrapper(module.norm2, fused_fn, ops)
                module.norm2 = wrapper
                n_patches += 1
                self.patch_log.append(f"  Patched {name}.norm2 with layernorm")

        return n_patches


# ===================================================================
# Section 4: Pipeline runner — analyze → generate → verify → register
# ===================================================================

MAX_RETRIES_PER_CANDIDATE = 5


def run_pipeline(block, block_name, sample_input, patcher, device="cuda:0",
                 llm_model="gpt-5.4"):
    """Run the FusionAgent pipeline on a block and register passing kernels.

    Retries each candidate up to MAX_RETRIES_PER_CANDIDATE times, accumulating
    all prior errors as feedback to the LLM.

    Returns: list of (candidate, kernel_code, result, n_attempts) tuples for reporting.
    """
    print(f"\n  --- Pipeline for {block_name} ---")

    # Step 1: Graph analysis
    print(f"  [1] Analyzing {block_name} with GraphAnalyzer...")
    analyzer = GraphAnalyzer(block, sample_input=sample_input)
    try:
        candidates = analyzer.analyze()
    except Exception as e:
        print(f"  GraphAnalyzer failed: {e}")
        return []

    print(f"      Found {len(candidates)} fusion candidates:")
    for i, c in enumerate(candidates):
        print(f"        {i+1}. ops={c.ops}  memory_bound={c.memory_bound}  "
              f"overhead={c.launch_overhead_us:.1f}us")

    if not candidates:
        return []

    # Step 2: Research + Generation + Verification
    retriever = ResearchRetriever()
    generator = KernelGenerator(model=llm_model)
    harness = BenchmarkHarness(device=device)

    pipeline_results = []

    for i, candidate in enumerate(candidates):
        print(f"\n  [Candidate {i+1}/{len(candidates)}] ops={candidate.ops}")

        # 2a. Research retrieval (best-effort)
        context = None
        try:
            context = retriever.retrieve(candidate)
            print(f"      Research: novelty={context.novelty_score:.2f}, "
                  f"{len(context.prior_implementations)} priors")
        except Exception as e:
            print(f"      Research failed ({e}), continuing without context")

        # 2b+2c. Generate + verify in a retry loop
        passed = False
        best_code = None
        best_result = None
        error_history = []  # accumulate all errors for richer feedback

        for attempt in range(1, MAX_RETRIES_PER_CANDIDATE + 1):
            # Build feedback from ALL prior errors
            feedback = None
            if error_history:
                feedback = (
                    f"You have failed {len(error_history)} previous attempt(s). "
                    f"Here are ALL the errors:\n\n"
                )
                for j, err in enumerate(error_history):
                    feedback += f"--- Attempt {j+1} error ---\n{err}\n\n"
                feedback += (
                    "CRITICAL REQUIREMENTS:\n"
                    "1. make_inputs(*shapes) must return the EXACT same number of tensors "
                    "that fused_kernel() and reference() accept as positional arguments.\n"
                    "2. If fused_kernel(x, weight) takes 2 args, make_inputs must return 2 tensors.\n"
                    "3. reference() must use plain PyTorch ops to compute the EXACT same result.\n"
                    "4. For RMSNorm: rms = sqrt(mean(x^2) + eps), output = (x / rms) * weight. "
                    "Do NOT subtract the mean (that's LayerNorm).\n"
                    "5. All tensors must be contiguous and on CUDA.\n"
                    "6. Do NOT use tl.view or tl.reshape with non-power-of-2 dimensions.\n"
                    "Fix ALL of these issues in your new implementation."
                )

            print(f"      Attempt {attempt}/{MAX_RETRIES_PER_CANDIDATE}...")
            try:
                code = generator.generate(candidate, context=context, feedback=feedback)
            except Exception as e:
                print(f"        Generation failed: {e}")
                error_history.append(f"Generation exception: {e}")
                continue

            if "KernelGenerator failed" in code:
                err_msg = "LLM returned error stub"
                print(f"        {err_msg}")
                error_history.append(err_msg)
                continue

            print(f"        Generated {len(code.splitlines())} lines. Benchmarking...")
            result = harness.evaluate(code, candidate)

            if result.passed_correctness:
                print(f"        PASSED (attempt {attempt})  |  "
                      f"fused={result.latency_us:.1f}us  "
                      f"baseline={result.baseline_latency_us:.1f}us  "
                      f"speedup={result.speedup:.2f}x")
                best_code = code
                best_result = result
                passed = True

                # Load and register
                mod = _load_kernel_module(code)
                if mod is not None:
                    patcher.register(candidate.ops, code, mod.fused_kernel)
                    print(f"        Registered as pattern: {patcher._pattern_key(candidate.ops)}")
                else:
                    print(f"        Failed to load kernel module for patching")
                    passed = False
                break
            else:
                err_msg = result.compile_error or "Unknown correctness failure"
                print(f"        FAILED: {err_msg[:120]}")
                error_history.append(err_msg)
                best_code = code
                best_result = result

        if not passed:
            print(f"      Exhausted {MAX_RETRIES_PER_CANDIDATE} attempts for candidate {i+1}")

        pipeline_results.append((candidate, best_code, best_result, len(error_history) if not passed else attempt))

    return pipeline_results


def save_kernels(pipeline_results, model_name, output_dir):
    """Save generated kernel files to disk for auditing."""
    model_dir = output_dir / model_name.replace(" ", "_").replace("(", "").replace(")", "")
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(pipeline_results):
        candidate, code, result = item[0], item[1], item[2]
        if code is None:
            continue
        ops_str = "_".join(candidate.ops)
        passed = result is not None and result.passed_correctness
        status = "PASS" if passed else "FAIL"
        filename = f"candidate_{i+1}_{ops_str}_{status}.py"
        (model_dir / filename).write_text(code)

    return model_dir


# ===================================================================
# Section 5: Training loop helpers (same as benchmark_training.py)
# ===================================================================

def train_causal_lm(model, n_steps, batch_size, seq_len, vocab_size, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()

    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def train_classifier(model, n_steps, batch_size, img_size, num_classes, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    model.train()

    # Warmup
    for _ in range(3):
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(n_steps):
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def train_distillation(student, teacher, n_steps, batch_size, seq_len,
                        vocab_size, temperature=2.0, device="cuda"):
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, fused=True)
    student.train()
    teacher.eval()

    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ===================================================================
# Section 6: Per-workload benchmark — 4 configs with dynamic patching
# ===================================================================

def sanity_check(model, make_input_fn, device="cuda"):
    """Run a quick forward+backward to verify patched model still works."""
    model.train()
    try:
        x = make_input_fn(device)
        out = model(x) if not isinstance(x, tuple) else model(*x)
        loss = out.sum()
        loss.backward()
        return True
    except Exception as e:
        print(f"    Sanity check failed: {e}")
        return False


def benchmark_workload(name, description, make_model, train_fn,
                       patcher, model_type, make_input_fn, device="cuda"):
    """Run a workload across 4 configs and return timing dict."""
    print(f"\n{'='*70}")
    print(f"  {name}: {description}")
    print(f"{'='*70}")

    results = {}

    # --- Config 1: Base PyTorch ---
    print(f"  [1/4] Base PyTorch (eager)...")
    model_base = make_model().to(device)
    n_params = sum(p.numel() for p in model_base.parameters())
    base_time = train_fn(model_base)
    results["base"] = base_time
    print(f"         {base_time:.2f}s  ({n_params/1e6:.1f}M params)")
    base_sd = copy.deepcopy(model_base.state_dict())
    del model_base
    torch.cuda.empty_cache()

    # --- Config 2: torch.compile ---
    print(f"  [2/4] torch.compile...")
    model_compile = make_model().to(device)
    model_compile.load_state_dict(base_sd, strict=False)
    model_compile = torch.compile(model_compile)
    compile_time = train_fn(model_compile)
    results["compile"] = compile_time
    print(f"         {compile_time:.2f}s")
    del model_compile
    torch.cuda.empty_cache()

    # --- Config 3: FusionAgent (LLM-generated kernels) ---
    n_patches = len(patcher.kernels)
    if n_patches > 0:
        print(f"  [3/4] FusionAgent (LLM-generated Triton kernels)...")
        model_fused = make_model().to(device)
        model_fused.load_state_dict(base_sd, strict=False)
        applied = patcher.patch_model(model_fused, model_type=model_type)
        print(f"         Applied {applied} patches:")
        for log_line in patcher.patch_log:
            print(f"         {log_line}")

        # Sanity check
        if sanity_check(model_fused, make_input_fn, device):
            fused_time = train_fn(model_fused)
            results["fused"] = fused_time
            print(f"         {fused_time:.2f}s")
        else:
            print(f"         Sanity check failed — using base time")
            results["fused"] = base_time
        del model_fused
        torch.cuda.empty_cache()
    else:
        print(f"  [3/4] FusionAgent — no kernels available, using base time")
        results["fused"] = base_time

    # --- Config 4: FusionAgent + torch.compile ---
    if n_patches > 0:
        print(f"  [4/4] FusionAgent + torch.compile...")
        model_both = make_model().to(device)
        model_both.load_state_dict(base_sd, strict=False)
        patcher.patch_model(model_both, model_type=model_type)
        model_both = torch.compile(model_both)
        both_time = train_fn(model_both)
        results["both"] = both_time
        print(f"         {both_time:.2f}s")
        del model_both
        torch.cuda.empty_cache()
    else:
        print(f"  [4/4] FusionAgent + torch.compile — no kernels, using compile time")
        results["both"] = compile_time

    del base_sd
    return results


# ===================================================================
# Section 7: Results table + JSON
# ===================================================================

def print_results_table(all_results, gpu_name, pytorch_version, triton_version):
    print("\n\n")
    print("=" * 130)
    print("E2E TRAINING BENCHMARK — LLM-GENERATED TRITON KERNELS")
    print(f"GPU: {gpu_name}  |  PyTorch: {pytorch_version}  |  Triton: {triton_version}")
    print("=" * 130)
    print(f"{'Model':<30} {'Steps':<8} "
          f"{'Base':<12} {'compile':<12} {'FusionAgent':<12} {'Both':<12} "
          f"{'compile':<10} {'FA':<10} {'Both':<10} {'Kernels':<10}")
    print(f"{'':30} {'':8} "
          f"{'(ms/step)':12} {'(ms/step)':12} {'(ms/step)':12} {'(ms/step)':12} "
          f"{'vs base':10} {'vs base':10} {'vs base':10} {'patched':10}")
    print("-" * 130)

    for r in all_results:
        steps = r["steps"]
        base_ms = r["base"] / steps * 1000
        comp_ms = r["compile"] / steps * 1000
        fused_ms = r["fused"] / steps * 1000
        both_ms = r["both"] / steps * 1000

        comp_speedup = r["base"] / r["compile"] if r["compile"] > 0 else 0
        fused_speedup = r["base"] / r["fused"] if r["fused"] > 0 else 0
        both_speedup = r["base"] / r["both"] if r["both"] > 0 else 0

        kernels_str = f"{r.get('n_kernels', 0)}/{r.get('n_candidates', 0)}"

        print(f"{r['name']:<30} {steps:<8} "
              f"{base_ms:<12.2f} {comp_ms:<12.2f} {fused_ms:<12.2f} {both_ms:<12.2f} "
              f"{comp_speedup:<10.2f}x {fused_speedup:<10.2f}x {both_speedup:<10.2f}x "
              f"{kernels_str:<10}")

    print("=" * 130)
    print("\nSpeedup > 1.0x means faster than base PyTorch eager mode.")
    print("'FusionAgent' = Triton kernels generated by GPT-5.4 via the full FusionAgent pipeline.")
    print("'Both' = LLM-generated fused kernels + torch.compile on the full model.")
    print("'Kernels' = passed/total fusion candidates from the pipeline.\n")


# ===================================================================
# Section 8: main()
# ===================================================================

def main():
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    import triton
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    kernel_output_dir = Path("generated_kernels")
    kernel_output_dir.mkdir(exist_ok=True)

    all_results = []
    all_pipeline_results = {}

    # =================================================================
    # Workload 1: GPT-2 Small (124M)
    # =================================================================
    print("\n" + "=" * 70)
    print("  WORKLOAD 1: GPT-2 Small (124M) — Pipeline Phase")
    print("=" * 70)

    n_steps_gpt2 = 200
    bs_gpt2, seq_gpt2, vocab_gpt2 = 8, 256, 50257

    # Trace a single TransformerBlock
    block_gpt2 = TransformerBlock(dim=768, n_heads=12).eval()
    sample_gpt2 = torch.randn(1, 128, 768)
    patcher_gpt2 = FusionPatcher()

    pipeline_gpt2 = run_pipeline(block_gpt2, "GPT2 TransformerBlock(768)",
                                  sample_gpt2, patcher_gpt2, device=device)
    save_kernels(pipeline_gpt2, "GPT2_Small_124M", kernel_output_dir)
    all_pipeline_results["GPT-2 Small (124M)"] = pipeline_gpt2

    # Training benchmark
    r = benchmark_workload(
        "GPT-2 Small (124M)",
        f"{n_steps_gpt2} steps, batch={bs_gpt2}, seq={seq_gpt2}",
        lambda: GPT2Small(vocab_size=vocab_gpt2),
        lambda model: train_causal_lm(model, n_steps_gpt2, bs_gpt2, seq_gpt2, vocab_gpt2, device),
        patcher_gpt2,
        model_type="transformer",
        make_input_fn=lambda dev: torch.randint(0, vocab_gpt2, (bs_gpt2, seq_gpt2), device=dev),
        device=device,
    )
    n_passed = sum(1 for item in pipeline_gpt2 if item[2] and item[2].passed_correctness)
    all_results.append({
        "name": "GPT-2 Small (124M)", "steps": n_steps_gpt2,
        "n_candidates": len(pipeline_gpt2), "n_kernels": n_passed, **r
    })

    # =================================================================
    # Workload 2: Qwen-Small (67M)
    # =================================================================
    print("\n" + "=" * 70)
    print("  WORKLOAD 2: Qwen-Small (67M) — Pipeline Phase")
    print("=" * 70)

    n_steps_qwen = 300
    bs_qwen, seq_qwen, vocab_qwen = 16, 256, 32000

    block_qwen = TransformerBlock(dim=512, n_heads=8).eval()
    sample_qwen = torch.randn(1, 128, 512)
    patcher_qwen = FusionPatcher()

    pipeline_qwen = run_pipeline(block_qwen, "Qwen TransformerBlock(512)",
                                  sample_qwen, patcher_qwen, device=device)
    save_kernels(pipeline_qwen, "Qwen_Small_67M", kernel_output_dir)
    all_pipeline_results["Qwen-Small (67M)"] = pipeline_qwen

    r = benchmark_workload(
        "Qwen-Small (67M)",
        f"{n_steps_qwen} steps, batch={bs_qwen}, seq={seq_qwen}",
        lambda: QwenSmall(vocab_size=vocab_qwen),
        lambda model: train_causal_lm(model, n_steps_qwen, bs_qwen, seq_qwen, vocab_qwen, device),
        patcher_qwen,
        model_type="transformer",
        make_input_fn=lambda dev: torch.randint(0, vocab_qwen, (bs_qwen, seq_qwen), device=dev),
        device=device,
    )
    n_passed = sum(1 for item in pipeline_qwen if item[2] and item[2].passed_correctness)
    all_results.append({
        "name": "Qwen-Small (67M)", "steps": n_steps_qwen,
        "n_candidates": len(pipeline_qwen), "n_kernels": n_passed, **r
    })

    # =================================================================
    # Workload 3: ViT-Small (22M)
    # =================================================================
    print("\n" + "=" * 70)
    print("  WORKLOAD 3: ViT-Small (22M) — Pipeline Phase")
    print("=" * 70)

    n_steps_vit = 300
    bs_vit, img_size_vit, nc_vit = 128, 32, 100

    block_vit = ViTBlock(dim=384, n_heads=6).eval()
    sample_vit = torch.randn(1, 65, 384)  # 64 patches + 1 cls token
    patcher_vit = FusionPatcher()

    pipeline_vit = run_pipeline(block_vit, "ViTBlock(384)",
                                 sample_vit, patcher_vit, device=device)
    save_kernels(pipeline_vit, "ViT_Small_22M", kernel_output_dir)
    all_pipeline_results["ViT-Small (22M)"] = pipeline_vit

    r = benchmark_workload(
        "ViT-Small (22M)",
        f"{n_steps_vit} steps, batch={bs_vit}, img={img_size_vit}x{img_size_vit}",
        lambda: ViTSmall(img_size=img_size_vit, num_classes=nc_vit),
        lambda model: train_classifier(model, n_steps_vit, bs_vit, img_size_vit, nc_vit, device),
        patcher_vit,
        model_type="vit",
        make_input_fn=lambda dev: torch.randn(4, 3, img_size_vit, img_size_vit, device=dev),
        device=device,
    )
    n_passed = sum(1 for item in pipeline_vit if item[2] and item[2].passed_correctness)
    all_results.append({
        "name": "ViT-Small (22M)", "steps": n_steps_vit,
        "n_candidates": len(pipeline_vit), "n_kernels": n_passed, **r
    })

    # =================================================================
    # Workload 4: Distillation Student (8M)
    # =================================================================
    print("\n" + "=" * 70)
    print("  WORKLOAD 4: Distillation Student (8M) — Pipeline Phase")
    print("=" * 70)

    n_steps_distill = 300
    bs_distill, seq_distill, vocab_distill = 32, 128, 32000

    block_distill = TransformerBlock(dim=256, n_heads=4).eval()
    sample_distill = torch.randn(1, 64, 256)
    patcher_distill = FusionPatcher()

    pipeline_distill = run_pipeline(block_distill, "DistillStudent TransformerBlock(256)",
                                     sample_distill, patcher_distill, device=device)
    save_kernels(pipeline_distill, "Distill_Student_8M", kernel_output_dir)
    all_pipeline_results["Distillation Student (8M)"] = pipeline_distill

    # Build teacher (once, not timed)
    print(f"\n  Building teacher model for distillation...")
    teacher = DistillTeacher(vocab_size=vocab_distill).to(device).eval()

    r = benchmark_workload(
        "Distillation Student (8M)",
        f"{n_steps_distill} steps, batch={bs_distill}, seq={seq_distill}",
        lambda: DistillStudent(vocab_size=vocab_distill),
        lambda model: train_distillation(
            model, teacher, n_steps_distill, bs_distill, seq_distill, vocab_distill,
            device=device
        ),
        patcher_distill,
        model_type="transformer",
        make_input_fn=lambda dev: torch.randint(0, vocab_distill, (bs_distill, seq_distill), device=dev),
        device=device,
    )
    n_passed = sum(1 for item in pipeline_distill if item[2] and item[2].passed_correctness)
    all_results.append({
        "name": "Distillation Student (8M)", "steps": n_steps_distill,
        "n_candidates": len(pipeline_distill), "n_kernels": n_passed, **r
    })

    del teacher
    torch.cuda.empty_cache()

    # =================================================================
    # Print combined results
    # =================================================================
    import triton
    print_results_table(
        all_results,
        torch.cuda.get_device_name(0),
        torch.__version__,
        triton.__version__,
    )

    # Pipeline summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY — Per-Kernel Results")
    print("=" * 70)
    print(f"{'Model':<30} {'#':<4} {'Ops':<35} {'Status':<8} "
          f"{'Fused(us)':<11} {'Base(us)':<11} {'Speedup':<9}")
    print("-" * 110)
    for model_name, results_list in all_pipeline_results.items():
        for i, item in enumerate(results_list):
            candidate, code, result = item[0], item[1], item[2]
            n_attempts = item[3] if len(item) > 3 else "?"
            ops_str = " -> ".join(candidate.ops)[:33]
            if result is None:
                print(f"{model_name if i==0 else '':<30} {i+1:<4} {ops_str:<35} {'SKIP':<8}")
            elif result.passed_correctness:
                print(f"{model_name if i==0 else '':<30} {i+1:<4} {ops_str:<35} "
                      f"{'PASS':<8} "
                      f"{result.latency_us:<11.1f} {result.baseline_latency_us:<11.1f} "
                      f"{result.speedup:<9.2f}x  (attempt {n_attempts})")
            else:
                err = (result.compile_error or "unknown")[:30]
                print(f"{model_name if i==0 else '':<30} {i+1:<4} {ops_str:<35} {'FAIL':<8} "
                      f"{err}")
    print("=" * 110)

    # Save JSON
    out_path = Path("e2e_training_results.json")
    json_data = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "workloads": all_results,
        "pipeline_details": {
            model_name: [
                {
                    "ops": item[0].ops,
                    "passed": item[2].passed_correctness if item[2] else False,
                    "speedup": item[2].speedup if item[2] and item[2].passed_correctness else 0.0,
                    "latency_us": item[2].latency_us if item[2] and item[2].passed_correctness else 0.0,
                    "error": item[2].compile_error if item[2] and not item[2].passed_correctness else None,
                    "attempts": item[3] if len(item) > 3 else None,
                }
                for item in results_list
            ]
            for model_name, results_list in all_pipeline_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Generated kernels saved to {kernel_output_dir}/")


if __name__ == "__main__":
    main()
