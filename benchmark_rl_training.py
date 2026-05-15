"""Training-level RL benchmark: safe production patching vs experimental results mode."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusionagent.generator.codegen import KernelGenerator
from fusionagent.graph.analyzer import GraphAnalyzer
from fusionagent.harness.benchmark import BenchmarkHarness
from fusionagent.packager import FusionPatcher, load_kernel_module
from fusionagent.packager.patch import (
    _FusedNormWrapper,
    _FusedSwiGLUWrapper,
    _is_norm_module,
    _is_swiglu_like,
    _is_transformer_like,
    _is_vit_like,
)
from fusionagent.research.retriever import ResearchRetriever
from fusionagent.rl import RLSearchLoop
from fusionagent.types import FusionCandidate, ResearchContext, SearchResult


def _load_local_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _fallback_context() -> ResearchContext:
    return ResearchContext(
        prior_implementations=[],
        known_pitfalls=[],
        suggested_tile_sizes=[],
        novelty_score=1.0,
    )


def _make_tokens(batch: int, seq: int, vocab: int, device: str) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq), device=device)


def _make_images(batch: int, size: int, device: str) -> torch.Tensor:
    return torch.randn(batch, 3, size, size, device=device)


class _StripGELUMLP(nn.Module):
    """Benchmark-only helper that removes GELU layers after a fused norm+GELU rewrite."""

    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        self.base_mlp = base_mlp

    def forward(self, x):
        if not isinstance(self.base_mlp, nn.Sequential):
            return self.base_mlp(x)

        out = x
        for layer in self.base_mlp:
            if isinstance(layer, nn.GELU):
                continue
            out = layer(out)
        return out


class ExperimentalFusionPatcher(FusionPatcher):
    """Benchmark-only semantics-relaxed patcher for historical results-style comparisons."""

    def patch_model(self, model: nn.Module, model_type: str = "auto") -> int:
        self.patch_log = []
        n_patches = 0
        n_patches += self._patch_transformer_like(model)
        n_patches += self._patch_vit_like(model)
        n_patches += self._patch_rwkv_like(model)
        return n_patches

    def _patch_transformer_like(self, model: nn.Module) -> int:
        n_patches = 0
        for name, module in list(model.named_modules()):
            if not _is_transformer_like(module):
                continue

            norm_key = None
            if self.has_pattern("rmsnorm_silu"):
                norm_key = "rmsnorm_silu"
            elif self.has_pattern("rmsnorm"):
                norm_key = "rmsnorm"

            if norm_key is not None:
                _, fused_fn, _ = self.kernels[norm_key]
                if _is_norm_module(module.attn_norm):
                    module.attn_norm = _FusedNormWrapper(module.attn_norm, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.attn_norm with {norm_key}")
                if _is_norm_module(module.ffn_norm):
                    module.ffn_norm = _FusedNormWrapper(module.ffn_norm, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.ffn_norm with {norm_key}")

            if self.has_pattern("silu_mul") and _is_swiglu_like(module.ffn):
                _, fused_fn, _ = self.kernels["silu_mul"]
                module.ffn = _FusedSwiGLUWrapper(module.ffn, fused_fn)
                n_patches += 1
                self.patch_log.append(f"Patched {name}.ffn with silu_mul")

        return n_patches

    def _patch_vit_like(self, model: nn.Module) -> int:
        n_patches = 0
        for name, module in list(model.named_modules()):
            if not _is_vit_like(module):
                continue

            norm_key = None
            if self.has_pattern("layernorm_gelu"):
                norm_key = "layernorm_gelu"
            elif self.has_pattern("layernorm"):
                norm_key = "layernorm"

            if norm_key is None:
                continue

            _, fused_fn, _ = self.kernels[norm_key]
            for attr in ("norm1", "norm2", "norm3"):
                if hasattr(module, attr) and _is_norm_module(getattr(module, attr)):
                    setattr(module, attr, _FusedNormWrapper(getattr(module, attr), fused_fn))
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.{attr} with {norm_key}")

            if norm_key == "layernorm_gelu" and hasattr(module, "mlp"):
                module.mlp = _StripGELUMLP(module.mlp)
                self.patch_log.append(f"Stripped GELU from {name}.mlp for experimental layernorm_gelu")

        return n_patches

    def _patch_rwkv_like(self, model: nn.Module) -> int:
        n_patches = 0
        norm_key = None
        if self.has_pattern("rmsnorm_silu"):
            norm_key = "rmsnorm_silu"
        elif self.has_pattern("rmsnorm"):
            norm_key = "rmsnorm"
        if norm_key is None:
            return 0

        _, fused_fn, _ = self.kernels[norm_key]
        for name, module in list(model.named_modules()):
            if not (hasattr(module, "ln1") and hasattr(module, "ln2")):
                continue
            if _is_norm_module(module.ln1):
                module.ln1 = _FusedNormWrapper(module.ln1, fused_fn)
                n_patches += 1
                self.patch_log.append(f"Patched {name}.ln1 with {norm_key}")
            if _is_norm_module(module.ln2):
                module.ln2 = _FusedNormWrapper(module.ln2, fused_fn)
                n_patches += 1
                self.patch_log.append(f"Patched {name}.ln2 with {norm_key}")
        return n_patches


@dataclass
class ProbeSpec:
    name: str
    extract_block: Callable[[nn.Module], nn.Module]
    sample_input: Callable[[str], Any]


@dataclass
class TrainingWorkload:
    name: str
    description: str
    steps: int
    make_model: Callable[[], nn.Module]
    train_fn: Callable[[nn.Module], float]
    make_input: Callable[[str], Any]
    probes: list[ProbeSpec] = field(default_factory=list)
    model_type: str = "auto"


def _load_canary_models():
    import benchmark_e2e_training as canary_models

    return canary_models


def _load_full_models():
    import benchmark_training as full_models

    return full_models


def _make_patcher(mode: str) -> FusionPatcher:
    if mode == "experimental":
        return ExperimentalFusionPatcher()
    return FusionPatcher()


def _retrieve_context(
    retriever: ResearchRetriever,
    candidate: FusionCandidate,
) -> ResearchContext:
    try:
        return retriever.retrieve(candidate)
    except Exception:
        return _fallback_context()


def _search_and_register(
    patcher: FusionPatcher,
    probe_model: nn.Module,
    probes: list[ProbeSpec],
    *,
    llm_model: str,
    device: str,
    cache_dir: Path,
    n_rounds: int,
    candidates_per_round: int,
    top_k: int,
) -> list[SearchResult]:
    search_results: list[SearchResult] = []
    retriever = ResearchRetriever()
    generator = KernelGenerator(model=llm_model)
    harness = BenchmarkHarness(device=device)
    search_loop = RLSearchLoop(generator, harness, cache_dir=str(cache_dir))
    seen_patterns: set[str] = set()

    for probe in probes:
        print(f"  Probe: {probe.name}", flush=True)
        block = probe.extract_block(probe_model)
        analyzer = GraphAnalyzer(block, sample_input=probe.sample_input("cpu"))
        try:
            candidates = analyzer.analyze()
        except Exception:
            print("    Analyzer failed; skipping probe", flush=True)
            continue

        for candidate in candidates:
            pattern_key = patcher._pattern_key(candidate.ops)
            if pattern_key in seen_patterns:
                continue
            seen_patterns.add(pattern_key)

            context = _retrieve_context(retriever, candidate)
            result = search_loop.run(
                candidate,
                context,
                n_rounds=n_rounds,
                candidates_per_round=candidates_per_round,
                top_k=top_k,
            )
            search_results.append(result)
            print(
                "    Search result "
                f"ops={candidate.ops} cache_hit={result.cache_hit} "
                f"confirmed={result.winner_correctness_confirmed} "
                f"best_speedup={result.best_speedup:.2f}x",
                flush=True,
            )

            if not result.best_kernel or not result.winner_correctness_confirmed:
                continue
            module = load_kernel_module(result.best_kernel)
            if module is None:
                continue
            patcher.register(candidate.ops, result.best_kernel, module.fused_kernel)

    return search_results


def _sanity_check(model: nn.Module, make_input: Callable[[str], Any], device: str) -> bool:
    model.train()
    try:
        inputs = make_input(device)
        output = model(inputs) if not isinstance(inputs, tuple) else model(*inputs)
        loss = output.sum()
        loss.backward()
        return True
    except Exception as exc:
        print(f"    Sanity check failed: {exc}", flush=True)
        return False


def _time_variant(
    workload: TrainingWorkload,
    *,
    base_state: dict[str, torch.Tensor],
    device: str,
    repeats: int,
    patcher: FusionPatcher | None = None,
    compile_model: bool = False,
) -> float:
    times: list[float] = []
    for _ in range(repeats):
        model = workload.make_model().to(device)
        model.load_state_dict(base_state, strict=False)
        if patcher is not None:
            patcher.patch_model(model, model_type=workload.model_type)
            if not _sanity_check(model, workload.make_input, device):
                del model
                torch.cuda.empty_cache()
                return 0.0
        if compile_model:
            model = torch.compile(model)
        times.append(workload.train_fn(model))
        del model
        torch.cuda.empty_cache()
    return sum(times) / len(times)


def _benchmark_workload(
    workload: TrainingWorkload,
    *,
    device: str,
    repeats: int,
    one_shot_patcher: FusionPatcher,
    rl_patcher: FusionPatcher,
) -> dict[str, Any]:
    probe_model = workload.make_model().to(device)
    n_params = sum(parameter.numel() for parameter in probe_model.parameters())
    base_state = copy.deepcopy(probe_model.state_dict())
    del probe_model
    torch.cuda.empty_cache()

    base = _time_variant(workload, base_state=base_state, device=device, repeats=repeats)
    compiled = _time_variant(
        workload,
        base_state=base_state,
        device=device,
        repeats=repeats,
        compile_model=True,
    )
    one_shot = _time_variant(
        workload,
        base_state=base_state,
        device=device,
        repeats=repeats,
        patcher=one_shot_patcher,
    )
    rl = _time_variant(
        workload,
        base_state=base_state,
        device=device,
        repeats=repeats,
        patcher=rl_patcher,
    )
    both = _time_variant(
        workload,
        base_state=base_state,
        device=device,
        repeats=repeats,
        patcher=rl_patcher,
        compile_model=True,
    )

    return {
        "name": workload.name,
        "steps": workload.steps,
        "n_params": n_params,
        "base": base,
        "compile": compiled,
        "one_shot": one_shot if one_shot > 0 else base,
        "rl": rl if rl > 0 else base,
        "both": both if both > 0 else compiled,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Model | Steps | Base | torch.compile | OneShot FA | RL FA | Both (RL + compile) | compile | OneShot | RL | Both | cache hits / candidates confirmed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        steps = row["steps"]
        base_ms = row["base"] / steps * 1000 if row["base"] else 0.0
        compile_ms = row["compile"] / steps * 1000 if row["compile"] else 0.0
        one_shot_ms = row["one_shot"] / steps * 1000 if row["one_shot"] else 0.0
        rl_ms = row["rl"] / steps * 1000 if row["rl"] else 0.0
        both_ms = row["both"] / steps * 1000 if row["both"] else 0.0
        compile_speedup = row["base"] / row["compile"] if row["compile"] else 0.0
        one_shot_speedup = row["base"] / row["one_shot"] if row["one_shot"] else 0.0
        rl_speedup = row["base"] / row["rl"] if row["rl"] else 0.0
        both_speedup = row["base"] / row["both"] if row["both"] else 0.0
        hits_confirmed = (
            f'OS {row["one_shot_cache_hits"]}/{row["one_shot_confirmed"]}, '
            f'RL {row["rl_cache_hits"]}/{row["rl_confirmed"]}'
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    str(steps),
                    f"{base_ms:.2f}",
                    f"{compile_ms:.2f}",
                    f"{one_shot_ms:.2f}",
                    f"{rl_ms:.2f}",
                    f"{both_ms:.2f}",
                    f"{compile_speedup:.2f}x",
                    f"{one_shot_speedup:.2f}x",
                    f"{rl_speedup:.2f}x",
                    f"{both_speedup:.2f}x",
                    hits_confirmed,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _build_canary_workloads(device: str) -> list[TrainingWorkload]:
    canary_models = _load_canary_models()
    teacher = canary_models.DistillTeacher(vocab_size=32000).to(device).eval()

    return [
        TrainingWorkload(
            name="GPT-2 Small (124M)",
            description="200 steps, batch=8, seq=256",
            steps=200,
            make_model=lambda: canary_models.GPT2Small(vocab_size=50257),
            train_fn=lambda model: canary_models.train_causal_lm(model, 200, 8, 256, 50257, device),
            make_input=lambda dev: _make_tokens(2, 128, 50257, dev),
            probes=[
                ProbeSpec(
                    name="transformer",
                    extract_block=lambda model: model.layers[0],
                    sample_input=lambda dev: torch.randn(2, 128, 768, device=dev),
                )
            ],
        ),
        TrainingWorkload(
            name="Qwen-Small (67M)",
            description="300 steps, batch=16, seq=256",
            steps=300,
            make_model=lambda: canary_models.QwenSmall(vocab_size=32000),
            train_fn=lambda model: canary_models.train_causal_lm(model, 300, 16, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[
                ProbeSpec(
                    name="transformer",
                    extract_block=lambda model: model.layers[0],
                    sample_input=lambda dev: torch.randn(2, 128, 512, device=dev),
                )
            ],
        ),
        TrainingWorkload(
            name="ViT-Small (22M)",
            description="300 steps, batch=128, img=32x32",
            steps=300,
            make_model=lambda: canary_models.ViTSmall(img_size=32, num_classes=100),
            train_fn=lambda model: canary_models.train_classifier(model, 300, 128, 32, 100, device),
            make_input=lambda dev: _make_images(2, 32, dev),
            probes=[
                ProbeSpec(
                    name="vit",
                    extract_block=lambda model: model.blocks[0],
                    sample_input=lambda dev: torch.randn(2, 65, 384, device=dev),
                )
            ],
        ),
        TrainingWorkload(
            name="Distillation Student (8M)",
            description="300 steps, batch=32, seq=128",
            steps=300,
            make_model=lambda: canary_models.DistillStudent(vocab_size=32000),
            train_fn=lambda model: canary_models.train_distillation(
                model,
                teacher,
                300,
                32,
                128,
                32000,
                device=device,
            ),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[
                ProbeSpec(
                    name="transformer",
                    extract_block=lambda model: model.layers[0],
                    sample_input=lambda dev: torch.randn(2, 128, 256, device=dev),
                )
            ],
        ),
    ]


def _build_full_workloads(device: str) -> list[TrainingWorkload]:
    full_models = _load_full_models()
    teacher = full_models.DistillTeacher(vocab_size=32000).to(device).eval()

    def train_t5(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        model.train()
        for _ in range(3):
            src = _make_tokens(16, 128, 32000, device)
            tgt = _make_tokens(16, 64, 32000, device)
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, 32000), tgt.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(200):
            src = _make_tokens(16, 128, 32000, device)
            tgt = _make_tokens(16, 64, 32000, device)
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, 32000), tgt.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 1000.0

    def train_clip(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        model.train()
        for _ in range(3):
            imgs = _make_images(64, 32, device)
            txts = _make_tokens(64, 32, 32000, device)
            img_f, txt_f, scale = model(imgs, txts)
            logits = scale * img_f @ txt_f.T
            labels = torch.arange(64, device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(200):
            imgs = _make_images(64, 32, device)
            txts = _make_tokens(64, 32, 32000, device)
            img_f, txt_f, scale = model(imgs, txts)
            logits = scale * img_f @ txt_f.T
            labels = torch.arange(64, device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 1000.0

    return [
        TrainingWorkload(
            name="GPT-2 Small (124M)",
            description="200 steps, batch=8, seq=256",
            steps=200,
            make_model=lambda: full_models.GPT2Small(vocab_size=50257, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 200, 8, 256, 50257, device),
            make_input=lambda dev: _make_tokens(2, 128, 50257, dev),
            probes=[ProbeSpec("transformer", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 768, device=dev))],
        ),
        TrainingWorkload(
            name="Qwen-Small (67M)",
            description="300 steps, batch=16, seq=256",
            steps=300,
            make_model=lambda: full_models.QwenSmall(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 300, 16, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("transformer", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 512, device=dev))],
        ),
        TrainingWorkload(
            name="ViT-Small (22M)",
            description="300 steps, batch=128, img=32x32",
            steps=300,
            make_model=lambda: full_models.ViTSmall(img_size=32, num_classes=100, fused=False),
            train_fn=lambda model: full_models.train_classifier(model, 300, 128, 32, 100, device),
            make_input=lambda dev: _make_images(2, 32, dev),
            probes=[ProbeSpec("vit", lambda model: model.blocks[0], lambda dev: torch.randn(2, 65, 384, device=dev))],
        ),
        TrainingWorkload(
            name="Distillation Student (8M)",
            description="300 steps, batch=32, seq=128",
            steps=300,
            make_model=lambda: full_models.DistillStudent(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_distillation(model, teacher, 300, 32, 128, 32000, device=device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("transformer", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 256, device=dev))],
        ),
        TrainingWorkload(
            name="GQA Transformer (85M)",
            description="200 steps, batch=8, seq=256",
            steps=200,
            make_model=lambda: full_models.GQATransformer(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 200, 8, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("gqa", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 1024, device=dev))],
        ),
        TrainingWorkload(
            name="MoE Transformer (120M)",
            description="200 steps, batch=8, seq=128",
            steps=200,
            make_model=lambda: full_models.MoETransformer(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 200, 8, 128, 32000, device),
            make_input=lambda dev: _make_tokens(2, 64, 32000, dev),
            probes=[ProbeSpec("moe", lambda model: model.layers[0], lambda dev: torch.randn(2, 64, 512, device=dev))],
        ),
        TrainingWorkload(
            name="LoRA Fine-tune (67M + r16)",
            description="300 steps, batch=16, seq=128",
            steps=300,
            make_model=lambda: full_models.LoRAFineTune(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 300, 16, 128, 32000, device),
            make_input=lambda dev: _make_tokens(2, 64, 32000, dev),
            probes=[ProbeSpec("lora", lambda model: model.layers[0], lambda dev: torch.randn(2, 64, 512, device=dev))],
        ),
        TrainingWorkload(
            name="MQA Language Model (55M)",
            description="300 steps, batch=16, seq=256",
            steps=300,
            make_model=lambda: full_models.MQALanguageModel(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 300, 16, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("mqa", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 768, device=dev))],
        ),
        TrainingWorkload(
            name="T5-style Enc-Dec (60M)",
            description="200 steps, batch=16, src=128, tgt=64",
            steps=200,
            make_model=lambda: full_models.T5Style(vocab_size=32000, fused=False),
            train_fn=train_t5,
            make_input=lambda dev: (_make_tokens(2, 128, 32000, dev), _make_tokens(2, 64, 32000, dev)),
            probes=[ProbeSpec("t5-enc", lambda model: model.enc_layers[0], lambda dev: torch.randn(2, 128, 512, device=dev))],
        ),
        TrainingWorkload(
            name="Deep-Narrow Transformer (30M)",
            description="300 steps, batch=32, seq=128",
            steps=300,
            make_model=lambda: full_models.DeepNarrowTransformer(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 300, 32, 128, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("deep", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 256, device=dev))],
        ),
        TrainingWorkload(
            name="CLIP Contrastive (2×22M)",
            description="200 steps, batch=64, img=32x32, seq=32",
            steps=200,
            make_model=lambda: full_models.CLIPStyleModel(vocab_size=32000, fused=False),
            train_fn=train_clip,
            make_input=lambda dev: (_make_images(2, 32, dev), _make_tokens(2, 32, 32000, dev)),
            probes=[
                ProbeSpec("clip-image", lambda model: model.image_enc.blocks[0], lambda dev: torch.randn(2, 65, 384, device=dev)),
                ProbeSpec("clip-text", lambda model: model.text_enc.layers[0], lambda dev: torch.randn(2, 32, 384, device=dev)),
            ],
        ),
        TrainingWorkload(
            name="RWKV-style LM (40M)",
            description="300 steps, batch=16, seq=256",
            steps=300,
            make_model=lambda: full_models.RWKVStyleModel(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 300, 16, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("rwkv", lambda model: model.blocks[0], lambda dev: torch.randn(2, 128, 512, device=dev))],
        ),
        TrainingWorkload(
            name="Mixture-of-Depths (70M)",
            description="200 steps, batch=8, seq=256",
            steps=200,
            make_model=lambda: full_models.MoDTransformer(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 200, 8, 256, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("mod", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 512, device=dev))],
        ),
        TrainingWorkload(
            name="Sliding-Window LM (80M)",
            description="200 steps, batch=8, seq=512",
            steps=200,
            make_model=lambda: full_models.SlidingWindowLM(vocab_size=32000, fused=False),
            train_fn=lambda model: full_models.train_causal_lm(model, 200, 8, 512, 32000, device),
            make_input=lambda dev: _make_tokens(2, 128, 32000, dev),
            probes=[ProbeSpec("sliding", lambda model: model.layers[0], lambda dev: torch.randn(2, 128, 768, device=dev))],
        ),
    ]


def _run_suite(
    workloads: list[TrainingWorkload],
    *,
    device: str,
    llm_model: str,
    cache_root: Path,
    patch_mode: str,
    repeats: int,
    search_rounds: int,
    candidates_per_round: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for workload in workloads:
        print(f"\nRunning workload: {workload.name}", flush=True)
        probe_model = workload.make_model()
        one_shot_patcher = _make_patcher(patch_mode)
        rl_patcher = _make_patcher(patch_mode)

        one_shot_results = _search_and_register(
            one_shot_patcher,
            probe_model,
            workload.probes,
            llm_model=llm_model,
            device=device,
            cache_dir=cache_root / "oneshot" / workload.name.replace(" ", "_"),
            n_rounds=1,
            candidates_per_round=1,
            top_k=1,
        )
        rl_results = _search_and_register(
            rl_patcher,
            probe_model,
            workload.probes,
            llm_model=llm_model,
            device=device,
            cache_dir=cache_root / "rl" / workload.name.replace(" ", "_"),
            n_rounds=search_rounds,
            candidates_per_round=candidates_per_round,
            top_k=top_k,
        )
        del probe_model
        torch.cuda.empty_cache()

        timings = _benchmark_workload(
            workload,
            device=device,
            repeats=repeats,
            one_shot_patcher=one_shot_patcher,
            rl_patcher=rl_patcher,
        )
        timings.update(
            {
                "one_shot_cache_hits": sum(result.cache_hit for result in one_shot_results),
                "one_shot_confirmed": sum(result.winner_correctness_confirmed for result in one_shot_results),
                "rl_cache_hits": sum(result.cache_hit for result in rl_results),
                "rl_confirmed": sum(result.winner_correctness_confirmed for result in rl_results),
                "one_shot_kernels": len(one_shot_patcher.kernels),
                "rl_kernels": len(rl_patcher.kernels),
            }
        )
        rows.append(timings)
        print(
            "Completed workload "
            f"{workload.name}: base={timings['base'] / workload.steps * 1000:.2f} ms/step "
            f"compile={timings['compile'] / workload.steps * 1000:.2f} "
            f"oneshot={timings['one_shot'] / workload.steps * 1000:.2f} "
            f"rl={timings['rl'] / workload.steps * 1000:.2f} "
            f"both={timings['both'] / workload.steps * 1000:.2f}",
            flush=True,
        )

    return rows


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["canary", "full"], default="canary")
    parser.add_argument("--patch-mode", choices=["safe", "experimental"], default="safe")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--search-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--cache-dir", default=".fusionagent_cache/benchmarks/rl_training")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_rl_training.py requires CUDA")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.set_float32_matmul_precision("high")
    cache_root = Path(args.cache_dir) / args.suite / args.patch_mode
    workloads = (
        _build_canary_workloads(args.device)
        if args.suite == "canary"
        else _build_full_workloads(args.device)
    )

    rows = _run_suite(
        workloads,
        device=args.device,
        llm_model=args.model,
        cache_root=cache_root,
        patch_mode=args.patch_mode,
        repeats=args.repeats,
        search_rounds=args.search_rounds,
        candidates_per_round=args.candidates_per_round,
        top_k=args.top_k,
    )

    payload = {
        "suite": args.suite,
        "patch_mode": args.patch_mode,
        "device": args.device,
        "model": args.model,
        "repeats": args.repeats,
        "search_rounds": args.search_rounds,
        "candidates_per_round": args.candidates_per_round,
        "top_k": args.top_k,
        "results": rows,
    }
    json_path = Path(f"rl_training_results_{args.suite}_{args.patch_mode}.json")
    md_path = Path(f"rl_training_results_{args.suite}_{args.patch_mode}.md")
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(
        md_path,
        rows,
        title=f"RL Training Benchmark ({args.suite}, {args.patch_mode})",
    )
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
