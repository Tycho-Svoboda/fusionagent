"""Kernel-level RL efficacy benchmark: one-shot generation vs measured search."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmark_comparison import GELU_ADD_CODE, RELU_ADD_CODE, SILU_MUL_CODE
from fusionagent.generator.codegen import KernelGenerator
from fusionagent.harness.benchmark import BenchmarkHarness
from fusionagent.research.retriever import ResearchRetriever
from fusionagent.rl import RLSearchLoop
from fusionagent.types import FusionCandidate, ResearchContext


def _load_local_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

LAYERNORM_CODE = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / N
        x_zm = x - mean
        var = tl.sum(x_zm * x_zm, axis=0) / N
        x_hat = x_zm / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row * N + offs, x_hat * w + b, mask=mask)

    def fused_kernel(x, w, b):
        x = x.contiguous()
        orig_shape = x.shape
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty_like(x)
        BLOCK = max(triton.next_power_of_2(N), 1)
        _layernorm_kernel[(M,)](x.view(M, N), w, b, out.view(M, N), M, N, eps=1e-5, BLOCK=BLOCK)
        return out.view(orig_shape)

    def reference(x, w, b):
        normed = torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)
        return normed

    def make_inputs(*shapes):
        data_shape = shapes[0]
        hidden = data_shape[-1]
        x = torch.randn(data_shape, dtype=torch.float32, device="cuda")
        w = torch.ones(hidden, dtype=torch.float32, device="cuda")
        b = torch.zeros(hidden, dtype=torch.float32, device="cuda")
        return x, w, b
""")

RMSNORM_CODE = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, M, N, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
        sq_mean = tl.sum(x * x, axis=0) / N
        x_hat = x / tl.sqrt(sq_mean + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0)
        tl.store(out_ptr + row * N + offs, x_hat * w, mask=mask)

    def fused_kernel(x, w):
        x = x.contiguous()
        orig_shape = x.shape
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty_like(x)
        BLOCK = max(triton.next_power_of_2(N), 1)
        _rmsnorm_kernel[(M,)](x.view(M, N), w, out.view(M, N), M, N, eps=1e-6, BLOCK=BLOCK)
        return out.view(orig_shape)

    def reference(x, w):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * rms * w

    def make_inputs(*shapes):
        data_shape = shapes[0]
        hidden = data_shape[-1]
        x = torch.randn(data_shape, dtype=torch.float32, device="cuda")
        w = torch.ones(hidden, dtype=torch.float32, device="cuda")
        return x, w
""")


@dataclass
class KernelBenchmarkSpec:
    name: str
    candidate: FusionCandidate
    handwritten_code: str | None = None


def _fallback_context() -> ResearchContext:
    return ResearchContext(
        prior_implementations=[],
        known_pitfalls=[],
        suggested_tile_sizes=[],
        novelty_score=1.0,
    )


def _fmt_latency(latency_us: float) -> str:
    if latency_us <= 0:
        return "n/a"
    return f"{latency_us:.1f} us"


def _fmt_speedup(speedup: float) -> str:
    if speedup <= 0:
        return "n/a"
    return f"{speedup:.2f}x"


def _geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _build_specs() -> list[KernelBenchmarkSpec]:
    return [
        KernelBenchmarkSpec(
            name="relu + add",
            candidate=FusionCandidate(
                ops=["relu", "add"],
                input_shapes=[(1024 * 1024,), (1024 * 1024,)],
                output_shape=(1024 * 1024,),
                memory_bound=True,
                launch_overhead_us=2.5,
                graph_position=0,
            ),
            handwritten_code=RELU_ADD_CODE,
        ),
        KernelBenchmarkSpec(
            name="silu * mul",
            candidate=FusionCandidate(
                ops=["silu", "mul"],
                input_shapes=[(1024 * 1024,), (1024 * 1024,)],
                output_shape=(1024 * 1024,),
                memory_bound=True,
                launch_overhead_us=2.5,
                graph_position=0,
            ),
            handwritten_code=SILU_MUL_CODE,
        ),
        KernelBenchmarkSpec(
            name="gelu + add",
            candidate=FusionCandidate(
                ops=["gelu", "add"],
                input_shapes=[(1024 * 1024,), (1024 * 1024,)],
                output_shape=(1024 * 1024,),
                memory_bound=True,
                launch_overhead_us=2.5,
                graph_position=0,
            ),
            handwritten_code=GELU_ADD_CODE,
        ),
        KernelBenchmarkSpec(
            name="layernorm",
            candidate=FusionCandidate(
                ops=["layernorm"],
                input_shapes=[(32, 512)],
                output_shape=(32, 512),
                memory_bound=True,
                launch_overhead_us=5.0,
                graph_position=0,
            ),
            handwritten_code=LAYERNORM_CODE,
        ),
        KernelBenchmarkSpec(
            name="rmsnorm",
            candidate=FusionCandidate(
                ops=["rmsnorm"],
                input_shapes=[(32, 512)],
                output_shape=(32, 512),
                memory_bound=True,
                launch_overhead_us=5.0,
                graph_position=0,
            ),
            handwritten_code=RMSNORM_CODE,
        ),
    ]


def _retrieve_context(
    retriever: ResearchRetriever,
    candidate: FusionCandidate,
) -> ResearchContext:
    try:
        return retriever.retrieve(candidate)
    except Exception:
        return _fallback_context()


def _write_markdown(path: Path, rows: list[dict], geomean_uplift: float) -> None:
    lines = [
        "# RL Kernel Benchmark Results",
        "",
        "| Pattern | Base | OneShot | RL | HandWritten | RL uplift vs OneShot |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    _fmt_latency(row["base_latency_us"]),
                    f'{_fmt_latency(row["one_shot_latency_us"])} ({_fmt_speedup(row["one_shot_speedup"])})',
                    f'{_fmt_latency(row["rl_latency_us"])} ({_fmt_speedup(row["rl_speedup"])})',
                    row["handwritten_display"],
                    _fmt_speedup(row["rl_uplift_vs_one_shot"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Geometric mean RL uplift vs OneShot: {_fmt_speedup(geomean_uplift)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=".fusionagent_cache/benchmarks/rl_kernels")
    parser.add_argument("--search-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_rl_kernels.py requires CUDA")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    harness = BenchmarkHarness(device=args.device)
    generator = KernelGenerator(model=args.model)
    retriever = ResearchRetriever()
    specs = _build_specs()

    results: list[dict] = []
    uplift_values: list[float] = []
    cache_root = Path(args.cache_dir)

    for spec in specs:
        print(f"Benchmarking {spec.name}...", flush=True)
        context = _retrieve_context(retriever, spec.candidate)

        one_shot = RLSearchLoop(
            generator,
            harness,
            cache_dir=str(cache_root / "oneshot"),
        ).run(
            spec.candidate,
            context,
            n_rounds=1,
            candidates_per_round=1,
            top_k=1,
        )
        rl = RLSearchLoop(
            generator,
            harness,
            cache_dir=str(cache_root / "rl"),
        ).run(
            spec.candidate,
            context,
            n_rounds=args.search_rounds,
            candidates_per_round=args.candidates_per_round,
            top_k=args.top_k,
        )

        handwritten_result = (
            harness.evaluate(spec.handwritten_code, spec.candidate)
            if spec.handwritten_code is not None
            else None
        )

        base_latency = 0.0
        for winner in (rl.winner_result, one_shot.winner_result, handwritten_result):
            if winner is not None and winner.baseline_latency_us > 0:
                base_latency = winner.baseline_latency_us
                break

        rl_uplift = 0.0
        if one_shot.best_speedup > 0 and rl.best_speedup > 0:
            rl_uplift = rl.best_speedup / one_shot.best_speedup
            uplift_values.append(rl_uplift)

        handwritten_display = "n/a"
        if handwritten_result is not None:
            handwritten_display = (
                f'{_fmt_latency(handwritten_result.latency_us)} '
                f'({_fmt_speedup(handwritten_result.speedup)})'
            )

        results.append(
            {
                "name": spec.name,
                "base_latency_us": base_latency,
                "one_shot_latency_us": one_shot.winner_result.latency_us if one_shot.winner_result else 0.0,
                "one_shot_speedup": one_shot.best_speedup,
                "rl_latency_us": rl.winner_result.latency_us if rl.winner_result else 0.0,
                "rl_speedup": rl.best_speedup,
                "handwritten_latency_us": handwritten_result.latency_us if handwritten_result else 0.0,
                "handwritten_speedup": handwritten_result.speedup if handwritten_result else 0.0,
                "handwritten_display": handwritten_display,
                "rl_uplift_vs_one_shot": rl_uplift,
                "one_shot_confirmed": one_shot.winner_correctness_confirmed,
                "rl_confirmed": rl.winner_correctness_confirmed,
            }
        )
        print(
            f"Completed {spec.name}: one_shot={_fmt_speedup(one_shot.best_speedup)} "
            f"rl={_fmt_speedup(rl.best_speedup)} "
            f"confirmed={rl.winner_correctness_confirmed}",
            flush=True,
        )

    geomean_uplift = _geometric_mean(uplift_values)
    payload = {
        "model": args.model,
        "device": args.device,
        "search_rounds": args.search_rounds,
        "candidates_per_round": args.candidates_per_round,
        "top_k": args.top_k,
        "results": results,
        "geometric_mean_rl_uplift_vs_one_shot": geomean_uplift,
    }

    json_path = Path("rl_kernel_benchmark_results.json")
    md_path = Path("rl_kernel_benchmark_results.md")
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(md_path, results, geomean_uplift)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Geometric mean RL uplift vs OneShot: {_fmt_speedup(geomean_uplift)}", flush=True)


if __name__ == "__main__":
    main()
