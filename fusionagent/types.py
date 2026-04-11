from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FusionCandidate:
    ops: List[str]
    input_shapes: List[tuple]
    output_shape: tuple
    memory_bound: bool
    launch_overhead_us: float
    graph_position: int
    is_cross_model: bool = False
    skip_if_grad_checkpoint: bool = False


@dataclass
class ResearchContext:
    prior_implementations: List[str]
    known_pitfalls: List[str]
    suggested_tile_sizes: List[tuple]
    novelty_score: float


@dataclass
class BenchmarkResult:
    passed_correctness: bool
    max_abs_error: float
    latency_us: float
    latency_std_us: float
    baseline_latency_us: float
    speedup: float
    compile_error: Optional[str]
    kernel_code: str = ""


@dataclass
class SearchResult:
    best_kernel: str
    best_speedup: float
    speedup_by_round: List[float]
    all_results: List[BenchmarkResult]
    winner_correctness_confirmed: bool
    candidate: FusionCandidate = None
