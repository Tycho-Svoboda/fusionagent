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
class SearchRoundSummary:
    round_index: int
    n_generated: int
    n_unique: int
    n_valid: int
    best_speedup: float
    median_speedup: float
    survivor_hashes: List[str] = field(default_factory=list)


@dataclass
class SearchCandidateRecord:
    round_index: int
    candidate_index: int
    temperature: float
    variation_hint: str
    code_hash: str
    reward: float
    result: Optional[BenchmarkResult] = None


@dataclass
class SearchResult:
    best_kernel: str = ""
    best_speedup: float = 0.0
    speedup_by_round: List[float] = field(default_factory=list)
    all_results: List[BenchmarkResult] = field(default_factory=list)
    winner_correctness_confirmed: bool = False
    candidate: Optional[FusionCandidate] = None
    winner_result: Optional[BenchmarkResult] = None
    round_summaries: List[SearchRoundSummary] = field(default_factory=list)
    candidate_records: List[SearchCandidateRecord] = field(default_factory=list)
    cache_hit: bool = False
    hardware: str = ""
    log_path: Optional[str] = None
    cache_path: Optional[str] = None
