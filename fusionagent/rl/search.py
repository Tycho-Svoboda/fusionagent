"""Reward-driven search loop for Triton kernel generation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import torch

from fusionagent.generator.codegen import _temperature_schedule, _variation_schedule
from fusionagent.types import (
    BenchmarkResult,
    FusionCandidate,
    ResearchContext,
    SearchCandidateRecord,
    SearchResult,
    SearchRoundSummary,
)

_CORRECTNESS_REQUIREMENTS = """\
CRITICAL REQUIREMENTS:
1. make_inputs(*shapes) must return the EXACT same number of tensors that fused_kernel() and reference() accept as positional arguments.
2. If fused_kernel(x, weight) takes 2 args, make_inputs must return 2 tensors.
3. reference() must use plain PyTorch ops to compute the EXACT same result.
4. For RMSNorm: rms = sqrt(mean(x^2) + eps), output = (x / rms) * weight. Do NOT subtract the mean (that's LayerNorm).
5. All tensors must be contiguous and on CUDA.
6. Do NOT use tl.view or tl.reshape with non-power-of-2 dimensions.
Fix ALL of these issues in your new implementation.
"""

_logger = logging.getLogger(__name__)


def _serialize_benchmark_result(
    result: BenchmarkResult | None,
    *,
    include_kernel_code: bool = False,
) -> dict[str, Any] | None:
    """Convert a benchmark result to JSON-safe metadata."""
    if result is None:
        return None

    payload = {
        "passed_correctness": result.passed_correctness,
        "max_abs_error": result.max_abs_error,
        "latency_us": result.latency_us,
        "latency_std_us": result.latency_std_us,
        "baseline_latency_us": result.baseline_latency_us,
        "speedup": result.speedup,
        "compile_error": result.compile_error,
    }
    if include_kernel_code:
        payload["kernel_code"] = result.kernel_code
    return payload


def _deserialize_benchmark_result(payload: dict[str, Any] | None) -> BenchmarkResult | None:
    """Rebuild a BenchmarkResult from serialized metadata."""
    if payload is None:
        return None
    return BenchmarkResult(
        passed_correctness=payload["passed_correctness"],
        max_abs_error=payload["max_abs_error"],
        latency_us=payload["latency_us"],
        latency_std_us=payload["latency_std_us"],
        baseline_latency_us=payload["baseline_latency_us"],
        speedup=payload["speedup"],
        compile_error=payload.get("compile_error"),
        kernel_code=payload.get("kernel_code", ""),
    )


def _candidate_metadata(candidate: FusionCandidate) -> dict[str, Any]:
    """Return JSON-safe candidate metadata used in logs and cache keys."""
    return {
        "ops": list(candidate.ops),
        "input_shapes": [list(shape) for shape in candidate.input_shapes],
        "output_shape": list(candidate.output_shape),
        "memory_bound": candidate.memory_bound,
        "launch_overhead_us": candidate.launch_overhead_us,
        "graph_position": candidate.graph_position,
        "is_cross_model": candidate.is_cross_model,
        "skip_if_grad_checkpoint": candidate.skip_if_grad_checkpoint,
    }


def _normalize_hardware_name() -> str:
    """Return a stable lowercase cache key for the current accelerator."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        name = "cpu"
    lowered = name.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "unknown"


def _pattern_hash(candidate: FusionCandidate, hardware: str) -> str:
    """Hash the full fusion pattern plus hardware identity."""
    payload = {
        "candidate": _candidate_metadata(candidate),
        "hardware": hardware,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _code_hash(code: str) -> str:
    """Return a stable hash for a kernel source string."""
    return hashlib.sha256(code.encode()).hexdigest()


def _stability(result: BenchmarkResult) -> float:
    """Approximate timing stability from median latency and observed variance."""
    if result.latency_us <= 0:
        return 0.0
    return max(0.0, 1.0 - result.latency_std_us / max(result.latency_us, 1e-9))


def _reward(result: BenchmarkResult, min_speedup: float) -> float:
    """Score a benchmark result using speedup and timing stability."""
    if not result.passed_correctness or result.speedup <= min_speedup:
        return 0.0
    return (result.speedup - 1.0) * _stability(result)


def _ranking_tuple(record: SearchCandidateRecord) -> tuple[float, float, float, float, str]:
    """Sort records by reward, then measured runtime quality, then deterministically."""
    result = record.result
    speedup = result.speedup if result is not None else 0.0
    latency = result.latency_us if result is not None and result.latency_us > 0 else float("inf")
    max_abs_error = result.max_abs_error if result is not None else float("inf")
    return (
        -record.reward,
        -speedup,
        latency,
        max_abs_error,
        record.code_hash,
    )


def _aggregate_feedback(records: list[SearchCandidateRecord]) -> str | None:
    """Compress prior-round failures into bounded retry guidance."""
    failures: list[str] = []
    seen: set[str] = set()

    for record in records:
        result = record.result
        if result is None or result.passed_correctness:
            continue
        error = (result.compile_error or "Unknown correctness failure").strip()
        if not error or error in seen:
            continue
        seen.add(error)
        failures.append(error)
        if len(failures) == 5:
            break

    if not failures:
        return None

    parts = [
        f"You have {len(failures)} distinct failure mode(s) to avoid.",
    ]
    for index, failure in enumerate(failures, start=1):
        parts.append(f"--- Failure {index} ---\n{failure}")
    parts.append(_CORRECTNESS_REQUIREMENTS)
    return "\n\n".join(parts)


class RLSearchLoop:
    """Measured multi-round search over LLM-generated Triton kernels."""

    def __init__(
        self,
        generator,
        harness,
        cache_dir: str = ".fusionagent_cache/search",
        max_eval_workers: int = 1,
        min_speedup: float = 1.0,
    ) -> None:
        self.generator = generator
        self.harness = harness
        self.cache_dir = Path(cache_dir)
        self.max_eval_workers = max(1, int(max_eval_workers))
        self.min_speedup = min_speedup

    def _evaluate_codes(
        self,
        unique_codes: list[tuple[str, str]],
        candidate: FusionCandidate,
    ) -> dict[str, BenchmarkResult]:
        """Benchmark one result per unique kernel hash."""
        if not unique_codes:
            return {}

        def _evaluate_one(item: tuple[str, str]) -> tuple[str, BenchmarkResult]:
            code_hash, code = item
            return code_hash, self.harness.evaluate(code, candidate)

        if self.max_eval_workers == 1 or len(unique_codes) == 1:
            return dict(_evaluate_one(item) for item in unique_codes)

        with ThreadPoolExecutor(max_workers=min(self.max_eval_workers, len(unique_codes))) as executor:
            return dict(executor.map(_evaluate_one, unique_codes))

    def _cache_paths(
        self,
        candidate: FusionCandidate,
        hardware: str,
    ) -> tuple[str, Path, Path, Path, Path]:
        """Resolve the per-pattern cache and run artifact paths."""
        pattern = _pattern_hash(candidate, hardware)
        cache_pattern_dir = self.cache_dir / hardware / pattern
        run_dir = Path("runs") / pattern
        return (
            pattern,
            cache_pattern_dir,
            cache_pattern_dir / "winner.py",
            cache_pattern_dir / "metadata.json",
            run_dir,
        )

    def _load_cache(
        self,
        cache_winner_path: Path,
        cache_metadata_path: Path,
    ) -> tuple[str, dict[str, Any]] | None:
        """Load a cached winner if it exists and is reusable."""
        if not cache_winner_path.exists() or not cache_metadata_path.exists():
            return None

        metadata = json.loads(cache_metadata_path.read_text())
        reusable = (
            metadata.get("winner_correctness_confirmed", False)
            and metadata.get("best_speedup", 0.0) > 1.0
        )
        if not reusable:
            return None

        return cache_winner_path.read_text(), metadata

    def _write_run_artifacts(
        self,
        run_dir: Path,
        winner_code: str,
        payload: dict[str, Any],
    ) -> Path:
        """Persist search.json plus the winning kernel source for the current run."""
        run_dir.mkdir(parents=True, exist_ok=True)
        if winner_code:
            (run_dir / "winner.py").write_text(winner_code)
        log_path = run_dir / "search.json"
        log_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return log_path

    def _write_cache_artifacts(
        self,
        cache_pattern_dir: Path,
        cache_winner_path: Path,
        cache_metadata_path: Path,
        winner_code: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a confirmed winner in the local hardware-keyed cache."""
        cache_pattern_dir.mkdir(parents=True, exist_ok=True)
        cache_winner_path.write_text(winner_code)
        cache_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def run(
        self,
        candidate: FusionCandidate,
        context: ResearchContext | None,
        n_rounds: int = 5,
        candidates_per_round: int = 8,
        top_k: int = 3,
    ) -> SearchResult:
        """Run multi-round measured search for a single fusion candidate."""
        hardware = _normalize_hardware_name()
        (
            pattern,
            cache_pattern_dir,
            cache_winner_path,
            cache_metadata_path,
            run_dir,
        ) = self._cache_paths(candidate, hardware)

        cached = self._load_cache(cache_winner_path, cache_metadata_path)
        if cached is not None:
            _logger.info(
                "RL search cache hit pattern=%s ops=%s hardware=%s",
                pattern,
                candidate.ops,
                hardware,
            )
            cached_code, cached_metadata = cached
            winner_result = _deserialize_benchmark_result(cached_metadata.get("winner_result"))
            result = SearchResult(
                best_kernel=cached_code,
                best_speedup=cached_metadata.get("best_speedup", 0.0),
                speedup_by_round=[],
                all_results=[winner_result] if winner_result is not None else [],
                winner_correctness_confirmed=cached_metadata.get("winner_correctness_confirmed", False),
                candidate=candidate,
                winner_result=winner_result,
                round_summaries=[],
                candidate_records=[],
                cache_hit=True,
                hardware=hardware,
                cache_path=str(cache_pattern_dir),
            )
            payload = {
                "candidate": _candidate_metadata(candidate),
                "hardware": hardware,
                "pattern_hash": pattern,
                "cache_hit": True,
                "round_summaries": [],
                "candidate_records": [],
                "winner_summary": _serialize_benchmark_result(winner_result),
                "winner_correctness_confirmed": result.winner_correctness_confirmed,
            }
            log_path = self._write_run_artifacts(run_dir, cached_code, payload)
            result.log_path = str(log_path)
            return result

        survivors: list[str] = []
        all_results: list[BenchmarkResult] = []
        candidate_records: list[SearchCandidateRecord] = []
        round_summaries: list[SearchRoundSummary] = []
        feedback: str | None = None
        code_by_hash: dict[str, str] = {}

        for round_index in range(n_rounds):
            _logger.info(
                "RL search round %d/%d pattern=%s ops=%s",
                round_index + 1,
                n_rounds,
                pattern,
                candidate.ops,
            )
            temperatures = _temperature_schedule(candidates_per_round)
            variation_hints = _variation_schedule(candidates_per_round)
            generated_codes = self.generator.generate_many(
                candidate,
                context=context,
                survivors=survivors,
                n=candidates_per_round,
                temperatures=temperatures,
                feedback=feedback,
            )

            unique_order: list[tuple[str, str]] = []
            seen_hashes: set[str] = set()
            for code in generated_codes:
                digest = _code_hash(code)
                code_by_hash.setdefault(digest, code)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                unique_order.append((digest, code))

            results_by_hash = self._evaluate_codes(unique_order, candidate)
            all_results.extend(results_by_hash.values())

            round_records: list[SearchCandidateRecord] = []
            for candidate_index, code in enumerate(generated_codes):
                digest = _code_hash(code)
                result = results_by_hash[digest]
                record = SearchCandidateRecord(
                    round_index=round_index,
                    candidate_index=candidate_index,
                    temperature=temperatures[candidate_index],
                    variation_hint=variation_hints[candidate_index],
                    code_hash=digest,
                    reward=_reward(result, self.min_speedup),
                    result=result,
                )
                round_records.append(record)
                candidate_records.append(record)

            unique_round_records = []
            unique_round_hashes: set[str] = set()
            for record in round_records:
                if record.code_hash in unique_round_hashes:
                    continue
                unique_round_hashes.add(record.code_hash)
                unique_round_records.append(record)

            valid_records = [
                record for record in unique_round_records
                if record.result is not None and record.result.passed_correctness
            ]
            valid_speedups = [record.result.speedup for record in valid_records if record.result is not None]
            ranked_valid_records = sorted(valid_records, key=_ranking_tuple)
            survivors = [
                next(code for code_hash, code in unique_order if code_hash == record.code_hash)
                for record in ranked_valid_records[:top_k]
            ]

            round_summaries.append(
                SearchRoundSummary(
                    round_index=round_index,
                    n_generated=len(generated_codes),
                    n_unique=len(unique_order),
                    n_valid=len(valid_records),
                    best_speedup=max(valid_speedups, default=0.0),
                    median_speedup=median(valid_speedups) if valid_speedups else 0.0,
                    survivor_hashes=[record.code_hash for record in ranked_valid_records[:top_k]],
                )
            )
            _logger.info(
                "RL round %d complete pattern=%s generated=%d unique=%d valid=%d best_speedup=%.2fx",
                round_index + 1,
                pattern,
                len(generated_codes),
                len(unique_order),
                len(valid_records),
                round_summaries[-1].best_speedup,
            )
            feedback = _aggregate_feedback(unique_round_records)

        unique_final_records = []
        seen_final_hashes: set[str] = set()
        for record in candidate_records:
            if record.code_hash in seen_final_hashes:
                continue
            seen_final_hashes.add(record.code_hash)
            unique_final_records.append(record)

        ranked_final_records = sorted(unique_final_records, key=_ranking_tuple)
        winner_record = ranked_final_records[0] if ranked_final_records else None
        winner_result = winner_record.result if winner_record is not None else None
        winner_code = ""
        if winner_record is not None:
            winner_code = code_by_hash.get(winner_record.code_hash, "")
        if not winner_code and winner_result is not None:
            winner_code = winner_result.kernel_code

        confirmed = False
        confirmation_error: str | None = None
        confirmation_max_abs_error = float("inf")
        if winner_code and winner_result is not None and winner_result.passed_correctness:
            _logger.info(
                "Confirming winner pattern=%s best_speedup=%.2fx",
                pattern,
                winner_result.speedup,
            )
            confirmed, confirmation_max_abs_error, confirmation_error = self.harness.confirm_correctness(
                winner_code,
                candidate,
                n_trials=100,
            )

        best_speedup = winner_result.speedup if winner_result is not None else 0.0
        metadata = {
            "candidate": _candidate_metadata(candidate),
            "hardware": hardware,
            "pattern_hash": pattern,
            "cache_hit": False,
            "round_summaries": [
                {
                    "round_index": summary.round_index,
                    "n_generated": summary.n_generated,
                    "n_unique": summary.n_unique,
                    "n_valid": summary.n_valid,
                    "best_speedup": summary.best_speedup,
                    "median_speedup": summary.median_speedup,
                    "survivor_hashes": summary.survivor_hashes,
                }
                for summary in round_summaries
            ],
            "candidate_records": [
                {
                    "round_index": record.round_index,
                    "candidate_index": record.candidate_index,
                    "temperature": record.temperature,
                    "variation_hint": record.variation_hint,
                    "code_hash": record.code_hash,
                    "reward": record.reward,
                    "result": _serialize_benchmark_result(record.result),
                }
                for record in candidate_records
            ],
            "winner_summary": _serialize_benchmark_result(winner_result),
            "winner_correctness_confirmed": confirmed,
            "confirmation_max_abs_error": confirmation_max_abs_error,
            "confirmation_error": confirmation_error,
        }

        log_path = self._write_run_artifacts(run_dir, winner_code, metadata)

        cache_path = str(cache_pattern_dir)
        if confirmed and winner_code:
            cache_metadata = {
                "candidate": _candidate_metadata(candidate),
                "hardware": hardware,
                "pattern_hash": pattern,
                "best_speedup": best_speedup,
                "winner_correctness_confirmed": True,
                "winner_result": _serialize_benchmark_result(winner_result),
            }
            self._write_cache_artifacts(
                cache_pattern_dir,
                cache_winner_path,
                cache_metadata_path,
                winner_code,
                cache_metadata,
            )
            _logger.info(
                "Cached confirmed winner pattern=%s speedup=%.2fx",
                pattern,
                best_speedup,
            )
        else:
            _logger.info(
                "Winner not cached pattern=%s confirmed=%s error=%s",
                pattern,
                confirmed,
                confirmation_error,
            )

        return SearchResult(
            best_kernel=winner_code,
            best_speedup=best_speedup,
            speedup_by_round=[summary.best_speedup for summary in round_summaries],
            all_results=all_results,
            winner_correctness_confirmed=confirmed,
            candidate=candidate,
            winner_result=winner_result,
            round_summaries=round_summaries,
            candidate_records=candidate_records,
            cache_hit=False,
            hardware=hardware,
            log_path=str(log_path),
            cache_path=cache_path,
        )
