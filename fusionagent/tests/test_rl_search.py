"""Tests for fusionagent.rl.search — measured multi-round kernel search."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from fusionagent.harness.benchmark import BenchmarkHarness
from fusionagent.rl.search import RLSearchLoop, _code_hash, _pattern_hash
from fusionagent.types import BenchmarkResult, FusionCandidate, ResearchContext


def _make_candidate(**overrides) -> FusionCandidate:
    defaults = dict(
        ops=["add"],
        input_shapes=[(1024,), (1024,)],
        output_shape=(1024,),
        memory_bound=True,
        launch_overhead_us=3.0,
        graph_position=0,
    )
    defaults.update(overrides)
    return FusionCandidate(**defaults)


def _make_context() -> ResearchContext:
    return ResearchContext(
        prior_implementations=["basic vector add"],
        known_pitfalls=["masking mistakes"],
        suggested_tile_sizes=[(1024,)],
        novelty_score=0.2,
    )


def _make_result(
    kernel_code: str,
    *,
    passed: bool = True,
    speedup: float = 1.2,
    latency_us: float = 10.0,
    latency_std_us: float = 1.0,
    max_abs_error: float = 0.0,
    compile_error: str | None = None,
) -> BenchmarkResult:
    baseline_latency_us = latency_us * speedup if latency_us > 0 else 0.0
    return BenchmarkResult(
        passed_correctness=passed,
        max_abs_error=max_abs_error,
        latency_us=latency_us,
        latency_std_us=latency_std_us,
        baseline_latency_us=baseline_latency_us,
        speedup=speedup,
        compile_error=compile_error,
        kernel_code=kernel_code,
    )


class FakeGenerator:
    def __init__(self, rounds: list[list[str]]):
        self.rounds = rounds
        self.calls: list[dict] = []

    def generate_many(
        self,
        candidate,
        context=None,
        survivors=None,
        n=8,
        temperatures=None,
        feedback=None,
    ) -> list[str]:
        call_index = len(self.calls)
        self.calls.append(
            {
                "candidate": candidate,
                "context": context,
                "survivors": list(survivors or []),
                "n": n,
                "temperatures": list(temperatures or []),
                "feedback": feedback,
            }
        )
        return list(self.rounds[call_index])


class FakeHarness:
    def __init__(
        self,
        results: dict[str, BenchmarkResult],
        confirm_result=(True, 0.0, None),
    ):
        self.results = results
        self.confirm_result = confirm_result
        self.evaluate_calls: list[str] = []
        self.confirm_calls: list[tuple[str, int]] = []

    def evaluate(self, kernel_code: str, candidate: FusionCandidate) -> BenchmarkResult:
        del candidate
        self.evaluate_calls.append(kernel_code)
        return self.results[kernel_code]

    def confirm_correctness(
        self,
        kernel_code: str,
        candidate: FusionCandidate,
        n_trials: int = 100,
    ):
        del candidate
        self.confirm_calls.append((kernel_code, n_trials))
        if callable(self.confirm_result):
            return self.confirm_result(kernel_code, n_trials)
        return self.confirm_result


FAST_VECTOR_ADD_CODE = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    def fused_kernel(x, y):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        _add_kernel[((n + BLOCK - 1) // BLOCK,)](x, y, out, n, BLOCK=BLOCK)
        return out

    def reference(x, y):
        return x + y
""")

SLOW_VECTOR_ADD_CODE = textwrap.dedent("""\
    import torch

    def fused_kernel(x, y):
        out = x + y
        return out.clone()

    def reference(x, y):
        return x + y
""")

BROKEN_VECTOR_ADD_CODE = textwrap.dedent("""\
    import torch

    def fused_kernel(x, y):
        return x + y + 1.0

    def reference(x, y):
        return x + y
""")


class TestRLSearchLoopOffline:
    def test_cache_hit_returns_without_generation_or_evaluation(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        hardware = "test_gpu"
        pattern = _pattern_hash(candidate, hardware)
        cache_dir = tmp_path / "cache"
        cache_pattern_dir = cache_dir / hardware / pattern
        cache_pattern_dir.mkdir(parents=True)
        (cache_pattern_dir / "winner.py").write_text("cached kernel")
        (cache_pattern_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "best_speedup": 1.5,
                    "winner_correctness_confirmed": True,
                    "winner_result": {
                        "passed_correctness": True,
                        "max_abs_error": 0.0,
                        "latency_us": 10.0,
                        "latency_std_us": 1.0,
                        "baseline_latency_us": 15.0,
                        "speedup": 1.5,
                        "compile_error": None,
                    },
                }
            )
        )
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: hardware)

        generator = FakeGenerator(rounds=[["should-not-run"]])
        harness = FakeHarness(results={})
        loop = RLSearchLoop(generator, harness, cache_dir=str(cache_dir))

        result = loop.run(candidate, _make_context())

        assert result.cache_hit is True
        assert result.best_kernel == "cached kernel"
        assert generator.calls == []
        assert harness.evaluate_calls == []
        assert Path(result.log_path).exists()

    def test_dedupes_candidates_and_ranks_by_reward(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        generator = FakeGenerator(rounds=[["dup", "dup", "fast", "broken"]])
        harness = FakeHarness(
            results={
                "dup": _make_result("dup", speedup=0.98),
                "fast": _make_result("fast", speedup=1.6, latency_us=8.0, latency_std_us=0.8),
                "broken": _make_result(
                    "broken",
                    passed=False,
                    speedup=0.0,
                    latency_us=0.0,
                    latency_std_us=0.0,
                    max_abs_error=9.0,
                    compile_error="Correctness failed",
                ),
            }
        )
        loop = RLSearchLoop(generator, harness, cache_dir=str(tmp_path / "cache"))

        result = loop.run(candidate, _make_context(), n_rounds=1, candidates_per_round=4, top_k=2)

        assert harness.evaluate_calls == ["dup", "fast", "broken"]
        assert len(result.all_results) == 3
        rewards_by_hash = {
            record.code_hash: record.reward for record in result.candidate_records
        }
        assert rewards_by_hash[_code_hash("dup")] == 0.0
        assert rewards_by_hash[_code_hash("broken")] == 0.0
        assert rewards_by_hash[_code_hash("fast")] > 0.0
        assert result.best_kernel == "fast"
        assert result.round_summaries[0].n_generated == 4
        assert result.round_summaries[0].n_unique == 3
        assert result.round_summaries[0].n_valid == 2
        assert result.round_summaries[0].survivor_hashes[0] == _code_hash("fast")

    def test_failure_feedback_is_bounded_and_carried_to_next_round(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        round_one = [f"bad_{idx}" for idx in range(6)]
        round_two = ["winner"]
        generator = FakeGenerator(rounds=[round_one, round_two])
        harness = FakeHarness(
            results={
                **{
                    code: _make_result(
                        code,
                        passed=False,
                        speedup=0.0,
                        latency_us=0.0,
                        latency_std_us=0.0,
                        max_abs_error=1.0,
                        compile_error=f"error-{idx}",
                    )
                    for idx, code in enumerate(round_one)
                },
                "winner": _make_result("winner", speedup=1.4, latency_us=9.0, latency_std_us=0.5),
            }
        )
        loop = RLSearchLoop(generator, harness, cache_dir=str(tmp_path / "cache"))

        result = loop.run(candidate, _make_context(), n_rounds=2, candidates_per_round=6, top_k=1)

        assert result.best_kernel == "winner"
        assert len(generator.calls) == 2
        feedback = generator.calls[1]["feedback"]
        assert feedback is not None
        assert "error-0" in feedback
        assert "error-4" in feedback
        assert "error-5" not in feedback
        assert "CRITICAL REQUIREMENTS" in feedback

    def test_confirmation_failure_prevents_cache_write(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        generator = FakeGenerator(rounds=[["winner"]])
        harness = FakeHarness(
            results={"winner": _make_result("winner", speedup=1.25)},
            confirm_result=(False, 42.0, "final mismatch"),
        )
        cache_dir = tmp_path / "cache"
        loop = RLSearchLoop(generator, harness, cache_dir=str(cache_dir))

        result = loop.run(candidate, _make_context(), n_rounds=1, candidates_per_round=1, top_k=1)

        assert result.winner_correctness_confirmed is False
        assert (Path(result.cache_path) / "metadata.json").exists() is False
        log_payload = json.loads(Path(result.log_path).read_text())
        assert log_payload["winner_correctness_confirmed"] is False
        assert log_payload["confirmation_error"] == "final mismatch"

    def test_rl_beats_one_shot_on_deterministic_candidates(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        results = {
            "slow": _make_result("slow", speedup=1.02, latency_us=10.0, latency_std_us=0.5),
            "fast": _make_result("fast", speedup=1.5, latency_us=8.0, latency_std_us=0.3),
        }

        one_shot = RLSearchLoop(
            FakeGenerator(rounds=[["slow"]]),
            FakeHarness(results=results),
            cache_dir=str(tmp_path / "cache_one_shot"),
        ).run(candidate, _make_context(), n_rounds=1, candidates_per_round=1, top_k=1)

        rl = RLSearchLoop(
            FakeGenerator(rounds=[["slow"], ["fast"]]),
            FakeHarness(results=results),
            cache_dir=str(tmp_path / "cache_rl"),
        ).run(candidate, _make_context(), n_rounds=2, candidates_per_round=1, top_k=1)

        assert rl.best_speedup > one_shot.best_speedup

    def test_tie_break_prefers_lower_latency_when_reward_matches(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        generator = FakeGenerator(rounds=[["lat10", "lat20"]])
        harness = FakeHarness(
            results={
                "lat10": _make_result("lat10", speedup=1.2, latency_us=10.0, latency_std_us=1.0),
                "lat20": _make_result("lat20", speedup=1.2, latency_us=20.0, latency_std_us=2.0),
            }
        )
        loop = RLSearchLoop(generator, harness, cache_dir=str(tmp_path / "cache"))

        result = loop.run(candidate, _make_context(), n_rounds=1, candidates_per_round=2, top_k=1)

        assert result.best_kernel == "lat10"

    def test_cache_reuse_after_first_successful_run(self, tmp_path, monkeypatch):
        candidate = _make_candidate()
        monkeypatch.setattr("fusionagent.rl.search._normalize_hardware_name", lambda: "test_gpu")
        cache_dir = tmp_path / "cache"
        first_generator = FakeGenerator(rounds=[["winner"]])
        first_harness = FakeHarness(results={"winner": _make_result("winner", speedup=1.3)})
        loop = RLSearchLoop(first_generator, first_harness, cache_dir=str(cache_dir))
        first = loop.run(candidate, _make_context(), n_rounds=1, candidates_per_round=1, top_k=1)

        second_generator = FakeGenerator(rounds=[["should-not-run"]])
        second_harness = FakeHarness(results={})
        second = RLSearchLoop(
            second_generator,
            second_harness,
            cache_dir=str(cache_dir),
        ).run(candidate, _make_context(), n_rounds=2, candidates_per_round=2, top_k=1)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.best_kernel == "winner"
        assert second_generator.calls == []
        assert second_harness.evaluate_calls == []


@pytest.mark.gpu
def test_rl_search_loop_gpu_smoke(tmp_path):
    candidate = _make_candidate(
        input_shapes=[(1024 * 1024,), (1024 * 1024,)],
        output_shape=(1024 * 1024,),
    )
    generator = FakeGenerator(
        rounds=[[SLOW_VECTOR_ADD_CODE, BROKEN_VECTOR_ADD_CODE, FAST_VECTOR_ADD_CODE]]
    )
    harness = BenchmarkHarness(device="cuda:0")
    loop = RLSearchLoop(generator, harness, cache_dir=str(tmp_path / "cache"))

    result = loop.run(candidate, _make_context(), n_rounds=1, candidates_per_round=3, top_k=2)

    assert result.best_kernel == FAST_VECTOR_ADD_CODE
    assert result.best_speedup > 1.0
    assert result.winner_correctness_confirmed is True
