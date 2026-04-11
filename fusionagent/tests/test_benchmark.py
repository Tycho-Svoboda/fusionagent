"""Tests for fusionagent.harness.benchmark — BenchmarkHarness.

CPU-only tests (TestBuildShapeMatrix, TestLoadModule) run without a GPU.
GPU tests are marked with ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from fusionagent.harness.benchmark import (
    BATCH_SIZES,
    SEQ_LENS,
    _build_shape_matrix,
    _cleanup_module,
    _load_module_from_path,
    _write_kernel_file,
)
from fusionagent.types import BenchmarkResult, FusionCandidate

# ---------------------------------------------------------------------------
# Kernel code fixtures
# ---------------------------------------------------------------------------

GOOD_VECTOR_ADD_CODE = textwrap.dedent("""\
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
        grid = ((n + BLOCK - 1) // BLOCK,)
        _add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
        return out

    def reference(x, y):
        return x + y
""")

GOOD_RELU_ADD_CODE = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _relu_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        result = tl.maximum(x, 0.0) + y
        tl.store(out_ptr + offs, result, mask=mask)

    def fused_kernel(x, y):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _relu_add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
        return out

    def reference(x, y):
        return torch.relu(x) + y
""")

BAD_SYNTAX_CODE = textwrap.dedent("""\
    def fused_kernel(x, y):
        return x +

    def reference(x, y):
        return x + y
""")

BAD_OUTPUT_CODE = textwrap.dedent("""\
    import torch

    def fused_kernel(x, y):
        return x + y + 999.0

    def reference(x, y):
        return x + y
""")

BAD_MISSING_EXPORT_CODE = textwrap.dedent("""\
    import torch

    def my_kernel(x, y):
        return x + y

    def reference(x, y):
        return x + y
""")

BAD_NAN_CODE = textwrap.dedent("""\
    import torch

    def fused_kernel(x, y):
        return torch.full_like(x, float('nan'))

    def reference(x, y):
        return x + y
""")

# Simple valid code for CPU-only load tests (no Triton dependency)
_VALID_SIMPLE_CODE = textwrap.dedent("""\
    def fused_kernel(x, y):
        return x + y

    def reference(x, y):
        return x + y
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(**overrides) -> FusionCandidate:
    """Create a FusionCandidate with sensible defaults."""
    defaults = dict(
        ops=["add"],
        input_shapes=[(4, 1024)],
        output_shape=(4, 1024),
        memory_bound=True,
        launch_overhead_us=3.5,
        graph_position=0,
    )
    defaults.update(overrides)
    return FusionCandidate(**defaults)


# ---------------------------------------------------------------------------
# CPU-only tests: TestBuildShapeMatrix
# ---------------------------------------------------------------------------

class TestBuildShapeMatrix:
    """Tests for _build_shape_matrix (no GPU required)."""

    def test_3d_cartesian_product(self):
        """3D template → cartesian product of batch x seqlen = 9 entries."""
        shapes = [(2, 64, 128)]
        result = _build_shape_matrix(shapes)
        assert len(result) == len(BATCH_SIZES) * len(SEQ_LENS)

    def test_2d_batch_only(self):
        """2D template → vary batch only = 3 entries."""
        shapes = [(4, 256)]
        result = _build_shape_matrix(shapes)
        assert len(result) == len(BATCH_SIZES)

    def test_1d_no_variation(self):
        """1D template → no variation, single entry."""
        shapes = [(1024,)]
        result = _build_shape_matrix(shapes)
        assert len(result) == 1
        assert result[0] == [(1024,)]

    def test_empty_shapes(self):
        """Empty input → empty output."""
        result = _build_shape_matrix([])
        assert result == []

    def test_multiple_inputs(self):
        """Multiple input shapes should all be varied together."""
        shapes = [(2, 64, 128), (2, 64, 128)]
        result = _build_shape_matrix(shapes)
        assert len(result) == 9
        # Each entry should have 2 shapes
        for entry in result:
            assert len(entry) == 2
            # Both shapes should match
            assert entry[0] == entry[1]

    def test_expected_batch_values(self):
        """2D template should produce shapes with the expected batch values."""
        shapes = [(8, 256)]
        result = _build_shape_matrix(shapes)
        produced_batches = [entry[0][0] for entry in result]
        assert produced_batches == BATCH_SIZES


# ---------------------------------------------------------------------------
# CPU-only tests: TestLoadModule
# ---------------------------------------------------------------------------

class TestLoadModule:
    """Tests for _write_kernel_file / _load_module_from_path (no GPU required)."""

    def test_syntax_error_raises(self):
        """SyntaxError in kernel code should propagate."""
        mod_name = f"_test_syntax_{id(self)}"
        path = _write_kernel_file(BAD_SYNTAX_CODE, mod_name)
        try:
            with pytest.raises(SyntaxError):
                _load_module_from_path(mod_name, path)
        finally:
            _cleanup_module(mod_name, path)

    def test_missing_export_raises(self):
        """Missing fused_kernel should raise ImportError."""
        mod_name = f"_test_missing_{id(self)}"
        path = _write_kernel_file(BAD_MISSING_EXPORT_CODE, mod_name)
        try:
            with pytest.raises(ImportError, match="fused_kernel"):
                _load_module_from_path(mod_name, path)
        finally:
            _cleanup_module(mod_name, path)

    def test_valid_code_loads(self):
        """Valid code with both exports should load successfully."""
        mod_name = f"_test_valid_{id(self)}"
        path = _write_kernel_file(_VALID_SIMPLE_CODE, mod_name)
        try:
            mod = _load_module_from_path(mod_name, path)
            assert callable(mod.fused_kernel)
            assert callable(mod.reference)
        finally:
            _cleanup_module(mod_name, path)


# ---------------------------------------------------------------------------
# GPU tests: TestBenchmarkHarnessCorrectKernel
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBenchmarkHarnessCorrectKernel:
    """GPU tests for correct kernels."""

    @pytest.fixture(autouse=True)
    def _setup_harness(self):
        from fusionagent.harness.benchmark import BenchmarkHarness
        self.harness = BenchmarkHarness(device="cuda:0")

    def test_vector_add_passes(self):
        """Trivially correct vector-add kernel should pass."""
        candidate = _make_candidate(
            ops=["add"],
            input_shapes=[(1024,), (1024,)],
            output_shape=(1024,),
        )
        result = self.harness.evaluate(GOOD_VECTOR_ADD_CODE, candidate)
        assert result.passed_correctness
        assert result.compile_error is None
        assert result.latency_us > 0
        assert result.baseline_latency_us > 0

    def test_relu_add_passes(self):
        """Fused relu+add kernel should pass correctness."""
        candidate = _make_candidate(
            ops=["relu", "add"],
            input_shapes=[(1024,), (1024,)],
            output_shape=(1024,),
        )
        result = self.harness.evaluate(GOOD_RELU_ADD_CODE, candidate)
        assert result.passed_correctness
        assert result.compile_error is None

    def test_speedup_is_sane(self):
        """Speedup should be a positive finite number."""
        candidate = _make_candidate(
            ops=["add"],
            input_shapes=[(1024,), (1024,)],
            output_shape=(1024,),
        )
        result = self.harness.evaluate(GOOD_VECTOR_ADD_CODE, candidate)
        assert result.speedup > 0
        assert result.speedup < 1000  # sanity upper bound


# ---------------------------------------------------------------------------
# GPU tests: TestBenchmarkHarnessBrokenKernels
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBenchmarkHarnessBrokenKernels:
    """GPU tests for broken / incorrect kernels."""

    @pytest.fixture(autouse=True)
    def _setup_harness(self):
        from fusionagent.harness.benchmark import BenchmarkHarness
        self.harness = BenchmarkHarness(device="cuda:0")

    def _make_default_candidate(self):
        return _make_candidate(
            ops=["add"],
            input_shapes=[(1024,), (1024,)],
            output_shape=(1024,),
        )

    def test_syntax_error(self):
        """Syntax error should be caught, not raised."""
        result = self.harness.evaluate(BAD_SYNTAX_CODE, self._make_default_candidate())
        assert not result.passed_correctness
        assert result.compile_error is not None
        assert "SyntaxError" in result.compile_error

    def test_wrong_output(self):
        """Numerically wrong output should fail correctness."""
        result = self.harness.evaluate(BAD_OUTPUT_CODE, self._make_default_candidate())
        assert not result.passed_correctness
        assert result.max_abs_error > 0

    def test_nan_output(self):
        """NaN output should fail correctness."""
        result = self.harness.evaluate(BAD_NAN_CODE, self._make_default_candidate())
        assert not result.passed_correctness
        assert "NaN" in (result.compile_error or "")

    def test_missing_export(self):
        """Missing fused_kernel export should be caught."""
        result = self.harness.evaluate(
            BAD_MISSING_EXPORT_CODE, self._make_default_candidate(),
        )
        assert not result.passed_correctness
        assert "fused_kernel" in (result.compile_error or "")

    def test_kernel_code_always_populated(self):
        """kernel_code field should always contain the source, even on failure."""
        result = self.harness.evaluate(BAD_SYNTAX_CODE, self._make_default_candidate())
        assert result.kernel_code == BAD_SYNTAX_CODE


# ---------------------------------------------------------------------------
# GPU tests: TestBenchmarkHarnessShapeVariation
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBenchmarkHarnessShapeVariation:
    """GPU test for 3D candidate shape matrix."""

    def test_3d_candidate_multiple_shapes(self):
        """A 3D candidate should exercise the full shape matrix."""
        from fusionagent.harness.benchmark import BenchmarkHarness
        harness = BenchmarkHarness(device="cuda:0")

        candidate = _make_candidate(
            ops=["add"],
            input_shapes=[(2, 64, 128), (2, 64, 128)],
            output_shape=(2, 64, 128),
        )
        result = harness.evaluate(GOOD_VECTOR_ADD_CODE, candidate)
        # Should pass across all 9 shape variants
        assert result.passed_correctness
        assert result.compile_error is None


# ---------------------------------------------------------------------------
# GPU tests: TestBenchmarkHarnessPerformance
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBenchmarkHarnessPerformance:
    """GPU tests for performance measurement sanity."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fusionagent.harness.benchmark import BenchmarkHarness
        self.harness = BenchmarkHarness(device="cuda:0")
        self.candidate = _make_candidate(
            ops=["add"],
            input_shapes=[(1024,), (1024,)],
            output_shape=(1024,),
        )
        self.result = self.harness.evaluate(GOOD_VECTOR_ADD_CODE, self.candidate)

    def test_latency_positive(self):
        """Latency measurements should be positive."""
        assert self.result.latency_us > 0
        assert self.result.baseline_latency_us > 0

    def test_std_reasonable_fraction(self):
        """Std should be a non-negative fraction of latency (< 100%)."""
        assert self.result.latency_std_us >= 0
        if self.result.latency_us > 0:
            ratio = self.result.latency_std_us / self.result.latency_us
            assert ratio < 1.0, f"std/latency ratio {ratio:.2f} >= 1.0"
