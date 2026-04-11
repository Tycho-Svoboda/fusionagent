"""Benchmark harness for evaluating generated Triton kernels.

Loads a kernel_code string (complete Python file exporting ``fused_kernel``
and ``reference`` functions), checks correctness across a shape matrix,
measures latency via ``triton.testing.do_bench``, and returns a
:class:`BenchmarkResult`.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from fusionagent.types import BenchmarkResult, FusionCandidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Correctness tolerances by dtype
_TOLERANCES = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float64: {"atol": 1e-4, "rtol": 1e-4},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}
_DEFAULT_TOL = {"atol": 1e-4, "rtol": 1e-4}

# do_bench parameters
_WARMUP = 25
_REP = 100
_QUANTILES = [0.5, 0.2, 0.8]

# Shape variation
BATCH_SIZES = [1, 4, 32]
SEQ_LENS = [1, 128, 1337]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_kernel_file(kernel_code: str, mod_name: str) -> Path:
    """Write *kernel_code* to a temp file and return its path."""
    tmp_dir = Path(tempfile.gettempdir()) / "fusionagent_kernels"
    tmp_dir.mkdir(exist_ok=True)
    file_path = tmp_dir / f"{mod_name}.py"
    file_path.write_text(kernel_code)
    return file_path


def _load_module_from_path(mod_name: str, file_path: Path):
    """Import the module at *file_path* and validate it exports required functions.

    Raises ``ImportError`` if ``fused_kernel`` or ``reference`` are missing.
    Raises ``SyntaxError`` for syntax problems in the source.
    """
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # may raise SyntaxError / ImportError

    for attr in ("fused_kernel", "reference"):
        if not hasattr(mod, attr) or not callable(getattr(mod, attr)):
            raise ImportError(
                f"Kernel module missing required callable '{attr}'"
            )
    return mod


def _cleanup_module(mod_name: str, file_path: Path) -> None:
    """Remove module from sys.modules and delete the temp file."""
    sys.modules.pop(mod_name, None)
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


def _build_shape_matrix(
    template_shapes: List[tuple],
) -> List[List[tuple]]:
    """Generate a list of shape-sets by varying batch/seqlen dimensions.

    Each entry in the returned list is a list of shapes (one per input tensor).
    For 3D+ templates we do a cartesian product over batch and seqlen; for 2D
    we vary only batch; for 1D/0D we return the template as-is.
    """
    if not template_shapes:
        return []

    first = template_shapes[0]
    ndim = len(first)

    if ndim >= 3:
        # Cartesian product of batch_sizes x seq_lens
        result = []
        for b in BATCH_SIZES:
            for s in SEQ_LENS:
                shapes = []
                for tmpl in template_shapes:
                    new_shape = (b, s) + tmpl[2:]
                    shapes.append(new_shape)
                result.append(shapes)
        return result

    if ndim == 2:
        # Vary batch dim only
        result = []
        for b in BATCH_SIZES:
            shapes = []
            for tmpl in template_shapes:
                new_shape = (b,) + tmpl[1:]
                shapes.append(new_shape)
            result.append(shapes)
        return result

    # 1D or scalar — just use the template as-is
    return [list(template_shapes)]


def _make_inputs(
    shapes: List[tuple],
    dtype: torch.dtype,
    device: str,
) -> List[torch.Tensor]:
    """Create random tensors for the given shapes."""
    return [torch.randn(s, dtype=dtype, device=device) for s in shapes]


def _make_inputs_from_module(
    mod,
    shapes: List[tuple],
    dtype: torch.dtype,
    device: str,
) -> List[torch.Tensor]:
    """Use the module's ``make_inputs`` if available, else fall back to ``_make_inputs``.

    ``mod.make_inputs(*shapes)`` is expected to return a tuple/list of all
    tensors the kernel needs (data + weights + biases), already on *device*.
    """
    if hasattr(mod, "make_inputs") and callable(mod.make_inputs):
        tensors = mod.make_inputs(*shapes)
        # Ensure list for consistency
        return list(tensors)
    return _make_inputs(shapes, dtype, device)


def _check_correctness(
    fused_fn: Callable,
    ref_fn: Callable,
    shape_matrix: List[List[tuple]],
    dtype: torch.dtype,
    device: str,
    mod=None,
) -> Tuple[bool, float, Optional[str]]:
    """Run correctness checks across all shapes.

    Returns (passed, max_abs_error, error_message).
    If *mod* is provided and has a ``make_inputs`` callable, it will be used
    to create inputs (including weights/biases) instead of plain random tensors.
    """
    tol = _TOLERANCES.get(dtype, _DEFAULT_TOL)
    max_abs_error = 0.0

    for shapes in shape_matrix:
        inputs = _make_inputs_from_module(mod, shapes, dtype, device) if mod else _make_inputs(shapes, dtype, device)

        try:
            torch.cuda.synchronize(device)
            fused_out = fused_fn(*inputs)
            torch.cuda.synchronize(device)

            ref_out = ref_fn(*inputs)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            return False, float("inf"), f"CUDA OOM with shapes {shapes}"
        except RuntimeError as exc:
            return False, float("inf"), f"RuntimeError with shapes {shapes}: {exc}"
        except Exception as exc:
            return False, float("inf"), f"Error with shapes {shapes}: {exc}"

        # Check NaN / Inf
        if torch.isnan(fused_out).any():
            return False, float("inf"), f"NaN in fused output with shapes {shapes}"
        if torch.isinf(fused_out).any():
            return False, float("inf"), f"Inf in fused output with shapes {shapes}"

        diff = (fused_out.float() - ref_out.float()).abs()
        cur_max = diff.max().item()
        max_abs_error = max(max_abs_error, cur_max)

        if not torch.allclose(
            fused_out.float(), ref_out.float(),
            atol=tol["atol"], rtol=tol["rtol"],
        ):
            return (
                False,
                max_abs_error,
                f"Correctness failed for shapes {shapes}: "
                f"max_abs_error={cur_max:.6e}",
            )

    return True, max_abs_error, None


def _measure_latency(
    fn: Callable,
    sample_inputs: List[torch.Tensor],
) -> Tuple[float, float]:
    """Measure latency using do_bench. Returns (latency_us, std_us)."""
    from triton.testing import do_bench

    ms_median, ms_p20, ms_p80 = do_bench(
        lambda: fn(*sample_inputs),
        warmup=_WARMUP,
        rep=_REP,
        quantiles=_QUANTILES,
    )

    latency_us = ms_median * 1000.0
    # Approximate std from IQR: std ≈ (p80 - p20) / 1.35
    std_us = (ms_p80 - ms_p20) / 1.35 * 1000.0
    return latency_us, std_us


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """Evaluate generated Triton kernel code for correctness and performance.

    Parameters
    ----------
    device : str
        CUDA device string, e.g. ``"cuda:0"``.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def evaluate(
        self,
        kernel_code: str,
        candidate: FusionCandidate,
    ) -> BenchmarkResult:
        """Evaluate *kernel_code* against *candidate* specification.

        This method **never raises** — all failures are captured in the
        returned :class:`BenchmarkResult` with ``passed_correctness=False``.
        """
        mod_name = f"_fusionagent_kernel_{uuid.uuid4().hex}"
        file_path: Optional[Path] = None

        def _fail(error: str, max_abs_error: float = 0.0) -> BenchmarkResult:
            return BenchmarkResult(
                passed_correctness=False,
                max_abs_error=max_abs_error,
                latency_us=0.0,
                latency_std_us=0.0,
                baseline_latency_us=0.0,
                speedup=0.0,
                compile_error=error,
                kernel_code=kernel_code,
            )

        try:
            # 1. Load module
            try:
                file_path = _write_kernel_file(kernel_code, mod_name)
                mod = _load_module_from_path(mod_name, file_path)
            except SyntaxError as exc:
                return _fail(f"SyntaxError: {exc}")
            except ImportError as exc:
                return _fail(f"ImportError: {exc}")
            except Exception as exc:
                return _fail(f"LoadError: {exc}")

            fused_fn = mod.fused_kernel
            ref_fn = mod.reference

            # 2. Build shape matrix
            shape_matrix = _build_shape_matrix(candidate.input_shapes)
            if not shape_matrix:
                # Fallback: use a single default shape
                shape_matrix = [[(128,)]]

            # 3. Correctness check (fp32)
            dtype = torch.float32
            try:
                passed, max_abs_error, err_msg = _check_correctness(
                    fused_fn, ref_fn, shape_matrix, dtype, self.device,
                    mod=mod,
                )
            except Exception as exc:
                return _fail(f"CorrectnessError: {exc}")

            if not passed:
                return _fail(err_msg or "Correctness check failed", max_abs_error)

            # 4. Warm JIT: one call to trigger Triton compilation
            sample_inputs = _make_inputs_from_module(mod, shape_matrix[0], dtype, self.device)
            try:
                fused_fn(*sample_inputs)
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                return _fail(f"JIT warmup failed: {exc}", max_abs_error)

            # 5. Measure fused kernel latency
            try:
                latency_us, latency_std_us = _measure_latency(
                    fused_fn, sample_inputs,
                )
            except Exception as exc:
                return _fail(f"Benchmark failed: {exc}", max_abs_error)

            # 6. Measure reference baseline latency
            try:
                baseline_latency_us, _ = _measure_latency(
                    ref_fn, sample_inputs,
                )
            except Exception as exc:
                return _fail(f"Baseline benchmark failed: {exc}", max_abs_error)

            # 7. Compute speedup
            if latency_us > 0:
                speedup = baseline_latency_us / latency_us
            else:
                speedup = 0.0

            return BenchmarkResult(
                passed_correctness=True,
                max_abs_error=max_abs_error,
                latency_us=latency_us,
                latency_std_us=latency_std_us,
                baseline_latency_us=baseline_latency_us,
                speedup=speedup,
                compile_error=None,
                kernel_code=kernel_code,
            )

        except Exception as exc:
            # Catch-all: evaluate() never raises
            return _fail(f"Unexpected error: {exc}")

        finally:
            if file_path is not None:
                _cleanup_module(mod_name, file_path)
