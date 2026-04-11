"""LLM-powered Triton kernel code generation.

Calls OpenAI to produce a complete Python file exporting ``fused_kernel``
and ``reference`` functions that the :class:`BenchmarkHarness` can evaluate.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time

from openai import OpenAI

from fusionagent.types import FusionCandidate, ResearchContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gpt-4o"
_MAX_TOKENS = 4096

_SYSTEM_PROMPT = """\
You are a Triton GPU kernel engineer. Respond with ONLY a complete Python file. \
No markdown fences, no explanation, no comments outside the code.

The file MUST define three module-level functions:
  def fused_kernel(*inputs) -> Tensor
  def reference(*inputs) -> Tensor
  def make_inputs(*shapes) -> tuple[Tensor, ...]

fused_kernel and reference accept the same positional tensor arguments and return a single tensor.

make_inputs receives the data tensor shapes (each argument is a tuple) and must return \
a tuple of ALL tensors needed to call fused_kernel/reference — including any weight and \
bias tensors the fused operation requires. All returned tensors must be on CUDA and float32. \
For example, if fused_kernel(x, W, B) fuses a linear + activation on input shape (B, D), \
make_inputs should create x of shape (B, D), W of shape (D, D), and B of shape (D,).

Rules:
- Begin with: import torch / import triton / import triton.language as tl
- Use @triton.jit for the inner kernel. Use tl.constexpr for BLOCK sizes.
- Always include bounds-checking masks (offs < n_elements).
- reference() must compute the same math using plain PyTorch ops.
- Both functions must handle non-contiguous inputs (.contiguous() if needed).
- Both functions must handle arbitrary tensor sizes (not just powers of 2).
- fused_kernel() should allocate its output, launch the Triton kernel, and return the result.
"""

_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL
)
_REQUIRED_FUSED = re.compile(r"^def fused_kernel\s*\(", re.MULTILINE)
_REQUIRED_REF = re.compile(r"^def reference\s*\(", re.MULTILINE)
_REQUIRED_MAKE_INPUTS = re.compile(r"^def make_inputs\s*\(", re.MULTILINE)
_CODE_START_RE = re.compile(r"^(import |from |@|def )", re.MULTILINE)

_ERROR_STUB_TEMPLATE = '''\
import torch

def fused_kernel(*args):
    raise RuntimeError("KernelGenerator failed: {reason}")

def reference(*args):
    raise RuntimeError("KernelGenerator failed: {reason}")

def make_inputs(*shapes):
    return tuple(torch.randn(s, dtype=torch.float32, device="cuda") for s in shapes)
'''

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _error_stub(reason: str) -> str:
    """Return a syntactically valid Python file that raises on call."""
    safe_reason = reason.replace('"', '\\"').replace("\n", " ")
    return _ERROR_STUB_TEMPLATE.format(reason=safe_reason)


def _strip_markdown_fences(text: str) -> str:
    """Extract code from markdown fences if present."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    # Strip leading non-code text before first import/def/@
    m2 = _CODE_START_RE.search(text)
    if m2:
        return text[m2.start():]
    return text


def _has_required_exports(code: str, *, strict: bool = True) -> bool:
    """Check that required functions are defined.

    When *strict* is True (default), also checks for ``make_inputs``.
    Use ``strict=False`` for backward-compat with older kernels that only
    export ``fused_kernel`` and ``reference``.
    """
    has_base = bool(_REQUIRED_FUSED.search(code)) and bool(_REQUIRED_REF.search(code))
    if not strict:
        return has_base
    return has_base and bool(_REQUIRED_MAKE_INPUTS.search(code))


def _is_valid_python(code: str) -> bool:
    """Check if the code parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _postprocess(raw: str) -> str:
    """Clean LLM output into a harness-compatible Python file."""
    if not raw or not raw.strip():
        return _error_stub("empty response")

    code = _strip_markdown_fences(raw).strip()

    if not _has_required_exports(code, strict=False):
        return _error_stub("missing fused_kernel or reference")

    if not _is_valid_python(code):
        return _error_stub("invalid Python syntax")

    # Warn but don't reject if make_inputs is missing — the harness has a
    # fallback.  Future codegen calls should always produce it since the
    # system prompt now requires it.
    if not _REQUIRED_MAKE_INPUTS.search(code):
        logger.warning("Generated code is missing make_inputs(); harness will use fallback")

    return code


def _format_shapes(shapes: list[tuple]) -> str:
    """Format a list of shapes for the prompt."""
    return ", ".join(str(s) for s in shapes)


def _build_user_prompt(
    candidate: FusionCandidate,
    context: ResearchContext | None = None,
    feedback: str | None = None,
) -> str:
    """Assemble the user prompt from candidate + optional context/feedback."""
    parts: list[str] = []

    # 1. Ops
    ops_str = ", ".join(candidate.ops)
    parts.append(f"Fuse these operations into a single Triton kernel: [{ops_str}]")

    # 2. Shapes
    parts.append(f"Input shapes: {_format_shapes(candidate.input_shapes)}")
    parts.append(f"Output shape: {candidate.output_shape}")

    # 3. Memory/compute guidance
    if candidate.memory_bound:
        parts.append(
            "This fusion is memory-bound. Minimize global memory accesses — "
            "maximize data reuse and coalesce loads/stores."
        )
    else:
        parts.append(
            "This fusion is compute-bound. Maximize compute throughput — "
            "use efficient math and maximize occupancy."
        )

    # 4. Research context (optional)
    if context is not None:
        if context.prior_implementations:
            parts.append(
                "Prior implementations for reference:\n"
                + "\n".join(f"- {impl}" for impl in context.prior_implementations)
            )
        if context.known_pitfalls:
            parts.append(
                "Known pitfalls to avoid:\n"
                + "\n".join(f"- {pit}" for pit in context.known_pitfalls)
            )
        if context.suggested_tile_sizes:
            tiles = ", ".join(str(t) for t in context.suggested_tile_sizes)
            parts.append(f"Suggested tile/block sizes: {tiles}")

    # 5. Feedback (optional, for retry loop)
    if feedback:
        parts.append(
            f"IMPORTANT — A previous attempt failed with this error:\n{feedback}\n"
            "Fix the issue in your new implementation."
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class KernelGenerator:
    """Generate fused Triton kernel code via an OpenAI LLM call.

    Parameters
    ----------
    model : str
        OpenAI model name (default ``"gpt-4o"``).
    max_retries : int
        Number of retries on transient API errors (timeout / rate-limit).
    """

    def __init__(self, model: str = _DEFAULT_MODEL, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set — generate() will return error stubs"
            )
            self._client = None
        else:
            self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        candidate: FusionCandidate,
        context: ResearchContext | None = None,
        temperature: float = 0.4,
        feedback: str | None = None,
    ) -> str:
        """Generate a Triton kernel file for *candidate*.

        This method **never raises** — all failures are captured as error
        stubs that the harness can safely import and evaluate.
        """
        if self._client is None:
            return _error_stub("OPENAI_API_KEY not set")

        user_prompt = _build_user_prompt(candidate, context, feedback)

        last_error: str | None = None
        for attempt in range(1 + self.max_retries):
            try:
                # GPT-5.4+ requires max_completion_tokens; older models use max_tokens
                token_kwarg = (
                    {"max_completion_tokens": _MAX_TOKENS}
                    if "5." in self.model
                    else {"max_tokens": _MAX_TOKENS}
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    **token_kwarg,
                )
                raw = response.choices[0].message.content
                return _postprocess(raw)

            except (TimeoutError, ConnectionError) as exc:
                last_error = f"API transient error: {exc}"
                logger.warning(
                    "OpenAI call failed (attempt %d/%d): %s",
                    attempt + 1,
                    1 + self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue

            except Exception as exc:
                # Check for rate-limit errors from the openai SDK
                if _is_rate_limit(exc) and attempt < self.max_retries:
                    last_error = f"Rate limit: {exc}"
                    logger.warning(
                        "Rate limit (attempt %d/%d): %s",
                        attempt + 1,
                        1 + self.max_retries,
                        exc,
                    )
                    time.sleep(2 ** attempt)
                    continue

                logger.error("OpenAI API error: %s", exc)
                return _error_stub(str(exc))

        return _error_stub(last_error or "max retries exceeded")


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception looks like a rate-limit error."""
    type_name = type(exc).__name__
    if "RateLimitError" in type_name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return False
