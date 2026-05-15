"""LLM-powered Triton kernel code generation.

Calls OpenAI to produce a complete Python file exporting ``fused_kernel``
and ``reference`` functions that the :class:`BenchmarkHarness` can evaluate.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised via reload tests
    OpenAI = None

from fusionagent.llm import codex_cli_available, run_codex_cli_prompt, using_codex_cli
from fusionagent.types import FusionCandidate, ResearchContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gpt-4o"
_MAX_TOKENS = 4096
_MAX_SURVIVOR_CHARS = 2000
_DEFAULT_TEMPERATURE_START = 0.30
_DEFAULT_TEMPERATURE_END = 0.80

_VARIATION_HINTS = [
    "Bias toward larger BLOCK sizes and simple coalesced tiles.",
    "Bias toward smaller BLOCK sizes to preserve occupancy on short shapes.",
    "Prefer wider vectorized loads and stores when masks allow.",
    "Minimize shared-memory usage and favor register reuse.",
    "Use shared-memory staging only when it clearly reduces redundant global loads.",
    "Favor straightforward row-major loop ordering over exotic indexing.",
    "Try an alternate loop ordering that prioritizes locality on the trailing dimension.",
    "Tune num_warps and launch configuration aggressively for stable occupancy.",
]

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


def _temperature_schedule(n: int) -> list[float]:
    """Return a deterministic temperature schedule for multi-candidate search."""
    if n <= 0:
        return []
    if n == 1:
        return [_DEFAULT_TEMPERATURE_START]

    step = (_DEFAULT_TEMPERATURE_END - _DEFAULT_TEMPERATURE_START) / (n - 1)
    return [
        round(_DEFAULT_TEMPERATURE_START + step * i, 2)
        for i in range(n)
    ]


def _variation_schedule(n: int) -> list[str]:
    """Return a deterministic prompt variation schedule for multi-candidate search."""
    if n <= 0:
        return []
    return [_VARIATION_HINTS[i % len(_VARIATION_HINTS)] for i in range(n)]


def _strip_imports_and_comments(code: str) -> str:
    """Reduce a survivor kernel to the core implementation for conditioning."""
    kept_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _summarize_survivor_kernels(kernels: list[str] | None) -> list[str]:
    """Prepare at most two prior survivor kernels for prompt conditioning."""
    if not kernels:
        return []

    summaries: list[str] = []
    for code in kernels[:2]:
        cleaned = _strip_imports_and_comments(code)
        if len(cleaned) > _MAX_SURVIVOR_CHARS:
            cleaned = f"{cleaned[:_MAX_SURVIVOR_CHARS].rstrip()}\n..."
        summaries.append(cleaned)
    return summaries


def _build_user_prompt(
    candidate: FusionCandidate,
    context: ResearchContext | None = None,
    feedback: str | None = None,
    *,
    variation_hint: str | None = None,
    survivors: list[str] | None = None,
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
    if variation_hint:
        parts.append(f"Search variation to explore: {variation_hint}")

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

    # 5. Survivor conditioning (optional, for multi-round search)
    survivor_summaries = _summarize_survivor_kernels(survivors)
    if survivor_summaries:
        formatted = []
        for idx, summary in enumerate(survivor_summaries, start=1):
            formatted.append(
                f"Survivor {idx} — these worked well:\n{summary}"
            )
        parts.append("\n\n".join(formatted))

    # 6. Feedback (optional, for retry loop / RL search)
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
        self._client = None
        self._unavailable_reason: str | None = None
        self._backend = "codex_cli" if using_codex_cli() else "openai"

        if self._backend == "codex_cli":
            if not codex_cli_available():
                logger.warning(
                    "Codex CLI not available — generate() will return error stubs"
                )
                self._unavailable_reason = "codex CLI not installed"
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set — generate() will return error stubs"
                )
                self._unavailable_reason = "OPENAI_API_KEY not set"
            elif OpenAI is None:
                logger.warning(
                    "openai package is not installed — generate() will return error stubs"
                )
                self._unavailable_reason = "openai package not installed"
            else:
                self._client = OpenAI(api_key=api_key)

    def _create_completion(self, user_prompt: str, temperature: float) -> str:
        """Issue a completion request and postprocess the returned content."""
        last_error: str | None = None
        for attempt in range(1 + self.max_retries):
            try:
                if self._backend == "codex_cli":
                    raw = run_codex_cli_prompt(
                        _SYSTEM_PROMPT,
                        user_prompt,
                        model=self.model,
                    )
                else:
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
        if self._unavailable_reason is not None:
            return _error_stub(self._unavailable_reason or "OpenAI client unavailable")

        user_prompt = _build_user_prompt(candidate, context, feedback)
        return self._create_completion(user_prompt, temperature)

    def generate_many(
        self,
        candidate: FusionCandidate,
        context: ResearchContext | None = None,
        survivors: list[str] | None = None,
        n: int = 8,
        temperatures: list[float] | None = None,
        feedback: str | None = None,
    ) -> list[str]:
        """Generate multiple kernel candidates in parallel for search."""
        if n <= 0:
            return []
        if self._unavailable_reason is not None:
            reason = self._unavailable_reason or "OpenAI client unavailable"
            return [_error_stub(reason) for _ in range(n)]

        resolved_temps = temperatures or _temperature_schedule(n)
        if len(resolved_temps) != n:
            raise ValueError("temperatures must match n")
        variation_hints = _variation_schedule(n)

        def _generate_one(index: int) -> str:
            prompt = _build_user_prompt(
                candidate,
                context,
                feedback,
                variation_hint=variation_hints[index],
                survivors=survivors,
            )
            return self._create_completion(prompt, resolved_temps[index])

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [executor.submit(_generate_one, i) for i in range(n)]
            return [future.result() for future in futures]


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception looks like a rate-limit error."""
    type_name = type(exc).__name__
    if "RateLimitError" in type_name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return False
