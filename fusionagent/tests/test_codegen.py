"""Tests for fusionagent.generator.codegen — KernelGenerator.

All tests are CPU-only and fully offline: OpenAI is mocked via
unittest.mock.patch.
"""

from __future__ import annotations

import ast
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from fusionagent.generator.codegen import (
    KernelGenerator,
    _build_user_prompt,
    _error_stub,
    _format_shapes,
    _has_required_exports,
    _is_valid_python,
    _postprocess,
    _strip_markdown_fences,
)
from fusionagent.types import FusionCandidate, ResearchContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(**overrides) -> FusionCandidate:
    defaults = dict(
        ops=["rmsnorm", "silu", "mul"],
        input_shapes=[(2, 64, 128)],
        output_shape=(2, 64, 128),
        memory_bound=True,
        launch_overhead_us=7.0,
        graph_position=0,
    )
    defaults.update(overrides)
    return FusionCandidate(**defaults)


def _make_research_context(**overrides) -> ResearchContext:
    defaults = dict(
        prior_implementations=["Liger-Kernel fused RMSNorm+SiLU"],
        known_pitfalls=["Shared memory bank conflicts"],
        suggested_tile_sizes=[(64, 64), (128, 128)],
        novelty_score=0.3,
    )
    defaults.update(overrides)
    return ResearchContext(**defaults)


_VALID_KERNEL_CODE = textwrap.dedent("""\
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
        _add_kernel[(n + 1023) // 1024,](x, y, out, n, BLOCK=1024)
        return out

    def reference(x, y):
        return x + y
""")


def _mock_openai_response(content: str):
    """Build a mock OpenAI chat completion response."""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ---------------------------------------------------------------------------
# TestBuildUserPrompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_ops_in_prompt(self):
        c = _make_candidate(ops=["rmsnorm", "silu", "mul"])
        prompt = _build_user_prompt(c)
        assert "rmsnorm" in prompt
        assert "silu" in prompt
        assert "mul" in prompt

    def test_shapes_in_prompt(self):
        c = _make_candidate(input_shapes=[(4, 256, 512)])
        prompt = _build_user_prompt(c)
        assert "(4, 256, 512)" in prompt

    def test_memory_bound_guidance(self):
        c = _make_candidate(memory_bound=True)
        prompt = _build_user_prompt(c)
        assert "memory" in prompt.lower()
        assert "minimize" in prompt.lower() or "reuse" in prompt.lower()

    def test_compute_bound_guidance(self):
        c = _make_candidate(memory_bound=False)
        prompt = _build_user_prompt(c)
        assert "compute" in prompt.lower()
        assert "throughput" in prompt.lower()

    def test_research_context_included(self):
        c = _make_candidate()
        ctx = _make_research_context()
        prompt = _build_user_prompt(c, context=ctx)
        assert "Liger-Kernel" in prompt
        assert "bank conflicts" in prompt
        assert "(64, 64)" in prompt

    def test_research_context_excluded(self):
        c = _make_candidate()
        prompt = _build_user_prompt(c, context=None)
        assert "Liger-Kernel" not in prompt
        assert "pitfalls" not in prompt.lower()

    def test_feedback_included(self):
        c = _make_candidate()
        prompt = _build_user_prompt(c, feedback="TypeError: bad arg")
        assert "TypeError: bad arg" in prompt
        assert "previous attempt failed" in prompt.lower()


# ---------------------------------------------------------------------------
# TestPostProcess
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_clean_code_passthrough(self):
        result = _postprocess(_VALID_KERNEL_CODE)
        assert "def fused_kernel(" in result
        assert "def reference(" in result

    def test_strip_markdown_fences(self):
        wrapped = f"```python\n{_VALID_KERNEL_CODE}```"
        result = _postprocess(wrapped)
        assert "def fused_kernel(" in result
        assert "```" not in result

    def test_missing_fused_kernel_rejected(self):
        code = "import torch\ndef reference(x):\n    return x\n"
        result = _postprocess(code)
        assert "KernelGenerator failed" in result
        assert "missing fused_kernel" in result

    def test_missing_reference_rejected(self):
        code = "import torch\ndef fused_kernel(x):\n    return x\n"
        result = _postprocess(code)
        assert "KernelGenerator failed" in result
        assert "missing fused_kernel or reference" in result

    def test_syntax_error_rejected(self):
        code = "def fused_kernel(x):\n    return x\ndef reference(x)\n    return x\n"
        result = _postprocess(code)
        assert "KernelGenerator failed" in result
        assert "invalid Python syntax" in result

    def test_leading_text_stripped(self):
        code = "Here is the code:\n\n" + _VALID_KERNEL_CODE
        result = _postprocess(code)
        assert "def fused_kernel(" in result
        assert "Here is the code" not in result

    def test_empty_response(self):
        result = _postprocess("")
        assert "KernelGenerator failed" in result
        assert "empty response" in result

    def test_error_stub_is_valid_python(self):
        stub = _error_stub("test reason")
        assert _is_valid_python(stub)
        assert _has_required_exports(stub)


# ---------------------------------------------------------------------------
# TestGenerate
# ---------------------------------------------------------------------------


class TestGenerate:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_happy_path(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_KERNEL_CODE
        )
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator()
        result = gen.generate(_make_candidate())

        assert "def fused_kernel(" in result
        assert "def reference(" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_api_error_returns_stub(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("API down")
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator(max_retries=0)
        result = gen.generate(_make_candidate())

        assert "KernelGenerator failed" in result
        assert "API down" in result

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_empty_response_returns_stub(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("")
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator()
        result = gen.generate(_make_candidate())

        assert "KernelGenerator failed" in result
        assert "empty response" in result

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_temperature_forwarded(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_KERNEL_CODE
        )
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator()
        gen.generate(_make_candidate(), temperature=0.8)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.8

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_feedback_changes_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_KERNEL_CODE
        )
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator()
        gen.generate(_make_candidate(), feedback="IndexError: out of bounds")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "IndexError: out of bounds" in user_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("fusionagent.generator.codegen.time.sleep")
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_retry_on_rate_limit(self, mock_openai_cls, mock_sleep):
        mock_client = MagicMock()

        # First call raises rate-limit-like error, second succeeds
        rate_err = Exception("rate limit exceeded")
        rate_err.status_code = 429
        mock_client.chat.completions.create.side_effect = [
            rate_err,
            _mock_openai_response(_VALID_KERNEL_CODE),
        ]
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator(max_retries=2)
        result = gen.generate(_make_candidate())

        assert "def fused_kernel(" in result
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# TestMissingApiKey
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    @patch.dict(os.environ, {}, clear=True)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_returns_error_stub_no_crash(self, mock_openai_cls):
        # Ensure OPENAI_API_KEY is absent
        os.environ.pop("OPENAI_API_KEY", None)
        gen = KernelGenerator()
        result = gen.generate(_make_candidate())

        assert "KernelGenerator failed" in result
        assert "OPENAI_API_KEY" in result
        # OpenAI client should not have been instantiated with real calls
        mock_openai_cls.return_value.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# TestFormatHelpers
# ---------------------------------------------------------------------------


class TestHasRequiredExports:
    def test_strict_requires_make_inputs(self):
        code_without = "def fused_kernel(x):\n    pass\ndef reference(x):\n    pass\n"
        assert _has_required_exports(code_without, strict=False) is True
        assert _has_required_exports(code_without, strict=True) is False

    def test_strict_passes_with_all_three(self):
        code = (
            "def fused_kernel(x):\n    pass\n"
            "def reference(x):\n    pass\n"
            "def make_inputs(*shapes):\n    pass\n"
        )
        assert _has_required_exports(code, strict=True) is True
        assert _has_required_exports(code, strict=False) is True

    def test_default_is_strict(self):
        code_without = "def fused_kernel(x):\n    pass\ndef reference(x):\n    pass\n"
        assert _has_required_exports(code_without) is False


class TestFormatHelpers:
    def test_3d_shapes_formatted(self):
        shapes = [(2, 64, 128), (2, 64, 128)]
        result = _format_shapes(shapes)
        assert "(2, 64, 128)" in result
        assert result.count("(2, 64, 128)") == 2

    def test_empty_research_lists_handled(self):
        c = _make_candidate()
        ctx = ResearchContext(
            prior_implementations=[],
            known_pitfalls=[],
            suggested_tile_sizes=[],
            novelty_score=0.5,
        )
        prompt = _build_user_prompt(c, context=ctx)
        # Should not crash; no "Prior implementations" section
        assert "Prior implementations" not in prompt
        assert "pitfalls" not in prompt.lower()

    def test_single_op_candidate(self):
        c = _make_candidate(ops=["relu"])
        prompt = _build_user_prompt(c)
        assert "relu" in prompt
        assert "Fuse these operations" in prompt
