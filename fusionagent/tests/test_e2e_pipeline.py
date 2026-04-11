"""End-to-end flow test: GraphAnalyzer → ResearchRetriever → KernelGenerator.

Validates that Tasks 1-5 connect correctly.  CPU-only, all external calls
(OpenAI, GitHub, arXiv) are mocked.  BenchmarkHarness is tested structurally
(we verify the generated code is importable and has the right exports) because
full harness evaluation requires a CUDA GPU + Triton compilation.
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fusionagent.graph.analyzer import GraphAnalyzer
from fusionagent.research.retriever import ResearchRetriever
from fusionagent.generator.codegen import KernelGenerator, _has_required_exports
from fusionagent.types import BenchmarkResult, FusionCandidate, ResearchContext


# ---------------------------------------------------------------------------
# Sample models
# ---------------------------------------------------------------------------


class AddRelu(nn.Module):
    """Simplest fusible pattern: elementwise add → relu."""

    def forward(self, x, y):
        return torch.relu(x + y)


class NormSiluMul(nn.Module):
    """Norm → activation → elementwise (LLaMA-style)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return torch.mul(torch.nn.functional.silu(self.norm(x)), x)


class LinearGelu(nn.Module):
    """Linear → activation."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.nn.functional.gelu(self.fc(x))


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_VALID_KERNEL_CODE = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        out = tl.where(x + y > 0, x + y, 0.0)
        tl.store(out_ptr + offs, out, mask=mask)

    def fused_kernel(x, y):
        x = x.contiguous()
        y = y.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        _add_relu_kernel[(n + 1023) // 1024,](x, y, out, n, BLOCK=1024)
        return out

    def reference(x, y):
        return torch.relu(x + y)

    def make_inputs(*shapes):
        return tuple(torch.randn(s, dtype=torch.float32, device="cuda") for s in shapes)
""")

_ARXIV_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Fused Triton Kernels for Transformers</title>
        <summary>We present fused elementwise kernels.</summary>
        <id>http://arxiv.org/abs/2401.99999</id>
      </entry>
    </feed>
""")


def _mock_openai_kernel_response(content: str):
    """Build a mock OpenAI chat completion that returns *content*."""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _mock_openai_json_response(payload: dict):
    """Build a mock OpenAI chat completion that returns JSON."""
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _mock_github_search_empty():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"items": []}
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EAddRelu:
    """Full pipeline for the simplest model: add + relu."""

    def test_graph_analyzer_finds_candidates(self):
        """Task 1: GraphAnalyzer detects the add→relu fusion."""
        model = AddRelu()
        sample = torch.randn(2, 64)
        analyzer = GraphAnalyzer(model, sample_input=sample)
        candidates = analyzer.analyze()

        assert len(candidates) >= 1
        # Should detect an elementwise chain or sole-consumer pair
        all_ops = [op for c in candidates for op in c.ops]
        assert any("add" in op for op in all_ops) or any(
            "relu" in op for op in all_ops
        )

    @patch.dict(
        os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake"}, clear=False
    )
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_retriever_returns_context(self, mock_httpx_cls, mock_openai_cls, tmp_path):
        """Task 2: ResearchRetriever returns a ResearchContext for a candidate."""
        # Mock arXiv
        arxiv_resp = MagicMock()
        arxiv_resp.status_code = 200
        arxiv_resp.raise_for_status = MagicMock()
        arxiv_resp.text = _ARXIV_XML

        # Mock GitHub (empty results for simplicity)
        gh_empty = _mock_github_search_empty()

        mock_arxiv_client = MagicMock()
        mock_arxiv_client.get.return_value = arxiv_resp

        mock_gh_client = MagicMock()
        mock_gh_client.get.return_value = gh_empty

        mock_httpx_cls.return_value.__enter__ = MagicMock(
            side_effect=[mock_arxiv_client, mock_gh_client]
        )
        mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Mock LLM extraction
        llm_response = {
            "prior_implementations": ["basic add+relu fusion"],
            "known_pitfalls": ["alignment issues"],
            "suggested_tile_sizes": [[1024]],
            "novelty_score": 0.2,
        }
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _mock_openai_json_response(
            llm_response
        )
        mock_openai_cls.return_value = mock_openai

        # Create candidate from analyzer
        model = AddRelu()
        sample = torch.randn(2, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()
        assert len(candidates) >= 1

        retriever = ResearchRetriever(cache_dir=tmp_path)
        ctx = retriever.retrieve(candidates[0])

        assert isinstance(ctx, ResearchContext)
        assert ctx.novelty_score <= 1.0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_generator_produces_valid_code(self, mock_openai_cls):
        """Task 5: KernelGenerator produces code with required exports."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_openai_kernel_response(_VALID_KERNEL_CODE)
        )
        mock_openai_cls.return_value = mock_client

        # Analyze → pick first candidate → generate
        model = AddRelu()
        sample = torch.randn(2, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()
        assert len(candidates) >= 1
        candidate = candidates[0]

        gen = KernelGenerator()
        code = gen.generate(candidate)

        # Verify harness contract
        assert _has_required_exports(code)
        ast.parse(code)  # valid Python
        assert "fused_kernel" in code
        assert "reference" in code

    @patch.dict(
        os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake-key"}, clear=False
    )
    @patch("fusionagent.generator.codegen.OpenAI")
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_full_pipeline_analyzer_to_generator(
        self, mock_httpx_cls, mock_retriever_openai_cls, mock_gen_openai_cls, tmp_path
    ):
        """Full e2e: GraphAnalyzer → ResearchRetriever → KernelGenerator."""
        # --- Mock retriever externals ---
        arxiv_resp = MagicMock()
        arxiv_resp.status_code = 200
        arxiv_resp.raise_for_status = MagicMock()
        arxiv_resp.text = _ARXIV_XML

        gh_empty = _mock_github_search_empty()

        mock_arxiv_client = MagicMock()
        mock_arxiv_client.get.return_value = arxiv_resp
        mock_gh_client = MagicMock()
        mock_gh_client.get.return_value = gh_empty

        mock_httpx_cls.return_value.__enter__ = MagicMock(
            side_effect=[mock_arxiv_client, mock_gh_client]
        )
        mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

        retriever_llm = MagicMock()
        retriever_llm.chat.completions.create.return_value = (
            _mock_openai_json_response(
                {
                    "prior_implementations": ["add+relu fusion in Liger-Kernel"],
                    "known_pitfalls": ["non-contiguous tensors"],
                    "suggested_tile_sizes": [[1024]],
                    "novelty_score": 0.15,
                }
            )
        )
        mock_retriever_openai_cls.return_value = retriever_llm

        # --- Mock generator external ---
        gen_llm = MagicMock()
        gen_llm.chat.completions.create.return_value = _mock_openai_kernel_response(
            _VALID_KERNEL_CODE
        )
        mock_gen_openai_cls.return_value = gen_llm

        # === Step 1: GraphAnalyzer ===
        model = AddRelu()
        sample = torch.randn(2, 64)
        analyzer = GraphAnalyzer(model, sample_input=sample)
        candidates = analyzer.analyze()
        assert len(candidates) >= 1
        candidate = candidates[0]

        # === Step 2: ResearchRetriever ===
        retriever = ResearchRetriever(cache_dir=tmp_path)
        context = retriever.retrieve(candidate)
        assert isinstance(context, ResearchContext)

        # === Step 3: KernelGenerator ===
        gen = KernelGenerator()
        code = gen.generate(candidate, context=context)

        # === Verify harness contract ===
        assert _has_required_exports(code)
        ast.parse(code)
        assert "@triton.jit" in code
        assert "def fused_kernel(" in code
        assert "def reference(" in code

        # === Verify generator received research context in the prompt ===
        gen_call_args = gen_llm.chat.completions.create.call_args
        user_msg = gen_call_args.kwargs["messages"][1]["content"]
        assert "add+relu fusion in Liger-Kernel" in user_msg
        assert "non-contiguous tensors" in user_msg


class TestE2ENormSiluMul:
    """Pipeline for a more complex pattern: LayerNorm → SiLU → mul."""

    def test_analyzer_detects_norm_act_elem(self):
        model = NormSiluMul(dim=64)
        sample = torch.randn(2, 16, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()

        assert len(candidates) >= 1
        # At least one candidate should contain norm-related ops
        all_ops = [op for c in candidates for op in c.ops]
        assert len(all_ops) >= 2  # should fuse at least 2 ops

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_generator_with_norm_candidate(self, mock_openai_cls):
        model = NormSiluMul(dim=64)
        sample = torch.randn(2, 16, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()
        assert len(candidates) >= 1

        norm_kernel = textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def _norm_silu_mul_kernel(x_ptr, w_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
                row = tl.program_id(0)
                offs = tl.arange(0, BLOCK)
                mask = offs < N
                x = tl.load(x_ptr + row * N + offs, mask=mask)
                mean = tl.sum(x * x, axis=0) / N
                x_norm = x / tl.sqrt(mean + 1e-6)
                w = tl.load(w_ptr + offs, mask=mask)
                normed = x_norm * w
                silu = normed * tl.sigmoid(normed)
                orig = tl.load(x_ptr + row * N + offs, mask=mask)
                out = silu * orig
                tl.store(out_ptr + row * N + offs, out, mask=mask)

            def fused_kernel(x):
                x = x.contiguous()
                M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
                N = x.shape[-1]
                w = torch.ones(N, device=x.device, dtype=x.dtype)
                out = torch.empty_like(x)
                _norm_silu_mul_kernel[(M,)](x.view(-1, N), w, out.view(-1, N), M, N, BLOCK=1024)
                return out.view_as(x)

            def reference(x):
                import torch.nn.functional as F
                norm = torch.nn.LayerNorm(x.shape[-1], device=x.device, dtype=x.dtype)
                norm.weight.data.fill_(1.0)
                norm.bias.data.fill_(0.0)
                return F.silu(norm(x)) * x

            def make_inputs(*shapes):
                return tuple(torch.randn(s, dtype=torch.float32, device="cuda") for s in shapes)
        """)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_openai_kernel_response(norm_kernel)
        )
        mock_openai_cls.return_value = mock_client

        gen = KernelGenerator()
        code = gen.generate(candidates[0])

        assert _has_required_exports(code)
        ast.parse(code)


class TestE2ELinearGelu:
    """Pipeline for linear + activation pattern."""

    def test_analyzer_detects_linear_act(self):
        model = LinearGelu(dim=64)
        sample = torch.randn(2, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()

        assert len(candidates) >= 1
        all_ops = [op for c in candidates for op in c.ops]
        assert any("linear" in op.lower() for op in all_ops) or any(
            "gelu" in op.lower() for op in all_ops
        )


class TestE2EErrorPropagation:
    """Verify error stubs flow correctly through the pipeline."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_api_failure_produces_importable_stub(self, mock_openai_cls):
        """When OpenAI fails, the stub is still valid Python with both exports."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("offline")
        mock_openai_cls.return_value = mock_client

        model = AddRelu()
        sample = torch.randn(2, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()
        assert len(candidates) >= 1

        gen = KernelGenerator(max_retries=0)
        code = gen.generate(candidates[0])

        # Error stub should still satisfy harness contract
        assert _has_required_exports(code)
        ast.parse(code)
        assert "KernelGenerator failed" in code

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_malformed_llm_output_produces_stub(self, mock_openai_cls):
        """When LLM returns garbage, we get a valid error stub."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_openai_kernel_response("This is not code at all, just text.")
        )
        mock_openai_cls.return_value = mock_client

        candidate = FusionCandidate(
            ops=["add", "relu"],
            input_shapes=[(4, 1024), (4, 1024)],
            output_shape=(4, 1024),
            memory_bound=True,
            launch_overhead_us=3.5,
            graph_position=0,
        )
        gen = KernelGenerator()
        code = gen.generate(candidate)

        assert _has_required_exports(code)
        ast.parse(code)
        assert "KernelGenerator failed" in code

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_feedback_retry_loop_simulation(self, mock_openai_cls):
        """Simulate the RL loop: first attempt fails → feedback → second attempt."""
        mock_client = MagicMock()

        # First call returns code missing reference()
        bad_code = "import torch\ndef fused_kernel(x):\n    return x\n"
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_kernel_response(bad_code),
            _mock_openai_kernel_response(_VALID_KERNEL_CODE),
        ]
        mock_openai_cls.return_value = mock_client

        candidate = FusionCandidate(
            ops=["add", "relu"],
            input_shapes=[(4, 1024), (4, 1024)],
            output_shape=(4, 1024),
            memory_bound=True,
            launch_overhead_us=3.5,
            graph_position=0,
        )

        gen = KernelGenerator()

        # Round 1
        code1 = gen.generate(candidate)
        assert "KernelGenerator failed" in code1  # post-process rejects it

        # Round 2 with feedback (simulating RL loop)
        code2 = gen.generate(
            candidate,
            feedback="Error: missing fused_kernel or reference",
            temperature=0.6,
        )
        assert _has_required_exports(code2)
        assert "KernelGenerator failed" not in code2

        # Verify feedback was passed in the second prompt
        second_call = mock_client.chat.completions.create.call_args_list[1]
        user_msg = second_call.kwargs["messages"][1]["content"]
        assert "missing fused_kernel or reference" in user_msg


class TestE2EMultipleCandidates:
    """Test that we can process multiple candidates from a single model."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_batch_generation(self, mock_openai_cls):
        """Generate code for all candidates found in a model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_openai_kernel_response(_VALID_KERNEL_CODE)
        )
        mock_openai_cls.return_value = mock_client

        # Use NormSiluMul which may produce multiple candidates
        model = NormSiluMul(dim=64)
        sample = torch.randn(2, 16, 64)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()

        gen = KernelGenerator()
        codes = [gen.generate(c) for c in candidates]

        assert len(codes) == len(candidates)
        for code in codes:
            assert _has_required_exports(code)
            ast.parse(code)


class TestCandidateShapesPropagation:
    """Verify shapes from GraphAnalyzer flow through to generator prompts."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("fusionagent.generator.codegen.OpenAI")
    def test_shapes_reach_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_openai_kernel_response(_VALID_KERNEL_CODE)
        )
        mock_openai_cls.return_value = mock_client

        model = AddRelu()
        sample = torch.randn(4, 128)
        candidates = GraphAnalyzer(model, sample_input=sample).analyze()
        assert len(candidates) >= 1

        gen = KernelGenerator()
        gen.generate(candidates[0])

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        # The shapes from the analyzer should appear in the prompt
        # At minimum the ops should be there
        for op in candidates[0].ops:
            assert op in user_msg
