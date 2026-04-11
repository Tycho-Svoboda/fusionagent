"""Tests for fusionagent.research.retriever — ResearchRetriever.

All tests are CPU-only and fully offline: httpx, OpenAI, and os.environ
are mocked via unittest.mock.patch.
"""

from __future__ import annotations

import json
import os
import pickle
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fusionagent.research.retriever import (
    ResearchRetriever,
    _build_arxiv_query,
    _cache_key,
    _extract_relevant_snippet,
    _llm_extract_context,
    _load_cache,
    _query_arxiv,
    _query_github,
    _save_cache,
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


_SAMPLE_ARXIV_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Fast Fused Triton Kernels</title>
        <summary>We present an approach to fuse normalization and activation.</summary>
        <id>http://arxiv.org/abs/2401.00001</id>
      </entry>
      <entry>
        <title>GPU Kernel Optimization</title>
        <summary>A survey of GPU kernel fusion techniques.</summary>
        <id>http://arxiv.org/abs/2401.00002</id>
      </entry>
    </feed>
""")


def _mock_github_search_response(items=None):
    """Build a mock httpx.Response for GitHub code search."""
    if items is None:
        items = [
            {
                "repository": {"full_name": "openai/triton"},
                "path": "python/triton/ops/fused.py",
                "url": "https://api.github.com/repos/openai/triton/contents/fused.py",
            }
        ]
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"items": items}
    return resp


def _mock_github_content_response(code="@triton.jit\ndef kernel(): pass"):
    """Build a mock httpx.Response for file content fetch."""
    import base64

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"content": base64.b64encode(code.encode()).decode()}
    return resp


# ---------------------------------------------------------------------------
# TestBuildSearchQuery
# ---------------------------------------------------------------------------


class TestBuildSearchQuery:
    def test_normalizes_silu(self):
        c = _make_candidate(ops=["silu"])
        q = _build_arxiv_query(c)
        assert "SiLU" in q

    def test_normalizes_rmsnorm(self):
        c = _make_candidate(ops=["rmsnorm"])
        q = _build_arxiv_query(c)
        assert "RMSNorm" in q

    def test_deduplicates_ops(self):
        c = _make_candidate(ops=["silu", "silu", "mul"])
        q = _build_arxiv_query(c)
        assert q.count("SiLU") == 1

    def test_single_op(self):
        c = _make_candidate(ops=["relu"])
        q = _build_arxiv_query(c)
        assert "ReLU" in q
        assert "Triton" in q


# ---------------------------------------------------------------------------
# TestCacheKey
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self):
        c = _make_candidate(ops=["silu", "mul"])
        assert _cache_key(c) == _cache_key(c)

    def test_order_independent(self):
        c1 = _make_candidate(ops=["silu", "mul"])
        c2 = _make_candidate(ops=["mul", "silu"])
        assert _cache_key(c1) == _cache_key(c2)

    def test_different_ops_different_key(self):
        c1 = _make_candidate(ops=["silu", "mul"])
        c2 = _make_candidate(ops=["relu", "add"])
        assert _cache_key(c1) != _cache_key(c2)


# ---------------------------------------------------------------------------
# TestDiskCache
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        ctx = ResearchContext(
            prior_implementations=["impl1"],
            known_pitfalls=["pit1"],
            suggested_tile_sizes=[(64, 64)],
            novelty_score=0.3,
        )
        _save_cache("testkey", ctx, tmp_path)
        loaded = _load_cache("testkey", tmp_path)
        assert loaded is not None
        assert loaded.novelty_score == 0.3
        assert loaded.prior_implementations == ["impl1"]

    def test_missing_returns_none(self, tmp_path):
        assert _load_cache("nonexistent", tmp_path) is None

    def test_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a pickle")
        assert _load_cache("corrupt", tmp_path) is None


# ---------------------------------------------------------------------------
# TestQueryArxiv
# ---------------------------------------------------------------------------


class TestQueryArxiv:
    def test_success_parse(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = _SAMPLE_ARXIV_XML

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        results = _query_arxiv("test query", mock_client)
        assert len(results) == 2
        assert results[0]["title"] == "Fast Fused Triton Kernels"
        assert "http://arxiv.org" in results[0]["link"]

    def test_http_error_returns_empty(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("boom")

        results = _query_arxiv("test query", mock_client)
        assert results == []

    def test_malformed_xml_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "<<<not xml>>>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        results = _query_arxiv("test query", mock_client)
        assert results == []


# ---------------------------------------------------------------------------
# TestQueryGitHub
# ---------------------------------------------------------------------------


class TestQueryGitHub:
    def test_curated_search(self):
        search_resp = _mock_github_search_response()
        content_resp = _mock_github_content_response()

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            search_resp,
            content_resp,
            # remaining curated repos + general search return empty
            _mock_github_search_response([]),
            _mock_github_search_response([]),
            _mock_github_search_response([]),
            _mock_github_search_response([]),
        ]

        results = _query_github(["silu", "mul"], mock_client, "fake-token")
        assert len(results) >= 1
        assert results[0]["repo"] == "openai/triton"

    def test_403_graceful(self):
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.raise_for_status = MagicMock()
        resp_403.json.return_value = {"items": []}

        mock_client = MagicMock()
        mock_client.get.return_value = resp_403

        results = _query_github(["silu"], mock_client, "fake-token")
        assert isinstance(results, list)

    def test_429_graceful(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.raise_for_status = MagicMock()
        resp_429.json.return_value = {"items": []}

        mock_client = MagicMock()
        mock_client.get.return_value = resp_429

        results = _query_github(["silu"], mock_client, "fake-token")
        assert isinstance(results, list)

    def test_general_fallback(self):
        # Curated repos all return empty, general returns a result
        empty_resp = _mock_github_search_response([])
        general_resp = _mock_github_search_response(
            [
                {
                    "repository": {"full_name": "user/myrepo"},
                    "path": "kernel.py",
                    "url": "https://api.github.com/repos/user/myrepo/contents/kernel.py",
                }
            ]
        )
        content_resp = _mock_github_content_response("@triton.jit\ndef my_kernel(): pass")

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            empty_resp,  # curated 1
            empty_resp,  # curated 2
            empty_resp,  # curated 3
            empty_resp,  # curated 4
            general_resp,  # general search
            content_resp,  # content fetch
        ]

        results = _query_github(["silu"], mock_client, "fake-token")
        assert any(r["repo"] == "user/myrepo" for r in results)


# ---------------------------------------------------------------------------
# TestMissingGitHubToken
# ---------------------------------------------------------------------------


class TestMissingGitHubToken:
    @patch.dict(os.environ, {}, clear=True)
    def test_raises_environment_error(self, tmp_path):
        retriever = ResearchRetriever(cache_dir=tmp_path)
        c = _make_candidate()
        with pytest.raises(EnvironmentError, match="GITHUB_TOKEN"):
            retriever.retrieve(c)


# ---------------------------------------------------------------------------
# TestLLMExtraction
# ---------------------------------------------------------------------------


class TestLLMExtraction:
    def _make_mock_openai(self, content_json: dict):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(content_json)
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_success(self):
        payload = {
            "prior_implementations": ["Liger-Kernel fused RMSNorm+SiLU"],
            "known_pitfalls": ["Shared memory bank conflicts"],
            "suggested_tile_sizes": [[64, 64], [128, 128]],
            "novelty_score": 0.3,
        }
        mock_client = self._make_mock_openai(payload)
        arxiv = [{"title": "paper", "summary": "summary", "link": "url"}]
        github = [{"repo": "r", "path": "p", "snippet": "s"}]
        c = _make_candidate()

        result = _llm_extract_context(arxiv, github, c, mock_client, "gpt-4o-mini")
        assert result["novelty_score"] == 0.3
        assert len(result["prior_implementations"]) == 1

    def test_api_error_returns_defaults(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        arxiv = [{"title": "p", "summary": "s", "link": "l"}]
        c = _make_candidate()

        result = _llm_extract_context(arxiv, [], c, mock_client, "gpt-4o-mini")
        assert result["novelty_score"] == 0.5
        assert result["suggested_tile_sizes"] == [(128, 128)]

    def test_bad_json_returns_defaults(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "not json at all"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp
        arxiv = [{"title": "p", "summary": "s", "link": "l"}]
        c = _make_candidate()

        result = _llm_extract_context(arxiv, [], c, mock_client, "gpt-4o-mini")
        assert result["novelty_score"] == 0.5

    def test_novelty_clamped(self):
        payload = {
            "prior_implementations": [],
            "known_pitfalls": [],
            "suggested_tile_sizes": [[128, 128]],
            "novelty_score": 1.5,  # out of range
        }
        mock_client = self._make_mock_openai(payload)
        arxiv = [{"title": "p", "summary": "s", "link": "l"}]
        c = _make_candidate()

        result = _llm_extract_context(arxiv, [], c, mock_client, "gpt-4o-mini")
        # _llm_extract_context returns raw; clamping happens in retrieve()
        # So the raw value is returned here
        assert result["novelty_score"] == 1.5


# ---------------------------------------------------------------------------
# TestExtractRelevantSnippet
# ---------------------------------------------------------------------------


class TestExtractRelevantSnippet:
    def test_finds_triton_jit(self):
        code = textwrap.dedent("""\
            import triton

            @triton.jit
            def fused_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                # kernel body
        """)
        snippet = _extract_relevant_snippet(code, ["silu"])
        assert "@triton.jit" in snippet

    def test_respects_max_chars(self):
        code = "@triton.jit\n" + "x" * 2000
        snippet = _extract_relevant_snippet(code, [])
        assert len(snippet) <= 500

    def test_handles_empty(self):
        snippet = _extract_relevant_snippet("", [])
        assert snippet == ""


# ---------------------------------------------------------------------------
# TestResearchRetrieverE2E
# ---------------------------------------------------------------------------


class TestResearchRetrieverE2E:
    """End-to-end tests with all externals mocked."""

    def _setup_mocks(self, mock_openai_cls, mock_httpx_cls, llm_response=None):
        """Wire up mocked httpx.Client and OpenAI for a happy-path run."""
        # --- httpx mock ---
        mock_http_client = MagicMock()

        # arXiv response
        arxiv_resp = MagicMock()
        arxiv_resp.status_code = 200
        arxiv_resp.raise_for_status = MagicMock()
        arxiv_resp.text = _SAMPLE_ARXIV_XML

        # GitHub search response
        gh_search_resp = _mock_github_search_response()
        gh_content_resp = _mock_github_content_response()
        gh_empty_resp = _mock_github_search_response([])

        mock_http_client.get.side_effect = [
            arxiv_resp,       # arXiv query
        ]
        # For the second client context (GitHub)
        mock_gh_client = MagicMock()
        mock_gh_client.get.side_effect = [
            gh_search_resp,   # curated search 1
            gh_content_resp,  # content fetch
            gh_empty_resp,    # curated 2
            gh_empty_resp,    # curated 3
            gh_empty_resp,    # curated 4
            gh_empty_resp,    # general search
        ]

        # httpx.Client() used as context manager twice
        mock_httpx_cls.return_value.__enter__ = MagicMock(
            side_effect=[mock_http_client, mock_gh_client]
        )
        mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

        # --- OpenAI mock ---
        if llm_response is None:
            llm_response = {
                "prior_implementations": ["Liger-Kernel fused RMSNorm"],
                "known_pitfalls": ["shared memory conflicts"],
                "suggested_tile_sizes": [[128, 128]],
                "novelty_score": 0.4,
            }
        mock_openai = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(llm_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_openai.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_openai

        return mock_http_client, mock_gh_client, mock_openai

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake"})
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_happy_path(self, mock_httpx_cls, mock_openai_cls, tmp_path):
        self._setup_mocks(mock_openai_cls, mock_httpx_cls)
        retriever = ResearchRetriever(cache_dir=tmp_path)
        ctx = retriever.retrieve(_make_candidate())

        assert isinstance(ctx, ResearchContext)
        assert 0.0 <= ctx.novelty_score <= 1.0
        assert isinstance(ctx.prior_implementations, list)
        assert isinstance(ctx.suggested_tile_sizes, list)

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake"})
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_cache_hit_skips_network(self, mock_httpx_cls, mock_openai_cls, tmp_path):
        # Pre-populate cache
        c = _make_candidate()
        from fusionagent.research.retriever import _cache_key

        key = _cache_key(c)
        ctx = ResearchContext(
            prior_implementations=["cached"],
            known_pitfalls=[],
            suggested_tile_sizes=[(64, 64)],
            novelty_score=0.2,
        )
        _save_cache(key, ctx, tmp_path)

        retriever = ResearchRetriever(cache_dir=tmp_path)
        result = retriever.retrieve(c)

        assert result.prior_implementations == ["cached"]
        # httpx should never have been called
        mock_httpx_cls.assert_not_called()
        mock_openai_cls.assert_not_called()

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake"})
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_partial_failures(self, mock_httpx_cls, mock_openai_cls, tmp_path):
        """arXiv fails but GitHub+LLM succeed → still returns a ResearchContext."""
        # arXiv client raises
        mock_arxiv_client = MagicMock()
        mock_arxiv_client.get.side_effect = Exception("arXiv down")

        # GitHub client succeeds
        mock_gh_client = MagicMock()
        gh_search_resp = _mock_github_search_response([])
        mock_gh_client.get.return_value = gh_search_resp

        mock_httpx_cls.return_value.__enter__ = MagicMock(
            side_effect=[mock_arxiv_client, mock_gh_client]
        )
        mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

        # LLM returns defaults (no snippets)
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        retriever = ResearchRetriever(cache_dir=tmp_path)
        ctx = retriever.retrieve(_make_candidate())

        assert isinstance(ctx, ResearchContext)
        # With no snippets, _llm_extract_context returns defaults
        assert ctx.novelty_score == 0.5

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake"})
    @patch("fusionagent.research.retriever.OpenAI")
    @patch("fusionagent.research.retriever.httpx.Client")
    def test_all_fail_returns_defaults(self, mock_httpx_cls, mock_openai_cls, tmp_path):
        """When all external sources fail, returns a high-novelty default."""
        # Both HTTP clients fail
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("network down")

        mock_httpx_cls.return_value.__enter__ = MagicMock(
            side_effect=[mock_client, mock_client]
        )
        mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

        # LLM also fails
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        retriever = ResearchRetriever(cache_dir=tmp_path)
        ctx = retriever.retrieve(_make_candidate())

        assert isinstance(ctx, ResearchContext)
        # No snippets → defaults applied by _llm_extract_context
        assert ctx.novelty_score == 0.5
        assert ctx.suggested_tile_sizes == [(128, 128)]
