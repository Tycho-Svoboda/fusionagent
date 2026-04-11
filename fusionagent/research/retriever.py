"""Research retriever — queries arXiv and GitHub for prior Triton implementations,
uses an LLM to extract structured design decisions, and returns a ResearchContext.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from fusionagent.types import FusionCandidate, ResearchContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
_GITHUB_SEARCH_URL = "https://api.github.com/search/code"
_CURATED_REPOS = [
    "openai/triton",
    "linkedin/Liger-Kernel",
    "unslothai/unsloth",
    "facebookresearch/xformers",
]
_CACHE_DIR = Path(".fusionagent_cache/research")
_LLM_MODEL = "gpt-4o-mini"
_MAX_SNIPPET_CHARS = 500

# Normalization table for op names in search queries.
_OP_DISPLAY_NAMES: Dict[str, str] = {
    "silu": "SiLU",
    "rmsnorm": "RMSNorm",
    "layernorm": "LayerNorm",
    "gelu": "GELU",
    "relu": "ReLU",
    "softmax": "Softmax",
    "matmul": "MatMul",
    "linear": "Linear",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_arxiv_query(candidate: FusionCandidate) -> str:
    """Build a search query string for arXiv from a FusionCandidate's ops."""
    seen: set[str] = set()
    normalized: list[str] = []
    for op in candidate.ops:
        key = op.lower()
        if key in seen:
            continue
        seen.add(key)
        display = _OP_DISPLAY_NAMES.get(key, op)
        normalized.append(display)
    ops_str = " ".join(normalized)
    return f"Triton kernel fused {ops_str} GPU optimization"


def _cache_key(candidate: FusionCandidate) -> str:
    """Deterministic cache key from sorted op names."""
    key_input = str(tuple(sorted(candidate.ops)))
    return hashlib.sha256(key_input.encode()).hexdigest()[:16]


def _load_cache(key: str, cache_dir: Path) -> Optional[ResearchContext]:
    """Load a cached ResearchContext from disk. Returns None on any failure."""
    path = cache_dir / f"{key}.pkl"
    try:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        logger.warning("Corrupt cache entry %s, ignoring", key)
    return None


def _save_cache(key: str, ctx: ResearchContext, cache_dir: Path) -> None:
    """Persist a ResearchContext to disk cache."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(ctx, f)
    except Exception:
        logger.warning("Failed to write cache entry %s", key)


def _query_arxiv(
    query: str, client: httpx.Client
) -> List[Dict[str, str]]:
    """Query arXiv Atom API and return list of {title, summary, link}."""
    try:
        resp = client.get(
            _ARXIV_BASE_URL,
            params={"search_query": f"all:{query}", "max_results": "5"},
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception:
        logger.warning("arXiv query failed for %r", query)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        logger.warning("Malformed XML from arXiv")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        link_el = entry.find("atom:id", ns)
        results.append(
            {
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "summary": (summary_el.text or "").strip()
                if summary_el is not None
                else "",
                "link": (link_el.text or "").strip() if link_el is not None else "",
            }
        )
    return results


def _query_github(
    ops: List[str],
    client: httpx.Client,
    token: str,
) -> List[Dict[str, str]]:
    """Search GitHub for Triton kernel code matching *ops*.

    Searches curated repos first, then falls back to general code search.
    Returns list of {repo, path, snippet}.  Gracefully degrades on 403/429.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    results: List[Dict[str, str]] = []

    # --- curated repo search ---
    for repo in _CURATED_REPOS:
        query = f"triton.jit {' '.join(ops)} repo:{repo}"
        try:
            resp = client.get(
                _GITHUB_SEARCH_URL,
                params={"q": query, "per_page": "3"},
                headers=headers,
                timeout=15.0,
            )
            if resp.status_code in (403, 429):
                logger.warning(
                    "GitHub rate-limited (%s) during curated search for %s",
                    resp.status_code,
                    repo,
                )
                return results  # return what we have so far
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", [])[:3]:
                snippet = _fetch_raw_content(item, client, headers)
                results.append(
                    {
                        "repo": item.get("repository", {}).get(
                            "full_name", repo
                        ),
                        "path": item.get("path", ""),
                        "snippet": snippet,
                    }
                )
        except Exception:
            logger.warning("GitHub curated search failed for %s", repo)

    # --- general search ---
    query = f"triton.jit {' '.join(ops)}"
    try:
        resp = client.get(
            _GITHUB_SEARCH_URL,
            params={"q": query, "per_page": "3"},
            headers=headers,
            timeout=15.0,
        )
        if resp.status_code in (403, 429):
            logger.warning(
                "GitHub rate-limited (%s) during general search", resp.status_code
            )
            return results
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", [])[:3]:
            snippet = _fetch_raw_content(item, client, headers)
            results.append(
                {
                    "repo": item.get("repository", {}).get("full_name", ""),
                    "path": item.get("path", ""),
                    "snippet": snippet,
                }
            )
    except Exception:
        logger.warning("GitHub general search failed")

    return results


def _fetch_raw_content(
    item: Dict[str, Any], client: httpx.Client, headers: Dict[str, str]
) -> str:
    """Fetch raw file content from a GitHub search result item."""
    url = item.get("url", "")
    if not url:
        return ""
    try:
        resp = client.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        import base64

        content = base64.b64decode(data.get("content", "")).decode(
            "utf-8", errors="replace"
        )
        return _extract_relevant_snippet(content, [])
    except Exception:
        return ""


def _extract_relevant_snippet(content: str, ops: List[str]) -> str:
    """Find ``@triton.jit`` decorated functions and return ≤ _MAX_SNIPPET_CHARS."""
    # Try to find @triton.jit blocks
    pattern = r"@triton\.jit.*?(?=\n(?:@|class |def (?!_))|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        snippet = matches[0]
    else:
        snippet = content
    return snippet[:_MAX_SNIPPET_CHARS]


def _llm_extract_context(
    arxiv_results: List[Dict[str, str]],
    github_results: List[Dict[str, str]],
    candidate: FusionCandidate,
    openai_client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    """Use an LLM to extract structured design decisions from research results.

    Returns a dict with prior_implementations, known_pitfalls,
    suggested_tile_sizes, novelty_score.  Falls back to safe defaults on any error.
    """
    defaults: Dict[str, Any] = {
        "prior_implementations": [],
        "known_pitfalls": [],
        "suggested_tile_sizes": [(128, 128)],
        "novelty_score": 0.5,
    }

    # Combine top 3 results for the prompt
    snippets: list[str] = []
    for r in arxiv_results[:3]:
        snippets.append(f"Paper: {r['title']}\n{r['summary'][:300]}")
    for r in github_results[:3]:
        snippets.append(f"Repo: {r['repo']} ({r['path']})\n{r['snippet'][:300]}")

    if not snippets:
        return defaults

    combined = "\n---\n".join(snippets)
    ops_str = ", ".join(candidate.ops)

    prompt = (
        f"Given these research results about fusing [{ops_str}] into a single Triton kernel:\n\n"
        f"{combined}\n\n"
        "Extract a JSON object with:\n"
        '- "prior_implementations": list of short descriptions of existing implementations\n'
        '- "known_pitfalls": list of known issues or pitfalls\n'
        '- "suggested_tile_sizes": list of [rows, cols] tile size pairs\n'
        '- "novelty_score": float 0-1 (0 = well-studied, 1 = novel)\n'
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        import json

        raw = json.loads(response.choices[0].message.content)
        return {
            "prior_implementations": raw.get("prior_implementations", []),
            "known_pitfalls": raw.get("known_pitfalls", []),
            "suggested_tile_sizes": raw.get("suggested_tile_sizes", [(128, 128)]),
            "novelty_score": raw.get("novelty_score", 0.5),
        }
    except Exception:
        logger.warning("LLM extraction failed, using defaults")
        return defaults


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ResearchRetriever:
    """Retrieves research context for a FusionCandidate from arXiv and GitHub."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        llm_model: str = _LLM_MODEL,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
        self._llm_model = llm_model

    def retrieve(self, candidate: FusionCandidate) -> ResearchContext:
        """Query external sources and return a ``ResearchContext``.

        Raises ``EnvironmentError`` if *GITHUB_TOKEN* is not set.
        All other external failures are handled gracefully.
        """
        # 1. Check disk cache
        key = _cache_key(candidate)
        cached = _load_cache(key, self._cache_dir)
        if cached is not None:
            logger.info("Cache hit for %s", key)
            return cached

        # 2. Validate env
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            raise EnvironmentError(
                "GITHUB_TOKEN environment variable is required for GitHub code search"
            )

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        # 3. Build search query
        query = _build_arxiv_query(candidate)

        # 4. Query arXiv
        with httpx.Client() as client:
            arxiv_results = _query_arxiv(query, client)

        # 5. Query GitHub
        with httpx.Client() as client:
            github_results = _query_github(candidate.ops, client, github_token)

        # 6. LLM extraction
        openai_client = OpenAI(api_key=openai_api_key)
        extracted = _llm_extract_context(
            arxiv_results,
            github_results,
            candidate,
            openai_client,
            self._llm_model,
        )

        # 7. Build ResearchContext
        novelty = extracted["novelty_score"]
        novelty = max(0.0, min(1.0, float(novelty)))

        tile_sizes = [
            tuple(t) if isinstance(t, list) else t
            for t in extracted["suggested_tile_sizes"]
        ]

        ctx = ResearchContext(
            prior_implementations=extracted["prior_implementations"],
            known_pitfalls=extracted["known_pitfalls"],
            suggested_tile_sizes=tile_sizes,
            novelty_score=novelty,
        )

        # 8. Save to cache
        _save_cache(key, ctx, self._cache_dir)
        return ctx
