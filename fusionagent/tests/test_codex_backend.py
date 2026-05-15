from __future__ import annotations

import subprocess
from unittest.mock import patch

from fusionagent.generator.codegen import KernelGenerator
from fusionagent.llm import extract_json_object, run_codex_cli_prompt
from fusionagent.research.retriever import _llm_extract_context
from fusionagent.types import FusionCandidate


def _make_candidate(**overrides) -> FusionCandidate:
    defaults = dict(
        ops=["relu", "add"],
        input_shapes=[(1024,), (1024,)],
        output_shape=(1024,),
        memory_bound=True,
        launch_overhead_us=2.5,
        graph_position=0,
    )
    defaults.update(overrides)
    return FusionCandidate(**defaults)


def test_extract_json_object_strips_fences():
    payload = '```json\n{"hello": "world"}\n```'
    assert extract_json_object(payload) == '{"hello": "world"}'


def test_run_codex_cli_prompt_returns_output_file_contents(tmp_path):
    output_path = tmp_path / "codex.txt"

    class _TempHandle:
        name = str(output_path)

        def __enter__(self):
            output_path.touch()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_run(cmd, input, text, capture_output, timeout, check):
        assert cmd[0] == "codex"
        assert cmd[-1] == "-"
        assert "SYSTEM" in input
        output_path.write_text("final answer")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch.dict("os.environ", {"FUSIONAGENT_CODEX_PATH": "codex"}, clear=False):
        with patch("fusionagent.llm.codex_cli_available", return_value=True):
            with patch("fusionagent.llm.tempfile.NamedTemporaryFile", return_value=_TempHandle()):
                with patch("fusionagent.llm.subprocess.run", side_effect=_fake_run):
                    result = run_codex_cli_prompt("sys", "user", model="gpt-5.4")

    assert result == "final answer"
    assert not output_path.exists()


def test_kernel_generator_uses_codex_cli_backend():
    kernel_code = "import torch\n\ndef fused_kernel(x):\n    return x\n\ndef reference(x):\n    return x\n"

    with patch.dict(
        "os.environ",
        {"FUSIONAGENT_LLM_BACKEND": "codex_cli"},
        clear=False,
    ):
        with patch("fusionagent.generator.codegen.codex_cli_available", return_value=True):
            with patch("fusionagent.generator.codegen.run_codex_cli_prompt", return_value=kernel_code) as mock_cli:
                generated = KernelGenerator(model="gpt-5.4").generate(_make_candidate(input_shapes=[(1024,)], output_shape=(1024,)))

    assert "def fused_kernel(" in generated
    mock_cli.assert_called_once()


def test_retriever_llm_extract_context_uses_codex_cli():
    arxiv_results = [{"title": "Paper", "summary": "Summary", "link": "http://example.com"}]
    github_results = [{"repo": "openai/triton", "path": "kernel.py", "snippet": "@triton.jit"}]
    response = """```json
    {
      "prior_implementations": ["impl"],
      "known_pitfalls": ["pitfall"],
      "suggested_tile_sizes": [[64, 64]],
      "novelty_score": 0.25
    }
    ```"""

    with patch.dict(
        "os.environ",
        {"FUSIONAGENT_LLM_BACKEND": "codex_cli"},
        clear=False,
    ):
        with patch("fusionagent.research.retriever.run_codex_cli_prompt", return_value=response) as mock_cli:
            result = _llm_extract_context(
                arxiv_results,
                github_results,
                _make_candidate(),
                openai_client=None,
                model="gpt-5.4",
            )

    assert result["prior_implementations"] == ["impl"]
    assert result["known_pitfalls"] == ["pitfall"]
    assert result["suggested_tile_sizes"] == [[64, 64]]
    assert result["novelty_score"] == 0.25
    mock_cli.assert_called_once()
