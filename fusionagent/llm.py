"""Shared LLM backend helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_BACKEND_OPENAI = "openai"
_BACKEND_CODEX_CLI = "codex_cli"
_DEFAULT_CODEX_PATH = "codex"
_DEFAULT_CODEX_SANDBOX = "read-only"


def selected_backend() -> str:
    """Return the configured LLM backend."""
    backend = os.environ.get("FUSIONAGENT_LLM_BACKEND", _BACKEND_OPENAI).strip().lower()
    return backend or _BACKEND_OPENAI


def using_codex_cli() -> bool:
    """Return True when Codex CLI should be used for LLM calls."""
    return selected_backend() == _BACKEND_CODEX_CLI


def codex_cli_path() -> str:
    """Return the configured Codex CLI binary path."""
    return os.environ.get("FUSIONAGENT_CODEX_PATH", _DEFAULT_CODEX_PATH)


def codex_cli_available() -> bool:
    """Return True when the Codex CLI binary is available."""
    path = codex_cli_path()
    if Path(path).exists():
        return True
    return shutil.which(path) is not None


def extract_json_object(text: str) -> str:
    """Extract the first JSON object-like payload from text."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Codex CLI response")
    return stripped[start : end + 1]


def run_codex_cli_prompt(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    timeout_s: int = 300,
) -> str:
    """Run a single non-interactive Codex CLI prompt and return the final message."""
    if not codex_cli_available():
        raise FileNotFoundError(
            f"Codex CLI not found at {codex_cli_path()!r}"
        )

    sandbox = os.environ.get("FUSIONAGENT_CODEX_SANDBOX", _DEFAULT_CODEX_SANDBOX)
    payload = (
        "Follow the SYSTEM instructions exactly.\n\n"
        "<SYSTEM>\n"
        f"{system_prompt.strip()}\n"
        "</SYSTEM>\n\n"
        "<USER>\n"
        f"{user_prompt.strip()}\n"
        "</USER>\n"
    )

    with tempfile.NamedTemporaryFile(
        prefix="fusionagent_codex_",
        suffix=".txt",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)

    cmd = [
        codex_cli_path(),
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
        "--color",
        "never",
        "--ignore-user-config",
        "-m",
        model,
        "-o",
        str(output_path),
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            log_tail = "\n".join(
                part.strip()
                for part in (proc.stdout, proc.stderr)
                if part and part.strip()
            )[-1200:]
            raise RuntimeError(
                f"codex exec failed with exit code {proc.returncode}: {log_tail}"
            )

        final_message = output_path.read_text().strip()
        if not final_message:
            raise RuntimeError("codex exec returned an empty final message")
        return final_message
    finally:
        output_path.unlink(missing_ok=True)
