"""Shared pytest configuration for FusionAgent tests."""

from __future__ import annotations

import pytest


def _gpu_test_available() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"Triton is unavailable: {exc}"

    return True, ""


def pytest_collection_modifyitems(config, items):
    available, reason = _gpu_test_available()
    if available:
        return

    skip_gpu = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
