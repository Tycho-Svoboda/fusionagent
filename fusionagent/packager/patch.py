"""Runtime patching helpers for verified fused kernels."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F


def load_kernel_module(kernel_code: str) -> ModuleType | None:
    """Load a kernel source string as a Python module.

    Returns ``None`` if the code cannot be imported or does not expose a
    callable ``fused_kernel``.
    """
    mod_name = f"_fusionagent_patch_kernel_{uuid.uuid4().hex}"
    tmp_dir = Path(tempfile.gettempdir()) / "fusionagent_patch_kernels"
    tmp_dir.mkdir(exist_ok=True)
    file_path = tmp_dir / f"{mod_name}.py"
    file_path.write_text(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "fused_kernel") and callable(mod.fused_kernel):
            return mod
        sys.modules.pop(mod_name, None)
        return None
    except Exception:
        sys.modules.pop(mod_name, None)
        return None


class _FusedNormWrapper(nn.Module):
    """Wrap a norm module with a fused callable and sticky eager fallback."""

    def __init__(self, original_module: nn.Module, fused_fn: Callable) -> None:
        super().__init__()
        self.weight = original_module.weight
        self.bias = getattr(original_module, "bias", None)
        self.fused_fn = fused_fn
        self.original_module = original_module
        self._failed = False

    def forward(self, x):
        if self._failed:
            return self.original_module(x)

        try:
            if self.bias is not None:
                return self.fused_fn(x, self.weight, self.bias)
            return self.fused_fn(x, self.weight)
        except Exception:
            self._failed = True
            return self.original_module(x)


class _FusedSiLUMulOp(nn.Module):
    """Wrap a fused ``silu(x) * y`` callable with sticky eager fallback."""

    def __init__(self, fused_fn: Callable) -> None:
        super().__init__()
        self.fused_fn = fused_fn
        self._failed = False

    def forward(self, gate, up):
        if self._failed:
            return F.silu(gate) * up
        try:
            return self.fused_fn(gate, up)
        except Exception:
            self._failed = True
            return F.silu(gate) * up


class _FusedSwiGLUWrapper(nn.Module):
    """Semantics-preserving wrapper for SwiGLU-like FFNs."""

    def __init__(self, base_ffn: nn.Module, fused_fn: Callable) -> None:
        super().__init__()
        self.base_ffn = base_ffn
        self.fused_op = _FusedSiLUMulOp(fused_fn)

    def forward(self, x):
        gate = self.base_ffn.gate_proj(x)
        up = self.base_ffn.up_proj(x)
        return self.base_ffn.down_proj(self.fused_op(gate, up))


def _is_transformer_like(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("attn_norm", "ffn_norm", "ffn"))


def _is_swiglu_like(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj"))


def _is_vit_like(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("norm1", "norm2", "mlp"))


def _is_norm_module(module: nn.Module) -> bool:
    return hasattr(module, "weight")


class FusionPatcher:
    """Registry of verified fused kernels plus safe runtime patching."""

    def __init__(self) -> None:
        self.kernels: dict[str, tuple[str, Callable, list[str]]] = {}
        self.patch_log: list[str] = []

    def register(self, ops, kernel_code, fused_fn) -> None:
        """Register a verified kernel for an ops pattern."""
        self.kernels[self._pattern_key(ops)] = (kernel_code, fused_fn, list(ops))

    def _pattern_key(self, ops) -> str:
        ops_lower = [str(op).lower() for op in ops]
        if "rmsnorm" in ops_lower:
            if "silu" in ops_lower:
                return "rmsnorm_silu"
            return "rmsnorm"
        if "layernorm" in ops_lower:
            if "gelu" in ops_lower:
                return "layernorm_gelu"
            return "layernorm"
        if "silu" in ops_lower and "mul" in ops_lower:
            return "silu_mul"
        return "_".join(ops_lower)

    def has_pattern(self, key: str) -> bool:
        return key in self.kernels

    def patch_model(self, model: nn.Module, model_type: str = "transformer") -> int:
        """Patch eligible modules in-place and return the number applied."""
        self.patch_log = []
        n_patches = 0

        if model_type in ("transformer", "auto"):
            n_patches += self._patch_transformer_like(model)
        if model_type in ("vit", "auto"):
            n_patches += self._patch_vit_like(model)

        return n_patches

    def _patch_transformer_like(self, model: nn.Module) -> int:
        n_patches = 0

        for name, module in list(model.named_modules()):
            if not _is_transformer_like(module):
                continue

            if self.has_pattern("rmsnorm"):
                _, fused_fn, _ = self.kernels["rmsnorm"]
                if _is_norm_module(module.attn_norm):
                    module.attn_norm = _FusedNormWrapper(module.attn_norm, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.attn_norm with rmsnorm")
                if _is_norm_module(module.ffn_norm):
                    module.ffn_norm = _FusedNormWrapper(module.ffn_norm, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.ffn_norm with rmsnorm")

            if self.has_pattern("rmsnorm_silu"):
                self.patch_log.append(
                    f"Skipped {name}: safe mode does not apply rmsnorm_silu to plain norm sites"
                )

            if self.has_pattern("silu_mul") and _is_swiglu_like(module.ffn):
                _, fused_fn, _ = self.kernels["silu_mul"]
                module.ffn = _FusedSwiGLUWrapper(module.ffn, fused_fn)
                n_patches += 1
                self.patch_log.append(f"Patched {name}.ffn with silu_mul")

        return n_patches

    def _patch_vit_like(self, model: nn.Module) -> int:
        n_patches = 0

        for name, module in list(model.named_modules()):
            if not _is_vit_like(module):
                continue

            if self.has_pattern("layernorm"):
                _, fused_fn, _ = self.kernels["layernorm"]
                if _is_norm_module(module.norm1):
                    module.norm1 = _FusedNormWrapper(module.norm1, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.norm1 with layernorm")
                if _is_norm_module(module.norm2):
                    module.norm2 = _FusedNormWrapper(module.norm2, fused_fn)
                    n_patches += 1
                    self.patch_log.append(f"Patched {name}.norm2 with layernorm")

            if self.has_pattern("layernorm_gelu"):
                self.patch_log.append(
                    f"Skipped {name}: safe mode does not apply layernorm_gelu to plain norm sites"
                )

        return n_patches


__all__ = ["FusionPatcher", "load_kernel_module"]
