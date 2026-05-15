"""Tests for the runtime packager patching surface."""

from __future__ import annotations

import copy
import textwrap

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusionagent.packager.patch import (
    FusionPatcher,
    _FusedNormWrapper,
    load_kernel_module,
)


_VALID_KERNEL_CODE = textwrap.dedent("""\
    def fused_kernel(x, y):
        return x + y
""")

_INVALID_KERNEL_CODE = textwrap.dedent("""\
    def fused_kernel(x, y)
        return x + y
""")


class ToyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class ToySwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 8, hidden: int = 16):
        super().__init__()
        self.attn_norm = ToyRMSNorm(dim)
        self.ffn_norm = ToyRMSNorm(dim)
        self.ffn = ToySwiGLUFFN(dim, hidden)

    def forward(self, x):
        h = self.attn_norm(x)
        h = self.ffn_norm(h)
        return x + self.ffn(h)


class ToyViTBlock(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        x = x + self.mlp(self.norm1(x))
        return self.norm2(x)


class TestLoadKernelModule:
    def test_valid_kernel_source_loads(self):
        mod = load_kernel_module(_VALID_KERNEL_CODE)
        assert mod is not None
        assert callable(mod.fused_kernel)

    def test_invalid_kernel_source_returns_none(self):
        assert load_kernel_module(_INVALID_KERNEL_CODE) is None


class TestFusedNormWrapper:
    def test_preserves_outputs_when_kernel_matches_reference(self):
        layer = nn.LayerNorm(4)
        wrapper = _FusedNormWrapper(
            layer,
            lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, layer.eps),
        )

        x = torch.randn(3, 4)
        torch.testing.assert_close(wrapper(x), layer(x))

    def test_falls_back_permanently_after_failure(self):
        layer = nn.LayerNorm(4)
        calls = {"count": 0}

        def broken_kernel(x, weight, bias):
            calls["count"] += 1
            raise RuntimeError("boom")

        wrapper = _FusedNormWrapper(layer, broken_kernel)
        x = torch.randn(2, 4)

        out1 = wrapper(x)
        out2 = wrapper(x)

        torch.testing.assert_close(out1, layer(x))
        torch.testing.assert_close(out2, layer(x))
        assert calls["count"] == 1


class TestFusionPatcherCPU:
    def test_silu_mul_patch_preserves_output_on_toy_transformer(self):
        torch.manual_seed(0)
        model = ToyTransformerBlock()
        patched = copy.deepcopy(model)
        patcher = FusionPatcher()
        patcher.register(["silu", "mul"], _VALID_KERNEL_CODE, lambda gate, up: F.silu(gate) * up)

        x = torch.randn(2, 3, 8)
        expected = model(x)
        n_patches = patcher.patch_model(patched)
        actual = patched(x)

        assert n_patches == 1
        assert any("silu_mul" in line for line in patcher.patch_log)
        torch.testing.assert_close(actual, expected)

    def test_safe_mode_skips_rmsnorm_silu(self):
        patcher = FusionPatcher()
        patcher.register(["rmsnorm", "silu"], _VALID_KERNEL_CODE, lambda x, weight: x)

        model = ToyTransformerBlock()
        n_patches = patcher.patch_model(model)

        assert n_patches == 0
        assert any("safe mode" in line for line in patcher.patch_log)

    def test_layernorm_patch_preserves_output_on_toy_vit_block(self):
        torch.manual_seed(0)
        model = ToyViTBlock()
        patched = copy.deepcopy(model)
        patcher = FusionPatcher()
        patcher.register(
            ["layernorm"],
            _VALID_KERNEL_CODE,
            lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b),
        )

        x = torch.randn(2, 8)
        expected = model(x)
        n_patches = patcher.patch_model(patched, model_type="vit")
        actual = patched(x)

        assert n_patches == 2
        torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_gpu_silu_mul_patch_supports_forward_and_backward():
    torch.manual_seed(0)
    device = "cuda:0"
    model = ToyTransformerBlock().to(device)
    patcher = FusionPatcher()
    patcher.register(["silu", "mul"], _VALID_KERNEL_CODE, lambda gate, up: F.silu(gate) * up)

    applied = patcher.patch_model(model)
    x = torch.randn(2, 3, 8, device=device, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert applied == 1
    assert x.grad is not None
