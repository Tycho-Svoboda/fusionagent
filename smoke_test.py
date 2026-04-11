import pytest
import torch

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


@pytest.mark.gpu
def test_triton_vector_add():
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = vector_add(x, y)
    torch.testing.assert_close(out, x + y)
