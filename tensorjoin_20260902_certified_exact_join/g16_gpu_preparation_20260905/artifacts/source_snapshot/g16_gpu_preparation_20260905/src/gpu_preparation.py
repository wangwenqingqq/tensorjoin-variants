"""Construct conservative G5 code/error metadata on the device."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

DIMENSION = 512
GAMMA64 = (2 * DIMENSION + 2) * 2.0**-53 / (1.0 - (2 * DIMENSION + 2) * 2.0**-53)


@triton.jit
def gpu_quantize_metadata(
    vectors, codes, scales, norms, errors, invalid,
    D: tl.constexpr, GAMMA: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, D)
    x = tl.load(vectors + row * D + column)
    maximum = tl.max(tl.abs(x), 0)
    scale = libdevice.div_rn(maximum, tl.full((), 127.0, tl.float32))
    scale = tl.where(scale == 0, 1.0, scale)
    quotient = libdevice.div_rn(x, scale)
    rounded = libdevice.rint(quotient)
    q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int32)
    tl.store(codes + row * D + column, q.to(tl.int8))
    code_norm = tl.sum(q * q, 0)
    scale64 = scale.to(tl.float64)
    squared_scale = scale64 * scale64
    norm32 = (code_norm.to(tl.float64) * squared_scale).to(tl.float32)
    reconstruction = q.to(tl.float64) * scale64
    delta = x.to(tl.float64) - reconstruction
    residual_sum = tl.sum(delta * delta, 0)
    adjusted = libdevice.div_rn(residual_sum, tl.full((), 1.0-GAMMA, tl.float64))
    root = libdevice.sqrt_rn(adjusted)
    root_upper = libdevice.nextafter(root, tl.full((), float('inf'), tl.float64))
    root_upper = tl.where(root == 0, 0.0, root_upper)
    error = root_upper.to(tl.float32)
    error = tl.where(error.to(tl.float64) < root_upper,
                     libdevice.nextafter(error, tl.full((), float('inf'), tl.float32)), error)
    error = tl.where((error > 0) & (error < 2.0**-126), 2.0**-126, error)
    tl.store(scales + row, scale)
    tl.store(norms + row, norm32)
    tl.store(errors + row, error)
    bad_input = tl.sum((~((x == x) & (tl.abs(x) <= 1.0))).to(tl.int32), 0) > 0
    bad_metadata = (~((scale > 0) & (scale < float('inf'))
                      & (norm32 >= 0) & (norm32 < float('inf'))
                      & (error >= 0) & (error < float('inf'))))
    tl.atomic_or(invalid, 1, mask=bad_input | bad_metadata, sem='relaxed')


def prepare_device(vectors):
    if (vectors.dtype != torch.float32 or vectors.ndim != 2 or vectors.shape[1] != DIMENSION
            or not vectors.is_cuda or not vectors.is_contiguous()):
        raise ValueError('Only contiguous GPU FP32 D512 is admitted')
    n = int(vectors.shape[0])
    if n < 1:
        raise ValueError('Empty input is outside this contract')
    codes = torch.empty((n, DIMENSION), dtype=torch.int8, device=vectors.device)
    scales = torch.empty(n, dtype=torch.float32, device=vectors.device)
    norms, errors = torch.empty_like(scales), torch.empty_like(scales)
    invalid = torch.zeros(1, dtype=torch.int32, device=vectors.device)
    gpu_quantize_metadata[(n,)](vectors, codes, scales, norms, errors, invalid,
                               D=DIMENSION, GAMMA=GAMMA64, num_warps=4,
                               enable_fp_fusion=False)
    if int(invalid.item()):
        raise RuntimeError('Invalid input or metadata: GPU preparation failed closed')
    codes_transposed = codes.T.contiguous()
    return codes, scales, norms, errors, codes_transposed
