"""Conservative GPU norm metadata for the unchanged pedantic FP32 control."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

GAMMA = (512 * 2.0**-52) / (1.0 - 512 * 2.0**-52)


@triton.jit
def gpu_norm_metadata(vectors, norms, radii, uppers, invalid, D: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(vectors + row * D + tl.arange(0, D))
    v = x.to(tl.float64)
    center = tl.sum(v * v, 0)
    radius = center * tl.full((), G, tl.float64) + tl.full((), 1024 * 2.0**-1022, tl.float64)
    root = libdevice.sqrt_rn(center + radius)
    upper = libdevice.nextafter(root * tl.full((), 1.0 + 8 * 2.0**-52, tl.float64),
                               tl.full((), float('inf'), tl.float64))
    tl.store(norms + row, center)
    tl.store(radii + row, radius)
    tl.store(uppers + row, upper)
    bad = tl.sum((~((x == x) & (tl.abs(x) <= 1.0))).to(tl.int32), 0) > 0
    bad = bad | (~((center >= 0) & (upper < float('inf')) & (radius > 0)))
    tl.atomic_or(invalid, 1, mask=bad, sem='relaxed')


def prepare_norm_device(vectors):
    if (vectors.dtype != torch.float32 or vectors.ndim != 2 or vectors.shape[1] != 512
            or not vectors.is_cuda or not vectors.is_contiguous() or vectors.shape[0] < 1):
        raise ValueError('Only nonempty contiguous GPU FP32 D512 is admitted')
    outputs = tuple(torch.empty(vectors.shape[0], device=vectors.device, dtype=torch.float64) for _ in range(3))
    invalid = torch.zeros(1, dtype=torch.int32, device=vectors.device)
    gpu_norm_metadata[(vectors.shape[0],)](vectors, *outputs, invalid, D=512, G=GAMMA,
                                         num_warps=4, enable_fp_fusion=False)
    if int(invalid.item()):
        raise RuntimeError('Invalid input or GPU norm metadata')
    return outputs
