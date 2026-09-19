"""Block32 producers and a native SM120 scaled Gram matrix."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from retained import cvup, cvdown


@triton.jit
def prepare_block32(x, q, scales, norm_lo, norm_hi, z_upper, residual_upper,
                    invalid, FORMAT: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 512)
    a = tl.load(x + row * 512 + k).reshape((16, 32))
    maximum = tl.max(tl.abs(a), 1)
    if FORMAT == 'e3m4':
        MAX: tl.constexpr = 30.0
    else:
        MAX: tl.constexpr = 448.0
    ratio = tl.maximum(maximum / MAX, 2.0**-100)
    exp = ((ratio.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32) - 127
    scale = ((exp + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scale = tl.where(maximum > scale * MAX, scale * 2.0, scale)
    scale = tl.where(maximum < 2.0**-100, 1.0, scale)
    sf = (scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(scales + row * 16 + tl.arange(0, 16), sf)
    v = libdevice.div_rn(a, scale[:, None])
    if FORMAT == 'e3m4':
        av = tl.abs(v)
        e = ((av.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32) - 127
        e = tl.maximum(e, -2)
        step = ((e - 4 + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        h = tl.minimum(libdevice.rint(av / step) * step, 30.0)
        he = ((h.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32) - 127
        he = tl.maximum(he, -2)
        inv = ((127 - he).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        mag = tl.where(h < 0.25, libdevice.rint(h * 64).to(tl.int32),
                       ((he + 3) << 4) + libdevice.rint((h * inv - 1.0) * 16).to(tl.int32))
        sign = (v.to(tl.uint32, bitcast=True) >> 24) & 128
        code = (mag.to(tl.uint32) | sign).to(tl.uint8)
        h = tl.where(sign != 0, -h, h)
    else:
        encoded = v.to(tl.float8e4nv)
        h = encoded.to(tl.float32)
        code = encoded.to(tl.uint8, bitcast=True)
    tl.store(q + row * 512 + k, code.reshape((512,)))

    # Same outward metadata as the retained per-row producer, but reconstruction
    # uses each element's block scale rather than one scale for the whole row.
    aa = a.to(tl.float64).reshape((512,))
    zz = (h.to(tl.float64) * scale[:, None].to(tl.float64)).reshape((512,))
    residual = aa - zz
    nx = tl.sum(aa * aa, 0); nz = tl.sum(zz * zz, 0); nr = tl.sum(residual * residual, 0)
    gamma = tl.full((), (512 * 2.0**-52) / (1 - 512 * 2.0**-52), tl.float64)
    tiny = tl.full((), 1024.0 * 2.0**-1022, tl.float64)
    rx = gamma * nx + tiny; rz = gamma * nz + tiny; rr = gamma * nr + tiny
    padding = tl.full((), 1.0 + 8.0 * 2.0**-52, tl.float64)
    zu = libdevice.nextafter(libdevice.sqrt_rn(nz + rz) * padding, tl.full((), float('inf'), tl.float64))
    eu = libdevice.nextafter(libdevice.sqrt_rn(nr + rr) * padding, tl.full((), float('inf'), tl.float64))
    tl.store(norm_lo + row, tl.maximum(cvdown(nx - rx), 0.0)); tl.store(norm_hi + row, cvup(nx + rx))
    tl.store(z_upper + row, cvup(zu)); tl.store(residual_upper + row, cvup(eu))
    bad = tl.sum((~((aa == aa) & (tl.abs(aa) <= 1.0))).to(tl.int32), 0) > 0
    tl.atomic_or(invalid, 1, mask=bad, sem='relaxed')


@triton.jit
def gram_block32(q, scales, out, N: tl.constexpr, D: tl.constexpr,
                 FA: tl.constexpr, FB: tl.constexpr,
                 BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 64):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    sk = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, D, BK):
        a = tl.load(q + i[:, None] * D + start + k[None, :], mask=i[:, None] < N, other=0)
        b = tl.load(q + j[None, :] * D + start + k[:, None], mask=j[None, :] < N, other=0)
        sa = tl.load(scales + i[:, None] * (D // 32) + start // 32 + sk[None, :],
                     mask=i[:, None] < N, other=127)
        sb = tl.load(scales + j[:, None] * (D // 32) + start // 32 + sk[None, :],
                     mask=j[:, None] < N, other=127)
        acc = tl.dot_scaled(a, sa, FA, b, sb, FB, acc, fast_math=False)
    tl.store(out + i[:, None] * N + j[None, :], acc,
             mask=(i[:, None] < N) & (j[None, :] < N))
