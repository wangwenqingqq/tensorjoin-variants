"""Fixed direct-FP32 refinement components; no Tensor Core or online router."""

import triton
import triton.language as tl


@triton.jit
def interval_state(distance, magnitude, different, lo, hi):
    radius = (2.0 ** -14) * tl.abs(distance) + (2.0 ** -22) * tl.abs(magnitude) + 4.81482486096809e-35
    radius = radius + (2.0 ** -22) * (tl.abs(distance) + radius) + 4.81482486096809e-35
    lower = tl.maximum(distance - radius, 0.0)
    upper = distance + radius
    accept = (different == 0) | (upper <= lo)
    reject = (different != 0) & (lower > hi)
    return tl.where(accept, 1, tl.where(reject, 0, 2)).to(tl.uint8)


@triton.jit
def pair_filter(X, IDs, OUT, Q: tl.constexpr, N: tl.constexpr,
                D: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
                B: tl.constexpr, K: tl.constexpr):
    slots = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(IDs + slots, slots < Q, 0)
    rows = ids // N
    cols = ids % N
    distance = tl.full((B,), 0, tl.float32)
    magnitude = tl.full((B,), 0, tl.float32)
    different = tl.full((B,), 0, tl.int32)
    for start in range(tl.cdiv(D, K)):
        features = start * K + tl.arange(0, K)
        x = tl.load(X + rows[:, None] * D + features[None, :],
                    (slots[:, None] < Q) & (features[None, :] < D), 0)
        y = tl.load(X + cols[:, None] * D + features[None, :],
                    (slots[:, None] < Q) & (features[None, :] < D), 0)
        delta = x - y
        absolute = tl.abs(x) + tl.abs(y)
        distance += tl.sum(delta * delta, 1)
        magnitude += tl.sum(absolute * absolute, 1)
        different += tl.sum((x != y).to(tl.int32), 1)
    state = interval_state(distance, magnitude, different, LO, HI)
    tl.store(OUT + slots, state, slots < Q)


@triton.jit
def tile_filter(X, TILES, SLOTS, OUT, N: tl.constexpr, D: tl.constexpr,
                LO: tl.constexpr, HI: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    tile = tl.load(TILES + pid)
    rows = (tile // (N // 4)) * 4 + tl.arange(0, 4)
    cols = (tile % (N // 4)) * 4 + tl.arange(0, 4)
    distance = tl.full((4, 4), 0, tl.float32)
    magnitude = tl.full((4, 4), 0, tl.float32)
    different = tl.full((4, 4), 0, tl.int32)
    for start in range(tl.cdiv(D, K)):
        features = start * K + tl.arange(0, K)
        x = tl.load(X + rows[:, None, None] * D + features[None, None, :],
                    features[None, None, :] < D, 0)
        y = tl.load(X + cols[None, :, None] * D + features[None, None, :],
                    features[None, None, :] < D, 0)
        delta = x - y
        absolute = tl.abs(x) + tl.abs(y)
        distance += tl.sum(delta * delta, 2)
        magnitude += tl.sum(absolute * absolute, 2)
        different += tl.sum((x != y).to(tl.int32), 2)
    state = interval_state(distance, magnitude, different, LO, HI)
    cells = tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :]
    slots = tl.load(SLOTS + pid * 16 + cells)
    tl.store(OUT + slots, state, slots >= 0)
