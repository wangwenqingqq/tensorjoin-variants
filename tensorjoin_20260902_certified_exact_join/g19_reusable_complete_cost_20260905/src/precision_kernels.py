"""Panel-local pedantic dot intervals and bounded upper-pair compaction."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['row_start', 'column_start'])
def classify_pedantic_panel(
    dots, norm_centers, norm_radii, norm_l2_upper,
    result_ids, ambiguous_ids, counters,
    lower_out, upper_out,
    row_start, column_start, threshold: tl.constexpr,
    N_: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
    CAPACITY: tl.constexpr, GAMMA: tl.constexpr, ABS_DOT: tl.constexpr,
    BLOCK: tl.constexpr, DUMP: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside_panel = offset < ROWS * COLS
    local_row = offset // COLS
    local_column = offset - local_row * COLS
    row = row_start + local_row
    column = column_start + local_column
    valid = inside_panel & (row <= column) & (row < N_) & (column < N_)
    dot = tl.load(dots + offset, mask=inside_panel, other=0).to(tl.float64)
    qn = tl.load(norm_centers + row, mask=inside_panel, other=0)
    bn = tl.load(norm_centers + column, mask=inside_panel, other=0)
    qnr = tl.load(norm_radii + row, mask=inside_panel, other=0)
    bnr = tl.load(norm_radii + column, mask=inside_panel, other=0)
    ql2 = tl.load(norm_l2_upper + row, mask=inside_panel, other=0)
    bl2 = tl.load(norm_l2_upper + column, mask=inside_panel, other=0)
    gamma = tl.full((), GAMMA, tl.float64)
    dot_radius = gamma * ql2 * bl2 + tl.full((), ABS_DOT, tl.float64)
    center = qn + bn - 2.0 * dot
    magnitude = tl.abs(qn) + tl.abs(bn) + 2.0 * tl.abs(dot) + qnr + bnr + 2.0 * dot_radius
    radius = (qnr + bnr + 2.0 * dot_radius
              + tl.full((), 32.0 * 2.0**-52, tl.float64) * magnitude
              + tl.full((), 1e-12, tl.float64)
              + tl.full((), 2.0**-38, tl.float64) * (tl.abs(qn) + tl.abs(bn) + qnr + bnr))
    lower = tl.maximum(center - radius, 0.0)
    upper = center + radius
    if DUMP:
        tl.store(lower_out + offset, lower, mask=inside_panel)
        tl.store(upper_out + offset, upper, mask=inside_panel)
    accept = valid & (upper <= tl.full((), threshold, tl.float64))
    reject = valid & (lower > tl.full((), threshold, tl.float64))
    ambiguous = valid & ~(accept | reject)
    accept_i = accept.to(tl.int32)
    ambiguous_i = ambiguous.to(tl.int32)
    accepts = tl.sum(accept_i, 0)
    ambiguities = tl.sum(ambiguous_i, 0)
    rejects = tl.sum(reject.to(tl.int32), 0)
    accept_base = tl.atomic_add(counters, accepts, mask=accepts > 0, sem='relaxed')
    ambiguous_base = tl.atomic_add(counters + 1, ambiguities, mask=ambiguities > 0, sem='relaxed')
    tl.atomic_add(counters + 3, rejects, mask=rejects > 0, sem='relaxed')
    accept_position = accept_base + tl.cumsum(accept_i, 0) - 1
    ambiguous_position = ambiguous_base + tl.cumsum(ambiguous_i, 0) - 1
    pair_id = row.to(tl.int64) * N_ + column.to(tl.int64)
    tl.store(result_ids + accept_position, pair_id, mask=accept & (accept_position < CAPACITY))
    tl.store(ambiguous_ids + ambiguous_position, pair_id, mask=ambiguous & (ambiguous_position < CAPACITY))
    overflow = tl.sum((accept & (accept_position >= CAPACITY)).to(tl.int32), 0)
    overflow += tl.sum((ambiguous & (ambiguous_position >= CAPACITY)).to(tl.int32), 0)
    tl.atomic_add(counters + 2, overflow, mask=overflow > 0, sem='relaxed')


@triton.jit
def refine_ambiguous_fp64_i64(
    query,
    base,
    fp64_ids,
    result_ids,
    counters,
    threshold_d2: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(fp64_ids + pair_index)
    query_row = pair_id // N_
    base_row = pair_id - query_row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_values = tl.load(query + query_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        base_values = tl.load(base + base_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        delta = query_values - base_values
        distance_d2 += tl.sum(delta * delta, axis=0)
    inside = distance_d2 <= tl.full((), threshold_d2, tl.float64)
    counter_lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(counters + counter_lane, 1, mask=inside, sem="relaxed")
    can_store = inside & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )

