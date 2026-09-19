"""Panel-local pedantic dot intervals and bounded upper-pair compaction."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['row_start', 'column_start'])
def classify_pedantic_panel(
    dots, norm_centers, norm_radii, norm_l2_upper,
    result_ids, ambiguous_ids, counters,
    lower_out, upper_out,
    row_start, column_start, threshold,
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
    accept = valid & (upper <= threshold)
    reject = valid & (lower > threshold)
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
