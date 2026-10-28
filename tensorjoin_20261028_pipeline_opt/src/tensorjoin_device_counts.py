"""Device-count scheduling variants; mathematical bodies match the frozen kernels."""
import triton
import triton.language as tl

@triton.jit
def scan_device_start(
    query_codes,
    base_codes_transposed,
    query_scales,
    base_scales,
    query_reconstructed_norm2,
    base_reconstructed_norm2,
    query_errors,
    base_errors,
    tile_rows,
    tile_columns,
    tile_start,
    result_ids,
    ambiguous_ids,
    counters,
    threshold_d2,
    bound_pad,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.load(tile_start) + tl.program_id(axis=0)
    program_m = tl.load(tile_rows + program)
    program_n = tl.load(tile_columns + program)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    query_ptrs = query_codes + offsets_m[:, None] * K + offsets_k[None, :]
    base_ptrs = base_codes_transposed + offsets_k[:, None] * N_ + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        query_block = tl.load(
            query_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        base_block = tl.load(
            base_ptrs,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(query_block, base_block, accumulator, out_dtype=tl.int32)
        query_ptrs += BLOCK_K
        base_ptrs += BLOCK_K * N_

    query_scale = tl.load(query_scales + offsets_m, mask=offsets_m < M, other=0.0)
    base_scale = tl.load(base_scales + offsets_n, mask=offsets_n < N_, other=0.0)
    query_norm2 = tl.load(
        query_reconstructed_norm2 + offsets_m, mask=offsets_m < M, other=0.0
    )
    base_norm2 = tl.load(
        base_reconstructed_norm2 + offsets_n, mask=offsets_n < N_, other=0.0
    )
    query_error = tl.load(query_errors + offsets_m, mask=offsets_m < M, other=0.0)
    base_error = tl.load(base_errors + offsets_n, mask=offsets_n < N_, other=0.0)
    dot = accumulator.to(tl.float32) * query_scale[:, None] * base_scale[None, :]
    reconstructed_d2 = query_norm2[:, None] + base_norm2[None, :] - 2.0 * dot
    reconstructed_d = tl.sqrt(tl.maximum(reconstructed_d2, 0.0))
    radius = query_error[:, None] + base_error[None, :] + bound_pad
    lower = tl.maximum(reconstructed_d - radius, 0.0)
    upper = reconstructed_d + radius
    lower_d2 = lower * lower
    upper_d2 = upper * upper
    valid = (
        (offsets_m[:, None] < M)
        & (offsets_n[None, :] < N_)
        & (offsets_m[:, None] <= offsets_n[None, :])
    )
    accept = valid & (upper_d2 <= threshold_d2)
    reject = valid & (lower_d2 > threshold_d2)
    ambiguous = valid & ~(accept | reject)
    pair_ids = (
        offsets_m[:, None].to(tl.int64) * N_ + offsets_n[None, :].to(tl.int64)
    )

    counter_lanes = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accept_positions = tl.atomic_add(
        counters + counter_lanes, 1, mask=accept, sem="relaxed"
    )
    accept_store = accept & (accept_positions < CAPACITY)
    tl.store(result_ids + accept_positions, pair_ids, mask=accept_store)
    tl.atomic_add(
        counters + 2 + counter_lanes,
        1,
        mask=accept & ~accept_store,
        sem="relaxed",
    )

    ambiguous_positions = tl.atomic_add(
        counters + 1 + counter_lanes, 1, mask=ambiguous, sem="relaxed"
    )
    ambiguous_store = ambiguous & (ambiguous_positions < CAPACITY)
    tl.store(ambiguous_ids + ambiguous_positions, pair_ids, mask=ambiguous_store)
    tl.atomic_add(
        counters + 2 + counter_lanes,
        1,
        mask=ambiguous & ~ambiguous_store,
        sem="relaxed",
    )


@triton.jit
def filter_device_count(
    query,
    base,
    ambiguous_ids,
    result_ids,
    fp64_ids,
    counters,
    threshold_d2,
    fp32_guard,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    count = tl.minimum(tl.load(counters + 1), CAPACITY)
    for pair_index in range(tl.program_id(0), count, tl.num_programs(0)):
        pair_id = tl.load(ambiguous_ids + pair_index)
        query_row = pair_id // N_
        base_row = pair_id - query_row * N_
        offsets_k = tl.arange(0, BLOCK_K)
        distance_d2 = tl.zeros((1,), dtype=tl.float32)
        different_coordinates = tl.zeros((1,), dtype=tl.int32)
        for block_start in range(0, K, BLOCK_K):
            k = block_start + offsets_k
            query_values = tl.load(query + query_row * K + k, mask=k < K, other=0.0)
            base_values = tl.load(base + base_row * K + k, mask=k < K, other=0.0)
            delta = query_values - base_values
            distance_d2 += tl.sum(delta * delta, axis=0)
            different_coordinates += tl.sum((query_values != base_values).to(tl.int32), axis=0)

        bitwise_equal = different_coordinates == 0
        accept = bitwise_equal | (distance_d2 <= threshold_d2 - fp32_guard)
        reject = (~bitwise_equal) & (distance_d2 > threshold_d2 + fp32_guard)
        needs_fp64 = ~(accept | reject)
        counter_lane = tl.zeros((1,), dtype=tl.int32)
        position = tl.atomic_add(counters + counter_lane, 1, mask=accept, sem="relaxed")
        can_store = accept & (position < CAPACITY)
        tl.store(result_ids + position, pair_id, mask=can_store)
        tl.atomic_add(
            counters + 2 + counter_lane,
            1,
            mask=accept & ~can_store,
            sem="relaxed",
        )
        fp64_position = tl.atomic_add(
            counters + 3 + counter_lane, 1, mask=needs_fp64, sem="relaxed"
        )
        fp64_store = needs_fp64 & (fp64_position < CAPACITY)
        tl.store(fp64_ids + fp64_position, pair_id, mask=fp64_store)
        tl.atomic_add(
            counters + 2 + counter_lane,
            1,
            mask=needs_fp64 & ~fp64_store,
            sem="relaxed",
        )
        tl.atomic_add(counters + 4 + counter_lane, 1, mask=bitwise_equal, sem="relaxed")


@triton.jit
def refine_device_count(
    query,
    base,
    fp64_ids,
    result_ids,
    counters,
    threshold_d2,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    count = tl.minimum(tl.load(counters + 3), CAPACITY)
    for pair_index in range(tl.program_id(0), count, tl.num_programs(0)):
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
        inside = distance_d2 <= threshold_d2
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


