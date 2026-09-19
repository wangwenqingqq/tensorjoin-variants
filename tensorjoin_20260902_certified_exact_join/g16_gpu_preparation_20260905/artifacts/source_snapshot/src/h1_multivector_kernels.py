#!/usr/bin/env python3
"""Triton kernels for the H1 exact multi-vector join screen."""

import triton
import triton.language as tl


@triton.jit
def token_l2_interval_rect_i64(
    query_codes,
    base_codes_transposed,
    query_scales,
    base_scales,
    query_reconstructed_norm2,
    base_reconstructed_norm2,
    query_residual_error_upper,
    base_residual_error_upper,
    lower_l2,
    upper_l2,
    expression_relative_radius,
    expression_absolute_radius,
    final_relative_radius,
    final_absolute_radius,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Materialize conservative L2 intervals for a rectangular token panel."""
    program_m = tl.program_id(axis=0)
    program_n = tl.program_id(axis=1)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_block = tl.load(
            query_codes + offsets_m[:, None] * K + k[None, :],
            mask=(offsets_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        base_block = tl.load(
            base_codes_transposed + k[:, None] * N_ + offsets_n[None, :],
            mask=(k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(
            query_block, base_block, accumulator, out_dtype=tl.int32
        )

    query_scale = tl.load(
        query_scales + offsets_m, mask=offsets_m < M, other=0.0
    )
    base_scale = tl.load(
        base_scales + offsets_n, mask=offsets_n < N_, other=0.0
    )
    query_norm = tl.load(
        query_reconstructed_norm2 + offsets_m,
        mask=offsets_m < M,
        other=0.0,
    )
    base_norm = tl.load(
        base_reconstructed_norm2 + offsets_n,
        mask=offsets_n < N_,
        other=0.0,
    )
    query_error = tl.load(
        query_residual_error_upper + offsets_m,
        mask=offsets_m < M,
        other=0.0,
    )
    base_error = tl.load(
        base_residual_error_upper + offsets_n,
        mask=offsets_n < N_,
        other=0.0,
    )
    scaled_dot = (
        accumulator.to(tl.float32)
        * query_scale[:, None]
        * base_scale[None, :]
    )
    reconstructed_d2 = (
        query_norm[:, None] + base_norm[None, :] - 2.0 * scaled_dot
    )
    term_abs_sum = (
        tl.abs(query_norm[:, None])
        + tl.abs(base_norm[None, :])
        + 2.0 * tl.abs(scaled_dot)
    )
    expression_radius = (
        expression_relative_radius * term_abs_sum + expression_absolute_radius
    )
    reconstructed_lower = tl.sqrt(
        tl.maximum(reconstructed_d2 - expression_radius, 0.0)
    )
    reconstructed_upper = tl.sqrt(
        tl.maximum(reconstructed_d2 + expression_radius, 0.0)
    )
    residual_radius = query_error[:, None] + base_error[None, :]
    final_magnitude = tl.maximum(reconstructed_upper + residual_radius, 1.0)
    final_radius = final_relative_radius * final_magnitude + final_absolute_radius
    lower = tl.maximum(
        reconstructed_lower - residual_radius - final_radius, 0.0
    )
    upper = reconstructed_upper + residual_radius + final_radius
    valid = (offsets_m[:, None] < M) & (offsets_n[None, :] < N_)
    locations = offsets_m[:, None] * N_ + offsets_n[None, :]
    tl.store(lower_l2 + locations, lower, mask=valid)
    tl.store(upper_l2 + locations, upper, mask=valid)


@triton.jit
def aggregate_interval_objects_i64(
    lower_l2,
    upper_l2,
    query_starts,
    query_counts,
    base_starts,
    base_counts,
    threshold_ptr,
    object_lower,
    object_upper,
    result_ids,
    ambiguous_ids,
    counters,
    square_absolute_guard,
    QUERY_OBJECTS: tl.constexpr,
    BASE_OBJECTS: tl.constexpr,
    BASE_TOKENS: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """Lift token L2 intervals through squared symmetric Chamfer."""
    pair_id = tl.program_id(axis=0)
    query_object = pair_id // BASE_OBJECTS
    base_object = pair_id - query_object * BASE_OBJECTS
    query_start = tl.load(query_starts + query_object)
    base_start = tl.load(base_starts + base_object)
    query_count = tl.load(query_counts + query_object)
    base_count = tl.load(base_counts + base_object)
    i = tl.arange(0, MAX_TOKENS)
    j = tl.arange(0, MAX_TOKENS)
    valid = (i[:, None] < query_count) & (j[None, :] < base_count)
    locations = (
        (query_start + i[:, None]) * BASE_TOKENS
        + base_start
        + j[None, :]
    )
    lower = tl.load(lower_l2 + locations, mask=valid, other=float("inf")).to(
        tl.float64
    )
    upper = tl.load(upper_l2 + locations, mask=valid, other=float("inf")).to(
        tl.float64
    )
    lower2 = tl.maximum(lower * lower - square_absolute_guard, 0.0)
    upper2 = upper * upper + square_absolute_guard

    lower_forward = tl.min(lower2, axis=1)
    lower_reverse = tl.min(lower2, axis=0)
    upper_forward = tl.min(upper2, axis=1)
    upper_reverse = tl.min(upper2, axis=0)
    valid_i = i < query_count
    valid_j = j < base_count
    lower_score = 0.5 * (
        tl.sum(tl.where(valid_i, lower_forward, 0.0), axis=0)
        / query_count.to(tl.float64)
        + tl.sum(tl.where(valid_j, lower_reverse, 0.0), axis=0)
        / base_count.to(tl.float64)
    )
    upper_score = 0.5 * (
        tl.sum(tl.where(valid_i, upper_forward, 0.0), axis=0)
        / query_count.to(tl.float64)
        + tl.sum(tl.where(valid_j, upper_reverse, 0.0), axis=0)
        / base_count.to(tl.float64)
    )
    tl.store(object_lower + pair_id, lower_score)
    tl.store(object_upper + pair_id, upper_score)

    threshold = tl.load(threshold_ptr)
    accept = upper_score <= threshold
    reject = lower_score > threshold
    ambiguous = ~(accept | reject)
    lane = tl.zeros((1,), dtype=tl.int32)
    accepted_position = tl.atomic_add(
        counters + lane, 1, mask=accept, sem="relaxed"
    )
    accepted_store = accept & (accepted_position < CAPACITY)
    tl.store(result_ids + accepted_position, pair_id, mask=accepted_store)
    tl.atomic_add(
        counters + 2 + lane,
        1,
        mask=accept & ~accepted_store,
        sem="relaxed",
    )
    ambiguous_position = tl.atomic_add(
        counters + 1 + lane, 1, mask=ambiguous, sem="relaxed"
    )
    ambiguous_store = ambiguous & (ambiguous_position < CAPACITY)
    tl.store(
        ambiguous_ids + ambiguous_position, pair_id, mask=ambiguous_store
    )
    tl.atomic_add(
        counters + 2 + lane,
        1,
        mask=ambiguous & ~ambiguous_store,
        sem="relaxed",
    )


@triton.jit
def exact_token_distances_fp64(
    query,
    base,
    output,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute every direct-difference squared token distance in FP64."""
    query_rows = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    base_rows = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    distance = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + k_offsets
        query_values = tl.load(
            query + query_rows[:, None] * K + k[None, :],
            mask=(query_rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float64)
        base_values = tl.load(
            base + base_rows[:, None] * K + k[None, :],
            mask=(base_rows[:, None] < N_) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float64)
        delta = query_values[:, None, :] - base_values[None, :, :]
        distance += tl.sum(delta * delta, axis=2)
    valid = (query_rows[:, None] < M) & (base_rows[None, :] < N_)
    locations = query_rows[:, None] * N_ + base_rows[None, :]
    tl.store(output + locations, distance, mask=valid)


@triton.jit
def exact_ambiguous_panels_fp64(
    query,
    base,
    ambiguous_ids,
    query_starts,
    query_counts,
    base_starts,
    base_counts,
    exact_panels,
    AMBIGUOUS_COUNT: tl.constexpr,
    BASE_OBJECTS: tl.constexpr,
    K: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute padded dense direct-FP64 panels for compacted object pairs."""
    cell_id = tl.program_id(axis=0)
    panel_id = cell_id // (MAX_TOKENS * MAX_TOKENS)
    local_cell = cell_id - panel_id * (MAX_TOKENS * MAX_TOKENS)
    local_query = local_cell // MAX_TOKENS
    local_base = local_cell - local_query * MAX_TOKENS
    pair_id = tl.load(ambiguous_ids + panel_id)
    query_object = pair_id // BASE_OBJECTS
    base_object = pair_id - query_object * BASE_OBJECTS
    query_start = tl.load(query_starts + query_object)
    base_start = tl.load(base_starts + base_object)
    query_count = tl.load(query_counts + query_object)
    base_count = tl.load(base_counts + base_object)
    valid = (local_query < query_count) & (local_base < base_count)
    query_row = query_start + local_query
    base_row = base_start + local_base
    k_offsets = tl.arange(0, BLOCK_K)
    distance = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + k_offsets
        query_values = tl.load(
            query + query_row * K + k,
            mask=valid & (k < K),
            other=0.0,
        ).to(tl.float64)
        base_values = tl.load(
            base + base_row * K + k,
            mask=valid & (k < K),
            other=0.0,
        ).to(tl.float64)
        delta = query_values - base_values
        distance += tl.sum(delta * delta, axis=0)
    lane = tl.arange(0, 1)
    tl.store(
        exact_panels + cell_id + lane,
        tl.where(valid, distance, float("inf")),
    )


@triton.jit
def aggregate_exact_matrix_i64(
    exact_tokens,
    query_starts,
    query_counts,
    base_starts,
    base_counts,
    threshold_ptr,
    object_scores,
    result_ids,
    counters,
    BASE_OBJECTS: tl.constexpr,
    BASE_TOKENS: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """Aggregate a full exact token matrix and compact qualifying objects."""
    pair_id = tl.program_id(axis=0)
    query_object = pair_id // BASE_OBJECTS
    base_object = pair_id - query_object * BASE_OBJECTS
    query_start = tl.load(query_starts + query_object)
    base_start = tl.load(base_starts + base_object)
    query_count = tl.load(query_counts + query_object)
    base_count = tl.load(base_counts + base_object)
    i = tl.arange(0, MAX_TOKENS)
    j = tl.arange(0, MAX_TOKENS)
    valid = (i[:, None] < query_count) & (j[None, :] < base_count)
    locations = (
        (query_start + i[:, None]) * BASE_TOKENS
        + base_start
        + j[None, :]
    )
    distances = tl.load(
        exact_tokens + locations, mask=valid, other=float("inf")
    )
    forward = tl.min(distances, axis=1)
    reverse = tl.min(distances, axis=0)
    valid_i = i < query_count
    valid_j = j < base_count
    score = 0.5 * (
        tl.sum(tl.where(valid_i, forward, 0.0), axis=0)
        / query_count.to(tl.float64)
        + tl.sum(tl.where(valid_j, reverse, 0.0), axis=0)
        / base_count.to(tl.float64)
    )
    tl.store(object_scores + pair_id, score)
    inside = score <= tl.load(threshold_ptr)
    lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(counters + lane, 1, mask=inside, sem="relaxed")
    can_store = inside & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )


@triton.jit
def aggregate_exact_panels_i64(
    exact_panels,
    ambiguous_ids,
    query_counts,
    base_counts,
    threshold_ptr,
    refined_scores,
    result_ids,
    counters,
    BASE_OBJECTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """Aggregate compacted exact panels and append qualifying object IDs."""
    panel_id = tl.program_id(axis=0)
    pair_id = tl.load(ambiguous_ids + panel_id)
    query_object = pair_id // BASE_OBJECTS
    base_object = pair_id - query_object * BASE_OBJECTS
    query_count = tl.load(query_counts + query_object)
    base_count = tl.load(base_counts + base_object)
    i = tl.arange(0, MAX_TOKENS)
    j = tl.arange(0, MAX_TOKENS)
    locations = panel_id * (MAX_TOKENS * MAX_TOKENS) + i[:, None] * MAX_TOKENS + j[None, :]
    distances = tl.load(exact_panels + locations)
    forward = tl.min(distances, axis=1)
    reverse = tl.min(distances, axis=0)
    valid_i = i < query_count
    valid_j = j < base_count
    score = 0.5 * (
        tl.sum(tl.where(valid_i, forward, 0.0), axis=0)
        / query_count.to(tl.float64)
        + tl.sum(tl.where(valid_j, reverse, 0.0), axis=0)
        / base_count.to(tl.float64)
    )
    tl.store(refined_scores + panel_id, score)
    inside = score <= tl.load(threshold_ptr)
    lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(counters + lane, 1, mask=inside, sem="relaxed")
    can_store = inside & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )
