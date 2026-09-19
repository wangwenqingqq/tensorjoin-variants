#!/usr/bin/env python3
"""Certified direct-FP32 baseline kernels for the H2A kill test."""

import triton
import triton.language as tl


@triton.jit
def token_d2_interval_fp32_rect(
    query,
    base,
    lower_d2,
    upper_d2,
    distance_relative_radius,
    input_magnitude_radius,
    final_relative_radius,
    absolute_radius,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Materialize conservative direct-FP32 squared-distance intervals."""
    rows_m = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_n = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    distance = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    magnitude = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        x = tl.load(
            query + rows_m[:, None] * K + k[None, :],
            mask=(rows_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        y = tl.load(
            base + rows_n[:, None] * K + k[None, :],
            mask=(rows_n[:, None] < N_) & (k[None, :] < K),
            other=0.0,
        )
        delta = x[:, None, :] - y[None, :, :]
        distance += tl.sum(delta * delta, axis=2)
        input_magnitude = tl.abs(x[:, None, :]) + tl.abs(y[None, :, :])
        magnitude += tl.sum(input_magnitude * input_magnitude, axis=2)

    radius = (
        distance_relative_radius * tl.abs(distance)
        + input_magnitude_radius * tl.abs(magnitude)
        + absolute_radius
    )
    final_magnitude = tl.abs(distance) + radius
    radius += final_relative_radius * final_magnitude + absolute_radius
    lower = tl.maximum(distance - radius, 0.0)
    upper = distance + radius
    valid = (rows_m[:, None] < M) & (rows_n[None, :] < N_)
    locations = rows_m[:, None] * N_ + rows_n[None, :]
    tl.store(lower_d2 + locations, lower, mask=valid)
    tl.store(upper_d2 + locations, upper, mask=valid)


@triton.jit
def aggregate_d2_interval_objects_i64(
    lower_d2,
    upper_d2,
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
    outward_absolute_guard,
    BASE_OBJECTS: tl.constexpr,
    BASE_TOKENS: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """Lift token squared-distance intervals to symmetric Chamfer objects."""
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
    lower = tl.load(lower_d2 + locations, mask=valid, other=float("inf")).to(
        tl.float64
    )
    upper = tl.load(upper_d2 + locations, mask=valid, other=float("inf")).to(
        tl.float64
    )
    lower = tl.maximum(lower - outward_absolute_guard, 0.0)
    upper = upper + outward_absolute_guard
    lower_forward = tl.min(lower, axis=1)
    lower_reverse = tl.min(lower, axis=0)
    upper_forward = tl.min(upper, axis=1)
    upper_reverse = tl.min(upper, axis=0)
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

