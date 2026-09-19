#!/usr/bin/env python3
"""Certified pedantic-SGEMM interval kernels for H2B."""

import triton
import triton.language as tl


@triton.jit
def dot_to_d2_interval_fp64_rect(
    dots,
    query_norm2_center,
    query_norm2_radius,
    query_l2_upper,
    base_norm2_center,
    base_norm2_radius,
    base_l2_upper,
    lower_d2,
    upper_d2,
    gamma_dot,
    dot_absolute_radius,
    fp64_relative_guard,
    absolute_guard,
    PAIRS: tl.constexpr,
    BASE_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize FP64 D2 intervals from a pedantic FP32 SGEMM dot matrix."""
    pair = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair < PAIRS
    query_token = pair // BASE_TOKENS
    base_token = pair - query_token * BASE_TOKENS
    dot = tl.load(dots + pair, mask=valid, other=0.0).to(tl.float64)
    qn = tl.load(query_norm2_center + query_token, mask=valid, other=0.0)
    bn = tl.load(base_norm2_center + base_token, mask=valid, other=0.0)
    qnr = tl.load(query_norm2_radius + query_token, mask=valid, other=0.0)
    bnr = tl.load(base_norm2_radius + base_token, mask=valid, other=0.0)
    ql2 = tl.load(query_l2_upper + query_token, mask=valid, other=0.0)
    bl2 = tl.load(base_l2_upper + base_token, mask=valid, other=0.0)
    dot_radius = gamma_dot * ql2 * bl2 + dot_absolute_radius
    center = qn + bn - 2.0 * dot
    reconstruction_magnitude = (
        tl.abs(qn) + tl.abs(bn) + 2.0 * tl.abs(dot) + qnr + bnr + 2.0 * dot_radius
    )
    radius = (
        qnr
        + bnr
        + 2.0 * dot_radius
        + fp64_relative_guard * reconstruction_magnitude
        + absolute_guard
    )
    lower = tl.maximum(center - radius, 0.0)
    upper = center + radius
    tl.store(lower_d2 + pair, lower, mask=valid)
    tl.store(upper_d2 + pair, upper, mask=valid)


@triton.jit
def aggregate_pedantic_dot_interval_objects_i64(
    dots,
    query_norm2_center,
    query_norm2_radius,
    query_l2_upper,
    base_norm2_center,
    base_norm2_radius,
    base_l2_upper,
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
    gamma_dot,
    dot_absolute_radius,
    fp64_relative_guard,
    absolute_guard,
    BASE_OBJECTS: tl.constexpr,
    BASE_TOKENS: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """Fuse D2 interval reconstruction with exact object-level aggregation."""
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
    query_token = query_start + i[:, None] + 0 * j[None, :]
    base_token = base_start + j[None, :] + 0 * i[:, None]
    locations = query_token * BASE_TOKENS + base_token

    dot = tl.load(dots + locations, mask=valid, other=0.0).to(tl.float64)
    qn = tl.load(query_norm2_center + query_token, mask=valid, other=0.0)
    bn = tl.load(base_norm2_center + base_token, mask=valid, other=0.0)
    qnr = tl.load(query_norm2_radius + query_token, mask=valid, other=0.0)
    bnr = tl.load(base_norm2_radius + base_token, mask=valid, other=0.0)
    ql2 = tl.load(query_l2_upper + query_token, mask=valid, other=0.0)
    bl2 = tl.load(base_l2_upper + base_token, mask=valid, other=0.0)
    dot_radius = gamma_dot * ql2 * bl2 + dot_absolute_radius
    center = qn + bn - 2.0 * dot
    reconstruction_magnitude = (
        tl.abs(qn) + tl.abs(bn) + 2.0 * tl.abs(dot) + qnr + bnr + 2.0 * dot_radius
    )
    radius = (
        qnr
        + bnr
        + 2.0 * dot_radius
        + fp64_relative_guard * reconstruction_magnitude
        + absolute_guard
    )
    lower = tl.maximum(center - radius, 0.0)
    upper = center + radius
    lower = tl.where(valid, lower, float("inf"))
    upper = tl.where(valid, upper, float("inf"))

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
    accepted_position = tl.atomic_add(counters + lane, 1, mask=accept, sem="relaxed")
    accepted_store = accept & (accepted_position < CAPACITY)
    tl.store(result_ids + accepted_position, pair_id, mask=accepted_store)
    tl.atomic_add(
        counters + 2 + lane, 1, mask=accept & ~accepted_store, sem="relaxed"
    )
    ambiguous_position = tl.atomic_add(
        counters + 1 + lane, 1, mask=ambiguous, sem="relaxed"
    )
    ambiguous_store = ambiguous & (ambiguous_position < CAPACITY)
    tl.store(ambiguous_ids + ambiguous_position, pair_id, mask=ambiguous_store)
    tl.atomic_add(
        counters + 2 + lane, 1, mask=ambiguous & ~ambiguous_store, sem="relaxed"
    )
