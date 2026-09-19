#!/usr/bin/env python3
"""Ragged-safe analytic INT8 certificate kernel for G4B-R1."""

import triton
import triton.language as tl

@triton.jit
def analytic_certificate_ragged_safe_i64(
    codes,
    codes_transposed,
    scales,
    reconstructed_norm2,
    residual_error_upper,
    tile_rows,
    tile_columns,
    result_ids,
    ambiguous_ids,
    counters,
    epsilon_lower,
    epsilon_upper,
    expression_relative_radius,
    expression_absolute_radius,
    final_relative_radius,
    final_absolute_radius,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(axis=0)
    program_m = tl.load(tile_rows + program)
    program_n = tl.load(tile_columns + program)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        a = tl.load(
            codes + offsets_m[:, None] * K + k[None, :],
            mask=(offsets_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            codes_transposed + k[:, None] * N_ + offsets_n[None, :],
            mask=(k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

    scale_i = tl.load(scales + offsets_m, mask=offsets_m < M, other=0.0)
    scale_j = tl.load(scales + offsets_n, mask=offsets_n < N_, other=0.0)
    norm_i = tl.load(
        reconstructed_norm2 + offsets_m, mask=offsets_m < M, other=0.0
    )
    norm_j = tl.load(
        reconstructed_norm2 + offsets_n, mask=offsets_n < N_, other=0.0
    )
    error_i = tl.load(
        residual_error_upper + offsets_m, mask=offsets_m < M, other=0.0
    )
    error_j = tl.load(
        residual_error_upper + offsets_n, mask=offsets_n < N_, other=0.0
    )
    scaled_dot = accumulator.to(tl.float32) * scale_i[:, None] * scale_j[None, :]
    reconstructed_d2 = norm_i[:, None] + norm_j[None, :] - 2.0 * scaled_dot
    term_abs_sum = (
        tl.abs(norm_i[:, None])
        + tl.abs(norm_j[None, :])
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
    residual_radius = error_i[:, None] + error_j[None, :]
    final_magnitude = tl.maximum(reconstructed_upper + residual_radius, 1.0)
    final_radius = final_relative_radius * final_magnitude + final_absolute_radius
    lower = tl.maximum(reconstructed_lower - residual_radius - final_radius, 0.0)
    upper = reconstructed_upper + residual_radius + final_radius

    valid = (
        (offsets_m[:, None] < M)
        & (offsets_n[None, :] < N_)
        & (offsets_m[:, None] <= offsets_n[None, :])
    )
    accept = valid & (upper <= epsilon_lower)
    reject = valid & (lower > epsilon_upper)
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

