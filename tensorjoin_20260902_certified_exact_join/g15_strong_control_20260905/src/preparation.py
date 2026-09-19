"""Byte-preserving host blocking around the existing audited quantizer."""

import numpy as np
from run_g4b_r1_public_opportunity import quantize_with_analytic_residual

BLOCK_ROWS = 4096


def original_prepare(vectors):
    codes, scales, norm2, errors = quantize_with_analytic_residual(vectors)
    return codes, scales, norm2, errors, codes.T.copy()


def chunked_prepare(vectors):
    n, dimension = vectors.shape
    codes = np.empty((n, dimension), dtype=np.int8)
    scales = np.empty(n, dtype=np.float32)
    norm2 = np.empty(n, dtype=np.float32)
    errors = np.empty(n, dtype=np.float32)
    transposed = np.empty((dimension, n), dtype=np.int8)
    for start in range(0, n, BLOCK_ROWS):
        stop = min(n, start + BLOCK_ROWS)
        q, s, norm, error = quantize_with_analytic_residual(vectors[start:stop])
        codes[start:stop] = q
        scales[start:stop] = s
        norm2[start:stop] = norm
        errors[start:stop] = error
        transposed[:, start:stop] = q.T
    return codes, scales, norm2, errors, transposed
