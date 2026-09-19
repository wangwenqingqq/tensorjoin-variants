"""Complete pageable-host exact-oracle join with a pedantic FP32-first scan."""

import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import triton

from run_h2b_p1_pedantic_correctness import CuBLAS, norm_metadata
from run_g2a_tensorjoin import refine_ambiguous_fp64_i64
from fp32_first_kernels import classify_pedantic_panel
from gpu_norms import prepare_norm_device

PANEL = 4096
CAPACITY = PANEL * PANEL
BLOCK = 1024
WARPS = 8
DIMENSION = 512
U32 = 2.0**-24
GAMMA = (2 * DIMENSION + 2) * U32 / (1 - (2 * DIMENSION + 2) * U32)
ABS_DOT = 4 * DIMENSION * float(np.finfo(np.float32).tiny)


class BoundCuBLAS(CuBLAS):
    """Admit only the stream used to obtain the existing typed FFI handle."""
    def __init__(self):
        super().__init__()
        self.stream = int(torch.cuda.current_stream().cuda_stream)

    def gemm(self, query, base, output):
        if int(torch.cuda.current_stream().cuda_stream) != self.stream:
            raise RuntimeError('An unvalidated CUDA stream was selected')
        return super().gemm(query, base, output)


def check_input(vectors, threshold):
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION or vectors.dtype != np.float32:
        raise ValueError('Only contiguous stored FP32 D512 vectors are admitted')
    if not vectors.flags.c_contiguous or not len(vectors):
        raise ValueError('Empty or strided input is outside this contract')
    if not np.isfinite(vectors).all() or np.max(np.abs(vectors)) > 1:
        raise ValueError('Only finite |x| <= 1 is admitted')
    if not np.isfinite(threshold) or threshold < 0 or float(np.float32(threshold)) != threshold:
        raise ValueError('Threshold must be nonnegative and exactly FP32 representable')


def panels(n):
    for row in range(0, n, PANEL):
        for column in range(row, n, PANEL):
            yield row, column, min(PANEL, n-row), min(PANEL, n-column)


def classify(dots, metadata, results, ambiguous, counters, row, column, threshold, n,
             lower=None, upper=None):
    rows, columns = dots.shape
    dump = lower is not None
    return classify_pedantic_panel[(triton.cdiv(rows*columns, BLOCK),)](
        dots, *metadata, results, ambiguous, counters,
        lower if dump else dots, upper if dump else dots,
        row, column, threshold, N_=n, ROWS=rows, COLS=columns,
        CAPACITY=CAPACITY, GAMMA=GAMMA, ABS_DOT=ABS_DOT,
        BLOCK=BLOCK, DUMP=dump, num_warps=WARPS,
    )


def refine(vectors, ambiguous, results, counters, count, threshold):
    if count:
        return refine_ambiguous_fp64_i64[(count,)](
            vectors, vectors, ambiguous, results, counters, threshold,
            N_=int(vectors.shape[0]), K=DIMENSION, CAPACITY=CAPACITY,
            BLOCK_K=256, num_warps=4)


def warmup(n, threshold, blas):
    # Exact full/tail specializations, without retained method-prepared inputs.
    size = min(PANEL, n)
    warm = torch.zeros((size, DIMENSION), dtype=torch.float32, device='cuda')
    temporary = prepare_norm_device(warm)
    del temporary
    norm = torch.ones(n, dtype=torch.float64, device='cuda')
    zero = torch.zeros(n, dtype=torch.float64, device='cuda')
    scores = torch.empty(CAPACITY, dtype=torch.float32, device='cuda')
    result = torch.empty(CAPACITY, dtype=torch.int64, device='cuda')
    ambiguous = torch.empty_like(result)
    counter = torch.zeros(4, dtype=torch.int32, device='cuda')
    for rows, columns in sorted({(r, c) for _, _, r, c in panels(n)}):
        counter.zero_()
        matrix = scores[:rows*columns].view(rows, columns)
        blas.gemm(warm[:rows], warm[:columns], matrix)
        classify(matrix, (norm, zero, norm), result, ambiguous, counter, 0, 0, threshold, n)
    counter.zero_()
    ambiguous[0] = 0
    # Match the full leading stride; only row zero is dereferenced in warmup.
    refine_ambiguous_fp64_i64[(1,)](
        warm, warm, ambiguous, result, counter, threshold,
        N_=n, K=DIMENSION, CAPACITY=CAPACITY, BLOCK_K=256, num_warps=4)
    torch.cuda.synchronize()
    del warm, norm, zero, scores, result, ambiguous, counter, matrix
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_operator(vectors, threshold, blas, return_trace=False):
    n = int(vectors.shape[0])
    started = time.perf_counter()
    vg = torch.from_numpy(vectors).to('cuda')
    h2d_done = time.perf_counter()
    metadata = prepare_norm_device(vg)
    metadata_done = time.perf_counter()
    scores = torch.empty(CAPACITY, dtype=torch.float32, device='cuda')
    results = torch.empty(CAPACITY, dtype=torch.int64, device='cuda')
    ambiguous = torch.empty_like(results)
    setup_done = time.perf_counter()
    parts, records = [], []
    for row, column, rows, columns in panels(n):
        matrix = scores[:rows*columns].view(rows, columns)
        counters = torch.zeros(4, dtype=torch.int32, device='cuda')
        blas.gemm(vg[row:row+rows], vg[column:column+columns], matrix)
        classify(matrix, metadata, results, ambiguous, counters, row, column, threshold, n)
        direct, unclear, overflow, rejected = map(int, counters.cpu().numpy())
        expected = rows*(rows+1)//2 if row == column else rows*columns
        if direct + unclear + rejected != expected or overflow or max(direct, unclear) > CAPACITY:
            raise RuntimeError('Panel classification/count/capacity gate failed')
        refine(vg, ambiguous, results, counters, unclear, threshold)
        final = counters.cpu().numpy()
        count = int(final[0])
        if int(final[2]) or count > CAPACITY:
            raise RuntimeError('Terminal/output capacity gate failed')
        parts.append(results[:count].cpu().numpy().astype(np.uint64, copy=True))
        records.append({'row': row, 'column': column, 'rows': rows, 'columns': columns,
                        'computed_dot_pairs': rows*columns, 'upper_pairs': expected,
                        'direct_accept': direct, 'ambiguous': unclear, 'direct_reject': rejected,
                        'accepted_upper': count, 'overflow': int(final[2])})
    torch.cuda.synchronize()
    device_done = time.perf_counter()
    accepted_upper = np.sort(np.concatenate(parts).astype(np.uint64, copy=False))
    upper_rows = accepted_upper // np.uint64(n)
    upper_columns = accepted_upper - upper_rows * np.uint64(n)
    nonself = upper_rows < upper_columns
    reverse = upper_columns[nonself] * np.uint64(n) + upper_rows[nonself]
    canonical = np.sort(np.concatenate((accepted_upper, reverse)).astype(np.uint64, copy=False))
    stopped = time.perf_counter()
    record = {'public_seconds': stopped-started,
              'phase_seconds': {'h2d_source': h2d_done-started, 'gpu_norm_metadata': metadata_done-h2d_done,
                                'alloc': setup_done-metadata_done,
                                'all_panels_and_readback': device_done-setup_done,
                                'canonicalize': stopped-device_done},
              'panel_count': len(records), 'panels': records,
              'computed_dot_pairs': sum(x['computed_dot_pairs'] for x in records),
              'upper_pairs': sum(x['upper_pairs'] for x in records),
              'ambiguous_upper_pairs': sum(x['ambiguous'] for x in records),
              'direct_accept_upper_pairs': sum(x['direct_accept'] for x in records),
              'direct_reject_upper_pairs': sum(x['direct_reject'] for x in records),
              'accepted_upper_pairs': int(accepted_upper.size),
              'overflow_events': sum(x['overflow'] for x in records),
              'input_device_pointer': int(vg.data_ptr())}
    return canonical, record


def cache_identity(cache):
    return {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(cache).rglob('*')) if p.suffix in ('.cubin', '.ptx')}
