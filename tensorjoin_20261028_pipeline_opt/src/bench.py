#!/usr/bin/env python3
"""Frozen full-workload A/B; no data/method state is reused across timed calls."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from g2b_public_common import (
    D, N, PROJECT, atomic_json, load_contract, load_pageable_source,
    validate_canonical, sha256_file,
)
from run_d1_triton import quantize_per_vector, upward_float32
from run_g2b_tensorjoin_public import (
    BLOCK_M, BLOCK_N, BLOCK_K, REFINE_BLOCK_K, TILES_PER_BATCH,
    BOUND_PAD, FP32_DISTANCE_GUARD, compile_exact_specializations,
    triangular_int8_certificate_compact_i64,
    filter_ambiguous_fp32_i64, refine_ambiguous_fp64_i64,
)
from tensorjoin_device_counts import (
    scan_device_start, filter_device_count, refine_device_count,
)

CAPACITY = TILES_PER_BATCH * BLOCK_M * BLOCK_N
FILTER_GRID = 16384
REFINE_GRID = 1024


def chunked_preprocess(vectors):
    """Bound the temporary working set without changing per-row NumPy operations."""
    n = len(vectors)
    codes = np.empty(vectors.shape, dtype=np.int8)
    scales = np.empty(n, dtype=np.float32)
    errors = np.empty(n, dtype=np.float64)
    norms = np.empty(n, dtype=np.float64)
    codes_t = np.empty((D, n), dtype=np.int8)
    for a in range(0, n, 1024):
        b = min(a + 1024, n)
        c, s, e = quantize_per_vector(vectors[a:b])
        codes[a:b], scales[a:b], errors[a:b] = c, s, e
        norms[a:b] = (np.sum(c.astype(np.int64) ** 2, axis=1).astype(np.float64)
                      * s.astype(np.float64) ** 2)
        codes_t[:, a:b] = c.T
    return codes, scales, errors, norms, codes_t


def compile_candidates(threshold):
    """JIT on a zero 64-row fixture, never compute the full timed input."""
    device = 'cuda'
    codes = torch.zeros((64,D), dtype=torch.int8, device=device)
    codes_t = torch.zeros((D,N), dtype=torch.int8, device=device)
    v = torch.ones(64, device=device)
    tile = torch.zeros(1, dtype=torch.int32, device=device)
    queues = [torch.zeros(4096, dtype=torch.int64, device=device) for _ in range(3)]
    counters = torch.zeros(5, dtype=torch.int32, device=device)
    vectors = torch.zeros((64,D), device=device)
    scan_device_start[(1,)](codes,codes_t,v,v,v,v,v,v,tile,tile,tile,
        queues[0],queues[1],counters,threshold,BOUND_PAD,
        M=N,N_=N,K=D,CAPACITY=CAPACITY,BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,
        num_warps=4,num_stages=3)
    counters.zero_()
    counters[1] = 1
    filter_device_count[(FILTER_GRID,)](vectors,vectors,queues[1],queues[0],queues[2],
        counters,threshold,FP32_DISTANCE_GUARD,N_=N,K=D,CAPACITY=CAPACITY,
        BLOCK_K=256,num_warps=4)
    counters.zero_()
    counters[3] = 1
    refine_device_count[(REFINE_GRID,)](vectors,vectors,queues[2],queues[0],counters,
        threshold,N_=N,K=D,CAPACITY=CAPACITY,BLOCK_K=256,num_warps=4)
    torch.cuda.synchronize()


def gpu_state():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)


def isolation_state():
    raw = subprocess.check_output(['nvidia-smi','--id=4',
        '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    foreign = [row for row in raw.splitlines() if row.strip() and
               int(row.split(',')[0].strip()) != os.getpid()]
    return {'compute_processes':raw,'foreign_processes':foreign,
            'eligible':not foreign,'gpu_state':gpu_state()}


def run_once(vectors, contract, variant):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    chunked = variant.startswith('chunked')
    graph_mode = 'graph' in variant
    device_counts = variant not in ('baseline', 'chunked_baseline')
    if chunked:
        codes,scales,errors,norms,codes_t = chunked_preprocess(vectors)
    else:
        codes,scales,errors = quantize_per_vector(vectors)
        norms = (np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
                 * scales.astype(np.float64) ** 2)
    extent = (N+63)//64
    tile_rows,tile_columns = np.triu_indices(extent)
    tile_rows = np.asarray(tile_rows,dtype=np.int32)
    tile_columns = np.asarray(tile_columns,dtype=np.int32)
    scheduled = len(tile_rows)
    preprocess_seconds = time.perf_counter() - started
    vectors_gpu = torch.from_numpy(vectors).to('cuda')
    codes_gpu = torch.from_numpy(codes).to('cuda')
    codes_t_gpu = torch.from_numpy(codes_t if chunked else codes.T.copy()).to('cuda')
    scales_gpu = torch.from_numpy(scales).to('cuda')
    norms_gpu = torch.from_numpy(norms.astype(np.float32)).to('cuda')
    errors_gpu = torch.from_numpy(upward_float32(errors)).to('cuda')
    rows_gpu = torch.from_numpy(tile_rows).to('cuda')
    columns_gpu = torch.from_numpy(tile_columns).to('cuda')
    result_ids,ambiguous_ids,fp64_ids = [
        torch.empty(CAPACITY,dtype=torch.int64,device='cuda') for _ in range(3)]
    threshold = float(contract['threshold_d2'])
    parts = []
    records = []
    graph_setup_seconds = 0.
    if device_counts:
        counters = torch.zeros(5,dtype=torch.int32,device='cuda')
        device_start = torch.zeros(1,dtype=torch.int32,device='cuda')

        def submit(tiles):
            counters.zero_()
            scan_device_start[(tiles,)](codes_gpu,codes_t_gpu,
                scales_gpu,scales_gpu,norms_gpu,norms_gpu,errors_gpu,errors_gpu,
                rows_gpu,columns_gpu,device_start,result_ids,ambiguous_ids,counters,
                threshold,BOUND_PAD,M=N,N_=N,K=D,CAPACITY=CAPACITY,
                BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3)
            filter_device_count[(FILTER_GRID,)](vectors_gpu,vectors_gpu,
                ambiguous_ids,result_ids,fp64_ids,counters,threshold,FP32_DISTANCE_GUARD,
                N_=N,K=D,CAPACITY=CAPACITY,BLOCK_K=256,num_warps=4)
            refine_device_count[(REFINE_GRID,)](vectors_gpu,vectors_gpu,fp64_ids,
                result_ids,counters,threshold,N_=N,K=D,CAPACITY=CAPACITY,
                BLOCK_K=256,num_warps=4)

        if graph_mode:
            t = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                submit(TILES_PER_BATCH)
            graph_setup_seconds = time.perf_counter() - t
    setup_seconds = time.perf_counter() - started
    for a in range(0,scheduled,TILES_PER_BATCH):
        b = min(a+TILES_PER_BATCH,scheduled)
        if device_counts:
            device_start.fill_(a)
            if graph_mode and b-a == TILES_PER_BATCH:
                graph.replay()
            else:
                submit(b-a)
            counts = counters.cpu().numpy().astype(np.int64)
            final_count = int(counts[0])
            overflow = int(counts[2]) + int(max(counts[0],counts[1],counts[3])>CAPACITY)
        else:
            counters = torch.zeros(5,dtype=torch.int32,device='cuda')
            triangular_int8_certificate_compact_i64[(b-a,)](codes_gpu,codes_t_gpu,
                scales_gpu,scales_gpu,norms_gpu,norms_gpu,errors_gpu,errors_gpu,
                rows_gpu[a:b],columns_gpu[a:b],result_ids,ambiguous_ids,counters,
                threshold,BOUND_PAD,M=N,N_=N,K=D,CAPACITY=CAPACITY,
                BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3)
            torch.cuda.synchronize()
            scan_counts = counters.cpu().numpy().astype(np.int64)
            if scan_counts[1]:
                filter_ambiguous_fp32_i64[(int(scan_counts[1]),)](vectors_gpu,vectors_gpu,
                    ambiguous_ids,result_ids,fp64_ids,counters,threshold,FP32_DISTANCE_GUARD,
                    N_=N,K=D,CAPACITY=CAPACITY,BLOCK_K=256,num_warps=4)
                torch.cuda.synchronize()
            fp32_counts = counters.cpu().numpy().astype(np.int64)
            if fp32_counts[3]:
                refine_ambiguous_fp64_i64[(int(fp32_counts[3]),)](vectors_gpu,vectors_gpu,
                    fp64_ids,result_ids,counters,threshold,N_=N,K=D,CAPACITY=CAPACITY,
                    BLOCK_K=256,num_warps=4)
                torch.cuda.synchronize()
            counts = counters.cpu().numpy().astype(np.int64)
            final_count = int(counts[0])
            overflow = int(counts[2]) + int(max(*scan_counts[:2],fp32_counts[0],fp32_counts[3],final_count)>CAPACITY)
        parts.append(result_ids[:min(final_count,CAPACITY)].cpu().numpy().astype(np.uint64,copy=True))
        records.append({'tile_start':a,'tiles':b-a,'final':final_count,
                        'ambiguous':int(counts[1]),'fp64':int(counts[3]),'overflow':overflow})
    torch.cuda.synchronize()
    before_sort = time.perf_counter()
    # Keep the two shared host sorts and reverse construction byte-for-byte equivalent.
    upper = np.sort(np.concatenate(parts).astype(np.uint64,copy=False))
    upper_rows = upper // np.uint64(N)
    upper_columns = upper - upper_rows * np.uint64(N)
    nonself = upper_rows < upper_columns
    reverse = upper_columns[nonself] * np.uint64(N) + upper_rows[nonself]
    canonical = np.sort(np.concatenate((upper,reverse)).astype(np.uint64,copy=False))
    stopped = time.perf_counter()
    correct = validate_canonical(canonical,contract)
    correct['upper_duplicates'] = int(np.count_nonzero(upper[1:]==upper[:-1]))
    correct['invalid_or_lower_upper_pairs'] = int(np.count_nonzero((upper>=np.uint64(N)*np.uint64(N)) | (upper_rows>upper_columns)))
    correct['overflow_events'] = sum(r['overflow'] for r in records)
    correct['exact_contract_pass'] = bool(correct['exact_contract_pass'] and
        not correct['upper_duplicates'] and not correct['invalid_or_lower_upper_pairs'] and not correct['overflow_events'])
    return {'variant':variant,'public_seconds':stopped-started,
            'shared_host_canonical_seconds':stopped-before_sort,
            'join_excluding_shared_sort_seconds':before_sort-started,
            'host_preprocess_seconds':preprocess_seconds,'setup_seconds':setup_seconds,
            'graph_setup_seconds':graph_setup_seconds,'batches':records,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'correctness':correct}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--record-id',required=True)
    parser.add_argument('--variants',default='baseline,nobubble,graph,chunked_graph')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--ab',action='store_true')
    args = parser.parse_args()
    import re
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}',args.record_id)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4'
    assert torch.cuda.device_count() == 1
    path = PROJECT / f'results/{args.record_id}.json'
    if path.exists():
        raise FileExistsError(path)
    variants = args.variants.split(',')
    assert set(variants) <= {'baseline','nobubble','graph','chunked_graph','chunked_nobubble','chunked_baseline'}
    admission = isolation_state()
    if not admission['eligible']:
        atomic_json(path, {'record_id':args.record_id,'ineligible':admission})
        raise RuntimeError('GPU4 has foreign compute processes; refusing launch')
    contract = load_contract()
    vectors = load_pageable_source()
    torch.cuda.init()
    compile_exact_specializations(CAPACITY,float(contract['threshold_d2']))
    if any(v!='baseline' for v in variants):
        compile_candidates(float(contract['threshold_d2']))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    order = []
    for i in range(args.repeats):
        order.extend(variants[::-1] if args.ab and i%2 else variants)
    result = {'record_id':args.record_id,'contract':contract,'gpu':torch.cuda.get_device_name(),
              'gpu_index':4,'host':platform.node(),'pid':os.getpid(),'torch':torch.__version__,
              'order':order,'samples':[],'before':gpu_state(),'admission':admission,
              'source_hashes':{name:sha256_file(PROJECT/'src'/name) for name in
                  ['bench.py','tensorjoin_device_counts.py','run_g2b_tensorjoin_public.py',
                   'run_g2a_tensorjoin.py','run_g2a2_tensorjoin_triangular.py','run_d1_triton.py','g2b_public_common.py']}}
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    for idx,variant in enumerate(order):
        gc.collect()
        # No method tensors, captured graph or computed answers survive this call.
        before = isolation_state()
        if not before['eligible']:
            result['interrupted_by_contention'] = before
            break
        sample = run_once(vectors,contract,variant)
        after = isolation_state()
        sample['isolation_before'] = before
        sample['isolation_after'] = after
        sample['eligible'] = bool(before['eligible'] and after['eligible'])
        sample['observation'] = idx
        sample['pair'] = idx//len(variants)
        sample['timestamp'] = time.time()
        result['samples'].append(sample)
        print('SAMPLE '+json.dumps(sample,sort_keys=True),flush=True)
        if not sample['correctness']['exact_contract_pass'] or not sample['eligible']:
            result['invalid'] = True
            break
    result['after'] = gpu_state()
    atomic_json(path,result)
    assert not result.get('invalid') and not result.get('interrupted_by_contention'), 'Candidate invalidated: exact output mismatch'

if __name__ == '__main__':
    main()
