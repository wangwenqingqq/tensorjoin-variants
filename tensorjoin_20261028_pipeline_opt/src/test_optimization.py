#!/usr/bin/env python3
"""Bit-exact preprocessing, queue boundaries, pointer churn and Graph replay tests."""
import argparse
import gc
import json
import os
import numpy as np
import torch

from bench import chunked_preprocess
from g2b_public_common import load_pageable_source
from run_d1_triton import quantize_per_vector, upward_float32
from run_g2a_tensorjoin import filter_ambiguous_fp32_i64, refine_ambiguous_fp64_i64
from run_g2a2_tensorjoin_triangular import triangular_int8_certificate_compact_i64
from tensorjoin_device_counts import scan_device_start, filter_device_count, refine_device_count


def bit_equal(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert np.array_equal(a.view(np.uint8).reshape(-1),b.view(np.uint8).reshape(-1))


def test_preprocess():
    full = load_pageable_source()
    c,s,e = quantize_per_vector(full)
    norms = np.sum(c.astype(np.int64)**2,axis=1).astype(np.float64)*s.astype(np.float64)**2
    cc,ss,ee,nn,ct = chunked_preprocess(full)
    for name,a,b in [('codes',c,cc),('scales',s,ss),('errors',e,ee),('norms',norms,nn),('transpose',c.T.copy(),ct)]:
        bit_equal(a,b)
        print('BIT_EXACT '+name,flush=True)
    del full,c,s,e,norms,cc,ss,ee,nn,ct
    gc.collect()


def sorted_queue(queue,count):
    return np.sort(queue[:min(int(count),queue.numel())].cpu().numpy())


def test_queues():
    rng = np.random.default_rng(20261028)
    x = rng.normal(size=(37,512)).astype(np.float32)/np.float32(np.sqrt(512))
    x[7]=x[2]
    t = torch.from_numpy(x).cuda()
    # Deliberately force both filters with unordered, repeated valid IDs.
    cap=131
    ids=rng.integers(0,37*37,size=cap,dtype=np.int64)
    ids[:4]=[0,2*37+7,7*37+2,3*37+3]
    for count in [0,1,6,7,8,130,131,137]:
        for stage in [2,3]:
            outputs=[]
            for optimized in [False,True]:
                q=torch.from_numpy(ids.copy()).cuda()
                result=torch.empty(cap,dtype=torch.int64,device='cuda')
                fp64=torch.empty_like(result)
                counters=torch.zeros(5,dtype=torch.int32,device='cuda')
                counters[1 if stage==2 else 3]=count
                active=min(count,cap)
                grid=7 if optimized else active
                if grid:
                    if stage==2:
                        fn=filter_device_count if optimized else filter_ambiguous_fp32_i64
                        fn[(grid,)](t,t,q,result,fp64,counters,2.,1e-3,
                            N_=37,K=512,CAPACITY=cap,BLOCK_K=256,num_warps=4)
                    else:
                        fn=refine_device_count if optimized else refine_ambiguous_fp64_i64
                        fn[(grid,)](t,t,q,result,counters,2.,
                            N_=37,K=512,CAPACITY=cap,BLOCK_K=256,num_warps=4)
                c=counters.cpu().numpy()
                outputs.append((c,sorted_queue(result,c[0]),sorted_queue(fp64,c[3]) if stage==2 else None))
            for a,b in zip(*outputs):
                if a is not None: assert np.array_equal(a,b),(count,stage,a,b)
    print('QUEUE_BOUNDARIES_PASS counts=0,1,6,7,8,130,131,137 cap=131 grid=7',flush=True)


def test_pipeline_graph():
    rng=np.random.default_rng(7123)
    for churn in range(3):
        n=129; d=512; cap=2*64*64
        x=rng.normal(size=(n,d)).astype(np.float32)/np.float32(np.sqrt(d))
        x[7]=x[2]
        codes,scales,errors,norms,ct=chunked_preprocess(x)
        tensors=[torch.from_numpy(a).cuda() for a in [x,codes,ct,scales,norms.astype(np.float32),upward_float32(errors)]]
        v,c,ct,s,norm,e=tensors
        rows,cols=[torch.from_numpy(a.astype(np.int32)).cuda() for a in np.triu_indices(3)]
        start=torch.zeros(1,dtype=torch.int32,device='cuda')
        result,amb,fp64=[torch.empty(cap,dtype=torch.int64,device='cuda') for _ in range(3)]
        counters=torch.zeros(5,dtype=torch.int32,device='cuda')
        threshold=2.
        def optimized():
            counters.zero_()
            scan_device_start[(2,)](c,ct,s,s,norm,norm,e,e,rows,cols,start,result,amb,counters,
                threshold,1e-4,M=n,N_=n,K=d,CAPACITY=cap,BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3)
            filter_device_count[(7,)](v,v,amb,result,fp64,counters,threshold,1e-3,N_=n,K=d,CAPACITY=cap,BLOCK_K=256,num_warps=4)
            refine_device_count[(7,)](v,v,fp64,result,counters,threshold,N_=n,K=d,CAPACITY=cap,BLOCK_K=256,num_warps=4)
        expected={}
        for a in [0,2,4]:
            counters.zero_()
            triangular_int8_certificate_compact_i64[(2,)](c,ct,s,s,norm,norm,e,e,rows[a:a+2],cols[a:a+2],result,amb,counters,
                threshold,1e-4,M=n,N_=n,K=d,CAPACITY=cap,BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3)
            count=int(counters[1].item())
            if count:
                filter_ambiguous_fp32_i64[(count,)](v,v,amb,result,fp64,counters,threshold,1e-3,N_=n,K=d,CAPACITY=cap,BLOCK_K=256,num_warps=4)
            count=int(counters[3].item())
            if count:
                refine_ambiguous_fp64_i64[(count,)](v,v,fp64,result,counters,threshold,N_=n,K=d,CAPACITY=cap,BLOCK_K=256,num_warps=4)
            cc=counters.cpu().numpy()
            expected[a]=(cc,sorted_queue(result,cc[0]))
        optimized()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): optimized()
        for i in range(30):
            a=[0,4,2][i%3]
            start.fill_(a)
            graph.replay()
            actual=counters.cpu().numpy()
            assert np.array_equal(actual,expected[a][0])
            assert np.array_equal(sorted_queue(result,actual[0]),expected[a][1])
        del graph
    print('GRAPH_POINTER_CHURN_PASS fixtures=3 replays=90 starts=0,4,2 ragged_rows=129',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-full-preprocess',action='store_true');args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
    if not args.skip_full_preprocess:test_preprocess()
    test_queues()
    test_pipeline_graph()
    print('ALL_TESTS_PASS',flush=True)

if __name__=='__main__': main()
