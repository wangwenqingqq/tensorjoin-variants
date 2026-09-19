"""Charged, sample-driven global routing prototype; not a production tile router."""
import time
import numpy as np
import torch
import triton
from kernels import metadata, terminal_fp64 as refine_ambiguous_fp64_i64
from retained import half_classify, certified_fp32_filter_i64


def graph_cost(fn):
    # Setup and calibration are deliberately charged to the routing call.
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    graph.replay();torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/10


def route(engine,x,threshold):
    from experiment import FORMATS,D,threshold32
    torch.cuda.synchronize();start=time.perf_counter()
    tfloor=threshold32(threshold)
    n=len(x);m=min(n,128);idx=np.linspace(0,n-1,m,dtype=np.int64)
    sample=x[idx].copy();v=torch.from_numpy(sample).cuda();cap=m*m;npairs=m*(m+1)//2
    out=torch.empty(cap,dtype=torch.int64,device='cuda');amb=torch.empty_like(out);terminal=torch.empty_like(out)
    counters=torch.zeros(6,dtype=torch.int32,device='cuda')
    ij=torch.triu_indices(m,m,device='cuda');allpairs=(ij[0]*m+ij[1]).contiguous()
    def f32():
        counters.zero_()
        certified_fp32_filter_i64[(npairs,)](v,allpairs,out,terminal,counters,tfloor,tfloor,2**-14,2**-22,2**-22,4096*2**-126,N_=m,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
    def f64():
        counters.zero_()
        refine_ambiguous_fp64_i64[(npairs,)](v,v,allpairs,out,counters,threshold,N_=m,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
    refine32=graph_cost(f32)/npairs;refine64=graph_cost(f64)/npairs
    totalpairs=n*(n+1)//2;predicted={'fp64':refine64*totalpairs};samples={}
    for fmt in FORMATS:
        # The sample's terminal result is not an oracle used to resolve full pairs.
        _,rec=engine.run(sample,threshold,fmt)
        q,s,md=engine.prepare(v,fmt);invalid=torch.zeros(1,dtype=torch.int32,device='cuda')
        def prep():
            metadata[(m,)](v,q,s,*md,invalid,D=D,FORMAT=fmt,num_warps=4,enable_fp_fusion=False)
        prep_ms=graph_cost(prep)
        counts=torch.zeros(4,dtype=torch.int32,device='cuda')
        def stage1():
            counts.zero_();scores=engine.dots(q,s,fmt)
            half_classify[(triton.cdiv(cap,1024),)](scores,*md,out,amb,counts,scores,scores,0,0,N=m,ROWS=m,COLS=m,CAP=cap,GAMMA=2**-21 if fmt=='int8' else .00012232370499987155,T=tfloor,BLOCK=1024,DUMP=False,num_warps=8,enable_fp_fusion=False)
        dense_ms=graph_cost(stage1)
        predicted[fmt]=prep_ms*n/m+dense_ms*(n/m)**2+(rec['fp32_pairs']*refine32+rec['fp64_pairs']*refine64)*totalpairs/npairs
        samples[fmt]={'sample_fp32_pairs':rec['fp32_pairs'],'sample_fp64_pairs':rec['fp64_pairs'],'prep_graph_ms':prep_ms,'dense_graph_ms':dense_ms}
    selected=min(predicted,key=predicted.get)
    torch.cuda.synchronize();calibration_ms=(time.perf_counter()-start)*1000
    result,rec=engine.run(x,threshold,selected)
    torch.cuda.synchronize();rec.update(format='route',selected=selected,selected_e2e_ms=rec['e2e_ms'],
        e2e_ms=(time.perf_counter()-start)*1000,routing_ms=calibration_ms,predicted_device_ms=predicted,
        sample_rows=m,samples=samples,refine32_ms_per_pair=refine32,refine64_ms_per_pair=refine64)
    return result,rec
