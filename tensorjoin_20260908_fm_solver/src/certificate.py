"""Conservative original-space lower bounds from actual projected point pairs."""
import numpy as np
import torch
import triton
import triton.language as tl
from problem import N,SQ_PAD


@triton.jit
def _minimum(Q,TR,TC,OUT,DIMS,LIMIT,NN:tl.constexpr,EARLY:tl.constexpr,B:tl.constexpr):
    tile=tl.program_id(0)
    tr=tl.load(TR+tile);tc=tl.load(TC+tile)
    offset=tl.arange(0,B)
    r=tr*64+offset//64;c=tc*64+offset%64
    valid=(r<NN)&(c<NN)
    acc=tl.full((B,),0,tl.float32)
    used=0;minimum=tl.full((),float('inf'),tl.float32)
    group=0;active=tl.full((),True,tl.int1)
    while (group<4)&active:
        for j in tl.static_range(8):
            k=group*8+j
            a=tl.load(Q+k*NN+r,r<NN,0)
            b=tl.load(Q+k*NN+c,c<NN,0)
            delta=a-b
            acc=tl.fma(delta,delta,acc)
        used+=8
        minimum=tl.min(tl.where(valid,acc,float('inf')),0)
        if EARLY:
            active=minimum.to(tl.float64)<=tl.load(LIMIT)
        group+=1
    tl.store(OUT+tile,minimum);tl.store(DIMS+tile,used)


def minima(q_gpu,tr,tc,threshold=None,margin=None,scale2=None):
    assert q_gpu.dtype==torch.float32 and q_gpu.is_contiguous() and q_gpu.shape==(32,N)
    if len(tr)==0:return np.empty(0,np.float32),np.empty(0,np.int32),None
    trg=torch.as_tensor(np.ascontiguousarray(tr,dtype=np.int32),device='cuda')
    tcg=torch.as_tensor(np.ascontiguousarray(tc,dtype=np.int32),device='cuda')
    out=torch.empty(len(tr),device='cuda',dtype=torch.float32)
    dims=torch.empty(len(tr),device='cuda',dtype=torch.int32)
    limit=float('inf') if threshold is None else (float(threshold)+SQ_PAD)*scale2+margin
    # Runtime FP64 limit avoids rounding a borderline threshold downward.
    lim=torch.tensor(limit,device='cuda',dtype=torch.float64)
    kernel=_minimum[(len(tr),)](q_gpu,trg,tcg,out,dims,lim,NN=N,EARLY=threshold is not None,B=4096,num_warps=8,enable_fp_fusion=True)
    return out.cpu().numpy().copy(),dims.cpu().numpy().copy(),kernel


def lower_bounds(values,margin,scale2):
    return np.maximum(0.,values.astype(np.float64)-float(margin))/float(scale2)


def certified_reject(values,threshold,margin,scale2):
    return lower_bounds(values,margin,scale2)>float(threshold)+SQ_PAD
