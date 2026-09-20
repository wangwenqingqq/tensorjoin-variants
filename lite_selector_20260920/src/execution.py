"""Timed fixed-format adapter; retained GPU kernels and certificates unchanged.

The run body follows the keeper's Engine.run. Its only semantic boundary
additions are a common timed host legality scan and output capacity assertion.
Peak memory and hashes are collected after the full host-to-host timer.
"""
import hashlib
import json
import time

import numpy as np
import torch
import triton
from experiment import Engine as Keeper, D, threshold32, canonical, digest, FORMATS, codebook
from kernels import terminal_fp64 as refine_ambiguous_fp64_i64
from retained import half_classify, certified_fp32_filter_i64


def validate_input(x, threshold):
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 512:
        raise ValueError('Input must be a NumPy FP32 N x 512 matrix')
    if not 1 <= len(x) <= 4096 or not x.flags.c_contiguous:
        raise ValueError('Input must be contiguous with 1 <= N <= 4096')
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError('Input must be finite with abs(x) <= 1')
    if not isinstance(threshold, (float, np.float64)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Threshold must be a nonnegative finite binary64 scalar')


class Engine(Keeper):
    def __init__(self):
        super().__init__()
        self.registry = {}
        self.used = {}
        self.keepalive = []

    def record_kernels(self):
        # Outside timing: retain every observed specialization without repeatedly
        # hashing the same loaded kernel object on every sample.
        self.used = {}
        for name, kernel in self.kernels.items():
            key = (name, id(kernel))
            if key not in self.registry:
                self.keepalive.append(kernel)
                kernel._init_handles()
                self.registry[key] = {'name': name, 'cubin_sha256': hashlib.sha256(kernel.kernel).hexdigest(),
                    'registers': kernel.n_regs, 'spills': kernel.n_spills, 'shared_bytes': kernel.metadata.shared,
                    'target': str(kernel.metadata.target),
                    'constants': {str(a): str(b) for a, b in kernel.src.constants.items()}}
            self.used[name] = self.registry[key]['cubin_sha256']
        return dict(self.used)

    def capture(self, path):
        self.record_kernels()
        rows = {r['name'] + ':' + r['cubin_sha256']: r for r in self.registry.values()}
        path.write_text(json.dumps(rows, indent=2)+'\n')

    def run(self,x,threshold,fmt,profile=True):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
        validate_input(x,threshold)
        if fmt not in ('int8','e3m4','fp16','fp64'):raise ValueError('Unsupported fixed format')
        self.kernels = {}
        n=len(x);cap=n*n;events=[];tfloor=threshold32(threshold)
        def mark():
            e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
        mark();self.calls+=1
        if self.churn:
            shift=32*(1+self.calls%16)
            storage=torch.empty(n*D+shift,device='cuda',dtype=torch.float32)
            v=storage[shift:].view(n,D);v.copy_(torch.from_numpy(x))
            self.addresses.add(v.data_ptr())
        else:
            v=torch.from_numpy(x).to('cuda')
        out=torch.empty(cap,device='cuda',dtype=torch.int64)
        amb=torch.empty_like(out);fp64=torch.empty_like(out)
        c2=torch.zeros(6,device='cuda',dtype=torch.int32)
        if fmt=='fp64':
            ij=torch.triu_indices(n,n,device='cuda');pairs=(ij[0]*n+ij[1]).contiguous()
            mark();mark();count=pairs.numel();n1=0;n2=count;direct=0;reject=0
            mark()
            k=refine_ambiguous_fp64_i64[(count,)](v,v,pairs,out,c2,threshold,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
            self.kernels['fp64']=k
        else:
            q,s,md=self.prepare(v,fmt);mark();scores=self.dots(q,s,fmt)
            c=torch.zeros(4,device='cuda',dtype=torch.int32)
            # Integer dot is exact through D=512; only int->FP32 and two scales round.
            gamma=2.0**-21 if fmt=='int8' else 0.00012232370499987155
            k=half_classify[(triton.cdiv(cap,1024),)](scores,*md,out,amb,c,scores,scores,0,0,N=n,ROWS=n,COLS=n,CAP=cap,GAMMA=gamma,T=tfloor,BLOCK=1024,DUMP=False,num_warps=8,enable_fp_fusion=False)
            self.kernels['classify_'+fmt]=k;mark()
            first=c.cpu().numpy();direct,n1,overflow,reject=map(int,first)
            assert direct+n1+reject==n*(n+1)//2 and overflow==0
            c2[0]=direct
            if n1:
                k=certified_fp32_filter_i64[(n1,)](v,amb,out,fp64,c2,tfloor,tfloor,2**-14,2**-22,2**-22,4096*2**-126,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
                self.kernels['fp32_refine']=k
            mid=c2.cpu().numpy();n2=int(mid[3]);assert mid[2]==0;mark()
            if n2:
                k=refine_ambiguous_fp64_i64[(n2,)](v,v,fp64,out,c2,threshold,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
                self.kernels['fp64']=k
        mark();last=c2.cpu().numpy();assert last[2]==0
        assert 0 <= int(last[0]) <= cap
        a=canonical(out[:int(last[0])].cpu().numpy(),n)
        torch.cuda.synchronize();wall=(time.perf_counter()-t0)*1e3
        phase=[events[i].elapsed_time(events[i+1]) for i in range(4)]
        return a,{'format':fmt,'n':n,'threshold_d2':threshold,'e2e_ms':wall,
            'transfer_allocate_prepare_ms':phase[0],'stage1_ms':phase[1],
            'host_counts_fp32_ms':phase[2],'fp64_ms':phase[3],
            'stage1_accept':direct,'stage1_reject':reject,'fp32_pairs':n1,'fp64_pairs':n2,
            'feature_ms':0.0,'inference_ms':0.0,'selected_format':fmt,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'output_count':len(a),'output_sha256':digest(a),'input_sha256':digest(x)}


def numerical_test(engine):
    rng=np.random.default_rng(9919)
    x=rng.normal(size=(97,512)).astype(np.float32)*np.float32(.04)
    x[::3]*=np.float32(2**-14);x[0]=0;x[1]=np.finfo(np.float32).smallest_subnormal
    x[2]=np.linspace(-1,1,512,dtype=np.float32)
    result={};v=torch.from_numpy(x).cuda()
    exact=np.sum(x.astype(np.float64)**2,axis=1)
    distances=np.sum((x.astype(np.float64)[:,None,:]-x.astype(np.float64)[None,:,:])**2,axis=2)
    for fmt in FORMATS:
        q,s,md=engine.prepare(v,fmt);ss=s.cpu().numpy().astype(np.float64);qq=q.cpu().numpy()
        h=codebook(fmt)[qq] if fmt.startswith('e') else qq.astype(np.float64)
        z=h*ss[:,None];dots=engine.dots(q,s,fmt).cpu().numpy()
        truth=z@z.T;err=np.abs(dots.astype(np.float64)-truth)
        gamma=2**-21 if fmt=='int8' else .00012232370499987155
        bound=gamma*np.linalg.norm(z,axis=1)[:,None]*np.linalg.norm(z,axis=1)[None,:]+2048*2**-126
        assert np.all(err<=bound),(fmt,'dot envelope')
        lo,hi,zu,eu=[m.cpu().numpy().astype(np.float64) for m in md]
        assert np.all(lo<=exact) and np.all(hi>=exact),fmt
        assert np.all(zu>=np.linalg.norm(z,axis=1)) and np.all(eu>=np.linalg.norm(x-z,axis=1)),fmt
        if fmt.startswith('e'):
            book=codebook(fmt)[:127 if fmt!='e5m2' else 124]
            av=np.abs(x.astype(np.float64)/ss[:,None]);dist=np.abs(av[:,:,None]-book[None,None,:]);mini=dist.min(axis=2)
            goterr=np.abs(np.abs(h)-av)
            assert np.all(goterr<=mini+1e-12),(fmt,'nearest representable')
        scores=torch.from_numpy(dots).cuda();n=len(x);cap=n*n
        out=torch.empty(cap,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);counts=torch.zeros(4,device='cuda',dtype=torch.int32)
        lower=torch.empty_like(scores);upper=torch.empty_like(scores)
        dumped = half_classify[(triton.cdiv(cap,1024),)](scores,*md,out,amb,counts,lower,upper,0,0,N=n,ROWS=n,COLS=n,CAP=cap,GAMMA=gamma,T=.3955230712890625,BLOCK=1024,DUMP=True,num_warps=8,enable_fp_fusion=False)
        engine.kernels['interval_dump_'+fmt] = dumped
        engine.record_kernels()
        assert np.all(lower.cpu().numpy()<=distances) and np.all(upper.cpu().numpy()>=distances),(fmt,'interval containment')
        result[fmt]={'pairs_checked':cap,'max_abs_dot_error':float(err.max()),'metadata_and_intervals':True}
    return result


