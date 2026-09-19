"""Complete-cost engines for the bounded G19 contract (one method per process)."""
import ctypes as C
import gc
import sys
from pathlib import Path
import numpy as np
import torch

H=Path(__file__).resolve().parents[1];P=H.parent
for folder in [P/'src',P/'g15_strong_control_20260905/src',P/'g16_gpu_preparation_20260905/src',P/'g18_rthiss_conservative_repair_20260905/src']:
    sys.path.append(str(folder))
from bounds import cuts
from precision_kernels import classify_pedantic_panel,refine_ambiguous_fp64_i64
import fp32_gpu_norm_operator as fp
import run_gpu_prepared_g5 as tc
from gpu_preparation import prepare_device

# Bounded in-memory wiring; original imported source files remain immutable.
fp.classify_pedantic_panel=classify_pedantic_panel
fp.refine_ambiguous_fp64_i64=refine_ambiguous_fp64_i64
tc.refine_ambiguous_fp64_i64=refine_ambiguous_fp64_i64

class RT:
    def __init__(self,diagnostic=False):
        self.path=H/('build_on_a0' if diagnostic else 'build_off_a0')/'libRT-HiSS.so'
        self.lib=C.CDLL(str(self.path),mode=C.RTLD_LOCAL)
        self.lib.g19Error.restype=C.c_char_p
        self.lib.g19Create.argtypes=[];self.lib.g19Create.restype=C.c_int
        self.lib.g19Destroy.argtypes=[];self.lib.g19Destroy.restype=C.c_int
        self.lib.g19Diagnostics.argtypes=[];self.lib.g19Diagnostics.restype=C.c_int
        self.lib.g19Run.argtypes=[C.c_void_p,C.c_uint32,C.c_double,C.c_float,C.c_float,C.c_float,C.c_char_p,C.POINTER(C.c_void_p)]
        self.lib.g19Run.restype=C.c_int
        self.lib.g19ResultData.argtypes=[C.c_void_p];self.lib.g19ResultData.restype=C.POINTER(C.c_uint64)
        self.lib.g19ResultCount.argtypes=[C.c_void_p];self.lib.g19ResultCount.restype=C.c_uint64
        self.lib.g19ResultRelease.argtypes=[C.c_void_p];self.lib.g19ResultRelease.restype=None
        assert self.lib.g19Diagnostics()==int(diagnostic)
        self.check(self.lib.g19Create());self.closed=False
    def check(self,status):
        if status:raise RuntimeError(self.lib.g19Error().decode())
    def prepare(self,n,case):
        # RT native kernels are ahead-of-time compiled; full calls warm OptiX.
        pass
    def run(self,x,case,directory=None):
        low,high=cuts(case['reference_threshold'])
        result=C.c_void_p()
        self.check(self.lib.g19Run(x.ctypes.data,len(x),case['reference_threshold'],case['epsilon'],
                                  float(low),float(high),str(directory).encode() if directory else None,C.byref(result)))
        try:
            count=self.lib.g19ResultCount(result)
            output=np.ctypeslib.as_array(self.lib.g19ResultData(result),shape=(count,)).copy()
        finally:self.lib.g19ResultRelease(result)
        return output,dict(pairs=len(output))
    def close(self):
        assert not self.closed
        self.check(self.lib.g19Destroy());self.closed=True

class FP32:
    def __init__(self):self.blas=fp.BoundCuBLAS()
    def prepare(self,n,case):fp.warmup(n,case['reference_threshold'],self.blas)
    def run(self,x,case,directory=None):return fp.run_operator(x,case['reference_threshold'],self.blas)
    def close(self):
        # The handle is borrowed from PyTorch, not owned by this wrapper.
        gc.collect();torch.cuda.synchronize()

class TC:
    def __init__(self):pass
    def prepare(self,n,case):
        tc.N=n
        el,eu=tc.outward_float32(case['epsilon'])
        lo,hi=tc.outward_float32(case['reference_threshold'])
        tc.compile_exact_specializations(float(el),float(eu),float(lo),float(hi),case['reference_threshold'])
        x=torch.zeros((n,512),dtype=torch.float32,device='cuda')
        metadata=prepare_device(x)
        torch.cuda.synchronize();del x,metadata
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    def run(self,x,case,directory=None):
        n=len(x);cap=tc.CAPACITY
        el,eu=tc.outward_float32(case['epsilon'])
        lo,hi=tc.outward_float32(case['reference_threshold'])
        extent=(n+tc.BLOCK_M-1)//tc.BLOCK_M
        rows,cols=np.triu_indices(extent)
        rows=np.asarray(rows,dtype=np.int32);cols=np.asarray(cols,dtype=np.int32)
        vg=torch.from_numpy(x).to('cuda')
        codes,scales,norms,errors,codes_t=prepare_device(vg)
        rg=torch.from_numpy(rows).to('cuda');cg=torch.from_numpy(cols).to('cuda')
        results=torch.empty(cap,dtype=torch.int64,device='cuda')
        ambiguous=torch.empty_like(results);terminal=torch.empty_like(results)
        parts=[];records=[]
        for start in range(0,len(rows),tc.TILES_PER_BATCH):
            stop=min(start+tc.TILES_PER_BATCH,len(rows));c=torch.zeros(6,dtype=torch.int32,device='cuda')
            tc.analytic_certificate_ragged_safe_i64[(stop-start,)](
                codes,codes_t,scales,norms,errors,rg[start:stop],cg[start:stop],results,ambiguous,c,
                float(el),float(eu),tc.EXPRESSION_RELATIVE_RADIUS,tc.EXPRESSION_ABSOLUTE_RADIUS,
                tc.SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,tc.SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
                M=n,N_=n,K=512,CAPACITY=cap,BLOCK_M=tc.BLOCK_M,BLOCK_N=tc.BLOCK_N,BLOCK_K=tc.BLOCK_K,
                num_warps=4,num_stages=3)
            a=c.cpu().numpy().astype(np.int64);direct,unclear=int(a[0]),int(a[1])
            if a[2] or max(direct,unclear)>cap:raise RuntimeError('TC stage1 capacity')
            if unclear:
                tc.certified_fp32_filter_i64[(unclear,)](
                    vg,ambiguous,results,terminal,c,float(lo),float(hi),tc.FP32_DISTANCE_RELATIVE_RADIUS,
                    tc.FP32_INPUT_MAGNITUDE_RADIUS,tc.FP32_FINAL_RELATIVE_RADIUS,tc.FP32_ABSOLUTE_RADIUS,
                    N_=n,K=512,CAPACITY=cap,BLOCK_K=tc.FP32_BLOCK_K,num_warps=4)
            b=c.cpu().numpy().astype(np.int64);fp64=int(b[3])
            if b[2] or max(int(b[0]),fp64)>cap:raise RuntimeError('TC stage2 capacity')
            if fp64:
                refine_ambiguous_fp64_i64[(fp64,)](vg,vg,terminal,results,c,case['reference_threshold'],
                    N_=n,K=512,CAPACITY=cap,BLOCK_K=256,num_warps=4)
            end=c.cpu().numpy().astype(np.int64);count=int(end[0])
            if end[2] or count>cap:raise RuntimeError('TC terminal capacity')
            parts.append(results[:count].cpu().numpy().astype(np.uint64,copy=True))
            records.append(dict(direct=direct,ambiguous=unclear,fp64=fp64,accepted_upper=count,
                                fp32_reject=int(b[4]),fp32_equal=int(b[5]),overflow=int(end[2])))
        torch.cuda.synchronize()
        upper=np.sort(np.concatenate(parts).astype(np.uint64,copy=False))
        r=upper//np.uint64(n);c=upper-r*np.uint64(n);nonself=r<c
        canonical=np.sort(np.concatenate((upper,c[nonself]*np.uint64(n)+r[nonself])).astype(np.uint64,copy=False))
        return canonical,dict(batches=records,scheduled_tiles=len(rows),pairs=len(canonical),input_device_pointer=int(vg.data_ptr()))
    def close(self):gc.collect();torch.cuda.synchronize()

def make(name):
    return RT(name=='rt_on') if name in ['rt','rt_on'] else TC() if name=='tc' else FP32()
