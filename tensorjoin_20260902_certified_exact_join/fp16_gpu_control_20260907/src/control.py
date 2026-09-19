"""Owned FP16 GPU control; original INT8 method remains a separate frozen path."""
import hashlib
import json
import time
import numpy as np
import torch
from frozen_driver import Driver, HERE, ROOT, sha, verify
from owned_half import OwnedCuBLAS
from retained_ops import Operators as OldOperators, full_source, output_check, canonical
from gpu_kernels import half_metadata, half_classify

N,D,CAP,T,EPS=60000,512,16777216,25921/65536,161/256
G=(512*2**-52)/(1-512*2**-52)
GAMMA=0.00012232370499987155


class Control(Driver):
    def __init__(self):
        super().__init__()
        self.blas=OwnedCuBLAS();self.kernels={}
        self.library_hashes={p:sha(__import__('pathlib').Path(p)) for p in json.loads((HERE/'library_manifest.json').read_text())}
        assert self.library_hashes==json.loads((HERE/'library_manifest.json').read_text())

    def prepare(self,v):
        assert v.shape==(N,D) and v.dtype==torch.float32 and v.is_contiguous()
        half=torch.empty((N,D),device='cuda',dtype=torch.float16)
        md=[torch.empty(N,device='cuda') for _ in range(4)]
        invalid=torch.zeros(1,device='cuda',dtype=torch.int32)
        k=half_metadata[(N,)](v,half,*md,invalid,D=D,G=G,num_warps=4,enable_fp_fusion=False)
        self.kernels['metadata']=k
        assert invalid.item()==0,'metadata admission'
        return half,md

    def classify(self,dots,md,out,amb,c,row,col,dump=False):
        rows,cols=dots.shape
        lo=torch.empty_like(dots) if dump else dots
        hi=torch.empty_like(dots) if dump else dots
        k=half_classify[((rows*cols+1023)//1024,)](dots,*md,out,amb,c,lo,hi,row,col,
            N=N,ROWS=rows,COLS=cols,CAP=CAP,GAMMA=GAMMA,T=T,BLOCK=1024,DUMP=dump,
            num_warps=8,enable_fp_fusion=False)
        self.kernels[f'class_{rows}_{cols}_{int(dump)}']=k
        return lo,hi

    def terminal(self,v,pairs,out,c,n):
        self.launch('terminal',n,[v,v,pairs,out,c],[T])

    def stage2(self,v,amb,out,fp64,c,n):
        self.launch('stage2',n,[v,amb,out,fp64,c],[T,T,2**-14,2**-22,2**-22,4096*2**-126])

    def run(self,method,x):
        if method=='A':return OldOperators.run(self,'A',x)
        assert method=='C'
        torch.cuda.synchronize();start=time.perf_counter()
        v=torch.from_numpy(x).to('cuda');half,md=self.prepare(v)
        score=torch.empty(CAP,device='cuda');out=torch.empty(CAP,device='cuda',dtype=torch.int64)
        amb=torch.empty_like(out);fp64=torch.empty_like(out);parts=[];records=[]
        for row in range(0,N,4096):
            for col in range(row,N,4096):
                rows,cols=min(4096,N-row),min(4096,N-col)
                dots=score[:rows*cols].view(rows,cols)
                self.blas.gemm(half[row:row+rows],half[col:col+cols],dots)
                c=torch.zeros(4,device='cuda',dtype=torch.int32)
                self.classify(dots,md,out,amb,c,row,col)
                first=c.cpu().numpy().copy();direct,n1,overflow,reject=map(int,first)
                expected=rows*(rows+1)//2 if row==col else rows*cols
                assert direct+n1+reject==expected and overflow==0 and max(direct,n1)<=CAP
                c2=torch.zeros(6,device='cuda',dtype=torch.int32);c2[0]=direct
                self.stage2(v,amb,out,fp64,c2,n1)
                mid=c2.cpu().numpy().copy();n2=int(mid[3]);assert mid[2]==0 and n2<=CAP
                self.terminal(v,fp64,out,c2,n2)
                last=c2.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and n<=CAP
                parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True))
                records.append([direct,n1,reject,int(mid[0]-direct),int(mid[4]),n2,n])
        torch.cuda.synchronize();a=canonical(parts);seconds=time.perf_counter()-start
        return a,{'seconds_diagnostic_until_timing_admitted':seconds,'batches':len(records),
                  'stage_counts':np.sum(np.array(records,dtype=np.int64),axis=0).tolist(),'input_pointer':v.data_ptr()}

    def capture(self):
        directory=HERE/'artifacts'/'compiled';directory.mkdir(exist_ok=True)
        rec={}
        for key,kernel in self.kernels.items():
            entry={}
            for kind in ['cubin','ptx','llir','ttgir','ttir']:
                value=kernel.asm[kind];data=value if isinstance(value,bytes) else value.encode()
                p=directory/(key+'.'+kind);h=hashlib.sha256(data).hexdigest()
                if p.exists():assert sha(p)==h,('compiled identity drift',key,kind)
                else:
                    with p.open('xb') as f:f.write(data)
                entry[kind]=h
            meta=kernel.metadata._asdict();meta.update(n_regs=kernel.n_regs,n_spills=kernel.n_spills)
            p=directory/(key+'.json')
            if p.exists():assert json.loads(p.read_text())==meta
            else:
                with p.open('x') as f:json.dump(meta,f,indent=2)
            entry['metadata']=meta;rec[key]=entry
        assert verify()==self.identities
        return rec

    def close(self):
        self.blas.close();self.blas=None
        import gc
        gc.collect();torch.cuda.empty_cache()
