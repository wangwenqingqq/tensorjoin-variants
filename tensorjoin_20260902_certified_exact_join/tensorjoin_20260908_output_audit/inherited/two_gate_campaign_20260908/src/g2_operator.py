"""Complete common-host 2x2 operator. Frozen F8 is not recompiled."""
import gc,hashlib,json,time
import numpy as np
import torch
from frozen_driver import Driver,HERE,ROOT,sha,verify
from retained_ops import canonical,full_source,output_check
from gpu_kernels import half_metadata
from g2_kernels import scan8,scan16
N,D,CAP,T,EPS=60000,512,16777216,25921/65536,161/256
PARAM=[EPS,EPS,2**-16,16*2**-126,2**-20,16*2**-126]
G=(512*2**-52)/(1-512*2**-52);GAMMA=0.00012232370499987155

class Matrix(Driver):
    def __init__(self):super().__init__();self.kernels={}
    def prepare(self,v,dtype):
        bad=torch.zeros(1,device='cuda',dtype=torch.int32)
        if dtype==8:
            q=torch.empty((N,D),device='cuda',dtype=torch.int8)
            md=[torch.empty(N,device='cuda') for _ in range(3)]
            self.launch('metadata',N,[v,q,*md,bad]);assert bad.item()==0
            return [q,q.T.contiguous(),*md]
        z=torch.empty((N,D),device='cuda',dtype=torch.float16)
        md=[torch.empty(N,device='cuda') for _ in range(4)]
        self.kernels['metadata']=half_metadata[(N,)](v,z,*md,bad,D=D,G=G,num_warps=4,enable_fp_fusion=False)
        assert bad.item()==0
        return [z,*md]
    def stage1(self,m,prep,tr,tc,out,amb,c,score,ld,ud,dump=False):
        tiles=len(tr)
        if m=='F8' and not dump:
            self.launch('stage1',tiles,[*prep,tr,tc,out,amb,c],PARAM);return
        is8=m.endswith('8');kernel=scan8 if is8 else scan16
        kwargs=dict(N=N,D=D,CAP=CAP,num_warps=4,num_stages=3,enable_fp_fusion=is8)
        if not is8:kwargs.update(GAMMA=GAMMA,T=T)
        modes=[0,1] if m.startswith('S') else [2]
        for mode in modes:
            key=f'{8 if is8 else 16}_{mode}_{int(dump and mode!=0)}'
            self.kernels[key]=kernel[(tiles,)](*prep,tr,tc,score,out,amb,c,ld,ud,
                *([] if not is8 else PARAM),MODE=mode,DUMP=dump and mode!=0,**kwargs)
    def stage2(self,v,amb,out,fp64,c,n):
        self.launch('stage2',n,[v,amb,out,fp64,c],[T,T,2**-14,2**-22,2**-22,4096*2**-126])
    def terminal(self,v,pairs,out,c,n):self.launch('terminal',n,[v,v,pairs,out,c],[T])
    def run(self,m,x):
        assert m in ['F8','S8','F16','S16']
        torch.cuda.synchronize();start=time.perf_counter()
        tr,tc=np.triu_indices(938);tr=tr.astype(np.int32);tc=tc.astype(np.int32)
        v=torch.from_numpy(x).to('cuda');prep=self.prepare(v,8 if m.endswith('8') else 16)
        trg=torch.from_numpy(tr).to('cuda');tcg=torch.from_numpy(tc).to('cuda')
        out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
        score=torch.empty(CAP if m.startswith('S') else 1,device='cuda',dtype=torch.int32 if m.endswith('8') else torch.float32)
        parts=[];records=[]
        for off in range(0,len(tr),4096):
            tiles=min(4096,len(tr)-off);c=torch.zeros(6,device='cuda',dtype=torch.int32)
            self.stage1(m,prep,trg[off:off+tiles],tcg[off:off+tiles],out,amb,c,score,score,score)
            s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=CAP
            self.stage2(v,amb,out,fp64,c,n1);s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=CAP
            self.terminal(v,fp64,out,c,n2);last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
            parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True))
            records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
        torch.cuda.synchronize();a=canonical(parts);seconds=time.perf_counter()-start
        return a,dict(seconds=seconds,stage_counts=np.sum(np.array(records,dtype=np.int64),axis=0).tolist(),batches=len(records),input_pointer=v.data_ptr())
    def capture(self):
        directory=HERE/'artifacts/g2_compiled_a0';directory.mkdir(exist_ok=True);rec={}
        for key,k in self.kernels.items():
            entry={}
            for kind in ['cubin','ptx','llir','ttgir','ttir']:
                a=k.asm[kind];data=a if isinstance(a,bytes) else a.encode();p=directory/(key+'.'+kind);h=hashlib.sha256(data).hexdigest()
                if p.exists():assert sha(p)==h,(key,kind)
                else:p.write_bytes(data)
                entry[kind]=h
            meta=json.loads(json.dumps(k.metadata._asdict(),default=str));meta.update(n_regs=k.n_regs,n_spills=k.n_spills)
            p=directory/(key+'.json')
            if p.exists():assert json.loads(p.read_text())==meta
            else:p.write_text(json.dumps(meta,indent=2))
            entry['metadata']=meta;rec[key]=entry
        assert verify()==self.identities
        return rec
    def close(self):gc.collect();torch.cuda.empty_cache()
