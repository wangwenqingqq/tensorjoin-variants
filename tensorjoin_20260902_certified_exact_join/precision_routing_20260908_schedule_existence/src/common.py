"""Host-only experiment utilities; all CUDA programs come from the keeper."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent;OLD=ROOT/'two_gate_campaign_20260908';DGS=ROOT/'dgs_directional_20260908_small_exact'
sys.path.insert(0,str(OLD/'src'))
from g2_operator_r1 import Matrix as OriginalMatrix
from retained_ops import full_source,output_check,canonical
from frozen_driver import sha,verify
N,D,B,M,CAP=60000,512,64,938,16777216
LAYOUTS=['original','clustered','interleaved'];PLANS=['F8','F16']
COUNTS={'F8':[1278504,1827007,713664,1111609,1734,1993039],'F16':[1950364,86029,41804,42491,1734,1993039]}

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
def load():
 x=full_source();archive=np.load(HERE/'artifacts/schedules.npz',allow_pickle=False)
 schedules={k:archive[k].astype(np.int64) for k in LAYOUTS}
 for ix in schedules.values():assert np.array_equal(np.sort(ix),np.arange(M*(M+1)//2))
 orders={k:np.arange(N,dtype=np.uint64) for k in LAYOUTS}
 return {k:(x,schedules[k]) for k in LAYOUTS},orders

def remap(a,order,unordered=False):
 r=order[(a//N).astype(np.int64)];c=order[(a%N).astype(np.int64)]
 return np.sort(np.minimum(r,c)*N+np.maximum(r,c) if unordered else r*N+c)

def check_old():
 for name,h in json.loads((OLD/'artifacts/g2_timing_freeze_a0.json').read_text()).items():assert sha(OLD/name)==h,name
 a=json.loads((OLD/'artifacts/g2_admission_a0.json').read_text());assert a['pass']
 for name,h in a['evidence_sha256'].items():assert sha(OLD/name)==h,name
 return verify()

def one(op,method,x,order):
 torch.cuda.synchronize();start=time.perf_counter()
 a,rec=op.run(method,x)
 logical=remap(a,order)
 elapsed=time.perf_counter()-start
 rec['inner_seconds']=rec.pop('seconds');rec['seconds']=elapsed
 assert rec['stage_counts']==COUNTS[method],(method,rec['stage_counts'])
 h=output_check(logical)
 return logical,dict(rec,hash=h,count=len(logical))

def distribution(ids):
 r=ids//N//B;c=ids%N//B
 index=(r*M-r*(r-1)//2+(c-r)).astype(np.int64)
 hist=np.bincount(index,minlength=M*(M+1)//2)
 assert hist.sum()==len(ids)
 count=len(hist);mass=int(hist.sum());top=int(np.ceil(count*.1))
 vals=np.sort(hist)
 return hist,{'pairs':mass,'occupied_tiles':int(np.count_nonzero(hist)),'total_tiles':count,'top10pct_mass':float(vals[-top:].sum()/mass) if mass else 0.,'max_pairs_per_tile':int(vals[-1]),'hist_quantiles':np.quantile(hist,[0,.5,.9,.99,1]).tolist()}

class Matrix(OriginalMatrix):
    def run(self,m,entry):
        x,schedule=entry
        assert m in ['F8','S8','F16','S16']
        torch.cuda.synchronize();start=time.perf_counter()
        tr,tc=np.triu_indices(938);tr=tr[schedule].astype(np.int32);tc=tc[schedule].astype(np.int32)
        v=torch.from_numpy(x).to('cuda');prep=self.prepare(v,8 if m.endswith('8') else 16)
        trg=torch.from_numpy(tr).to('cuda');tcg=torch.from_numpy(tc).to('cuda')
        out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
        score=torch.empty(CAP if m.startswith('S') else 1,device='cuda',dtype=torch.int32 if m.endswith('8') else torch.float32)
        dump_pointer=score.view(torch.float32)
        parts=[];records=[]
        for off in range(0,len(tr),4096):
            tiles=min(4096,len(tr)-off);c=torch.zeros(6,device='cuda',dtype=torch.int32)
            self.stage1(m,prep,trg[off:off+tiles],tcg[off:off+tiles],out,amb,c,score,dump_pointer,dump_pointer)
            s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=CAP
            self.stage2(v,amb,out,fp64,c,n1);s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=CAP
            self.terminal(v,fp64,out,c,n2);last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
            parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True))
            records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
        torch.cuda.synchronize();a=canonical(parts);seconds=time.perf_counter()-start
        return a,dict(seconds=seconds,stage_counts=np.sum(np.array(records,dtype=np.int64),axis=0).tolist(),batches=len(records),input_pointer=v.data_ptr())

class Collector(Matrix):
 def __init__(self):super().__init__();self.u1=[];self.u2=[];self.batch_u1=[];self.batch_u2=[]
 def stage2(self,v,amb,out,fp64,c,n):
  self.u1.append(amb[:n].cpu().numpy().astype(np.uint64,copy=True));self.batch_u1.append(n)
  return super().stage2(v,amb,out,fp64,c,n)
 def terminal(self,v,pairs,out,c,n):
  self.u2.append(pairs[:n].cpu().numpy().astype(np.uint64,copy=True));self.batch_u2.append(n)
  return super().terminal(v,pairs,out,c,n)
