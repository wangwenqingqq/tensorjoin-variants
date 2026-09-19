"""Host-only experiment utilities; all CUDA programs come from the keeper."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent;OLD=ROOT/'two_gate_campaign_20260908';DGS=ROOT/'dgs_directional_20260908_small_exact'
sys.path.insert(0,str(OLD/'src'))
from g2_operator_r1 import Matrix
from retained_ops import full_source,output_check
from frozen_driver import sha,verify
N,D,B,M,CAP=60000,512,64,938,16777216
LAYOUTS=['original','raw_tree','hadamard_tree'];PLANS=['F8','F16']
COUNTS={'F8':[1278504,1827007,713664,1111609,1734,1993039],'F16':[1950364,86029,41804,42491,1734,1993039]}

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
def load():
 x=full_source();archive=np.load(DGS/'artifacts/orders.npz',allow_pickle=False)
 orders={k:archive[k].astype(np.uint64) for k in LAYOUTS}
 for o in orders.values():assert np.array_equal(np.sort(o),np.arange(N))
 return {k:np.ascontiguousarray(x[orders[k].astype(np.int64)]) for k in LAYOUTS},orders

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

class Collector(Matrix):
 def __init__(self):super().__init__();self.u1=[];self.u2=[];self.batch_u1=[];self.batch_u2=[]
 def stage2(self,v,amb,out,fp64,c,n):
  self.u1.append(amb[:n].cpu().numpy().astype(np.uint64,copy=True));self.batch_u1.append(n)
  return super().stage2(v,amb,out,fp64,c,n)
 def terminal(self,v,pairs,out,c,n):
  self.u2.append(pairs[:n].cpu().numpy().astype(np.uint64,copy=True));self.batch_u2.append(n)
  return super().terminal(v,pairs,out,c,n)
