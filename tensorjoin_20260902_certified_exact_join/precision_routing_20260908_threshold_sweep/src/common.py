import gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OLD=ROOT/'two_gate_campaign_20260908'
sys.path.insert(0,str(OLD/'src'))
from g2_operator_r1 import Matrix as OriginalMatrix,N,D,CAP,GAMMA
from frozen_driver import Driver,sha,verify
from retained_ops import canonical,full_source
from g2_kernels import scan16
CELLS=json.loads((HERE/'artifacts/thresholds.json').read_text())['cells']
METHODS=['F8','F16']
def digest(a):return hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
def write_json(p,obj):p.open('x').write(json.dumps(obj,indent=2,default=str))
def verify_freeze():
 for rel,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():assert sha(ROOT/rel)==h,rel
def refpath(cell):return HERE/'artifacts'/f"reference_{cell['name']}.npy"

class Matrix(OriginalMatrix):
 def set_cell(self,cell):self.cell=cell;self.eps=cell['eps'];self.T=cell['T']
 def stage1(self,m,prep,tr,tc,out,amb,c,score,ld,ud,dump=False):
  assert m in METHODS and not dump
  if m=='F8':
   self.launch('stage1',len(tr),[*prep,tr,tc,out,amb,c],[self.eps,self.eps,2**-16,16*2**-126,2**-20,16*2**-126]);return
  key=f"scan16_m{self.cell['m']}"
  self.kernels[key]=scan16[(len(tr),)](*prep,tr,tc,score,out,amb,c,ld,ud,MODE=2,DUMP=False,N=N,D=D,CAP=CAP,GAMMA=GAMMA,T=self.T,num_warps=4,num_stages=3,enable_fp_fusion=False)
 def stage2(self,v,amb,out,fp64,c,n):self.launch('stage2',n,[v,amb,out,fp64,c],[self.T,self.T,2**-14,2**-22,2**-22,4096*2**-126])
 def terminal(self,v,pairs,out,c,n):self.launch('terminal',n,[v,v,pairs,out,c],[self.T])
 def capture(self):
  directory=HERE/'artifacts/compiled';directory.mkdir(exist_ok=True);rec={}
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
   else:write_json(p,meta)
   assert meta['n_spills']==0 and meta['global_scratch_size']==meta['profile_scratch_size']==0
   entry['metadata']=meta;rec[key]=entry
  assert verify()==self.identities
  return rec
 def diagnostic(self,m,x):
  torch.cuda.synchronize();start=time.perf_counter()
  tr,tc=np.triu_indices(938);tr=tr.astype(np.int32);tc=tc.astype(np.int32)
  v=torch.from_numpy(x).to('cuda');prep=self.prepare(v,8 if m=='F8' else 16)
  trg=torch.from_numpy(tr).to('cuda');tcg=torch.from_numpy(tc).to('cuda')
  out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
  score=torch.empty(1,device='cuda',dtype=torch.int32 if m=='F8' else torch.float32);dummy=score.view(torch.float32)
  torch.cuda.synchronize();prepared=time.perf_counter();events=[];parts=[];counts=[];download=0
  for off in range(0,len(tr),4096):
   c=torch.zeros(6,device='cuda',dtype=torch.int32)
   ev=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
   ev[0].record();self.stage1(m,prep,trg[off:off+4096],tcg[off:off+4096],out,amb,c,score,dummy,dummy);ev[1].record()
   s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and n1<=CAP
   ev[2].record();self.stage2(v,amb,out,fp64,c,n1);ev[3].record()
   s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and n2<=CAP
   ev[4].record();self.terminal(v,fp64,out,c,n2);ev[5].record()
   last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and n<=CAP
   d=time.perf_counter();parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True));download+=time.perf_counter()-d
   counts.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n]);events.append(ev)
  torch.cuda.synchronize();sorting=time.perf_counter();a=canonical(parts);end=time.perf_counter()
  return a,dict(seconds=end-start,preparation_seconds=prepared-start,canonicalization_seconds=end-sorting,output_download_seconds=download,stage_gpu_ms=np.sum([[e[0].elapsed_time(e[1]),e[2].elapsed_time(e[3]),e[4].elapsed_time(e[5])] for e in events],axis=0).tolist(),stage_counts=np.sum(counts,axis=0).tolist())

def check_output(a,cell,exact=False):
 ref=json.loads((HERE/'artifacts/reference_manifest.json').read_text())['cells'][cell['name']]
 h=digest(a);assert len(a)==ref['count'] and h==ref['sha256'],(cell['name'],len(a),h,ref)
 if exact:assert np.array_equal(a,np.load(refpath(cell),mmap_mode='r'))
 return h
