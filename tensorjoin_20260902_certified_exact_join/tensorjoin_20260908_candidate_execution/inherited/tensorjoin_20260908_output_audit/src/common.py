import ctypes as C,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OLD=ROOT/'two_gate_campaign_20260908';SWEEP=ROOT/'precision_routing_20260908_threshold_sweep'
sys.path.insert(0,str(OLD/'src'))
from g2_operator_r1 import Matrix as OriginalMatrix,N,D,CAP,GAMMA
from g2_kernels import scan16
from frozen_driver import sha,verify
from retained_ops import Operators,canonical,full_source,PANEL
CELLS=json.loads((SWEEP/'artifacts/thresholds.json').read_text())['cells']
REF=json.loads((SWEEP/'artifacts/reference_manifest.json').read_text())
METHODS=['F8','F16','F32'];OUTPUTS=['legacy','cpu_single','gpu_radix']
def write_json(p,r):p.open('x').write(json.dumps(r,indent=2,default=str))
def digest(a):return hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
def check_output(a,cell,exact=False):
 r=REF['cells'][cell['name']];h=digest(a);assert len(a)==r['count'] and h==r['sha256'],(cell['name'],len(a),h)
 if exact:assert np.array_equal(a,np.load(SWEEP/'artifacts'/f"reference_{cell['name']}.npy",mmap_mode='r'))
 return h
def verify_freeze():
 for rel,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():assert sha(ROOT/rel)==h,rel

class Radix:
 def __init__(self):
  build=json.loads((HERE/'artifacts/build.json').read_text());assert sha(HERE/'artifacts/output.so')==build['library_sha256']
  self.lib=C.CDLL(str(HERE/'artifacts/output.so'))
  self.lib.output_mirror.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_uint,C.c_void_p,C.c_void_p];self.lib.output_mirror.restype=C.c_int
  self.lib.output_sort.argtypes=[C.c_void_p,C.POINTER(C.c_size_t),C.c_void_p,C.c_void_p,C.c_int,C.c_void_p];self.lib.output_sort.restype=C.c_int
  self.lib.output_cub_version.restype=C.c_int
  self.version=self.lib.output_cub_version()
 def mirror(self,upper,expanded,status,n=N):
  rc=self.lib.output_mirror(upper.data_ptr(),expanded.data_ptr(),len(upper),n,status.data_ptr(),torch.cuda.current_stream().cuda_stream);assert rc==0,('mirror',rc)
 def sort(self,expanded,sorted_out):
  stream=torch.cuda.current_stream().cuda_stream;size=C.c_size_t()
  rc=self.lib.output_sort(None,C.byref(size),expanded.data_ptr(),sorted_out.data_ptr(),len(expanded),stream);assert rc==0,('CUB query',rc)
  temp=torch.empty(max(size.value,1),device='cuda',dtype=torch.uint8)
  rc=self.lib.output_sort(temp.data_ptr(),C.byref(size),expanded.data_ptr(),sorted_out.data_ptr(),len(expanded),stream);assert rc==0,('CUB sort',rc)
  return temp,size.value

class OutputSink:
 def __init__(self,mode,radix,diagnostic=False):
  self.mode=mode;self.radix=radix;self.parts=[];self.diagnostic=diagnostic;self.collect_seconds=0.;self.download_bytes=0;self.gpu_copy_bytes=0
 def push(self,out,n):
  t=time.perf_counter() if self.diagnostic else 0
  if self.mode=='gpu_radix':self.parts.append(out[:n].clone());self.gpu_copy_bytes+=8*n
  else:self.parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True));self.download_bytes+=8*n
  if self.diagnostic:self.collect_seconds+=time.perf_counter()-t
 def finish(self):
  t=time.perf_counter();n=sum(len(p) for p in self.parts);scratch=0;selfs=None
  if self.mode=='legacy':a=canonical(self.parts);self.parts.clear()
  elif self.mode=='cpu_single':
   u=np.concatenate(self.parts).astype(np.uint64,copy=False);self.parts.clear();r,c=u//N,u%N
   a=np.sort(np.concatenate([u,c[r<c]*N+r[r<c]]))
  else:
   upper=torch.cat(self.parts) if self.parts else torch.empty(0,device='cuda',dtype=torch.int64);self.parts.clear()
   expanded=torch.empty(2*n,device='cuda',dtype=torch.int64);status=torch.zeros(2,device='cuda',dtype=torch.int32)
   self.radix.mirror(upper,expanded,status);diag,bad=map(int,status.cpu().numpy());assert bad==0 and diag<=n
   selfs=diag;del upper
   sorted_out=torch.empty_like(expanded);temp,scratch=self.radix.sort(expanded,sorted_out)
   a=sorted_out[:2*n-diag].cpu().numpy().view(np.uint64);self.download_bytes+=len(a)*8
  return a,dict(output_mode=self.mode,upper_count=n,self_count=selfs,output_count=len(a),output_finalize_seconds=time.perf_counter()-t,output_collect_seconds=self.collect_seconds if self.diagnostic else None,output_download_bytes=self.download_bytes,output_gpu_copy_bytes=self.gpu_copy_bytes,cub_scratch_bytes=scratch)

class Matrix(OriginalMatrix):
 def set_cell(self,cell):self.cell=cell;self.T=cell['T'];self.eps=cell['eps']
 def stage1(self,m,prep,tr,tc,out,amb,c,score,ld,ud,dump=False):
  assert m in ['F8','F16'] and not dump
  if m=='F8':self.launch('stage1',len(tr),[*prep,tr,tc,out,amb,c],[self.eps,self.eps,2**-16,16*2**-126,2**-20,16*2**-126]);return
  self.kernels[f"scan16_m{self.cell['m']}"]=scan16[(len(tr),)](*prep,tr,tc,score,out,amb,c,ld,ud,MODE=2,DUMP=False,N=N,D=D,CAP=CAP,GAMMA=GAMMA,T=self.T,num_warps=4,num_stages=3,enable_fp_fusion=False)
 def stage2(self,v,amb,out,fp64,c,n):self.launch('stage2',n,[v,amb,out,fp64,c],[self.T,self.T,2**-14,2**-22,2**-22,4096*2**-126])
 def terminal(self,v,pairs,out,c,n):self.launch('terminal',n,[v,v,pairs,out,c],[self.T])
 def capture(self):
  rec={}
  for key,k in self.kernels.items():
   entry={}
   for ext in ['cubin','ptx','llir','ttgir','ttir']:
    data=k.asm[ext];data=data if isinstance(data,bytes) else data.encode();h=hashlib.sha256(data).hexdigest();p=SWEEP/'artifacts/compiled'/(key+'.'+ext);assert sha(p)==h,(key,ext);entry[ext]=h
   assert k.n_spills==0;rec[key]=entry
  assert verify()==self.identities
  return rec
 def execute(self,m,x,sink,diagnostic=False):
  torch.cuda.synchronize();start=time.perf_counter()
  tr,tc=np.triu_indices(938);tr=tr.astype(np.int32);tc=tc.astype(np.int32)
  v=torch.from_numpy(x).to('cuda');prep=self.prepare(v,8 if m=='F8' else 16)
  trg=torch.from_numpy(tr).to('cuda');tcg=torch.from_numpy(tc).to('cuda')
  out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
  score=torch.empty(1,device='cuda',dtype=torch.int32 if m=='F8' else torch.float32);dummy=score.view(torch.float32)
  records=[];events=[];prepared=time.perf_counter()
  for off in range(0,len(tr),4096):
   c=torch.zeros(6,device='cuda',dtype=torch.int32)
   if diagnostic:ev=[torch.cuda.Event(enable_timing=True) for _ in range(6)];ev[0].record()
   self.stage1(m,prep,trg[off:off+4096],tcg[off:off+4096],out,amb,c,score,dummy,dummy)
   if diagnostic:ev[1].record()
   s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=CAP
   if diagnostic:ev[2].record()
   self.stage2(v,amb,out,fp64,c,n1)
   if diagnostic:ev[3].record()
   s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=CAP
   if diagnostic:ev[4].record()
   self.terminal(v,fp64,out,c,n2)
   if diagnostic:ev[5].record();events.append(ev)
   last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
   sink.push(out,n);records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
  torch.cuda.synchronize();before_output=time.perf_counter();a,sr=sink.finish();end=time.perf_counter()
  result=dict(seconds=end-start,stage_counts=np.sum(records,axis=0).tolist(),batches=len(records),**sr)
  if diagnostic:result.update(preparation_seconds=prepared-start,pre_finalize_seconds=before_output-start,stage_gpu_ms=np.sum([[e[i].elapsed_time(e[i+1]) for i in [0,2,4]] for e in events],axis=0).tolist())
  return a,result

class FP32(Operators):
 def set_cell(self,cell):self.cell=cell;self.T=cell['T']
 def terminal(self,v,pairs,out,c,n):self.launch('terminal',n,[v,v,pairs,out,c],[self.T])
 def panel(self,v,norms,score,out,amb,c,row,col,rows,cols,prefix=None):
  assert torch.cuda.current_stream().cuda_stream==self.stream
  matrix=score[:rows*cols].view(rows,cols);self.blas.gemm(v[row:row+rows],v[col:col+cols],matrix)
  self.baseline(f'dense_{rows}x{cols}',(rows*cols+1023)//1024 if prefix is None else prefix,[matrix,*norms,out,amb,c,matrix,matrix],[row,col,self.T])
 def execute(self,m,x,sink,diagnostic=False):
  assert m=='F32';torch.cuda.synchronize();start=time.perf_counter()
  v=torch.from_numpy(x).to('cuda');norms=self.norm(v);score=torch.empty(CAP,device='cuda',dtype=torch.float32)
  out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);records=[];events=[];prepared=time.perf_counter()
  for row in range(0,N,PANEL):
   for col in range(row,N,PANEL):
    rows,cols=min(PANEL,N-row),min(PANEL,N-col);c=torch.zeros(4,device='cuda',dtype=torch.int32)
    if diagnostic:ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ev[0].record()
    self.panel(v,norms,score,out,amb,c,row,col,rows,cols)
    if diagnostic:ev[1].record()
    direct,n1,overflow,rejected=map(int,c.cpu().numpy());expect=rows*(rows+1)//2 if row==col else rows*cols
    assert direct+n1+rejected==expect and overflow==0 and max(direct,n1)<=CAP
    if diagnostic:ev[2].record()
    self.terminal(v,amb,out,c,n1)
    if diagnostic:ev[3].record();events.append(ev)
    last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
    sink.push(out,n);records.append([direct,n1,rejected,n])
  torch.cuda.synchronize();before_output=time.perf_counter();a,sr=sink.finish();end=time.perf_counter()
  result=dict(seconds=end-start,stage_counts=np.sum(records,axis=0).tolist(),batches=len(records),**sr)
  if diagnostic:result.update(preparation_seconds=prepared-start,pre_finalize_seconds=before_output-start,stage_gpu_ms=np.sum([[e[i].elapsed_time(e[i+1]) for i in [0,2]] for e in events],axis=0).tolist())
  return a,result

def call(op,radix,cell,method,output,x,exact=False,diagnostic=False):
 op.set_cell(cell);torch.cuda.reset_peak_memory_stats();sink=OutputSink(output,radix,diagnostic)
 a,r=op.execute(method,x,sink,diagnostic);r.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 r.update(output_sha256=check_output(a,cell,exact),method=method,cell=cell['name'],diagnostic=diagnostic)
 del a,sink
 return r
