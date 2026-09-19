from common import *
import traceback

def output_tests(radix):
 rng=np.random.default_rng(2026090806);cases=[np.array([],np.uint64),np.array([0,N+1,N*N-1],np.uint64),np.array([1,N-1,(N-2)*N+N-1,N*N-1,0],np.uint64)]
 r=rng.integers(0,N,10007,dtype=np.uint64);c=rng.integers(0,N,10007,dtype=np.uint64);a=np.minimum(r,c)*N+np.maximum(r,c);rng.shuffle(a);cases.append(a)
 records=[]
 for ids in cases:
  want=canonical([ids]);g=torch.from_numpy(ids.view(np.int64)).to('cuda');n=len(g)
  expanded_all=torch.full((2*n+64,),117,device='cuda',dtype=torch.int64);sorted_all=torch.full_like(expanded_all,117);expanded=expanded_all[32:-32];sorted_out=sorted_all[32:-32];status=torch.zeros(2,device='cuda',dtype=torch.int32)
  radix.mirror(g,expanded,status);diag,bad=map(int,status.cpu().numpy());assert bad==0 and diag==int(np.count_nonzero(ids//N==ids%N))
  temp,sz=radix.sort(expanded,sorted_out);got=sorted_out[:2*n-diag].cpu().numpy().view(np.uint64)
  assert np.array_equal(got,want)
  for t in [expanded_all,sorted_all]:assert torch.all(t[:32]==117).item() and torch.all(t[-32:]==117).item()
  for mode in OUTPUTS:
   sink=OutputSink(mode,radix)
   for part in torch.tensor_split(g,3):sink.push(part,len(part))
   got,rec=sink.finish();assert np.array_equal(got,want)
  records.append(dict(input_count=n,output_count=len(want),self_count=diag,cub_scratch_bytes=sz,pass_=True))
 bad=torch.tensor([N, N*N],device='cuda',dtype=torch.int64);ex=torch.empty(4,device='cuda',dtype=torch.int64);status=torch.zeros(2,device='cuda',dtype=torch.int32);radix.mirror(bad,ex,status);assert status[1].item()==1
 return records

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
 result=dict(started=time.time(),pass_=False,samples=[],diagnostics=[])
 try:
  x=full_source();tc=Matrix();fp=FP32();radix=Radix();result['output_tests']=output_tests(radix)
  for cell in CELLS:
   for method in METHODS:
    op=fp if method=='F32' else tc;expected=None
    for mode in OUTPUTS:
     rec=call(op,radix,cell,method,mode,x,exact=True)
     if expected is None:expected=rec['stage_counts']
     else:assert rec['stage_counts']==expected
     result['samples'].append(rec);print(json.dumps(rec),flush=True)
    rec=call(op,radix,cell,method,'gpu_radix',x,exact=True,diagnostic=True);assert rec['stage_counts']==expected;result['diagnostics'].append(rec)
   tc.capture();gc.collect();torch.cuda.empty_cache()
  result.update(pass_=True,compiled=tc.capture(),fp32_identities=fp.identities,fp32_library=fp.library_record,fp32_library_hashes=fp.library_hashes,cub_version=radix.version,ended=time.time());result['pass']=True
  write_json(HERE/'artifacts/admission.json',result)
  paths=list((HERE/'src').glob('*'))+[HERE/'PROTOCOL.md',HERE/'targets_r2.json',HERE/'artifacts/build.json',HERE/'artifacts/output.so',HERE/'artifacts/admission.json',SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json']
  paths+=list((OLD/'src').glob('*.py'))+list(OLD.glob('*manifest.json'))
  paths+=[SWEEP/'artifacts'/f"reference_{c['name']}.npy" for c in CELLS]
  paths+=list((SWEEP/'artifacts/compiled').glob('*'))
  freeze={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()};freeze.update(fp.identities)
  write_json(HERE/'artifacts/timing_freeze.json',freeze)
  print(json.dumps(dict(admission_pass=True,frozen_files=len(freeze))),flush=True)
 except Exception:
  result['pass']=False;result['exception']=traceback.format_exc();result['ended']=time.time();write_json(HERE/'results/admission_failed.json',result);raise
if __name__=='__main__':main()
