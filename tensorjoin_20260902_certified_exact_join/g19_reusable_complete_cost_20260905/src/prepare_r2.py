"""Add explicit post-engine allocator cleanup, preserving all R1 files."""
from common import *
cleanup='''
def close_torch_engine():
    gc.collect();torch.cuda.synchronize()
    receipt={'before':dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),host=torch.cuda.memory.host_memory_stats())}
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect();torch.cuda.synchronize()
    receipt['after_workspace_clear']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    torch.cuda.empty_cache()
    library=C.CDLL(str(H/'artifacts/host_allocator.so'))
    library.g19EmptyPinnedCache.argtypes=[];library.g19EmptyPinnedCache.restype=None
    library.g19EmptyPinnedCache();torch.cuda.synchronize()
    receipt['after']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),host=torch.cuda.memory.host_memory_stats())
    assert receipt['after']['allocated']==0 and receipt['after']['reserved']==0
    return receipt
'''
s=(H/'src/operators_r1.py').read_text()
s=s.replace('class RT:',cleanup+'\nclass RT:',1)
s=s.replace('        gc.collect();torch.cuda.synchronize()','        return close_torch_engine()',1)
s=s.replace('    def close(self):gc.collect();torch.cuda.synchronize()','    def close(self):return close_torch_engine()',1)
assert s.count('return close_torch_engine()')==2
(H/'src/operators_r2.py').write_text(s)
for original,destination,replacements in [
 ('run_slot_r1.py','run_slot_r2.py',[('from operators_r1 import make','from operators_r2 import make'),
  ('from common_r1 import check_r1','from common_r1 import check_r1\nfrom common_r2 import check_r2'),
  ('check_frozen();check_r1();root=','check_frozen();check_r1();check_r2();root='),
  ('results/admission_r1.json','results/admission_r2.json'),
  ("assert max(m['torch_allocated'] for m in stable)==min(m['torch_allocated'] for m in stable)==0",
   "assert max(m['torch_allocated'] for m in stable)==min(m['torch_allocated'] for m in stable)==engine_record['after_prepare_memory']['torch_allocated']"),
  ('engine.close();del engine;',"engine_record['close_receipt']=engine.close();del engine;")]),
 ('supervise_r1.py','supervise_r2.py',[('src/run_slot_r1.py','src/run_slot_r2.py'),
  ('from common_r1 import check_r1','from common_r1 import check_r1\nfrom common_r2 import check_r2'),
  ('check_frozen();check_r1();command=','check_frozen();check_r1();check_r2();command=')])]:
    s=(H/'src'/original).read_text()
    for a,b in replacements:assert a in s,a;s=s.replace(a,b)
    p=H/'src'/destination;assert not p.exists();p.write_text(s)
print('R2 source prepared; operations and all kernels unchanged')
