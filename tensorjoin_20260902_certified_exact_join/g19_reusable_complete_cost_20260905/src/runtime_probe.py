"""Bind actual owned Triton launches to their compiled objects; diagnostic only."""
import argparse
import gc
import os
import platform
import time
import triton
import torch
from triton.compiler import CompiledKernel
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3
from operators_r3 import make

a=argparse.ArgumentParser();a.add_argument('--method',choices=['tc','fp32'],required=True);a.add_argument('--record-id',required=True);args=a.parse_args()
check_frozen();check_r1();check_r2();check_r3();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
assert os.environ['NVIDIA_TF32_OVERRIDE']=='0' and platform.node()=='gpu-host-8'
cache=H/'artifacts/cache'/args.record_id;assert not cache.exists();cache.mkdir(parents=True)
os.environ['TRITON_CACHE_DIR']=str(cache)
torch.set_num_threads(1);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False
case=next(c for c in cases() if c['name']=='cifar4096');source,oracle=load(case)
engine=make(args.method);engine.prepare(len(source),case)
output,_=engine.run(source.copy(),case);assert validate(output,oracle,len(source))['exact'];del output
torch.cuda.synchronize()
observed=[];objects={};old=CompiledKernel.launch_metadata
def capture(self,grid,stream,*kernel_args):
    self._init_handles()
    files={k:dict(path=str(Path(v).relative_to(P)),sha256=sha(v)) for k,v in self.metadata_group.items()}
    constants={self.src.fn.arg_names[k[0]]:dict(value=v,hex=v.hex() if isinstance(v,float) else None)
               for k,v in self.src.constants.items() if len(k)==1 and isinstance(v,(int,float,bool,str))}
    obj=dict(kernel=self.name,hash=self.hash,function=int(self.function),module=int(self.module),
             files=files,constants=constants,metadata=self.metadata._asdict())
    obj['metadata']['target']=str(obj['metadata']['target'])
    objects[self.hash]=obj
    observed.append(dict(name=self.name,hash=self.hash,function=int(self.function),grid=list(grid),stream=int(stream)))
    return old(self,grid,stream,*kernel_args)
CompiledKernel.launch_metadata=capture
audits=[]
try:
    for i in range(2):
        x=source.copy();torch.cuda.nvtx.range_push('G19_BIND_'+args.method)
        output,work=engine.run(x,case);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
        audit=validate(output,oracle,len(source));assert audit['exact'];audits.append(dict(audit=audit,work=work))
        del output
finally:CompiledKernel.launch_metadata=old
library=engine.blas.record() if args.method=='fp32' else None
engine.close();del engine;gc.collect();torch.cuda.synchronize()
assert observed and objects
for item in objects.values():
    for name,c in item['constants'].items():
        if name in ['threshold_d2','threshold']:assert c['value']==case['reference_threshold'] and c['hex']==case['reference_threshold'].hex()
write(H/'results'/f'inner_{args.record_id}.json',dict(method=args.method,observed=observed,objects=objects,audits=audits,
    library=library,correctness=dict(exact_contract_pass=True),diagnostic_only=True,performance_admitted=False,
    hook_scope='In-memory diagnostic interception of CompiledKernel.launch_metadata, recording the actual launched compiled object; never installed in timing process.'))
print('RUNTIME_BIND',args.method,len(observed),len(objects),flush=True)
