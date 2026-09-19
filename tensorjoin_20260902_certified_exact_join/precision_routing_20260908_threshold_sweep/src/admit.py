from common import *
import collections,ctypes as C,re,subprocess

def audit(op):
 op.bind('cuModuleUnload',[C.c_void_p]);op.bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p]);records=[]
 for f in sorted((HERE/'artifacts/compiled').glob('*.cubin')):
  md=json.loads(f.with_suffix('.json').read_text());ptx=f.with_suffix('.ptx').read_text();name=md['name']
  sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','-sass',str(f)],text=True)
  resource=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','-res-usage',str(f)],text=True)
  f.with_suffix('.sass.txt').write_text(sass);f.with_suffix('.resource.txt').write_text(resource)
  ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([\w.]+)',sass))
  assert md['n_spills']==0 and md['global_scratch_size']==md['profile_scratch_size']==0
  assert not any(k.split('.')[0] in ['LDL','STL'] for k in ops)
  assert 'STACK:0' in resource and 'LOCAL:0' in resource
  if f.stem.startswith('scan16'):
   assert '.ftz' not in ptx and '.FTZ' not in sass and '.f64' not in ptx
   assert any(k.startswith('HMMA') for k in ops)
   assert ops['FADD.RP'] and ops['FADD.RM'] and ops['FMUL.RP']
  decl=ptx.split('.visible .entry '+name+'(')[1].split(')')[0];kinds=re.findall(r'\.param \.(u64|u32|f32)\b',decl)
  assert kinds==['u64']*(9 if name=='half_metadata' else 15) and '.reqntid 128' in ptx
  mod,fun=C.c_void_p(),C.c_void_p();op.check(op.lib.cuModuleLoad(C.byref(mod),str(f).encode()),'load')
  try:
   op.check(op.lib.cuModuleGetFunction(C.byref(fun),mod,name.encode()),'function');params=[]
   for i in range(len(kinds)):
    off,got=C.c_size_t(),C.c_size_t();op.check(op.lib.cuFuncGetParamInfo(fun,i,C.byref(off),C.byref(got)),'parameter');assert (off.value,got.value)==(8*i,8);params.append([off.value,got.value])
   attrs={}
   for key,code in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers',4)]:
    value=C.c_int();op.check(op.lib.cuFuncGetAttribute(C.byref(value),code,fun),'attribute');attrs[key]=value.value
   assert attrs['registers']==md['n_regs'] and attrs['local_bytes']==0 and attrs['max_threads']>=128
   records.append(dict(kernel=f.stem,cubin_sha256=sha(f),parameters=params,attributes=attrs,opcodes=dict(ops),resource=resource))
  finally:op.check(op.lib.cuModuleUnload(mod),'unload')
 assert len(records)==7
 return records

def main():
 reference=json.loads((HERE/'artifacts/reference_manifest.json').read_text());assert reference['pass']
 x=full_source();op=Matrix();result=dict(started=time.time(),samples=[],pass_=False)
 for cell in CELLS:
  op.set_cell(cell)
  for method in METHODS:
   a,r=op.run(method,x);h=check_output(a,cell,exact=True);del a
   op.capture()
   a,diag=op.diagnostic(method,x);assert diag['stage_counts']==r['stage_counts'];check_output(a,cell,exact=True);del a
   rec=dict(cell=cell,method=method,output_sha256=h,stage_counts=r['stage_counts'],diagnostic=diag)
   result['samples'].append(rec);print(json.dumps(rec),flush=True)
 result.update(compiled=op.capture(),audit=audit(op),ended=time.time(),pass_=True);result['pass']=True
 write_json(HERE/'artifacts/admission.json',result)
 freeze={}
 paths=list((HERE/'src').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'targets_r2.json',HERE/'artifacts/thresholds.json',HERE/'artifacts/reference_manifest.json',HERE/'artifacts/admission.json']
 paths+=list((HERE/'artifacts/compiled').glob('*'))+list((OLD/'src').glob('*.py'))+list(OLD.glob('*manifest.json'))
 paths+=[refpath(c) for c in CELLS]
 for p in paths:freeze[str(p.relative_to(ROOT))]=sha(p)
 freeze.update(op.identities)
 write_json(HERE/'artifacts/timing_freeze.json',freeze)
 print(json.dumps({'admission_pass':True,'frozen_files':len(freeze)}),flush=True)
if __name__=='__main__':main()
