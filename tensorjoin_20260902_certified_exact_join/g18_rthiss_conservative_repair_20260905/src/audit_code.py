"""Bind runtime-selected repaired refinement to the exact frozen executable."""
import collections,hashlib,json,re,sqlite3,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;G=P/'g17_rthiss_pair_contract_20260905'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(text):
 out={}
 for block in re.split(r'\s*Function : ',text)[1:]:
  name=block.splitlines()[0].strip();inst=[]
  for line in block.splitlines()[1:]:
   m=re.search(r'/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;',line)
   if m:inst.append(' '.join(m.group(1).split())+';')
  if inst:out[name]=inst
 return out
assert json.loads((H/'results/campaign_gpu3_a3.json').read_text())['passed']
frozen=json.loads((H/'artifacts/frozen_execution.json').read_text())
for name,v in frozen.items():assert sha(P/name)==v,name
binary=H/'build_r1/RT-HiSS';sas=H/'raw/repaired_full.sass';assert not sas.exists()
sas.write_text(subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(binary)],text=True))
ptx=H/'raw/repaired_full.ptx';ptx.write_text(subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-ptx',str(binary)],text=True))
func=functions(sas.read_text());dbpath=H/'artifacts/nsys_repair_gpu3_a3.sqlite';db=sqlite3.connect(dbpath)
launches=db.execute('''select s.value,k.registersPerThread,k.gridX,k.blockX,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread,count(*)
 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
 group by s.value,k.registersPerThread,k.gridX,k.blockX,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread''').fetchall()
prior=json.loads((G/'results/code_audit_a0.json').read_text());rows=[]
for target in ['identifyNeighborsGridPrimitiveSharedQueryShared','compressResultMask']:
 match=[(k,v) for k,v in func.items() if target in k];assert len(match)==1
 symbol,inst=match[0];runtime=[r for r in launches if target in r[0]];assert len(runtime)==1
 h=hashlib.sha256(('\n'.join(inst)+'\n').encode()).hexdigest();old=next(r for r in prior['selected'] if r['variant']=='adapter' and r['target']==target)
 ops=collections.Counter(re.sub(r'^@!?\w+\s+','',s).split()[0].rstrip(';') for s in inst)
 assert not any(re.match(r'^(?:LDL|STL|[A-Z]*MMA)',o) for o in ops),'unexpected local/MMA code'
 if target=='compressResultMask':assert h==old['normalized_sass_sha256'],'compression changed'
 else:
  assert h!=old['normalized_sass_sha256'] and ops['FFMA']>0 and ops['DADD']>0 and ops['DMUL']>0
  assert not any(o.startswith(('FADD','FFMA','DADD','DMUL')) and '.FTZ' in o for o in ops)
  assert runtime[0][1]<=64 and runtime[0][3]==1024 and runtime[0][5]==47104 and runtime[0][6]==0
 rows.append(dict(target=target,symbol=symbol,normalized_sass_sha256=h,instructions=len(inst),old_instructions=old['instruction_count'],opcodes=dict(ops),runtime=runtime))
result=dict(runtime_selected_code_pass=True,selected=rows,binary_sha256=sha(binary),libowl_sha256=sha(H/'build_r1/OWL/owl/libowl.so'),
 nsys_sqlite_sha256=sha(dbpath),full_sass_sha256=sha(sas),full_ptx_sha256=sha(ptx),normalization='G17 same normalization: strip address/encoding/metadata; retain predicates, operands, constants, reuse flags and branch offsets',
 scope='adapter runtime-observed CUDA symbols and frozen static executable; changed refinement, unchanged compression; no performance or OptiX driver-JIT identity claim')
(H/'results/code_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
