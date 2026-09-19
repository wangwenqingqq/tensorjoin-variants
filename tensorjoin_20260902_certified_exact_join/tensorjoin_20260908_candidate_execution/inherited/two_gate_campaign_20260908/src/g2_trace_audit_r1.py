"""Runtime-selected SASS identity and source-correlated instruction ledger."""
import collections,csv,hashlib,json,re,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
METHODS={'F8':['original'],'S8':['8_0_0','8_1_0'],'F16':['16_2_0'],'S16':['16_0_0','16_1_0']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(rows,base):
 out=[]
 for address,ins in rows:
  ins=' '.join(ins.strip().removesuffix(';').split())
  ins=ins.replace('.reuse','')
  ins=re.sub(r'`\(\.L_x_\d+\)\s*','',ins)
  op=re.sub(r'^@!?P\d+\s+','',ins).split()[0]
  if op.split('.')[0] in ['BRA','CALL','JMP','JMX','BSSY']:ins=re.sub(r'0x[0-9a-f]+',lambda m:hex(int(m[0],16)-base),ins)
  if op.startswith('LDGSTS'):
   prefix=ins[:ins.index(op)+len(op)];args=[x.strip() for x in ins[len(prefix):].strip().split(',')]
   if len(args)==3 and args[0].startswith('desc['):ins=prefix+' '+', '.join([args[2],args[0],args[1]])
  out.append(f'{address-base:06x} {ins}')
 return ('\n'.join(out)+'\n').encode()
records=[]
for method,keys in METHODS.items():
 label=f'g2_trace_{method}_a1';report=HERE/'artifacts'/f'{label}.ncu-rep';assert report.exists()
 source=HERE/'raw'/f'{label}_source.csv';raw=HERE/'raw'/f'{label}_metrics.csv'
 for page,dest in [('source',source),('raw',raw)]:
  cmd=['ncu','--import',str(report),'--page',page,'--csv']+(['--print-source','sass'] if page=='source' else [])
  data=subprocess.check_output(cmd)
  if dest.exists():assert dest.read_bytes()==data
  else:dest.write_bytes(data)
 groups=[]
 for row in csv.reader(source.open()):
  if row and row[0]=='Kernel Name':groups.append(dict(name=row[1],rows=[]))
  elif row and row[0]=='Address':groups[-1]['header']=row
  elif row and row[0].startswith('0x'):groups[-1]['rows'].append(row)
 assert len(groups)==len(keys),(method,len(groups))
 result=json.loads((HERE/'results'/f'{label}.json').read_text());guard=json.loads((HERE/'results'/f'{label}_guard.json').read_text())
 assert result.get('pass',result.get('pass_')) is True and guard['pass'] and guard['exit_code']==0
 for action,(group,key) in enumerate(zip(groups,keys)):
  if key=='original':
   spec=json.loads((HERE/'frozen_manifest.json').read_text())['stage1'];cubin=ROOT/(spec['stem']+'.cubin')
  else:cubin=HERE/'artifacts/g2_compiled_a0'/f'{key}.cubin'
  static_sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','-sass',str(cubin)],text=True)
  staticrows=[]
  for line in static_sass.splitlines():
   m=re.search(r'/\*([0-9a-f]+)\*/\s+(.+?)\s*;',line)
   if m:staticrows.append((int(m[1],16),m[2]))
  rows=group['rows'];base=int(rows[0][0],16)
  actual=normalized([(int(x[0],16),x[1]) for x in rows],base);expected=normalized(staticrows,0)
  (HERE/'artifacts'/f'{label}_{action}_runtime_r1.normalized.sass').write_bytes(actual)
  (HERE/'artifacts'/f'{label}_{action}_static_r1.normalized.sass').write_bytes(expected)
  assert actual==expected,('selected SASS identity',label,action)
  static=collections.Counter();dyn=collections.Counter();threads=collections.Counter();pred=collections.Counter();stores=0
  head=group['header']
  def number(row,col):
   value=row[head.index(col)].replace(',','');return float(value or 0)
  for x in rows:
   op=re.sub(r'^@!?P\d+\s+','',x[1].strip()).split()[0];static[op]+=1
   dyn[op]+=int(number(x,'Instructions Executed'));threads[op]+=int(number(x,'Thread Instructions Executed'));pred[op]+=int(number(x,'Predicated-On Thread Instructions Executed'))
   if op.startswith('STG'):
    value=x[head.index('Access Size')].replace(',','')
    # Access Size is a profiler field; keep raw data if it is unavailable.
    if value not in ['','-']:stores+=number(x,'Predicated-On Thread Instructions Executed')*float(value)
  assert not any(k.split('.')[0] in ['LDL','STL'] for k in static)
  mma={k:v for k,v in dyn.items() if 'MMA' in k}
  records.append(dict(label=label,method=method,action=action,kernel=key,name=group['name'],cubin_sha256=sha(cubin),report_sha256=sha(report),selected_normalized_sha256=hashlib.sha256(actual).hexdigest(),source_sha256=sha(source),static_opcodes=dict(static),dynamic_warp_instructions=dict(dyn),thread_instructions=dict(threads),predicated_on_thread_instructions=dict(pred),mma=mma,stg_logical_bytes_from_source_fields=stores,pass_=True))
assert len(records)==6
for method in ['S8','S16']:
 r=next(x for x in records if x['method']==method and x['action']==0)
 assert r['stg_logical_bytes_from_source_fields']==4096*4096*4,('score-store bytes',method,r['stg_logical_bytes_from_source_fields'])
result=dict(pass_=True,scope='same-round fixed first 4096-tile batch; instruction/source evidence not performance, DRAM bytes or novelty',normalization='relative PCs/targets; NCU comparable syntax: omit unavailable reuse hints, canonicalize LDGSTS operand display order and decorative local BRA labels; preserve operand values and predicates; full cubin/static evidence retained',records=records)
with (HERE/'results/g2_trace_audit.json').open('x') as f:json.dump(result,f,indent=2)
for r in records:print(r['method'],r['action'],r['mma'],r['stg_logical_bytes_from_source_fields'])
