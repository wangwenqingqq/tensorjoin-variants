"""Run only the predeclared bounded admission slots under one GPU lock."""
import fcntl,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
out=H/'results/campaign.json';assert not out.exists()
names=[v['name'] for v in json.loads((P/'g17_rthiss_pair_contract_20260905/data/manifest_r1.json').read_text())['inputs']]
slots=[dict(kind='probe',rid='predicate_ab_a0',iterations=2)]
slots += [dict(kind='case',name=n,rid='correct_'+n+'_a0',tool='none') for n in names]
slots += [dict(kind='case',name=n,rid=t+'_'+n+'_a0',tool=t) for t in ['memcheck','synccheck'] for n in ['boundary_zero32','cifar4096']]
slots += [dict(kind='case',name=n,rid=f'repeat{j}_'+n+'_a0',tool='none') for j,order in enumerate([['cifar4096','boundary_zero32'],['boundary_zero32','cifar4096']]) for n in order]
slots += [dict(kind='probe',rid='predicate_stress_a0',iterations=1000),dict(kind='case',name='boundary_zero32',rid='nsys_boundary_zero32_a0',tool='nsys')]
(H/'artifacts/frozen_orchestration.json').write_text(json.dumps(dict(slots=slots,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
records=[]
with open('/tmp/tensorjoin_gpu2_campaign.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for slot in slots:
  rid=slot['rid'];label='g18_'+rid
  if slot['kind']=='probe':
   path=H/'results'/f'{rid}.json';child=[PY,str(H/'src/run_probe.py'),'--record-id',rid,'--iterations',str(slot['iterations'])]
  else:
   path=H/'results'/f'case_{rid}.json';child=[PY,str(H/'src/run_case.py'),'--case',slot['name'],'--record-id',rid,'--sanitizer',slot['tool'] if slot['tool']!='nsys' else 'none']
   if slot['tool']=='nsys':child=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=false','-o',str(H/'artifacts/nsys_repair_a0')]+child
  command=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(path.relative_to(P)),'--physical-gpu','2','--']+child
  print('SLOT_START '+json.dumps(command),flush=True);run=subprocess.run(command)
  guardpath=P/'results'/f'g5_guard_{label}.json';guard=json.loads(guardpath.read_text()) if guardpath.exists() else {}
  passed=run.returncode==0 and guard.get('admitted') and not guard['foreign_rows'] and not guard['postflight_compute_rows']
  row=dict(slot=slot,command=command,returncode=run.returncode,passed=bool(passed),result_path=str(path.relative_to(P)),guard_path=str(guardpath.relative_to(P)))
  records.append(row)
  out.write_text(json.dumps(dict(complete=len(records)==len(slots),passed=all(r['passed'] for r in records) and len(records)==len(slots),records=records,performance_admitted=False),indent=2)+'\n')
  print('SLOT_COMPLETE '+json.dumps(row),flush=True)
  if not passed:break
 if len(records)==len(slots) and records[-1]['passed']:
  with (H/'raw/nsys_export.log').open('x') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(H/'artifacts/nsys_repair_a0.sqlite'),str(H/'artifacts/nsys_repair_a0.nsys-rep')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('CAMPAIGN_END',len(records),len(slots),all(r['passed'] for r in records),flush=True)
