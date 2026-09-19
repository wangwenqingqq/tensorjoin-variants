"""Resume only unmeasured slots after a retained foreign-preflight block."""
import fcntl,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((H/'results/campaign.json').read_text());original=json.loads((H/'artifacts/frozen_orchestration.json').read_text())
assert len(old['records'])==3 and all(r['passed'] for r in old['records'][:2]) and not old['records'][2]['passed']
assert not (P/old['records'][2]['result_path']).exists() and not (P/old['records'][2]['guard_path']).exists()
log=(H/'raw/campaign.log').read_text();assert 'Physical GPU2 is occupied before G5' in log and '1788163' in log
assert sha(H/'src/campaign.py')==original['sha256']
slots=original['slots'][2:];assert len(slots)==18 and slots[0]['rid']=='correct_signed31_a0';slots[0]['rid']='correct_signed31_a1'
f=H/'artifacts/frozen_resume_a1.json';assert not f.exists();f.write_text(json.dumps(dict(slots=slots,hashes={str(p.relative_to(P)):sha(p) for p in [Path(__file__),H/'ADDENDUM_OCCUPANCY_A1.md',H/'results/campaign.json',H/'raw/campaign.log']}),indent=2)+'\n')
records=list(old['records'][:2]);out=H/'results/campaign_resume_a1.json';assert not out.exists()
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
  row=dict(slot=slot,command=command,returncode=run.returncode,passed=bool(passed),result_path=str(path.relative_to(P)),guard_path=str(guardpath.relative_to(P)));records.append(row)
  out.write_text(json.dumps(dict(complete=len(records)==20,passed=all(r['passed'] for r in records) and len(records)==20,records=records,retained_original_passes=2,prior_prelaunch_block='results/campaign.json',performance_admitted=False),indent=2)+'\n')
  print('SLOT_COMPLETE '+json.dumps(row),flush=True)
  if not passed:break
 if len(records)==20 and records[-1]['passed']:
  with (H/'raw/nsys_export.log').open('x') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(H/'artifacts/nsys_repair_a0.sqlite'),str(H/'artifacts/nsys_repair_a0.nsys-rep')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('CAMPAIGN_END',len(records),20,all(r['passed'] for r in records),flush=True)
