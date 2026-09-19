"""Resume only unmeasured slots after a retained foreign-preflight block."""
import fcntl,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((H/'results/campaign_gpu3_a2.json').read_text());original=json.loads((H/'artifacts/frozen_gpu3_a2.json').read_text())
assert len(old['records'])==5 and all(r['passed'] for r in old['records'][:4]) and not old['records'][4]['passed']
failed=json.loads((P/old['records'][4]['result_path']).read_text());assert failed['error']=="AssertionError('point_map.u32')" and failed['audit']['frozen_reference_pass']
controls=[json.loads((H/'results'/f'native_tie_g3_d{i}.json').read_text()) for i in [1,2]]
for i in [1,2]:assert json.loads((P/'results'/f'g5_guard_g18_native_tie_g3_d{i}.json').read_text())['admitted']
assert all(r['all_reordered_value_bytes_identical'] for r in controls) and any(not r['same_point_map_as_g17_gpu2'] for r in controls)
slots=original['slots'][4:];assert len(slots)==16
for slot in slots:slot['rid']=slot['rid'].replace('g3_a2','g3_a3')
files=[Path(__file__),H/'src/run_case_gpu3_a3.py',H/'ADDENDUM_MAP_EQUIVALENCE_A3.md',H/'results/campaign_gpu3_a2.json',H/'raw/campaign_gpu3_a2.log']
files += [H/'results'/f'native_tie_g3_d{i}.json' for i in [1,2]]
f=H/'artifacts/frozen_gpu3_a3.json';assert not f.exists();f.write_text(json.dumps(dict(slots=slots,hashes={str(p.relative_to(P)):sha(p) for p in files}),indent=2)+'\n')
records=list(old['records'][:4]);out=H/'results/campaign_gpu3_a3.json';assert not out.exists()
with open('/tmp/tensorjoin_gpu3_campaign.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for slot in slots:
  rid=slot['rid'];label='g18_'+rid
  if slot['kind']=='probe':
   path=H/'results'/f'{rid}.json';child=[PY,str(H/'src/run_probe_gpu3.py'),'--record-id',rid,'--iterations',str(slot['iterations'])]
  else:
   path=H/'results'/f'case_{rid}.json';child=[PY,str(H/'src/run_case_gpu3_a3.py'),'--case',slot['name'],'--record-id',rid,'--sanitizer',slot['tool'] if slot['tool']!='nsys' else 'none']
   if slot['tool']=='nsys':child=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=false','-o',str(H/'artifacts/nsys_repair_gpu3_a3')]+child
  command=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(path.relative_to(P)),'--physical-gpu','3','--']+child
  print('SLOT_START '+json.dumps(command),flush=True);run=subprocess.run(command)
  guardpath=P/'results'/f'g5_guard_{label}.json';guard=json.loads(guardpath.read_text()) if guardpath.exists() else {}
  passed=run.returncode==0 and guard.get('admitted') and not guard['foreign_rows'] and not guard['postflight_compute_rows']
  row=dict(slot=slot,command=command,returncode=run.returncode,passed=bool(passed),result_path=str(path.relative_to(P)),guard_path=str(guardpath.relative_to(P)));records.append(row)
  out.write_text(json.dumps(dict(complete=len(records)==20,passed=all(r['passed'] for r in records) and len(records)==20,records=records,retained_original_passes=4,prior_failed_proxy='results/campaign_gpu3_a2.json',performance_admitted=False),indent=2)+'\n')
  print('SLOT_COMPLETE '+json.dumps(row),flush=True)
  if not passed:break
 if len(records)==20 and records[-1]['passed']:
  with (H/'raw/nsys_export.log').open('x') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(H/'artifacts/nsys_repair_gpu3_a3.sqlite'),str(H/'artifacts/nsys_repair_gpu3_a3.nsys-rep')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('CAMPAIGN_END',len(records),20,all(r['passed'] for r in records),flush=True)
