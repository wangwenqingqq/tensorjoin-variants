"""Close finite G18 evidence without promoting timings or general exactness."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;G=P/'g17_rthiss_pair_contract_20260905'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
out=H/'results/closure_checks.json';assert not out.exists()
c=load(H/'results/campaign_gpu3_a2.json');assert c['complete'] and c['passed'] and len(c['records'])==20
frozen=load(H/'artifacts/frozen_execution.json')
for rel,v in frozen.items():assert sha(P/rel)==v,rel
orchestration=load(H/'artifacts/frozen_orchestration.json');assert sha(H/'src/campaign.py')==orchestration['sha256']
assert len(orchestration['slots'])==20
resume=load(H/'artifacts/frozen_resume_a1.json')
for rel,v in resume['hashes'].items():assert sha(P/rel)==v,rel
original=load(H/'results/campaign.json');assert not original['passed'] and not original['records'][2]['passed']
a1=load(H/'results/campaign_resume_a1.json');assert not a1['passed']
for rel,v in load(H/'artifacts/frozen_gpu3_a2.json')['hashes'].items():assert sha(P/rel)==v,rel
code=load(H/'results/code_audit.json');assert code['runtime_selected_code_pass']
assert sha(H/'build_r1/RT-HiSS')==code['binary_sha256']
assert sha(H/'build_r1/OWL/owl/libowl.so')==code['libowl_sha256']
assert sha(H/'artifacts/nsys_repair_gpu3_a2.sqlite')==code['nsys_sqlite_sha256']
rows=[];guards=[];archive=H/'artifacts/guards';archive.mkdir(exist_ok=False)
for row in c['records']:
 result=load(P/row['result_path']);guard=load(P/row['guard_path'])
 assert guard['admitted'] and not guard['foreign_rows'] and not guard['postflight_compute_rows'] and guard['returncode']==0
 assert sha(P/row['result_path'])==guard['expected_result_sha256']
 assert result['correctness']['exact_contract_pass'] and result['public_seconds'] is None
 for key in ['raw_log','occupancy_log','preflight_log']:
  p=P/guard[key];assert sha(p)==guard[key+'_sha256'];shutil.copyfile(p,archive/p.name)
 shutil.copyfile(P/row['guard_path'],archive/Path(row['guard_path']).name)
 guards.append(dict(label=guard['label'],returncode=0,occupancy_clean=True))
 if row['slot']['kind']=='case':
  a=result['audit'];assert a['frozen_reference_pass'] and not a['missing_pairs'] and not a['extra_pairs'] and not a['rt_candidate_missing_frozen_pairs']
  assert result['binary_sha256']==code['binary_sha256'] and result['libowl_sha256']==code['libowl_sha256']
  for rel,v in result['raw_files'].items():
   p=P/rel;assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
  if result['sanitizer']!='none':assert 'ERROR SUMMARY: 0 errors' in (H/'raw'/result['record_id']/'program.log').read_text()
  rows.append(dict(record_id=result['record_id'],case=result['case']['name'],pairs=a['output_pairs'],sha256=a['output_sha256'],work=result['work'],sanitizer=result['sanitizer']))
assert len(rows)==18
historical=[]
for row in original['records'][:2]:
 guard=load(P/row['guard_path']);assert guard['admitted'] and not guard['foreign_rows'] and not guard['postflight_compute_rows']
 assert sha(P/row['result_path'])==guard['expected_result_sha256']
 shutil.copyfile(P/row['guard_path'],archive/Path(row['guard_path']).name)
 for key in ['raw_log','occupancy_log','preflight_log']:
  p=P/guard[key];assert sha(p)==guard[key+'_sha256'];shutil.copyfile(p,archive/p.name)
 historical.append(dict(label=guard['label'],gpu=2,admitted=True))
real=[r for r in rows if r['case']=='cifar4096'];assert len(real)==5 and len({r['sha256'] for r in real})==1
assert len({json.dumps(r['work'],sort_keys=True) for r in real})==1
probe=load(H/'results/predicate_ab_g3_a2.json');stress=load(H/'results/predicate_stress_g3_a2.json')
assert probe['full_result_sha256']==stress['full_result_sha256'] and stress['iterations']==1000
# Previously closed native G17 evidence must remain unchanged.
g17=load(G/'artifacts/raw_evidence_manifest.json')
for v in g17['files']:assert sha(P/v['path'])==v['sha256'],v['path']
# Verify original core and historical manuscript against the previous G16 archive.
g16=load(P/'g16_gpu_preparation_20260905/artifacts/raw_evidence_manifest.json')
prior={k:v for k,v in g16['files'].items() if k.startswith(('src/','g15_strong_control_20260905/src/','g16_gpu_preparation_20260905/src/')) or k=='overleaf_bundle/main.tex'}
for k,v in prior.items():assert sha(P/k)==v['sha256'],k
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
assert 'GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2' not in processes
state=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=index,uuid,name,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.draw','--format=csv'],text=True)
post=H/'raw/postflight.txt';post.write_text(subprocess.check_output(['date','-u'],text=True)+processes+'\n'+state)
r=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_file_count=len(frozen),guard_count=len(guards),guards=guards,retained_prelaunch_occupancy_blocks=2,historical_gpu2_passes=historical,
 full_operator_observations=rows,strict_probe_pairs=8192,stress_invocations=1000,stress_scope='alternating two-buffer/stride predicate, not full-operator sustained work',
 real_terminal_fraction=real[0]['work']['terminal_fraction'],real_cheap_repair_screen_pass=real[0]['work']['terminal_fraction']<.01,
 native_g17_files_unchanged=len(g17['files']),prior_core_and_manuscript_files_unchanged=len(prior),bounded_repaired_reference_admitted=True,
 general_rt_candidate_certificate=False,exact_real_claim=False,performance_admitted=False,novelty_pass=False,postflight_gpu3_idle=True,postflight_sha256=sha(post))
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
