"""Close A3 composite admission without relabeling any earlier failed attempt."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;G=P/'g17_rthiss_pair_contract_20260905'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
out=H/'results/closure_checks.json';assert not out.exists()
c=load(H/'results/campaign_gpu3_a3.json');assert c['complete'] and c['passed'] and len(c['records'])==20
for name in ['frozen_execution.json','frozen_map_diagnostic.json','frozen_native_tie_r1.json']:
 for rel,v in load(H/'artifacts'/name).items():assert sha(P/rel)==v,rel
for name in ['frozen_resume_a1.json','frozen_gpu3_a2.json','frozen_gpu3_a3.json']:
 for rel,v in load(H/'artifacts'/name)['hashes'].items():assert sha(P/rel)==v,rel
original=load(H/'results/campaign.json');a1=load(H/'results/campaign_resume_a1.json');a2=load(H/'results/campaign_gpu3_a2.json')
assert not original['passed'] and not a1['passed'] and not a2['passed']
assert len(original['records'])==3 and len(a1['records'])==3 and len(a2['records'])==5
assert all(v['passed'] for v in a2['records'][:4])
code=load(H/'results/code_audit.json');assert code['runtime_selected_code_pass']
assert sha(H/'build_r1/RT-HiSS')==code['binary_sha256'] and sha(H/'build_r1/OWL/owl/libowl.so')==code['libowl_sha256']
assert sha(H/'artifacts/nsys_repair_gpu3_a3.sqlite')==code['nsys_sqlite_sha256']
archive=H/'artifacts/guards';archive.mkdir(exist_ok=False)
def keep_guard(path,admitted):
 path=P/path;g=load(path);assert bool(g['admitted'])==admitted
 if admitted:assert not g['foreign_rows'] and not g['postflight_compute_rows'] and g['returncode']==0
 if 'expected_result_sha256' in g:assert sha(P/g['expected_result'])==g['expected_result_sha256']
 shutil.copyfile(path,archive/path.name)
 for key in ['raw_log','occupancy_log','preflight_log']:
  q=P/g[key];assert sha(q)==g[key+'_sha256'];shutil.copyfile(q,archive/q.name)
 return dict(label=g['label'],gpu=g['physical_gpu'],admitted=g['admitted'],foreign_rows=g['foreign_rows'])
rows=[];guards=[]
for entry in c['records']:
 r=load(P/entry['result_path']);assert r['correctness']['exact_contract_pass'] and r['public_seconds'] is None
 guards.append(keep_guard(entry['guard_path'],True))
 if entry['slot']['kind']=='case':
  a=r['audit'];assert a['frozen_reference_pass'] and a['missing_pairs']==a['extra_pairs']==a['rt_candidate_missing_frozen_pairs']==0
  assert r['binary_sha256']==code['binary_sha256'] and r['libowl_sha256']==code['libowl_sha256']
  for rel,v in r['raw_files'].items():assert sha(P/rel)==v['sha256'] and (P/rel).stat().st_size==v['bytes']
  if r['sanitizer']!='none':assert 'ERROR SUMMARY: 0 errors' in (H/'raw'/r['record_id']/'program.log').read_text()
  rows.append(dict(record_id=r['record_id'],case=r['case']['name'],pairs=a['output_pairs'],sha256=a['output_sha256'],work=r['work'],sanitizer=r['sanitizer'],map_diagnostic=r.get('map_diagnostic',{'old_stricter_snapshot_map_check_passed':True})))
assert len(rows)==18 and len({r['case'] for r in rows})==9
real=[r for r in rows if r['case']=='cifar4096'];assert len(real)==5 and len({r['sha256'] for r in real})==1 and len({json.dumps(r['work'],sort_keys=True) for r in real})==1
historical=[keep_guard(r['guard_path'],True) for r in original['records'][:2]]
controls=[keep_guard(f'results/g5_guard_g18_native_tie_g3_d{i}.json',True) for i in [1,2]]
failed=[keep_guard(a2['records'][-1]['guard_path'],False),keep_guard('results/g5_guard_g18_native_tie_g3_d0.json',False)]
probe=load(H/'results/predicate_ab_g3_a2.json');stress=load(H/'results/predicate_stress_g3_a3.json')
assert probe['full_result_sha256']==stress['full_result_sha256'] and stress['iterations']==1000
assert probe['full_result_sha256']==load(H/'results/predicate_ab_a0.json')['full_result_sha256']
g17=load(G/'artifacts/raw_evidence_manifest.json')
for v in g17['files']:assert sha(P/v['path'])==v['sha256'],v['path']
g16=load(P/'g16_gpu_preparation_20260905/artifacts/raw_evidence_manifest.json')
prior={k:v for k,v in g16['files'].items() if k.startswith(('src/','g15_strong_control_20260905/src/','g16_gpu_preparation_20260905/src/')) or k=='overleaf_bundle/main.tex'}
for k,v in prior.items():assert sha(P/k)==v['sha256'],k
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
assert 'GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2' not in processes
state=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=index,uuid,name,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.draw','--format=csv'],text=True)
post=H/'raw/postflight.txt';post.write_text(subprocess.check_output(['date','-u'],text=True)+processes+'\n'+state)
r=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_gpu=3,main_guard_count=len(guards),main_guards=guards,full_operator_observations=rows,
 historical_gpu2_passes=historical,native_map_controls=controls,retained_failed_guards=failed,retained_prelaunch_blocks=2,
 strict_probe_pairs=8192,exact_rational_prefix_checks=probe['exact_rational_prefix_checks'],stress_invocations=1000,
 stress_scope='two-buffer/stride predicate only, not full-operator sustained work',real_work=real[0]['work'],real_cheap_repair_screen_pass=real[0]['work']['terminal_fraction']<.01,
 native_g17_unchanged_files=len(g17['files']),prior_core_and_manuscript_unchanged_files=len(prior),bounded_repaired_reference_admitted=True,
 general_rt_candidate_certificate=False,exact_real_claim=False,performance_admitted=False,novelty_pass=False,postflight_gpu3_idle=True,postflight_sha256=sha(post))
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
