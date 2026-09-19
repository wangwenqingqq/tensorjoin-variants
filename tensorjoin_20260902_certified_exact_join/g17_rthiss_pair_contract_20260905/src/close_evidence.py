"""Verify frozen G17 evidence and retain a bounded campaign closure receipt."""
import datetime,hashlib,json,pathlib,shutil,subprocess
HERE=pathlib.Path(__file__).resolve().parents[1];PROJECT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
out=HERE/'results/closure_checks.json';assert not out.exists()
frozen=[]
for name in ['frozen_execution.json','frozen_execution_r1.json','frozen_diagnostic_a0.json']:
 entries=load(HERE/'artifacts'/name)
 for name_,expected in entries.items():assert sha(PROJECT/name_)==expected,name_
 frozen.append(dict(manifest=name,files=len(entries),all_match=True))
upstream=load(HERE/'artifacts/upstream_sources.json')
for name,v in upstream['files'].items():assert sha(PROJECT/'adapters/rthiss_g7_a0'/name)==v['g7_sha256']
cases=[];guards=[];safety=[]
for p in sorted((HERE/'results').glob('case_*.json')):
 r=load(p);a=r['audit'];assert r['returncode']==0 and r['correctness']['adapter_structure_pass'] and r['public_seconds'] is None
 assert r['binary_sha256']==sha(HERE/'build_a0/RT-HiSS')
 assert r['libowl_sha256']==sha(HERE/'build_a0/OWL/owl/libowl.so')
 for rel,receipt in r['raw_files'].items():
  raw=PROJECT/rel
  assert sha(raw)==receipt['sha256'] and raw.stat().st_size==receipt['bytes'],rel
 cases.append(dict(record_id=r['record_id'],n=r['case']['n'],structure_pass=True,reference_pass=a['frozen_reference_pass'],pairs=a['output_pairs'],missing=a['missing_pairs'],extra=a['extra_pairs'],candidate_pairs=a['unique_rt_candidate_pairs']))
 if r['sanitizer'] in ['memcheck','synccheck']:
  log=HERE/'raw'/r['record_id']/'program.log';assert 'ERROR SUMMARY: 0 errors' in log.read_text()
  safety.append(dict(record_id=r['record_id'],zero_errors=True,log_sha256=sha(log)))
assert len(cases)==16 and len(safety)==6
labels=['g17_'+r['record_id'] for r in cases]+['g17_pair_replay_a0']
archive=HERE/'artifacts/guards';archive.mkdir(exist_ok=False)
for label in labels:
 p=PROJECT/'results'/('g5_guard_'+label+'.json');g=load(p)
 assert not g['foreign_rows'] and not g['postflight_compute_rows']
 assert sha(PROJECT/g['expected_result'])==g['expected_result_sha256']
 assert sha(PROJECT/'src/run_g5_guarded_process.py')==g['runner_sha256']
 shutil.copyfile(p,archive/p.name)
 for key in ['raw_log','occupancy_log','preflight_log']:
  src=PROJECT/g[key];assert sha(src)==g[key+'_sha256'];shutil.copyfile(src,archive/src.name)
 guards.append(dict(label=label,returncode=g['returncode'],admitted_for_own_contract=g['admitted'],occupancy_clean=True))
assert len(guards)==17
replay=load(HERE/'results/pair_replay_a0.json');assert replay['correctness']['exact_contract_pass']
code=load(HERE/'results/code_audit_a0.json');assert code['cuda_selected_code_identity_pass']
for variant,folder in [('native','build_upstream_d512_a0'),('adapter','build_a0')]:
 assert sha(HERE/folder/'RT-HiSS')==code['containers'][variant]['binary_sha256']
 assert sha(HERE/folder/'OWL/owl/libowl.so')==code['containers'][variant]['libowl_sha256']
assert sha(HERE/'artifacts/nsys_adapter_a0.sqlite')==code['nsys_sqlite_sha256']
query=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv']
processes=subprocess.check_output(query,text=True)
assert 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245' not in processes
state=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=index,uuid,name,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.draw','--format=csv'],text=True)
post=HERE/'raw/postflight.txt';post.write_text(subprocess.check_output(['date','-u'],text=True)+processes+'\n'+state)
result=dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_files=frozen,g7_source_hashes_match=True,
 adapter_cases=cases,guard_count=len(guards),guards=guards,sanitizers=safety,same_launch_diagnostic_pass=True,selected_cuda_code_identity_pass=True,
 native_fp64_reference_admitted=False,performance_admitted=False,novelty_pass=False,postflight_gpu2_idle=True,
 postflight_sha256=sha(post),scope='G17 bounded compatibility/safety/code audit; native numerical mismatches retained; diagnostic guard does not certify native join')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
