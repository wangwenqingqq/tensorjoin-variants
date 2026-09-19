"""Verify composite G19 gates and preserve every accepted/rejected guard."""
import datetime
import shutil
import subprocess
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3
check_frozen();check_r1();check_r2();check_r3()
a=json.loads((H/'results/admission_r3.json').read_text());assert a['complete'] and a['passed'] and len(a['records'])==14
code=json.loads((H/'results/code_audit.json').read_text());assert code['passed']
screen=json.loads((H/'results/screen.json').read_text());assert screen['complete']
failed=[]
for name in ['admission.json','admission_r1.json','admission_r2.json']:
 r=json.loads((H/'results'/name).read_text());assert not r['passed']
 failed.append(dict(path=name,failed_record=r['records'][-1]['record_id'],sha256=sha(H/'results'/name)))
for f in ['frozen_runtime_probe.json']:
 for p,h in json.loads((H/'artifacts'/f).read_text()).items():assert sha(P/p)==h,p
fs=json.loads((H/'artifacts/frozen_screen.json').read_text())
for p,h in fs['source_hashes'].items():assert sha(P/p)==h,p
for p,h in fs['libraries'].items():assert sha(p)==h,p
archive=H/'artifacts/guards';archive.mkdir(exist_ok=False);guards=[]
for p in sorted((P/'results').glob('g5_guard_g19_*.json')):
 g=json.loads(p.read_text());shutil.copy2(p,archive/p.name)
 for key in ['raw_log','occupancy_log','preflight_log']:
  if key not in g:continue
  q=P/g[key];assert sha(q)==g[key+'_sha256'];shutil.copy2(q,archive/q.name)
 if 'expected_result_sha256' in g:assert sha(P/g['expected_result'])==g['expected_result_sha256']
 if g['admitted']:assert not g['foreign_rows'] and not g['postflight_compute_rows'] and g['returncode']==0
 guards.append(dict(label=g['label'],admitted=g['admitted'],gpu=g['physical_gpu'],result_sha256=g.get('expected_result_sha256')))
observations=[];engines=[]
for slot in a['records']:
 assert slot['passed']
 outer=json.loads((P/slot['result_path']).read_text());assert outer['correctness']['exact_contract_pass']
 inner=json.loads((P/outer['inner_path']).read_text());assert inner['correctness']['exact_contract_pass']
 assert sha(P/outer['inner_path'])==outer['inner_sha256']
 for row in inner['rows']:
  assert row['audit']['exact'] and row['audit']['missing']==row['audit']['extra']==0
  observations.append(dict(method=slot['method'],kind=slot['kind'],record_id=slot['record_id'],case=row['case'],cycle=row['cycle'],iteration=row['iteration'],audit=row['audit']))
 engines += [dict(method=slot['method'],kind=slot['kind'],record_id=slot['record_id'],**r) for r in inner['engine_records']]
 if slot['tool'] in ['memcheck','synccheck']:
  log=(H/'raw'/f"{slot['record_id']}.log").read_text()
  assert 'ERROR SUMMARY: 0 errors' in log
  if slot['tool']=='memcheck':assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in log
old_counts={}
for name in ['g17_rthiss_pair_contract_20260905','g18_rthiss_conservative_repair_20260905']:
 m=json.loads((P/name/'artifacts/raw_evidence_manifest.json').read_text())
 for row in m['files']:assert sha(P/row['path'])==row['sha256'],row['path']
 old_counts[name]=len(m['files'])
g16=json.loads((P/'g16_gpu_preparation_20260905/artifacts/raw_evidence_manifest.json').read_text())
prior={p:r for p,r in g16['files'].items() if p.startswith(('src/','g15_strong_control_20260905/src/','g16_gpu_preparation_20260905/src/')) or p=='overleaf_bundle/main.tex'}
for p,r in prior.items():assert sha(P/p)==r['sha256'],p
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
assert 'GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2' not in processes
state=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=index,uuid,name,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.draw','--format=csv'],text=True)
post=H/'raw/postflight.txt';assert not post.exists();post.write_text(subprocess.check_output(['date','-u'],text=True)+processes+'\n'+state)
r=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),composite_admission_slots=14,
 observations=observations,engine_records=engines,retained_failed_campaigns=failed,guards=guards,
 unchanged_archives=old_counts,old_core_and_manuscript_unchanged_files=len(prior),
 screen_pairs=screen['pairs'],narrow_screen_pass=screen['narrow_screen_pass'],
 bounded_full_shutdown_memcheck_pass=True,bounded_reusable_comparison_admitted=True,
 general_rt_candidate_certificate=False,exact_real_claim=False,formal_or_sustained_promotion=False,
 novelty_pass=False,paper_edited=False,postflight_gpu3_idle=True,postflight_sha256=sha(post))
write(H/'results/closure_checks.json',r)
print(json.dumps({k:v for k,v in r.items() if k not in ['observations','engine_records','guards']},indent=2))
