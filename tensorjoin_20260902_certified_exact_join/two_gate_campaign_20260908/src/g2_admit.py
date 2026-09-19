"""Fail-closed evidence admission and immutable pre-timing inventory."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parents[1]
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def passed(r):return r.get('pass',r.get('pass_')) is True
labels=['g2_fixture_a0','g2_public_a0','g2_abi_a0','g2_full_a1','g2_stress_a1']+[f'g2_{c}_a1' for c in ['memcheck','synccheck','initcheck','racecheck']]+[f'g2_trace_{m}_a1' for m in ['F8','S8','F16','S16']]
evidence={}
for label in labels:
 for suffix in ['', '_guard']:
  p=HERE/'results'/f'{label}{suffix}.json';r=j(p);assert passed(r),p
  if suffix:assert r['exit_code']==0
  evidence[str(p.relative_to(HERE))]=sha(p)
for name in ['g2_offline_replay','g2_bitwise_check','g2_trace_audit']:
 p=HERE/'results'/f'{name}.json';assert passed(j(p));evidence[str(p.relative_to(HERE))]=sha(p)
for mode in ['memcheck','synccheck','initcheck','racecheck']:
 p=HERE/'raw'/f'g2_{mode}_a1.log';s=p.read_text()
 assert ('0 errors, 0 warnings' in s if mode=='racecheck' else 'ERROR SUMMARY: 0 errors' in s),p
 evidence[str(p.relative_to(HERE))]=sha(p)
full=j(HERE/'results/g2_full_a1.json');assert full['counts_match'] and len(full['results'])==4
assert all(x['count']==3926078 and x['hash']=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495' for x in full['results'])
stress=j(HERE/'results/g2_stress_a1.json');assert sum(len(x.get('stress_runs',[])) for x in stress['results'])==128
assert len(j(HERE/'results/g2_abi_a0.json')['records'])==10
assert len(j(HERE/'results/g2_trace_audit.json')['records'])==6
p=HERE/'artifacts/g2_admission_a0.json';assert not p.exists()
p.write_text(json.dumps(dict(pass_=True,pass_evidence_only=True,evidence_sha256=evidence,retained_failures=['g1_p00_a0 CPU-only preflight','g1 a1 prerequisite sync preflight','g2_full_a0 unused dump-pointer IR identity and exception serialization'],scope='finite numerical, safety and instruction qualification; not novelty or performance',qualified_source='g2_operator_r1.py; unchanged g2_kernels.py',pass_timing=False)|{'pass':True},indent=2))
files=list((HERE/'src').glob('*.py'))+list((HERE/'artifacts/g2_compiled_a0').glob('*'))+[HERE/'schedule_g2.json',HERE/'PROTOCOL_G2.md',HERE/'DESIGN_G2.md',HERE/'ADMISSION_G2_A1.md',HERE/'targets_r2.json',HERE/'frozen_manifest.json',p]
freeze=HERE/'artifacts/g2_timing_freeze_a0.json';assert not freeze.exists()
freeze.write_text(json.dumps({str(p.relative_to(HERE)):sha(p) for p in files if p.is_file()},indent=2))
assert not list((HERE/'results').glob('g2_p??_a0.json'))
print('ADMITTED for frozen 2x2 timing; no novelty promotion')
