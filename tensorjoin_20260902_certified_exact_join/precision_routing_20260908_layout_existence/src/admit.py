"""Fail-closed admission; retain original evidence and freeze timing sources."""
import json,re,time
from common import *

def main():
 evidence={};c=json.loads((HERE/'results/census_a0.json').read_text());assert c['pass'] and c['manipulation_pass']
 for label in ['census_a0','memcheck_a0','initcheck_a0','synccheck_a0','racecheck_a0']:
  result=HERE/'results'/f'{label}.json';guard=HERE/'results'/f'{label}_guard.json';log=HERE/'raw'/f'{label}.log'
  r=json.loads(result.read_text());g=json.loads(guard.read_text());assert r['pass'] and g['pass'] and g['exit_code']==0,(label,r.get('exception'),g)
  if label!='census_a0':
   text=log.read_text();assert ('ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text),label
   assert not re.search(r'ERROR SUMMARY: [1-9]|RACECHECK SUMMARY: [1-9]',text),label
  for f in [result,guard,log]:evidence[str(f.relative_to(HERE))]=sha(f)
 freeze=json.loads((HERE/'artifacts/source_freeze_a0.json').read_text())
 for rel,spec in freeze['files'].items():assert sha(HERE/rel)==spec['sha256'],rel
 (HERE/'artifacts/admission_a0.json').open('x').write(json.dumps({'time':time.time(),'pass':True,'evidence':evidence,'scope':'unchanged eager programs; full logical ID invariance plus changed-layout sanitizer composites'},indent=2))
 files=[HERE/'PROTOCOL.md',HERE/'targets_r2.json',*sorted((HERE/'src').glob('*.py'))]
 (HERE/'artifacts/timing_freeze_a0.json').open('x').write(json.dumps({str(f.relative_to(HERE)):sha(f) for f in files},indent=2))
 print('ADMISSION_PASS',flush=True)
if __name__=='__main__':main()
