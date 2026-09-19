"""Sequential guarded processes; stop immediately on any failed run."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'artifacts/admission.json').read_text())['pass']
results=[]
for phase,count in [('explore',2),('confirm',6)]:
 for index in range(count):
  label=f'{phase}_{index:02d}_a0'
  cmd=[sys.executable,str(HERE/'src/guard_r2.py'),'--label',label,'--gpu','2','--target','8p_gpu2','--',sys.executable,str(HERE/'src/runner.py'),'--phase',phase,'--index',str(index)]
  print(json.dumps({'starting':label,'time':time.time()}),flush=True)
  p=subprocess.run(cmd,check=False)
  results.append(dict(label=label,exit_code=p.returncode,ended=time.time()))
  (HERE/'results/campaign_progress.json').write_text(json.dumps(results,indent=2))
  if p.returncode:raise SystemExit(p.returncode)
print(json.dumps({'campaign_complete':True,'processes':len(results)}),flush=True)
