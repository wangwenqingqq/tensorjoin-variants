import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'artifacts/admission.json').read_text())['pass'];results=[]
for phase,count in [('explore',2),('confirm',6)]:
 if phase=='confirm':subprocess.run([sys.executable,str(HERE/'src/select.py')],check=True)
 for index in range(count):
  label=f'{phase}_{index:02d}_a0';cmd=[sys.executable,str(HERE/'src/guard_r2.py'),'--label',label,'--gpu','2','--target','8p_gpu2','--',sys.executable,str(HERE/'src/runner.py'),'--phase',phase,'--index',str(index)]
  print(json.dumps(dict(starting=label,time=time.time())),flush=True)
  env=os.environ.copy();env['NVIDIA_TF32_OVERRIDE']='0';p=subprocess.run(cmd,env=env,check=False)
  results.append(dict(label=label,exit_code=p.returncode,ended=time.time()));(HERE/'results/campaign_progress.json').write_text(json.dumps(results,indent=2))
  if p.returncode:raise SystemExit(p.returncode)
print(json.dumps(dict(campaign_complete=True,processes=len(results))),flush=True)
