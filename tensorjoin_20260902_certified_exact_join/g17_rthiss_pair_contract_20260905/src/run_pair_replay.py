"""Same-process same-kernel A/B localizes the four real disputed decisions."""
import json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
r=json.loads((HERE/'results/case_native_cifar4096_a0.json').read_text());a=r['audit'];cases=a['scalar_fmaf_replays'];assert len(cases)==4
raw=HERE/'raw/pair_replay_a0';raw.mkdir(exist_ok=False)
x=np.fromfile(PROJECT/r['case']['path'],dtype='<f4').reshape(4096,512)
dims=np.fromfile(HERE/'raw/native_cifar4096_a0/dimension_map.u32',dtype='<u4');inverse=np.argsort(dims).astype('<u4')
pairs=np.stack([np.stack([x[v['row'],dims],x[v['column'],dims]]) for v in cases]).astype('<f4')
pairs.tofile(raw/'pairs.f32');inverse.tofile(raw/'inverse.u32')
cmd=[str(HERE/'artifacts/pair_replay'),str(raw/'pairs.f32'),str(raw/'inverse.u32'),str(a['native_threshold']),str(a['reference_threshold'])]
run=subprocess.run(cmd,text=True,capture_output=True);(raw/'stdout.log').write_text(run.stdout);(raw/'stderr.log').write_text(run.stderr)
assert run.returncode==0
observed=json.loads(run.stdout)
for want,got in zip(cases,observed):
 assert got['native']==want['observed_gpu_decision'] and got['native_sum']==want['native_fp32_sum']
 assert got['reference']==(want['kind']=='missing')
result=dict(scope='same GPU launch native predicate versus FP64 direct-difference replay on four disputed pairs; not performance or arbitrary-input certification',command=cmd,cases=cases,observed=observed,correctness=dict(exact_contract_pass=True,meaning='reproduced diagnostic divergence, not native join correctness'),binary_sha256=hashlib.sha256((HERE/'artifacts/pair_replay').read_bytes()).hexdigest())
(HERE/'results/pair_replay_a0.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
