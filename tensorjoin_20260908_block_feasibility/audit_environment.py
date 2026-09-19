"""Read-only environment capture and independent extended-precision spot audit."""
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join')
sys.path.insert(0,str(HERE/'src'))
from geometry import projection, PROJ_PAD

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
for filename,cmd in [('gpu0_environment.xml',['nvidia-smi','-i','0','-q','-x']),
                     ('cpu_environment.json',['lscpu','-J'])]:
    with (HERE/'raw'/filename).open('x') as f:
        subprocess.run(cmd,stdout=f,check=True)
x = np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy')
p,scale2 = projection(x)
assert np.max(np.abs(p)) <= 1
wide = p.astype(np.longdouble)
gram = wide.T @ wide
residual = np.sqrt(np.sum((gram-np.eye(p.shape[1],dtype=np.longdouble))**2))
assert np.longdouble(scale2) >= 1+residual
rng = np.random.default_rng(2026090812)
rows = rng.choice(len(x),256,replace=False)
normal = x[rows].astype(np.float64) @ p
extended = x[rows].astype(np.longdouble) @ wide
error = np.max(np.abs(normal.astype(np.longdouble)-extended))
assert error < PROJ_PAD
selection = json.loads((HERE/'artifacts/selection.json').read_text())['selected']
prefix = HERE/'artifacts'/('cifar60000_'+selection)
perm = np.load(str(prefix)+'_permutation.npy')
b = dict(np.load(str(prefix)+'_bounds.npz'))
tile = rng.integers(0,len(b['tr']),8192)
i0 = b['tr'][tile].astype(np.int64)*64
j0 = b['tc'][tile].astype(np.int64)*64
i = i0+(rng.random(len(tile))*np.minimum(64,len(x)-i0)).astype(np.int64)
j = j0+(rng.random(len(tile))*np.minimum(64,len(x)-j0)).astype(np.int64)
diff = x[perm[i]].astype(np.longdouble)-x[perm[j]].astype(np.longdouble)
truth = np.sum(diff*diff,axis=1)
assert np.all(b['combined'][tile].astype(np.longdouble) <= truth)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer): np.show_config()
import torch,triton
record = dict(host=platform.node(),platform=platform.platform(),python=sys.version,
    numpy=np.__version__,torch=torch.__version__,triton=triton.__version__,
    cuda_toolkit=torch.version.cuda,blas_threads={k:os.environ.get(k) for k in
    ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']},numpy_config=buffer.getvalue(),
    longdouble_bits=int(np.finfo(np.longdouble).nmant),projection_max_abs=float(np.max(np.abs(p))),
    projection_scale2=scale2,wide_gram_residual=float(residual),
    projection_spot_rows=len(rows),max_projection_abs_error=float(error),projection_pad=PROJ_PAD,
    lower_bound_spot_pairs=len(tile),minimum_lower_bound_slack=float(np.min(truth-b['combined'][tile])),
    pass_=True,scope='Independent extended-precision spot checks supplement the source argument and full output validation.')
with (HERE/'artifacts/environment.json').open('x') as f:
    json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='numpy_config'},indent=2))
