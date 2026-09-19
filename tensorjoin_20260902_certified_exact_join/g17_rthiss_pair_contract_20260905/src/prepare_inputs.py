"""Freeze the G17 fixture matrix; never select a favorable threshold or subset."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
source=np.load(PROJECT/'data/g2a_cifar4096/vectors_f32.npy')
meta=json.loads((PROJECT/'data/g2a_cifar4096/metadata.json').read_text());rng=np.random.default_rng(17052026)
inputs=[]
def add(name,x,eps,threshold=None,oracle=None):
    x=np.ascontiguousarray(x,dtype=np.float32);path=HERE/f'data/{name}.f32'
    assert not path.exists();x.tofile(path)
    inputs.append(dict(name=name,n=len(x),d=512,path=str(path.relative_to(PROJECT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epsilon=eps,reference_threshold=threshold if threshold is not None else eps*eps,oracle_path=oracle))
add('real31',source[:31].copy(),.62890625)
add('signed31',rng.uniform(-.1,.1,(31,512)).astype(np.float32),.62890625)
x=np.zeros((31,512),np.float32)
for i in range(31):x[i,(i//3)%10]=1
x=x[rng.permutation(31)];add('shuffled_onehot31',x,1.)
add('zeros31',np.zeros((31,512),np.float32),2**-12)
x=np.zeros((31,512),np.float32);x[0,0]=1;x[1,:2]=[1,2**-13];x[2,:2]=[1,2**-27];x[3,0]=np.nextafter(np.float32(1),np.float32(0));x[4,0]=-1
for i in range(5,31):x[i,i]=1
add('boundary31',x,1.)
x=(np.float32(.5)+rng.integers(-2,3,(31,512)).astype(np.float32)*np.float32(2**-12))
add('cancellation31',x,2**-5)
tiny=np.nextafter(np.float32(0),np.float32(1));x=rng.integers(-16384,16385,(31,512)).astype(np.float32)*tiny
add('subnormal31',x,2**-12)
add('cifar4096',source,float(meta['radius']['epsilon']),float(meta['radius']['effective_epsilon_d2']),'data/g2a_cifar4096/oracle_pairs_u64.npy')
(HERE/'data/manifest.json').write_text(json.dumps(dict(seed=17052026,source_npy_sha256=hashlib.sha256((PROJECT/'data/g2a_cifar4096/vectors_f32.npy').read_bytes()).hexdigest(),inputs=inputs),indent=2)+'\n')
