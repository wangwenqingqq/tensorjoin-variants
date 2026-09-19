"""Deterministic falsification inputs, not tuned to observed G18 results."""
import hashlib,json
from pathlib import Path
import numpy as np
from bounds import cuts,g,h,B,F
H=Path(__file__).resolve().parents[1];P=H.parent;out=H/'data';out.mkdir(exist_ok=False)
rng=np.random.default_rng(18052026);n=8192;d=512
x=rng.uniform(-1,1,(n,2,d)).astype(np.float32)
x[1024:2048]*=np.float32(.03125)
x[2048:3072,0]=np.float32(.5);x[2048:3072,1]=x[2048:3072,0]+rng.uniform(-2**-12,2**-12,(1024,d)).astype(np.float32)
x[3072:4096]=np.ldexp(rng.uniform(-1,1,(1024,2,d)).astype(np.float32),rng.integers(-149,0,(1024,2,d))).astype(np.float32)
x[4096:5120]=(rng.integers(-200,201,(1024,2,d))*float(np.nextafter(np.float32(0),np.float32(1)))).astype(np.float32)
x[5120:6144]=0
x[6144:7168]=0;x[6144:7168,0,0]=1;x[6144:7168,0,1]=np.ldexp(np.float32(1),-rng.integers(8,40,1024)).astype(np.float32)
ref=np.zeros(n,np.float64)
for j in range(d):
 delta=x[:,0,j].astype(np.float64)-x[:,1,j].astype(np.float64);ref+=delta*delta
T=ref.copy();T[::4]=np.nextafter(T[::4],-np.inf);T[1::4]=np.nextafter(T[1::4],np.inf)
T[2::4]*=.5;T[3::4]*=1.5;T=np.clip(T,0,2048)
# Four retained real discrepancies occupy the first four rows, not replaced later.
r=json.loads((P/'g17_rthiss_pair_contract_20260905/results/case_native_cifar4096_a0.json').read_text());real=np.fromfile(P/r['case']['path'],dtype='<f4').reshape(4096,512)
for i,c in enumerate(r['audit']['scalar_fmaf_replays']):x[i]=real[[c['row'],c['column']]];T[i]=r['audit']['reference_threshold']
ref.fill(0)
for j in range(d):
 delta=x[:,0,j].astype(np.float64)-x[:,1,j].astype(np.float64);ref+=delta*delta
lo,hi=np.array([cuts(t) for t in T],dtype=np.float32).T
# Also exactly check cutoff rounding across a wider threshold exponent range.
checked=0
for t in [0.,1.,2048.,*np.ldexp(np.ones(400),np.arange(-350,50)).tolist()]:
 if t>2048:continue
 a,b=cuts(t);q=F(t)
 assert F(float(a))<=(1-g)*q/(1+h)-B and F(float(b))>=B+(1+g)*q/(1-h);checked+=1
perm=rng.permutation(512).astype('<u4');inverse=np.argsort(perm).astype('<u4')
for name,a in [('q.f32',x[:,0,perm].copy()),('b.f32',x[:,1,perm].copy()),('threshold.f64',T),('low.f32',lo),('high.f32',hi),('reference.f64',ref),('inverse.u32',inverse)]:np.ascontiguousarray(a).tofile(out/name)
manifest=dict(n=n,d=d,seed=18052026,cutoff_rounding_tests=checked,constants={k:str(v) for k,v in [('g',g),('h',h),('B',B)]},groups=['signed','small signed','cancellation','random exponents','subnormal','zeros','unit-boundary','signed'],files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
(out/'probe_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
