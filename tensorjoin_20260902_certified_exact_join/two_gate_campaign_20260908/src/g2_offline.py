"""Separate CPU rational replay of saved samples, without importing GPU code."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parents[1]
records=[]
for f in sorted((HERE/'results').glob('g2_*a0_*_numeric.npz')):
    if not f.name.startswith(('g2_fixture_','g2_public_')):continue
    a=np.load(f,allow_pickle=False);p=a['p'];rec=dict(file=str(f.relative_to(HERE)),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    if a['q'].dtype==np.int8:
        exact=(a['q'].astype(np.int64)*a['b'].astype(np.int64)).sum(axis=1);assert np.array_equal(p,exact)
        rec.update(integer_dot_checks=len(p))
    else:
        gamma=Q(1026,2**23-1026)
        for u,v,pred,low,high,md in zip(a['q'],a['b'],p,a['lo'],a['hi'],a['pair_metadata'].transpose(2,0,1)):
            u=[Q(float(x)) for x in u];v=[Q(float(x)) for x in v];pred=Q(float(pred))
            error=pred-sum(x*y for x,y in zip(u,v));norm=sum(x*x for x in u)*sum(x*x for x in v)
            assert error*error<=gamma*gamma*norm
            il,jl,ih,jh,iz,jz,ie,je=[Q(float(x)) for x in md.ravel()]
            b=Q(0.00012232370499987155)*iz*jz+ie*jz+iz*je+ie*je+Q(2048,2**126)
            r=2*b+(ih+jh)*Q(1,2**38)+Q(1.0000001044244144e-12)
            assert Q(float(low))<=max(Q(0),il+jl-2*pred-r) and Q(float(high))>=ih+jh-2*pred+r
        for x,z,md in zip(a['x'],a['z'],a['metadata'].T):
            x=[Q(float(t)) for t in x];z=[Q(float(t)) for t in z];l,h,zu,eu=[Q(float(t)) for t in md]
            assert l<=sum(t*t for t in x)<=h and zu*zu>=sum(t*t for t in z) and eu*eu>=sum((u-v)**2 for u,v in zip(x,z))
        rec.update(conditional_dot_checks=len(p),interval_checks=len(p),metadata_rows=len(a['rows']))
    rec['pass']=True;records.append(rec)
assert len(records)==12,len(records)
result=dict(pass_=True,scope='separate CPU replay of frozen finite samples, not a universal dot-model proof',records=records)
with (HERE/'results/g2_offline_replay.json').open('x') as f:json.dump(result,f,indent=2)
print('PASS',len(records),'sample bundles')
