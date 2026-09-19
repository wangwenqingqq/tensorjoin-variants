"""Independent saved-artifact rational checks; no GPU and no latency ranking."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parents[1]
checks={'dot_samples':0,'metadata_rows':0,'interval_samples':0,'panel_outputs':0};sources={}
def read(path):
    sources[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return np.load(path,allow_pickle=False)
def metadata(x,z,m):
    for i in range(len(x)):
        a=[F(float(t)) for t in x[i]];b=[F(float(t)) for t in z[i]]
        nx=sum(t*t for t in a);nz=sum(t*t for t in b);nr=sum((u-v)**2 for u,v in zip(a,b))
        lo,hi,lz,er=[F(float(t[i])) for t in m]
        assert lo<=nx<=hi and lz*lz>=nz and er*er>=nr
        checks['metadata_rows']+=1
for label in ['ncu_half_0_a1','ncu_half_1_a1','ncu_half_2_a1','ncu_half_3_a2','gpu0_stress_a0_input0','gpu0_stress_a0_input1']:
    f=HERE/'results'/(label+('_dot_samples.npz' if label.startswith('ncu') else '_dots.npz'));s=read(f)
    if 'metadata' in s:metadata(s['x'],s['z'],s['metadata'])
    g=F(1026,2**23-1026)
    for a,b,p in zip(s['q'],s['b'],s['p']):
        aa=[F(float(t)) for t in a];bb=[F(float(t)) for t in b]
        err=F(float(p))-sum(u*v for u,v in zip(aa,bb));n=sum(t*t for t in aa)*sum(t*t for t in bb)
        assert err*err<=g*g*n;checks['dot_samples']+=1
for f in sorted((HERE/'results').glob('gpu0_*a2*_metadata.npz')):
    s=read(f);metadata(s['x'],s['z'],[s[k] for k in ['lo','hi','lz','er']])
for f in sorted((HERE/'results').glob('gpu0_*a2*_interval.npz')):
    s=read(f)
    for k,p in enumerate(s['p']):
        m=[[F(float(s['metadata'][i,j,k])) for j in range(2)] for i in range(4)]
        (a,b),(c,d),(li,lj),(ei,ej)=m;p=F(float(p))
        B=F(0.00012232370499987155)*li*lj+ei*lj+li*ej+ei*ej+F(2048,2**126)
        R=2*B+(c+d)*F(1,2**38)+F(1.0000001044244144e-12)
        assert F(float(s['lo'][k]))<=max(F(0),a+b-2*p-R)
        assert F(float(s['hi'][k]))>=c+d-2*p+R
        checks['interval_samples']+=1
for f in sorted((HERE/'results').glob('gpu0_*a2*_ids.npz')):
    s=read(f);assert np.array_equal(s['reference'],s['actual'])
    assert not np.setdiff1d(s['direct'],s['reference']).size
    assert not np.setdiff1d(s['reference'],np.r_[s['direct'],s['ambiguous']]).size
    assert len(np.unique(s['actual']))==len(s['actual']);checks['panel_outputs']+=1
old=json.loads((HERE.parent/'fp16_dense_gate0_20260907/results/public_a0.json').read_text())['datasets'][0]['panels']
new=json.loads((HERE/'results/gpu0_public_a2.json').read_text())['datasets'][0]['panels']
counts=[]
for a,b in zip(old,new):
    assert a['reference_hash']==b['hash'] and a['pairs']==b['pairs']
    assert a['C']['uncertain']==b['ambiguous'] and a['C']['fp64_inputs']==b['fp64_inputs']
    counts.append({'offset':b['offset'],'pairs':b['pairs'],'output':b['count'],'A_ambiguity':a['A']['uncertain'],'C_ambiguity':b['ambiguous'],'A_fp64':a['A']['fp64_inputs'],'C_fp64':b['fp64_inputs']})
result={'pass':True,'checks':checks,'public_panel_count_comparison':counts,'sources':sources,'scope':'finite saved samples and selected-panel identities; not universal library theorem or performance'}
with (HERE/'results/offline_review.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'pass':True,'checks':checks,'counts':counts}))
