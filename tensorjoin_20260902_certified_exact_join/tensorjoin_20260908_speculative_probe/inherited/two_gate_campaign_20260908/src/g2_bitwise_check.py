"""Strict bit-pattern audit, including signed zeros, of saved paired dumps."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parents[1];records=[]
for prefix,whichs in [('g2_fixture_a0',[0,1]),('g2_public_a0',[0])]:
 for w in whichs:
  for precision in [8,16]:
   paths=[HERE/'results'/f'{prefix}_{w}_{m}{precision}_ids.npz' for m in ['F','S']]
   a,b=[np.load(p,allow_pickle=False) for p in paths];counts={}
   for key in ['dots','lo','hi']:
    aa,bb=a[key],b[key];assert aa.dtype==bb.dtype and aa.shape==bb.shape
    aa,bb=aa.view(np.uint32),bb.view(np.uint32);counts[key]=int(np.count_nonzero(aa!=bb));assert counts[key]==0,(paths,key)
   for key in ['yes','ambiguous','truth','tr','tc']:assert np.array_equal(a[key],b[key])
   records.append(dict(prefix=prefix,input=w,precision=precision,mismatched_bits=counts,cells=len(a['dots']),files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},pass_=True))
with (HERE/'results/g2_bitwise_check.json').open('x') as f:json.dump(dict(pass_=True,records=records),f,indent=2)
print('PASS',len(records),'paired bit-exact dump comparisons')
