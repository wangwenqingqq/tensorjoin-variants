import hashlib,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 path=ROOT/'data/g2b_cifar60000/vectors_f32.npy'
 assert sha(path)=='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
 x=np.load(path,allow_pickle=False);assert x.shape==(60000,512) and x.dtype==np.float32
 rng=np.random.default_rng(2026090801);n=2**22
 i=rng.integers(0,60000,n,dtype=np.int32);j=rng.integers(0,59999,n,dtype=np.int32);j+=j>=i
 lo=np.minimum(i,j);hi=np.maximum(i,j);ds=np.empty(n,np.float64)
 for off in range(0,n,8192):
  d=x[lo[off:off+8192]].astype(np.float64)-x[hi[off:off+8192]].astype(np.float64)
  ds[off:off+8192]=np.einsum('ij,ij->i',d,d)
 sorted_ds=np.sort(ds);ms=np.arange(1,4096,dtype=np.int64);ts=ms*ms/2**24
 degrees=np.searchsorted(sorted_ds,ts,side='right')*59999/n
 cells=[]
 for target in [4,16,64,256,1024]:
  k=int(np.argmin(np.abs(degrees-target)));m=int(ms[k])
  cells.append(dict(name=f'k{target}',target_degree=target,m=m,eps=m/4096,T=m*m/2**24,sample_degree=float(degrees[k])))
 m=2576;cells.append(dict(name='original',target_degree=None,m=m,eps=m/4096,T=m*m/2**24,sample_degree=float(degrees[m-1])))
 assert len(set(c['m'] for c in cells))==6
 for c in cells:assert float(np.float32(c['eps']))==c['eps'] and float(np.float32(c['T']))==c['T']
 rec=dict(created=time.time(),protocol_sha256=sha(HERE/'PROTOCOL.md'),input_sha256=sha(path),sample_seed=2026090801,sample_size=n,sample_pairs_sha256=hashlib.sha256(np.stack([lo,hi],axis=1).tobytes()).hexdigest(),sample_distances_sha256=hashlib.sha256(ds.tobytes()).hexdigest(),cells=sorted(cells,key=lambda c:c['T']))
 (HERE/'artifacts/thresholds.json').open('x').write(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()
