"""Exhaustive frozen-terminal reference, with monotone threshold reuse."""
from common import *
def main():
 result={'started':time.time(),'pass':False,'cells':{}}
 x=full_source();op=Matrix();v=torch.from_numpy(x).to('cuda');del x
 largest=CELLS[-1];op.set_cell(largest)
 directory=HERE/'artifacts/reference_upper';directory.mkdir(exist_ok=True)
 out=torch.empty(128*N,device='cuda',dtype=torch.int64);col=np.arange(N,dtype=np.int64)
 covered=0;shards=[]
 for row0 in range(0,N,4096):
  path=directory/f'rows_{row0:05d}.npy';meta=path.with_suffix('.json');stop=min(N,row0+4096)
  expect=sum(N-r for r in range(row0,stop))
  if path.exists():
   r=json.loads(meta.read_text());assert r['pairs']==expect and r['file_sha256']==sha(path)
  else:
   parts=[];pairs=0
   for row in range(row0,stop,128):
    rows=np.arange(row,min(stop,row+128),dtype=np.int64)[:,None]
    ids=(rows*N+col[None,:])[col[None,:]>=rows].copy();pairs+=len(ids)
    pairs_gpu=torch.from_numpy(ids).to('cuda');c=torch.zeros(6,device='cuda',dtype=torch.int32)
    op.terminal(v,pairs_gpu,out,c,len(ids));ct=c.cpu().numpy();n=int(ct[0]);assert ct[2]==0 and n<=len(out)
    parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True))
   assert pairs==expect;upper=np.sort(np.concatenate(parts));assert np.all(upper[1:]>upper[:-1])
   np.save(path,upper,allow_pickle=False);r=dict(row_start=row0,row_stop=stop,pairs=pairs,accepted=len(upper),file_sha256=sha(path),threshold=largest,terminal_identities=op.identities);write_json(meta,r)
  covered+=expect;shards.append(r)
  print(json.dumps(dict(reference_rows=stop,pairs_covered=covered,elapsed=time.time()-result['started'])),flush=True)
 assert covered==N*(N+1)//2
 upper=np.concatenate([np.load(directory/f'rows_{r:05d}.npy') for r in range(0,N,4096)])
 assert np.all(upper[1:]>upper[:-1]);np.save(HERE/'artifacts/reference_upper_max.npy',upper,allow_pickle=False)
 for cell in CELLS[::-1]:
  op.set_cell(cell)
  if cell==largest:selected=upper
  else:
   parts=[]
   for off in range(0,len(upper),4_000_000):
    ids=upper[off:off+4_000_000].astype(np.int64);g=torch.from_numpy(ids).to('cuda');c=torch.zeros(6,device='cuda',dtype=torch.int32)
    op.terminal(v,g,out,c,len(ids));ct=c.cpu().numpy();assert ct[2]==0 and int(ct[0])<=len(out)
    parts.append(out[:int(ct[0])].cpu().numpy().astype(np.uint64,copy=True))
   selected=np.concatenate(parts)
  a=canonical([selected]);assert np.all(a[1:]>a[:-1]);assert np.count_nonzero(a//N==a%N)==N
  if cell['name']=='original':assert len(a)==3926078 and digest(a)=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495'
  np.save(refpath(cell),a,allow_pickle=False)
  rec=dict(count=len(a),nonself_mean_degree=(len(a)-N)/N,sha256=digest(a),file_sha256=sha(refpath(cell)),upper_count=len(selected),threshold=cell)
  result['cells'][cell['name']]=rec;print(json.dumps(rec),flush=True)
  del a,selected
 result.update(pass_=True,ended=time.time(),pairs_covered=covered,shards=shards,terminal_identities=op.identities);result['pass']=True
 write_json(HERE/'artifacts/reference_manifest.json',result)
if __name__=='__main__':main()
