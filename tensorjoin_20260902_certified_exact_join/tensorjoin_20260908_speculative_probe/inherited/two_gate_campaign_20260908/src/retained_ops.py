"""Complete warmed host-to-host operators using retained GPU programs only."""
import ctypes as C,json,re,time
import numpy as np
import torch
from frozen_driver import Driver,HERE,ROOT,sha
from pedantic_binding import CuBLAS
N,D,CAP,PANEL=60000,512,16777216,4096
EPS,T,TAU=161/256,25921/65536,2.0**-126
BASE=json.loads((HERE/'baseline_manifest.json').read_text())

def full_source():
    p=ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    assert sha(p)=='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
    x=np.ascontiguousarray(np.load(p,allow_pickle=False));assert x.shape==(N,D) and x.dtype==np.float32 and np.isfinite(x).all() and np.max(np.abs(x))<=1
    return x

def canonical(parts):
    a=np.sort(np.concatenate(parts).astype(np.uint64,copy=False));r,c=a//N,a%N
    return np.sort(np.concatenate([a,c[r<c]*N+r[r<c]]))

def output_check(a):
    import hashlib
    h=hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
    assert len(a)==3926078 and h=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495',('canonical',len(a),h)
    return h

class Operators(Driver):
    def __init__(self):
        super().__init__()
        for key,spec in BASE.items():
            for ext,h in spec['hashes'].items():
                p=ROOT/(spec['stem']+ext);assert sha(p)==h,p;self.identities[str(p.relative_to(ROOT))]=h
            text=(ROOT/(spec['stem']+'.ptx')).read_text()
            kinds=re.findall(r'\.param \.(u64|u32|f32)\b',text.split('.visible .entry '+spec['name']+'(')[1].split(')')[0])
            want=['u64']*7 if key=='dense_metadata' else ['u64']*9+['u32','u32','f32','u64','u64']
            assert kinds==want
            meta=json.loads((ROOT/(spec['stem']+'.json')).read_text());assert meta['global_scratch_size']==meta['profile_scratch_size']==0
            mod,fun=C.c_void_p(),C.c_void_p()
            self.check(self.lib.cuModuleLoad(C.byref(mod),str(ROOT/(spec['stem']+'.cubin')).encode()),key)
            self.check(self.lib.cuModuleGetFunction(C.byref(fun),mod,spec['name'].encode()),key)
            self.modules.append(mod);self.functions[key]=fun;position=0;info=[]
            for i,kind in enumerate(kinds):
                size=8 if kind=='u64' else 4;position=(position+size-1)//size*size
                off,got=C.c_size_t(),C.c_size_t();self.check(self.lib.cuFuncGetParamInfo(fun,i,C.byref(off),C.byref(got)),'baseline param')
                assert (off.value,got.value)==(position,size);info.append([off.value,got.value]);position+=size
            self.abi[key]={'parameters':info,'block':spec['block'],'shared':spec['shared']}
        self.blas=CuBLAS();self.stream=torch.cuda.current_stream().cuda_stream
        self.library_record=self.blas.record()
        self.library_hashes={p:sha(__import__('pathlib').Path(p)) for p in json.loads((HERE/'library_manifest.json').read_text())}
        assert self.library_hashes==json.loads((HERE/'library_manifest.json').read_text()),'cuBLAS containers changed'
    def baseline(self,key,grid,pointers,extras=()):
        spec=BASE[key]
        vals=[C.c_uint64(t.data_ptr()) for t in pointers]
        if extras:vals += [C.c_uint32(extras[0]),C.c_uint32(extras[1]),C.c_float(extras[2])]
        vals += [C.c_uint64(0),C.c_uint64(0)]
        expected=7 if key=='dense_metadata' else 14;assert len(vals)==expected
        params=(C.c_void_p*len(vals))(*(C.cast(C.byref(x),C.c_void_p) for x in vals))
        assert torch.cuda.current_stream().cuda_stream==self.stream
        self.check(self.lib.cuLaunchKernel(self.functions[key],grid,1,1,spec['block'],1,1,spec['shared'],self.stream,params,None),key)
    def norm(self,v):
        outs=[torch.empty(N,device='cuda',dtype=torch.float64) for _ in range(3)];bad=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.baseline('dense_metadata',N,[v,*outs,bad]);assert bad.item()==0
        return outs
    def terminal(self,v,pairs,out,c,n):
        self.launch('terminal',n,[v,v,pairs,out,c],[T])
    def panel(self,v,norms,score,out,amb,c,row,col,rows,cols,prefix=None):
        assert torch.cuda.current_stream().cuda_stream==self.stream
        matrix=score[:rows*cols].view(rows,cols)
        self.blas.gemm(v[row:row+rows],v[col:col+cols],matrix)
        grid=(rows*cols+1023)//1024 if prefix is None else prefix
        self.baseline(f'dense_{rows}x{cols}',grid,[matrix,*norms,out,amb,c,matrix,matrix],[row,col,T])
    def run(self,method,x):
        assert method in ['A','B']
        torch.cuda.synchronize();start=time.perf_counter()
        if method=='A':
            tr,tc=np.triu_indices(938);tr=tr.astype(np.int32);tc=tc.astype(np.int32)
        v=torch.from_numpy(x).to('cuda')
        if method=='A':
            q=torch.empty((N,D),device='cuda',dtype=torch.int8);s=torch.empty(N,device='cuda');h=torch.empty_like(s);e=torch.empty_like(s);bad=torch.zeros(1,device='cuda',dtype=torch.int32)
            self.launch('metadata',N,[v,q,s,h,e,bad]);assert bad.item()==0
            qt=q.T.contiguous();trg=torch.from_numpy(tr).to('cuda');tcg=torch.from_numpy(tc).to('cuda')
        else:norms=self.norm(v);score=torch.empty(CAP,device='cuda',dtype=torch.float32)
        out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out)
        if method=='A':fp64=torch.empty_like(out)
        parts=[];records=[]
        if method=='A':
            for off in range(0,len(tr),4096):
                tiles=min(4096,len(tr)-off);c=torch.zeros(6,device='cuda',dtype=torch.int32)
                self.launch('stage1',tiles,[q,qt,s,h,e,trg[off:],tcg[off:],out,amb,c],[EPS,EPS,2**-16,16*TAU,2**-20,16*TAU])
                s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=CAP
                self.launch('stage2',n1,[v,amb,out,fp64,c],[T,T,2**-14,2**-22,2**-22,4096*TAU])
                s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=CAP
                self.terminal(v,fp64,out,c,n2);last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
                parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True));records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
        else:
            for row in range(0,N,PANEL):
                for col in range(row,N,PANEL):
                    rows,cols=min(PANEL,N-row),min(PANEL,N-col);c=torch.zeros(4,device='cuda',dtype=torch.int32)
                    self.panel(v,norms,score,out,amb,c,row,col,rows,cols)
                    first=c.cpu().numpy().copy();direct,n1,overflow,rejected=map(int,first)
                    expect=rows*(rows+1)//2 if row==col else rows*cols
                    assert direct+n1+rejected==expect and overflow==0 and max(direct,n1)<=CAP
                    self.terminal(v,amb,out,c,n1);last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
                    parts.append(out[:n].cpu().numpy().astype(np.uint64,copy=True));records.append([direct,n1,rejected,n])
        torch.cuda.synchronize();a=canonical(parts);seconds=time.perf_counter()-start
        return a,{'seconds':seconds,'stage_counts':np.sum(np.array(records,dtype=np.int64),axis=0).tolist(),'batches':len(records),'input_pointer':v.data_ptr()}
    def prefix_probe(self):
        rng=np.random.default_rng(927);x=np.zeros((N,D),np.float32)
        x[4:1024]=rng.uniform(-.08,.08,(1020,D)).astype(np.float32)
        x[128]=0;x[128,0]=EPS;x[129]=x[128];x[129,1:4]=2**-28
        x[57344:58368]=x[:1024];x[57344]=0
        v=torch.from_numpy(x).to('cuda');norms=self.norm(v);score=torch.empty(CAP,device='cuda')
        out=torch.full((CAP+64,),-17,device='cuda',dtype=torch.int64);amb=torch.full_like(out,-17);res,unc=out[32:-32],amb[32:-32]
        reports=[]
        for row,col,rows,cols in [(0,0,4096,4096),(0,57344,4096,2656),(57344,57344,2656,2656)]:
            c=torch.zeros(4,device='cuda',dtype=torch.int32);self.panel(v,norms,score,res,unc,c,row,col,rows,cols,prefix=16)
            first=c.cpu().numpy().copy();direct,n,over,rej=map(int,first);assert over==0
            accepted=np.sort(res[:direct].cpu().numpy().copy());unclear=np.sort(unc[:n].cpu().numpy().copy())
            offsets=np.arange(16*1024,dtype=np.int64);rr=row+offsets//cols;cc=col+offsets%cols;allids=rr[rr<=cc]*N+cc[rr<=cc]
            assert direct+n+rej==len(allids)
            refout=torch.empty(len(allids),device='cuda',dtype=torch.int64);refc=torch.zeros(6,device='cuda',dtype=torch.int32);rg=torch.from_numpy(allids).to('cuda')
            self.terminal(v,rg,refout,refc,len(allids));truth=np.sort(refout[:int(refc[0].item())].cpu().numpy().copy())
            assert np.isin(accepted,truth).all() and np.isin(truth,np.concatenate([accepted,unclear])).all()
            self.terminal(v,unc,res,c,n);got=np.sort(res[:int(c[0].item())].cpu().numpy().copy());assert np.array_equal(got,truth)
            assert torch.all(out[:32]==-17).item() and torch.all(out[-32:]==-17).item() and torch.all(amb[:32]==-17).item() and torch.all(amb[-32:]==-17).item()
            reports.append({'shape':[rows,cols],'offset':[row,col],'pairs':len(allids),'direct':direct,'ambiguous':n,'reject':rej,'accepted':len(truth),'pass':True})
        return reports
