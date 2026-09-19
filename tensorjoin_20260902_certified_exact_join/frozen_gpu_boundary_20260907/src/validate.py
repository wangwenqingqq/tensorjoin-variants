"""Correctness-only diagnostics of four immutable G16 GPU programs."""
import argparse,gc,hashlib,json,os,platform,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from frozen_driver import Driver,HERE,ROOT,verify,sha
N,D,CAP=60000,512,4096*64*64
EPS=161/256;T=25921/65536;TAU=2.0**-126
SENTINEL=-70274321

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<u8').tobytes()).hexdigest()
def write_new(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True)
def valid_rows(n):
    if not 1<=n<=2**22:raise ValueError('Metadata row-offset admission')
def admitted_tiles(tr,tc):
    if len(tr)!=len(tc) or not 0<len(tr)<=4096:raise ValueError('Tile capacity admission')
    if not np.all((tr>=0)&(tr<=tc)&(tc<938)):raise ValueError('Tile range admission')
    if len(set(zip(map(int,tr),map(int,tc))))!=len(tr):raise ValueError('Duplicate tiles')
def selected_ids(tr,tc):
    chunks=[]
    for a,b in zip(tr,tc):
        r=np.arange(int(a)*64,min(N,(int(a)+1)*64),dtype=np.int64)
        c=np.arange(int(b)*64,min(N,(int(b)+1)*64),dtype=np.int64)
        m=r[:,None]<=c[None,:]
        chunks.append((r[:,None]*N+c[None,:])[m])
    return np.concatenate(chunks)
def ids(t,n):
    assert 0<=n<=CAP
    a=np.sort(t[:n].cpu().numpy().copy())
    assert len(a)==len(np.unique(a)),'duplicate output'
    assert np.all((a>=0)&(a<N*N)),'invalid ID'
    return a

def fixture():
    rng=np.random.default_rng(20260907)
    x=np.zeros((N,D),np.float32)
    x[1]=np.float32(-0.0)
    x[64:128]=1.0
    # Threshold ties, neighbors, cancellation witnesses, coordinate positions.
    for i in range(64):
        r=128+i;k=(i*71)%512
        e=np.float32(EPS)
        steps=(i%9)-4
        for _ in range(abs(steps)):e=np.nextafter(e,np.float32(np.inf if steps>0 else -np.inf))
        x[r,k]=e
        if i%3==0:x[r,(k+1)%512]=2.0**-28
    x[128]=0;x[128,0]=EPS
    x[129]=x[128];x[129,1:4]=2.0**-28
    x[130]=0;x[130,0]=np.nextafter(np.float32(EPS),np.float32(-np.inf))
    x[131]=0;x[131,0]=np.nextafter(np.float32(EPS),np.float32(np.inf))
    x[192:256]=rng.uniform(-.045,.045,(64,D)).astype(np.float32)
    for i in range(192,224):
        x[i+32]=x[i]; x[i+32,(i*37)%D]=np.nextafter(x[i+32,(i*37)%D],np.float32(np.inf))
    tinybits=[0,1,2,0x7ffffe,0x7fffff,0x800000,0x800001]
    for i in range(64):
        r=256+i
        if i<14:
            v=np.array(tinybits[i%7],np.uint32).view(np.float32)
            x[r]=v if i<7 else -v
        elif i<28:
            bits=rng.integers(1,0x800001,size=D,dtype=np.uint32)
            x[r]=bits.view(np.float32)*rng.choice(np.array([-1,1],np.float32),D)
        elif i<44:
            x[r]=np.ldexp(rng.uniform(-1,1,D),rng.integers(-149,1,D)).astype(np.float32)
        else:
            x[r]=rng.uniform(-1,1,D).astype(np.float32)
            if i%2:x[r]=np.nextafter(x[r-1],np.float32(0))
    x[-32:]=rng.uniform(-.04,.04,(32,D)).astype(np.float32)
    x[-2]=0;x[-2,511]=EPS;x[-1]=x[129]
    labels=['zero_and_signedzero','all_one_far','threshold_ties_neighbors','high_entropy_adjacent','subnormal_exponent_cancellation','ragged_last32']
    tiles=np.array([0,1,2,3,4,937],np.int32)
    a,b=np.triu_indices(len(tiles));tr,tc=tiles[a],tiles[b]
    return x,tr,tc,labels

class Engine:
    def __init__(self,driver,x,offset=0):
        assert x.shape==(N,D) and x.dtype==np.float32 and x.flags.c_contiguous
        assert np.isfinite(x).all() and np.max(np.abs(x))<=1
        valid_rows(N)
        self.d=driver;self.cpu=x;self.owners=[]
        # Aligned pointer offset and overlapping allocation lifetimes exercise placement.
        v=torch.empty(N*D+offset,device='cuda',dtype=torch.float32)
        self.owners.append(v);self.x=v[offset:].view(N,D);self.x.copy_(torch.from_numpy(x))
        self.q=torch.empty((N,D),device='cuda',dtype=torch.int8)
        self.qt=torch.empty((D,N),device='cuda',dtype=torch.int8)
        self.s=torch.empty(N,device='cuda');self.h=torch.empty_like(self.s);self.e=torch.empty_like(self.s)
        self.invalid=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.c=torch.zeros(6,device='cuda',dtype=torch.int32)
        self.refc=torch.zeros_like(self.c)
        for name in ['out','amb','fp64','refout']:
            base=torch.full((CAP+64,),SENTINEL,device='cuda',dtype=torch.int64)
            self.owners.append(base);setattr(self,name,base[32:-32])
        self.prepare();torch.cuda.synchronize()
        assert self.invalid.item()==0
    def prepare(self):
        self.invalid.zero_()
        self.d.launch('metadata',N,[self.x,self.q,self.s,self.h,self.e,self.invalid])
        self.qt.copy_(self.q.T)
    def canaries(self):
        for x in self.owners[1:]:
            assert torch.all(x[:32]==SENTINEL).item() and torch.all(x[-32:]==SENTINEL).item(),'queue canary'
    def stage1(self,tr,tc):
        self.d.launch('stage1',tr.numel(),[self.q,self.qt,self.s,self.h,self.e,tr,tc,self.out,self.amb,self.c],[EPS,EPS,2**-16,16*TAU,2**-20,16*TAU])
    def stage2(self,pairs,n):
        self.d.launch('stage2',n,[self.x,pairs,self.out,self.fp64,self.c],[T,T,2**-14,2**-22,2**-22,4096*TAU])
    def terminal(self,pairs,n,reference=False,threshold=T):
        self.d.launch('terminal',n,[self.x,self.x,pairs,self.refout if reference else self.out,self.refc if reference else self.c],[threshold])
    def direct(self,pair_ids):
        assert len(pair_ids)<=CAP and np.all((pair_ids>=0)&(pair_ids<N*N))
        p=torch.from_numpy(np.asarray(pair_ids,np.int64)).to('cuda')
        self.refc.zero_();self.terminal(p,len(pair_ids),True)
        c=self.refc.cpu().numpy();assert c[2]==0
        return ids(self.refout,int(c[0]))
    def cascade(self,tr,tc,truth=None,check_intermediate=False):
        admitted_tiles(tr,tc)
        a=torch.from_numpy(np.ascontiguousarray(tr,np.int32)).to('cuda');b=torch.from_numpy(np.ascontiguousarray(tc,np.int32)).to('cuda')
        self.c.zero_();self.stage1(a,b);s1=self.c.cpu().numpy().copy();assert s1[2]==0
        accept1=ids(self.out,int(s1[0])) if check_intermediate else None
        amb=ids(self.amb,int(s1[1])) if check_intermediate else None
        allids=selected_ids(tr,tc) if check_intermediate else None
        if check_intermediate:
            assert np.isin(accept1,allids).all() and np.isin(amb,allids).all()
            assert not np.intersect1d(accept1,amb).size
            assert np.isin(accept1,truth).all(),'stage1 false accept'
            assert np.isin(truth,np.concatenate([accept1,amb])).all(),'stage1 false reject'
        self.stage2(self.amb,int(s1[1]));s2=self.c.cpu().numpy().copy();assert s2[2]==0
        if check_intermediate:
            accept2=ids(self.out,int(s2[0]));fallback=ids(self.fp64,int(s2[3]))
            assert np.isin(accept2,allids).all() and np.isin(fallback,amb).all()
            assert np.isin(accept1,accept2).all() and not np.intersect1d(accept2,fallback).size
            assert np.isin(accept2,truth).all(),'stage2 false accept'
            assert np.isin(truth,np.concatenate([accept2,fallback])).all(),'stage2 false reject'
            assert s2[0]-s1[0]+s2[3]+s2[4]==s1[1]
        self.terminal(self.fp64,int(s2[3]));c=self.c.cpu().numpy().copy();assert c[2]==0
        got=ids(self.out,int(c[0]))
        if truth is not None:assert np.array_equal(got,truth),'final vs same-cubin terminal'
        self.canaries()
        return got,{'stage1_accept':int(s1[0]),'stage1_ambiguous':int(s1[1]),'stage2_accept':int(s2[0]-s1[0]),'stage2_reject':int(s2[4]),'stage2_fp64':int(s2[3]),'stage2_numeric_equal':int(s2[5]),'terminal_accept':int(c[0]-s2[0]),'final_accept':int(c[0]),'overflow':int(c[2]),'hash':digest(got)}
    def force_stage2(self,pairs,truth):
        p=torch.from_numpy(np.ascontiguousarray(pairs,np.int64)).to('cuda')
        self.c.zero_();self.stage2(p,len(pairs));c=self.c.cpu().numpy().copy()
        a=ids(self.out,int(c[0]));f=ids(self.fp64,int(c[3]))
        assert c[2]==0 and c[0]+c[3]+c[4]==len(pairs)
        assert np.isin(a,pairs).all() and np.isin(f,pairs).all() and not np.intersect1d(a,f).size
        assert np.isin(a,truth).all() and np.isin(truth,np.concatenate([a,f])).all()
        self.terminal(self.fp64,int(c[3]));last=self.c.cpu().numpy().copy()
        assert np.array_equal(ids(self.out,int(last[0])),truth)
        self.canaries()
        return {'accept':int(c[0]),'reject':int(c[4]),'fp64':int(c[3]),'numeric_equal':int(c[5]),'overflow':int(last[2])}
    def metadata_audit(self,indices,out_path):
        q=self.q.cpu().numpy();s=self.s.cpu().numpy();h=self.h.cpu().numpy();e=self.e.cpu().numpy()
        rows=[];u=F(1,2**24);nu=2*u/(1-2*u)
        for r in indices:
            scale=F(float(s[r]));err=F(float(e[r]));norm=F(float(h[r]));values=self.cpu[r]
            residual=sum((F(float(v))-int(z)*scale)**2 for v,z in zip(values,q[r]))
            rec=sum(int(z)**2 for z in q[r])*scale**2
            checks={'codes_bounded':bool(np.abs(q[r].astype(np.int32)).max()<=127),'scale_normal_positive':F(TAU)<=scale<=1,'residual_enclosed':err**2>=residual,'norm_bound':abs(norm-rec)<=nu*rec+F(1,2**150)}
            assert all(checks.values()),(r,checks)
            rows.append({'row':int(r),'scale_hex':float(s[r]).hex(),'norm_hex':float(h[r]).hex(),'error_hex':float(e[r]).hex(),'exact_residual_squared':str(residual),'checks':checks})
        np.savez_compressed(out_path.with_suffix('.npz'),rows=np.array(indices),vectors=self.cpu[indices],codes=q[indices],scales=s[indices],norms=h[indices],errors=e[indices])
        write_new(out_path,rows)
        return {'rows':len(rows),'checks_per_row':4,'all_pass':True,'raw':str(out_path.name)}
    def graph(self,tr,tc,stats,truth,replays):
        a=torch.from_numpy(tr).to('cuda');b=torch.from_numpy(tc).to('cuda')
        g=torch.cuda.CUDAGraph(); stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm the exact kernel and torch operation paths before capture.
            self.prepare();self.c.zero_();self.stage1(a,b);self.stage2(self.amb,stats['stage1_ambiguous']);self.terminal(self.fp64,stats['stage2_fp64'])
        stream.synchronize()
        with torch.cuda.graph(g,stream=stream):
            self.prepare();self.c.zero_();self.stage1(a,b);self.stage2(self.amb,stats['stage1_ambiguous']);self.terminal(self.fp64,stats['stage2_fp64'])
        for i in range(replays):
            g.replay();torch.cuda.synchronize();c=self.c.cpu().numpy()
            assert self.invalid.item()==0 and c[2]==0
            assert np.array_equal(ids(self.out,int(c[0])),truth),('graph replay',i)
            self.canaries()
        return {'pass':True,'replays':replays,'includes':['metadata','transpose','stage1','stage2','terminal'],'scope':'immutable input, calibrated stage counts'}

def host_admission_tests():
    checks=0
    for n in [0,2**22+1]:
        try:valid_rows(n)
        except ValueError:checks+=1
        else:raise AssertionError('invalid row count admitted')
    valid_rows(2**22)
    for a,b in [(np.zeros(4097,np.int32),np.zeros(4097,np.int32)),(np.array([0,0]),np.array([1,1])),(np.array([0]),np.array([938]))]:
        try:admitted_tiles(a,b)
        except ValueError:checks+=1
        else:raise AssertionError('invalid tiles admitted')
    return {'rejections':checks,'largest_metadata_row_count_admitted':2**22,'no_invalid_gpu_launch':True,'scope':'new diagnostic harness, not original API retrofit'}

def boundary(driver,args,result):
    x,tr,tc,labels=fixture();allids=selected_ids(tr,tc)
    assert len(allids)==len(np.unique(allids))
    np.savez_compressed(HERE/'results'/f'{args.label}_fixture.npz',rows=np.r_[np.arange(320),np.arange(N-32,N)],vectors=x[np.r_[np.arange(320),np.arange(N-32,N)]],tile_rows=tr,tile_columns=tc)
    eng=Engine(driver,x)
    result['fixture']={'families':labels,'pair_count':len(allids),'tiles':len(tr),'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'raw':f'{args.label}_fixture.npz'}
    # ABI smoke and known predicates before accepting this launcher as a reference.
    smoke_pairs=np.array([0,64,128,129,130,131,N*(N-1)+(N-1)],np.int64)
    smoke_expected=np.sort(np.array([0,128,129,130,N*(N-1)+(N-1)],np.int64))
    smoke=eng.direct(smoke_pairs);assert np.array_equal(smoke,smoke_expected),(smoke,smoke_expected)
    result['abi_smoke']={'pass':True,'tested_ids':smoke_pairs.tolist(),'accepted':smoke.tolist(),'main_semantic_witness_gpu_accepted':129 in smoke}
    truth=eng.direct(allids)
    result['direct_terminal']={'accepted':len(truth),'hash':digest(truth)}
    got,stats=eng.cascade(tr,tc,truth,True);result['cascade']=stats
    result['forced_stage2']=eng.force_stage2(allids,truth)
    assert all(result['forced_stage2'][k]>0 for k in ['accept','reject','fp64','numeric_equal'])
    assert stats['stage1_accept']>0 and stats['stage1_ambiguous']>0 and len(allids)>stats['stage1_accept']+stats['stage1_ambiguous']
    result['dense_empty_tiles']={}
    for name,a,b in [('dense',0,0),('empty',0,1),('ragged',937,937)]:
        ii=selected_ids([a],[b]);tt=eng.direct(ii);_,ss=eng.cascade(np.array([a],np.int32),np.array([b],np.int32),tt,True)
        if name=='dense':assert len(tt)==len(ii)
        if name=='empty':assert len(tt)==0
        result['dense_empty_tiles'][name]={'pairs':len(ii),'accepted':len(tt),'cascade':ss}
    indices=sorted(set(list(range(0,5))+list(range(64,68))+list(range(128,144))+list(range(192,200))+list(range(256,320))+list(range(N-8,N))))
    result['metadata']=eng.metadata_audit(indices,HERE/'results'/f'{args.label}_metadata.json')
    rng=np.random.default_rng(917)
    stress=[];pointers=[eng.x.data_ptr()]
    for i in range(args.repeats):
        if i in [1,3]:
            old=eng;eng=Engine(driver,x,offset=128*(i+1));assert eng.x.data_ptr()!=old.x.data_ptr();pointers.append(eng.x.data_ptr());del old;gc.collect()
        perm=rng.permutation(len(tr));_,ss=eng.cascade(tr[perm],tc[perm],truth,True)
        fp=eng.force_stage2(allids[rng.permutation(len(allids))],truth)
        stress.append({'iteration':i,'tile_permutation':perm.tolist(),'cascade':ss,'force_stage2':fp})
    result['stress']={'runs':stress,'distinct_vector_pointers':len(set(pointers)),'pointer_addresses':pointers}
    if args.graph:result['graph']=eng.graph(tr,tc,stats,truth,8)
    else:result['graph']={'status':'not_requested_for_this_process'}
    result['host_admission']=host_admission_tests()
    return eng

def full(driver,args,result):
    p=ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    expected='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
    assert sha(p)==expected,'full input identity'
    x=np.ascontiguousarray(np.load(p,allow_pickle=False));eng=Engine(driver,x)
    a,b=np.triu_indices(938);a=a.astype(np.int32);b=b.astype(np.int32)
    allout=[];batches=[]
    for start in range(0,len(a),4096):
        got,stats=eng.cascade(a[start:start+4096],b[start:start+4096]);allout.append(got)
        stats.update(start_tile=start,tiles=min(4096,len(a)-start));batches.append(stats)
        print(json.dumps({'full_batch':len(batches),'stats':stats}),flush=True)
    upper=np.sort(np.concatenate(allout));assert len(upper)==len(np.unique(upper))
    row,col=upper//N,upper%N;assert np.all(row<=col)
    mirrored=col[row!=col]*N+row[row!=col]
    canonical=np.sort(np.concatenate([upper,mirrored]).astype('<u8'))
    actual=digest(canonical)
    result['full']={'input_sha256':expected,'upper_accepted':len(upper),'canonical_count':len(canonical),'canonical_sha256':actual,'batches':batches,'scheduled_tiles':len(a),'upper_pair_count':N*(N+1)//2}
    assert len(canonical)==3926078 and actual=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495','retained full-output mismatch'
    return eng

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['boundary','full'],default='boundary');p.add_argument('--repeats',type=int,default=4);p.add_argument('--graph',action='store_true');args=p.parse_args()
    assert args.label.replace('_','').isalnum() and 0<=args.repeats<=16
    path=HERE/'results'/f'{args.label}.json';assert not path.exists()
    r={'label':args.label,'mode':args.mode,'pass':False,'started':time.time(),'pid':os.getpid(),'host':platform.node(),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':np.__version__,'python':sys.version,'timing_claim':False}
    try:
        assert r['host']=='gpu-host-8' and r['visible_gpu']=='3'
        driver=Driver();r['loaded_identity']=driver.identities;r['runtime_abi']=driver.abi
        print(json.dumps({'phase':'loaded_frozen_cubins','abi':driver.abi}),flush=True)
        eng=boundary(driver,args,r) if args.mode=='boundary' else full(driver,args,r)
        torch.cuda.synchronize();eng.canaries()
        r['identity_after']=verify();assert r['identity_after']==r['loaded_identity']
        r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        assert r['peak_allocated_bytes']<4*1024**3
        r['pass']=True
    except Exception:
        r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        r['ended']=time.time();write_new(path,r)
    print(json.dumps({'result':str(path),'pass':r['pass']}),flush=True)
    return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
