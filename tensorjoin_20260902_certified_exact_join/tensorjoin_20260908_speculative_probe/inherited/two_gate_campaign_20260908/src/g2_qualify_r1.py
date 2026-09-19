"""Qualification only: outputs, exact sample arithmetic, churn and Graphs."""
import argparse,gc,hashlib,json,os,time,traceback
from fractions import Fraction as F
import numpy as np
import torch
from g2_operator_r1 import Matrix,HERE,ROOT,N,D,CAP,T,GAMMA,full_source,output_check,sha
METHODS=['F8','S8','F16','S16'];SENTINEL=-70274321

def ids(t,n):
    a=np.sort(t[:n].cpu().numpy().copy());assert len(np.unique(a))==len(a);return a

def fixture(which):
    x=np.zeros((N,D),np.float32);x[:512]=np.load(ROOT/'fp16_dense_gate0_20260907/results/fixture_a0_fixture512.npy')
    if which:
        rng=np.random.default_rng(202609081)
        x[32:128]=rng.uniform(-1,1,(96,D)).astype(np.float32)
        x[128:256]=np.ldexp(rng.uniform(-1,1,(128,D)),rng.integers(-149,1,(128,D))).astype(np.float32)
        x[256:384]=-x[128:256];x[384:512]=np.nextafter(x[128:256],np.float32(0))
    x[59968:]=x[:32]
    return x

def tiles():
    a,b=np.triu_indices(8)
    return np.r_[a,0,937,11,10].astype(np.int32),np.r_[b,937,937,257,10].astype(np.int32)

class Probe:
    def __init__(self,o,x,method):
        self.o=o;self.x=x;self.m=method;self.v=torch.from_numpy(x).to('cuda')
        self.prep=o.prepare(self.v,8 if method.endswith('8') else 16)
        self.tr,self.tc=tiles();self.trg=torch.from_numpy(self.tr).to('cuda');self.tcg=torch.from_numpy(self.tc).to('cuda')
        self.cells=len(self.tr)*4096
        self.owners=[torch.full((CAP+64,),SENTINEL,device='cuda',dtype=torch.int64) for _ in range(3)]
        self.out,self.amb,self.fp64=[a[32:-32] for a in self.owners]
        self.score=torch.zeros(self.cells,device='cuda',dtype=torch.int32 if method.endswith('8') else torch.float32)
        self.lo=torch.zeros(self.cells,device='cuda');self.hi=torch.zeros_like(self.lo);self.c=torch.zeros(6,device='cuda',dtype=torch.int32)
        ii=self.tr[:,None,None]*64+np.arange(64)[None,:,None]
        jj=self.tc[:,None,None]*64+np.arange(64)[None,None,:]
        ii,jj=np.broadcast_arrays(ii,jj);valid=(ii<N)&(jj<N)&(ii<=jj)
        self.ii,self.jj=ii.ravel(),jj.ravel();self.valid=valid.ravel()
        p=torch.from_numpy((ii*N+jj)[valid].astype(np.int64)).to('cuda')
        truth=torch.empty_like(p);rc=torch.zeros(6,device='cuda',dtype=torch.int32)
        o.terminal(self.v,p,truth,rc,len(p));assert rc[2].item()==0
        self.truth=ids(truth,int(rc[0].item()));self.graph=None
    def canaries(self):
        for a in self.owners:assert torch.all(a[:32]==SENTINEL).item() and torch.all(a[-32:]==SENTINEL).item()
    def first(self,dump=False):self.o.stage1(self.m,self.prep,self.trg,self.tcg,self.out,self.amb,self.c,self.score,self.lo,self.hi,dump=dump)
    def run(self,dump=False,reprepare=True):
        if reprepare:self.prep=self.o.prepare(self.v,8 if self.m.endswith('8') else 16)
        self.c.zero_();self.first(dump);s1=self.c.cpu().numpy().copy();direct,n1=int(s1[0]),int(s1[1]);assert s1[2]==0
        yes=ids(self.out,direct);unc=ids(self.amb,n1)
        assert not np.setdiff1d(yes,self.truth).size and not np.setdiff1d(self.truth,np.r_[yes,unc]).size
        assert not np.intersect1d(yes,unc).size
        self.o.stage2(self.v,self.amb,self.out,self.fp64,self.c,n1);s2=self.c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0
        assert int(s2[0])-direct+int(s2[4])+n2==n1
        self.o.terminal(self.v,self.fp64,self.out,self.c,n2);last=self.c.cpu().numpy().copy();assert last[2]==0
        got=ids(self.out,int(last[0]));assert np.array_equal(got,self.truth);self.canaries();self.calibrated=n1,n2
        rec=dict(method=self.m,pass_=True,pointer=self.v.data_ptr(),direct=direct,ambiguous=n1,fp64=n2,accepted=len(got),pairs=int(self.valid.sum()),hash=hashlib.sha256(got.astype('<i8').tobytes()).hexdigest())
        return rec,yes,unc
    def numeric(self,tag):
        rng=np.random.default_rng(202609082);valid=np.flatnonzero(self.valid)
        selected=np.r_[valid[:256],rng.choice(valid,256,replace=False)]
        ii,jj=self.ii[selected],self.jj[selected];p=self.score.cpu().numpy()[selected]
        if self.m.endswith('8'):
            q=self.prep[0].cpu().numpy();exact=np.sum(q[ii].astype(np.int64)*q[jj].astype(np.int64),axis=1)
            assert np.array_equal(p,exact)
            np.savez_compressed(HERE/'results'/f'{tag}_numeric.npz',q=q[ii],b=q[jj],p=p,i=ii,j=jj)
            return dict(int8_exact_dot_checks=len(p),pass_=True)
        z=self.prep[0].cpu().numpy();md=[t.cpu().numpy() for t in self.prep[1:]]
        expected=self.x.astype(np.float16);expected[np.abs(expected)<2**-14]=0;assert np.array_equal(z,expected)
        rows=sorted(set(list(range(16))+[32,64,127,128,192,255,256,383,511,59968,59999]))
        for i in rows:
            a=[F(float(t)) for t in self.x[i]];b=[F(float(t)) for t in z[i]]
            nx=sum(t*t for t in a);nz=sum(t*t for t in b);nr=sum((u-v)**2 for u,v in zip(a,b))
            lo,hi,lz,er=[F(float(t[i])) for t in md];assert lo<=nx<=hi and lz*lz>=nz and er*er>=nr
        gamma=F(1026,2**23-1026);maxfrac=0.;lower=self.lo.cpu().numpy()[selected];upper=self.hi.cpu().numpy()[selected]
        for k,(i,j) in enumerate(zip(ii,jj)):
            a=[F(float(t)) for t in z[i]];b=[F(float(t)) for t in z[j]]
            err=abs(F(float(p[k]))-sum(u*v for u,v in zip(a,b)));norm=sum(t*t for t in a)*sum(t*t for t in b)
            assert err*err<=gamma*gamma*norm,('conditional dot model',i,j)
            if norm:maxfrac=max(maxfrac,float(err*err/(gamma*gamma*norm)))
            il,ih,iz,ie=[F(float(t[i])) for t in md];jl,jh,jz,je=[F(float(t[j])) for t in md]
            rad=F(GAMMA)*iz*jz+ie*jz+iz*je+ie*je+F(2048,2**126)
            rad=2*rad+(ih+jh)*F(1,2**38)+F(1.0000001044244144e-12)
            lo=max(F(0),il+jl-2*F(float(p[k]))-rad);hi=ih+jh-2*F(float(p[k]))+rad
            assert F(float(lower[k]))<=lo and F(float(upper[k]))>=hi,('outward interval',i,j)
        np.savez_compressed(HERE/'results'/f'{tag}_numeric.npz',q=z[ii],b=z[jj],p=p,i=ii,j=jj,lo=lower,hi=upper,rows=rows,x=self.x[rows],z=z[rows],metadata=np.array([t[rows] for t in md]),pair_metadata=np.array([[t[ii],t[jj]] for t in md]))
        return dict(metadata_rows=len(rows),exact_dot_checks=len(p),exact_interval_checks=len(p),max_squared_fraction_of_budget=maxfrac,pass_=True)
    def graph_check(self,replays):
        n1,n2=self.calibrated
        def body():
            self.c.zero_();self.first();self.o.stage2(self.v,self.amb,self.out,self.fp64,self.c,n1);self.o.terminal(self.v,self.fp64,self.out,self.c,n2)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):body()
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):body()
        for _ in range(replays):
            g.replay();torch.cuda.synchronize();c=self.c.cpu().numpy();assert c[2]==0 and np.array_equal(ids(self.out,int(c[0])),self.truth);self.canaries()
        self.graph=g;return dict(replays=replays,pass_=True,scope='immutable prepared input; custom dot, optional split, predicate, queues, calibrated refinement; not dynamic-input Graph')

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['fixture','public','full','stress','probe'],required=True);a=p.parse_args()
    dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    r=dict(label=a.label,mode=a.mode,started=time.time(),pid=os.getpid(),pass_=False,speed_claim=False,results=[]);o=None
    try:
        for name,h in json.loads((HERE/'artifacts/g2_source_freeze_a1.json').read_text()).items():assert sha(HERE/name)==h,name
        o=Matrix();r['original_identities']=o.identities
        if a.mode=='full':
            x=full_source();counts={}
            for m in METHODS:
                out,rec=o.run(m,x);rec['hash']=output_check(out);rec['method']=m;rec['count']=len(out);rec['seconds_diagnostic_not_performance']=rec.pop('seconds');r['results'].append(rec);counts[m]=rec['stage_counts']
            assert counts['F8']==counts['S8'] and counts['F16']==counts['S16'];r['counts_match']=True
        else:
            sources=[full_source()] if a.mode=='public' else [fixture(0),fixture(1)]
            cross={}
            for which,x in enumerate(sources):
                for m in METHODS:
                    probe=Probe(o,x,m);rec,yes,unc=probe.run(dump=True)
                    dots=probe.score.cpu().numpy().copy();lo=probe.lo.cpu().numpy().copy();hi=probe.hi.cpu().numpy().copy()
                    numeric=probe.numeric(f'{a.label}_{which}_{m}') if a.mode in ['fixture','public'] else None
                    prod,py,pu=probe.run(dump=False);assert np.array_equal(yes,py) and np.array_equal(unc,pu)
                    key=(which,8 if m.endswith('8') else 16)
                    if key in cross:
                        a_yes,a_unc,a_dot,a_lo,a_hi=cross[key]
                        assert np.array_equal(yes,a_yes) and np.array_equal(unc,a_unc),'fusion class sets'
                        assert np.array_equal(dots,a_dot),'fusion dot bits'
                        assert np.array_equal(lo,a_lo) and np.array_equal(hi,a_hi),'fusion interval bits'
                    else:cross[key]=yes,unc,dots,lo,hi
                    np.savez_compressed(HERE/'results'/f'{a.label}_{which}_{m}_ids.npz',yes=yes,ambiguous=unc,truth=probe.truth,dots=dots,lo=lo,hi=hi,tr=probe.tr,tc=probe.tc)
                    r['results'].append(dict(which=which,method=m,diagnostic=rec,production=prod,numeric=numeric,graph=probe.graph_check(2)))
                    del probe;gc.collect();torch.cuda.empty_cache()
            if a.mode=='stress':
                for m in METHODS:
                    probes=[Probe(o,x,m) for x in sources];assert probes[0].v.data_ptr()!=probes[1].v.data_ptr();runs=[]
                    for i in range(32):runs.append(probes[i%2].run()[0])
                    r['results'].append(dict(method=m,stress_runs=runs,graphs=[q.graph_check(8) for q in probes]))
                    del probes;gc.collect();torch.cuda.empty_cache()
        r['compiled']=o.capture()
        assert all(v['metadata']['n_spills']==0 for v in r['compiled'].values()),'spill gate'
        r['pass']=True
    except Exception:r['pass']=False;r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        if o is not None:
            
            try:r['compiled']=o.capture()
            except Exception:r['pass']=False;r['final_capture_exception']=traceback.format_exc()
            o.close()
        gc.collect();torch.cuda.empty_cache();r['ended']=time.time();dest.write_text(json.dumps(r,indent=2))
    print(json.dumps(dict(pass_=r['pass'],label=a.label)),flush=True);return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
