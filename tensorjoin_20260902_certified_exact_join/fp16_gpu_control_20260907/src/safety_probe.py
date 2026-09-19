"""Bounded composite safety, wide-exponent dots, and immutable Graph checks."""
import argparse,gc,hashlib,json,os,time,traceback
from fractions import Fraction as F
import numpy as np
import torch
from control_r2 import Control,HERE,ROOT,N,D,CAP,T
SENTINEL=-70274321


def ids(x,n):
    a=np.sort(x[:n].cpu().numpy().copy())
    assert len(np.unique(a))==len(a)
    return a


def fixture(which):
    x=np.zeros((N,D),np.float32)
    x[:512]=np.load(ROOT/'fp16_dense_gate0_20260907/results/fixture_a0_fixture512.npy')
    if which:
        rng=np.random.default_rng(202609081)
        x[32:128]=rng.uniform(-1,1,(96,D)).astype(np.float32)
        x[128:256]=np.ldexp(rng.uniform(-1,1,(128,D)),rng.integers(-149,1,(128,D))).astype(np.float32)
        x[256:384]=-x[128:256]
        x[384:512]=np.nextafter(x[128:256],np.float32(0))
    return x


class Probe:
    def __init__(self,o,which,label):
        self.o=o;self.x=fixture(which);self.v=torch.from_numpy(self.x).to('cuda')
        self.z,self.md=o.prepare(self.v)
        self.dots=torch.empty((512,512),device='cuda');o.blas.gemm(self.z[:512],self.z[:512],self.dots)
        self.owners=[torch.full((CAP+64,),SENTINEL,device='cuda',dtype=torch.int64) for _ in range(3)]
        self.out,self.amb,self.fp64=[v[32:-32] for v in self.owners]
        self.c=torch.zeros(4,device='cuda',dtype=torch.int32);self.c2=torch.zeros(6,device='cuda',dtype=torch.int32)
        ii,jj=np.triu_indices(512);p=torch.from_numpy((ii*N+jj).astype(np.int64)).to('cuda')
        truth=torch.empty_like(p);rc=torch.zeros(6,device='cuda',dtype=torch.int32)
        o.terminal(self.v,p,truth,rc,len(p));assert rc[2].item()==0
        self.truth=ids(truth,int(rc[0].item()));self.graph=None
        self.label=label;self.which=which

    def canaries(self):
        for a in self.owners:
            assert torch.all(a[:32]==SENTINEL).item() and torch.all(a[-32:]==SENTINEL).item()

    def classify(self):
        self.o.classify(self.dots,self.md,self.out,self.amb,self.c,0,0)

    def run(self):
        self.z,self.md=self.o.prepare(self.v)
        self.o.blas.gemm(self.z[:512],self.z[:512],self.dots)
        self.c.zero_();self.classify();first=self.c.cpu().numpy().copy()
        direct,n1,over,reject=map(int,first);assert over==0 and direct+n1+reject==131328
        self.c2.zero_();self.c2[0]=direct
        self.o.stage2(self.v,self.amb,self.out,self.fp64,self.c2,n1)
        mid=self.c2.cpu().numpy().copy();n2=int(mid[3]);assert mid[2]==0
        assert int(mid[0])-direct+int(mid[4])+n2==n1
        self.o.terminal(self.v,self.fp64,self.out,self.c2,n2)
        last=self.c2.cpu().numpy();assert last[2]==0
        got=ids(self.out,int(last[0]));assert np.array_equal(got,self.truth)
        self.canaries();self.calibrated=(direct,n1,n2)
        return {'which':self.which,'pointer':self.v.data_ptr(),'direct':direct,'ambiguous':n1,'fp64':n2,'accepted':len(got),'pass':True}

    def dot_and_metadata(self):
        z=self.z[:512].cpu().numpy();md=[m[:512].cpu().numpy() for m in self.md];p=self.dots.cpu().numpy()
        assert np.array_equal(z,np.where(abs(self.x[:512].astype(np.float16))<2**-14,0,self.x[:512].astype(np.float16)))
        rows=list(range(16))+[32,64,127,128,192,255,256,383,511];checks=[]
        for i in rows:
            a=[F(float(t)) for t in self.x[i]];b=[F(float(t)) for t in z[i]]
            nx=sum(t*t for t in a);nz=sum(t*t for t in b);nr=sum((u-v)**2 for u,v in zip(a,b))
            lo,hi,lz,er=[F(float(t[i])) for t in md]
            assert lo<=nx<=hi and lz*lz>=nz and er*er>=nr
            checks.append(i)
        rng=np.random.default_rng(202609082)
        cells=[(i,j) for i in range(16) for j in range(16)]+list(zip(rng.integers(512,size=256),rng.integers(512,size=256)))
        gamma=F(1026,2**23-1026);max_relative=0
        for i,j in cells:
            a=[F(float(t)) for t in z[i]];b=[F(float(t)) for t in z[j]]
            error=abs(F(float(p[i,j]))-sum(u*v for u,v in zip(a,b)))
            n=sum(t*t for t in a)*sum(t*t for t in b)
            assert error*error<=gamma*gamma*n,('conditional dot envelope',i,j)
            if n:max_relative=max(max_relative,float(error*error/(gamma*gamma*n)))
        ai=np.array([i for i,j in cells]);bj=np.array([j for i,j in cells])
        np.savez_compressed(HERE/'results'/f'{self.label}_input{self.which}_dots.npz',q=z[ai],b=z[bj],p=p[ai,bj],rows=rows,x=self.x[rows],z=z[rows],metadata=np.array([m[rows] for m in md]))
        return {'metadata_rows':len(checks),'exact_dot_checks':len(cells),'max_squared_fraction_of_budget':max_relative,'pass':True}

    def graph_check(self,replays):
        # Owned GEMM stays on its frozen default stream; Graph tests its consumer.
        # Inputs/dots/metadata are immutable and launch counts are calibrated.
        _,n1,n2=self.calibrated
        def body():
            self.c.zero_();self.c2.zero_();self.classify();self.c2[0].copy_(self.c[0])
            self.o.stage2(self.v,self.amb,self.out,self.fp64,self.c2,n1)
            self.o.terminal(self.v,self.fp64,self.out,self.c2,n2)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):body()
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):body()
        for i in range(replays):
            g.replay();torch.cuda.synchronize();last=self.c2.cpu().numpy()
            assert last[2]==0 and np.array_equal(ids(self.out,int(last[0])),self.truth)
            self.canaries()
        self.graph=g
        return {'pass':True,'replays':replays,'scope':'immutable precomputed dots/metadata; classifier, counter reset/transfer, stage2 and terminal; not GEMM or dynamic-input Graph'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['stress','probe'],required=True);a=p.parse_args()
    dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    r={'label':a.label,'mode':a.mode,'pass':False,'pid':os.getpid(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started':time.time(),'speed_claim':False};o=None;probes=[]
    try:
        assert json.loads((HERE/'results/gpu0_fixture_a2.json').read_text())['pass']
        o=Control();r['identities']=o.identities;r['binding']=o.blas.record()
        for i in range(2):probes.append(Probe(o,i,a.label))
        assert probes[0].v.data_ptr()!=probes[1].v.data_ptr()
        r['runs']=[]
        for i in range(32 if a.mode=='stress' else 2):r['runs'].append(probes[i%2].run())
        r['numeric']=[q.dot_and_metadata() for q in probes]
        r['graphs']=[q.graph_check(8 if a.mode=='stress' else 2) for q in probes]
        r['compiled']=o.capture();r['pass']=True
    except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        probes.clear()
        # List-comprehension and function scopes do not retain probe references.
        gc.collect();torch.cuda.synchronize()
        if o is not None:o.close()
        o=None;gc.collect();torch.cuda.empty_cache();r['ended']=time.time()
        with dest.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({'pass':r['pass'],'path':str(dest)}),flush=True)
    return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
