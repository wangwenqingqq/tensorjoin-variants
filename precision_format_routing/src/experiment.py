"""Run the frozen small-scale native-format / exact-refinement experiment."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import triton
from kernels import metadata, dot, terminal_fp64 as refine_ambiguous_fp64_i64
from retained import half_classify, certified_fp32_filter_i64
from native import patch, field_words

FORMATS=['int8','e3m4','e4m3','e5m2','fp16','fp32']
DTYPES={'int8':torch.int8,'e3m4':torch.uint8,'e4m3':torch.uint8,'e5m2':torch.uint8,'fp16':torch.float16,'fp32':torch.float32}
D=512
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ['TENSORJOIN_SOURCE']) if 'TENSORJOIN_SOURCE' in os.environ else None


def threshold32(t):
    value=np.float32(t)
    if float(value)>t:value=np.nextafter(value,np.float32(-np.inf))
    return float(value)


def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def canonical(upper,n):
    upper=np.asarray(upper,dtype=np.uint64);i=upper//n;j=upper%n
    a=np.sort(np.concatenate([upper,(j*n+i)[i!=j]]))
    assert not np.any(a[1:]==a[:-1]);return a


def codebook(format):
    c=np.arange(256);a=c&127;sign=np.where(c<128,1.,-1.)
    if format=='e3m4':
        e=a>>4;m=a&15;v=np.where(e==0,m/64.,(16.+m)*np.exp2(e-7.));v[a==127]=np.nan
    elif format=='e4m3':
        e=a>>3;m=a&7;v=np.where(e==0,m*2.**-9,(8.+m)*np.exp2(e-10.));v[a==127]=np.nan
    else:
        e=a>>2;m=a&3;v=np.where(e==0,m*2.**-16,(4.+m)*np.exp2(e-17.));v[(e==31)&(m==0)]=np.inf;v[(e==31)&(m!=0)]=np.nan
    return sign*v


class Engine:
    def __init__(self):self.patched={};self.kernels={};self.churn=False;self.calls=0;self.addresses=set()

    def prepare(self,v,fmt):
        n=v.shape[0];q=torch.empty_like(v,dtype=DTYPES[fmt]);s=torch.empty(n,device='cuda')
        md=[torch.empty(n,device='cuda') for _ in range(4)];invalid=torch.zeros(1,dtype=torch.int32,device='cuda')
        k=metadata[(n,)](v,q,s,*md,invalid,D=D,FORMAT=fmt,num_warps=4,enable_fp_fusion=False)
        self.kernels['metadata_'+fmt]=k
        assert invalid.item()==0
        return q,s,md

    def dots(self,q,s,fmt,fmt_b=None):
        n,d=q.shape;fmt_b=fmt if fmt_b is None else fmt_b
        out=torch.empty((n,n),device='cuda');grid=(triton.cdiv(n,32),triton.cdiv(n,32),1)
        args=(q,s,out,n,d,fmt,fmt_b,32,32,64)
        if fmt=='e3m4':
            key=(n,d,fmt_b)
            if key not in self.patched:
                k=dot.warmup(*args,grid=grid,num_warps=4,num_stages=2,enable_fp_fusion=False)
                self.patched[key]=patch(k)
            k=self.patched[key];k[grid](*args)
        else:
            k=dot[grid](*args,num_warps=4,num_stages=2,enable_fp_fusion=False)
        self.kernels[f'dot_{fmt}_{fmt_b}_{n}_{d}']=k
        return out

    def run(self,x,threshold,fmt,profile=True):
        if fmt=='route':
            from routing import route
            return route(self,x,threshold)
        n=len(x);cap=n*n
        torch.cuda.synchronize();t0=time.perf_counter();events=[];tfloor=threshold32(threshold)
        def mark():
            e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
        mark();self.calls+=1
        if self.churn:
            shift=32*(1+self.calls%16)
            storage=torch.empty(n*D+shift,device='cuda',dtype=torch.float32)
            v=storage[shift:].view(n,D);v.copy_(torch.from_numpy(x))
            self.addresses.add(v.data_ptr())
        else:
            v=torch.from_numpy(x).to('cuda')
        out=torch.empty(cap,device='cuda',dtype=torch.int64)
        amb=torch.empty_like(out);fp64=torch.empty_like(out)
        c2=torch.zeros(6,device='cuda',dtype=torch.int32)
        if fmt=='fp64':
            ij=torch.triu_indices(n,n,device='cuda');pairs=(ij[0]*n+ij[1]).contiguous()
            mark();mark();count=pairs.numel();n1=0;n2=count;direct=0;reject=0
            mark()
            k=refine_ambiguous_fp64_i64[(count,)](v,v,pairs,out,c2,threshold,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
            self.kernels['fp64']=k
        else:
            q,s,md=self.prepare(v,fmt);mark();scores=self.dots(q,s,fmt)
            c=torch.zeros(4,device='cuda',dtype=torch.int32)
            # Integer dot is exact through D=512; only int->FP32 and two scales round.
            gamma=2.0**-21 if fmt=='int8' else 0.00012232370499987155
            k=half_classify[(triton.cdiv(cap,1024),)](scores,*md,out,amb,c,scores,scores,0,0,N=n,ROWS=n,COLS=n,CAP=cap,GAMMA=gamma,T=tfloor,BLOCK=1024,DUMP=False,num_warps=8,enable_fp_fusion=False)
            self.kernels['classify_'+fmt]=k;mark()
            first=c.cpu().numpy();direct,n1,overflow,reject=map(int,first)
            assert direct+n1+reject==n*(n+1)//2 and overflow==0
            c2[0]=direct
            if n1:
                k=certified_fp32_filter_i64[(n1,)](v,amb,out,fp64,c2,tfloor,tfloor,2**-14,2**-22,2**-22,4096*2**-126,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
                self.kernels['fp32_refine']=k
            mid=c2.cpu().numpy();n2=int(mid[3]);assert mid[2]==0;mark()
            if n2:
                k=refine_ambiguous_fp64_i64[(n2,)](v,v,fp64,out,c2,threshold,N_=n,K=D,CAPACITY=cap,BLOCK_K=256,num_warps=4,enable_fp_fusion=False)
                self.kernels['fp64']=k
        mark();last=c2.cpu().numpy();assert last[2]==0
        a=canonical(out[:int(last[0])].cpu().numpy(),n)
        torch.cuda.synchronize();wall=(time.perf_counter()-t0)*1e3
        phase=[events[i].elapsed_time(events[i+1]) for i in range(4)]
        return a,{'format':fmt,'n':n,'threshold_d2':threshold,'e2e_ms':wall,
            'transfer_allocate_prepare_ms':phase[0],'stage1_ms':phase[1],
            'host_counts_fp32_ms':phase[2],'fp64_ms':phase[3],
            'stage1_accept':direct,'stage1_reject':reject,'fp32_pairs':n1,'fp64_pairs':n2,
            'output_count':len(a),'output_sha256':digest(a),'input_sha256':digest(x)}

    def capture(self,path):
        from native import inspect_binary
        rows={}
        for name,k in self.kernels.items():
            k._init_handles();binary=k.kernel
            entry={'cubin_sha256':hashlib.sha256(binary).hexdigest(),'n_regs':k.n_regs,'n_spills':k.n_spills,'shared':k.metadata.shared,'target':str(k.metadata.target)}
            if name.startswith('dot_'):
                import subprocess,tempfile,re
                from native import CUOBJDUMP
                with tempfile.TemporaryDirectory() as temp:
                    p=Path(temp)/'k.cubin';p.write_bytes(binary)
                    sass=subprocess.check_output([CUOBJDUMP,'-sass',str(p)],text=True)
                entry['ptx_target']=[line.strip() for line in k.asm['ptx'].splitlines() if line.startswith('.target')]
                entry['opcodes']={op:len(re.findall(r'\b'+op+r'\.',sass)) for op in ['QMMA','IMMA','HMMA','FFMA','FADD','FMUL','LDL','STL']}
            rows[name]=entry
        path.write_text(json.dumps(rows,indent=2))


def fixtures():
    if SOURCE is None:raise RuntimeError('Set TENSORJOIN_SOURCE to the certified TensorJoin directory.')
    manifest=json.loads((SOURCE/'g17_rthiss_pair_contract_20260905/data/manifest_r1.json').read_text())
    for c in manifest['inputs']:
        if c['n']<=4096:
            p=SOURCE/c['path'];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==c['source_sha256']
            yield c['name'],np.frombuffer(b,dtype='<f4').reshape(c['n'],512).copy(),float(c['reference_threshold'])
    edge=np.zeros((31,512),np.float32);edge[1::2,0]=np.float32(.75411)
    exact_t=float(np.float64(edge[1,0])**2)
    for name,t in [('below',np.nextafter(exact_t,-np.inf)),('equal',exact_t),('above',np.nextafter(exact_t,np.inf))]:
        yield 'fp64_threshold_'+name,edge.copy(),float(t)
    rng=np.random.default_rng(19092026)
    for name,spread in [('synthetic_clustered',False),('synthetic_outlier',True)]:
        x=rng.normal(size=(1024,512)).astype(np.float32)*np.float32(.01)
        # Identical task and threshold across formats; synthetic negative control,
        # not a surrogate for the real distribution.
        x[:,0]=.5 if spread else .02
        x+=rng.normal(size=(16,512)).astype(np.float32)[np.arange(1024)%16]*np.float32(.01)
        yield name,x,.1024


def native_probe(engine):
    result={}
    for fmt in ['e4m3','e5m2','e3m4']:
        q=torch.arange(256,dtype=torch.int16,device='cuda').to(torch.uint8)[:,None].expand(256,64).contiguous()
        s=torch.ones(256,device='cuda');got=engine.dots(q,s,fmt).cpu().numpy()
        v=codebook(fmt)
        with np.errstate(invalid='ignore',over='ignore'):expected=(v[:,None]*v[None,:]*64).astype(np.float32)
        assert np.array_equal(np.isnan(got),np.isnan(expected)),fmt
        finite=np.isfinite(expected)
        assert np.allclose(got[finite],expected[finite],rtol=2e-6,atol=0),fmt
        assert np.array_equal(np.isinf(got),np.isinf(expected)),fmt
        result[fmt]={'codebook_both_operands':256,'max_abs_error':float(np.max(np.abs(got[finite]-expected[finite]))),'raw_0x10_self_dot_k64':float(got[16,16])}
    # Compiler controls identify the documented A/B low selector fields.
    q=torch.zeros((32,64),dtype=torch.uint8,device='cuda');s=torch.ones(32,device='cuda')
    for a,b in [('e4m3','e4m3'),('e5m2','e4m3'),('e4m3','e5m2')]:engine.dots(q,s,a,b)
    base=field_words(engine.kernels['dot_e4m3_e4m3_32_64'].kernel)
    for a,b,bit in [('e5m2','e4m3',78),('e4m3','e5m2',79)]:
        other=field_words(engine.kernels[f'dot_{a}_{b}_32_64'].kernel)
        assert len(base)==len(other)
        allowed=sum(1<<v for v in [78,82,83,79,84,85])
        assert all(((x^y)&allowed)==1<<bit for x,y in zip(base,other)),(a,b)
    result['documented_selector_control']='A bit 78 / B bit 79 verified; E3M4 relative bits 82/84 validated by exhaustive native codebook'
    return result


def producer_test(engine):
    rng=np.random.default_rng(919);book=codebook('e3m4')[:127]
    # Max=30/32 fixes E3M4 scale=1/32. Test all codeword midpoints and neighbors.
    mid=(book[:-1]+book[1:])/2
    vals=np.concatenate([book,mid,np.nextafter(mid.astype(np.float32),np.float32(np.inf)),np.nextafter(mid.astype(np.float32),np.float32(-np.inf))]).astype(np.float32)/32
    x=np.zeros((4,512),np.float32);x[:,0]=30/32
    for row in range(4):x[row,1:len(vals)+1]=vals if row%2==0 else -vals
    q,s,md=engine.prepare(torch.from_numpy(x).cuda(),'e3m4');codes=q.cpu().numpy();scale=s.cpu().numpy()
    decoded=codebook('e3m4')[codes]*scale[:,None]
    for i in range(4):
        av=np.abs(x[i].astype(np.float64)/scale[i]);dist=np.abs(av[:,None]-book[None,:]);minimum=dist.min(axis=1)
        tied=dist==minimum[:,None]
        chosen=np.array([next((k for k in np.flatnonzero(t) if k%2==0),np.flatnonzero(t)[0]) for t in tied])
        assert np.array_equal(codes[i]&127,chosen),i
    return {'midpoints_neighbor_values':len(vals),'tie_rule':'nearest even','max_error':float(np.abs(decoded-x).max())}


def numerical_test(engine):
    rng=np.random.default_rng(9919)
    x=rng.normal(size=(97,512)).astype(np.float32)*np.float32(.04)
    x[::3]*=np.float32(2**-14);x[0]=0;x[1]=np.finfo(np.float32).smallest_subnormal
    x[2]=np.linspace(-1,1,512,dtype=np.float32)
    result={};v=torch.from_numpy(x).cuda()
    exact=np.sum(x.astype(np.float64)**2,axis=1)
    distances=np.sum((x.astype(np.float64)[:,None,:]-x.astype(np.float64)[None,:,:])**2,axis=2)
    for fmt in FORMATS:
        q,s,md=engine.prepare(v,fmt);ss=s.cpu().numpy().astype(np.float64);qq=q.cpu().numpy()
        h=codebook(fmt)[qq] if fmt.startswith('e') else qq.astype(np.float64)
        z=h*ss[:,None];dots=engine.dots(q,s,fmt).cpu().numpy()
        truth=z@z.T;err=np.abs(dots.astype(np.float64)-truth)
        gamma=2**-21 if fmt=='int8' else .00012232370499987155
        bound=gamma*np.linalg.norm(z,axis=1)[:,None]*np.linalg.norm(z,axis=1)[None,:]+2048*2**-126
        assert np.all(err<=bound),(fmt,'dot envelope')
        lo,hi,zu,eu=[m.cpu().numpy().astype(np.float64) for m in md]
        assert np.all(lo<=exact) and np.all(hi>=exact),fmt
        assert np.all(zu>=np.linalg.norm(z,axis=1)) and np.all(eu>=np.linalg.norm(x-z,axis=1)),fmt
        if fmt.startswith('e'):
            book=codebook(fmt)[:127 if fmt!='e5m2' else 124]
            av=np.abs(x.astype(np.float64)/ss[:,None]);dist=np.abs(av[:,:,None]-book[None,None,:]);mini=dist.min(axis=2)
            goterr=np.abs(np.abs(h)-av)
            assert np.all(goterr<=mini+1e-12),(fmt,'nearest representable')
        scores=torch.from_numpy(dots).cuda();n=len(x);cap=n*n
        out=torch.empty(cap,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);counts=torch.zeros(4,device='cuda',dtype=torch.int32)
        lower=torch.empty_like(scores);upper=torch.empty_like(scores)
        half_classify[(triton.cdiv(cap,1024),)](scores,*md,out,amb,counts,lower,upper,0,0,N=n,ROWS=n,COLS=n,CAP=cap,GAMMA=gamma,T=.3955230712890625,BLOCK=1024,DUMP=True,num_warps=8,enable_fp_fusion=False)
        assert np.all(lower.cpu().numpy()<=distances) and np.all(upper.cpu().numpy()>=distances),(fmt,'interval containment')
        result[fmt]={'pairs_checked':cap,'max_abs_dot_error':float(err.max()),'metadata_and_intervals':True}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['probe','validate','stress','bench'],required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--case',default='all');parser.add_argument('--order',type=int,default=0);parser.add_argument('--limit-n',type=int,default=0)
    args=parser.parse_args();engine=Engine();engine.churn=args.mode=='stress';result={'mode':args.mode,'label':args.label,'torch':torch.__version__,'triton':triton.__version__,'records':[]}
    from triton.backends.nvidia.compiler import get_ptxas
    compiler=get_ptxas(120);assert compiler.version=='13.1',compiler
    result['ptxas_version']=compiler.version
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    if args.mode=='probe':
        result['native']=native_probe(engine);result['producer']=producer_test(engine);result['numeric']=numerical_test(engine)
    else:
        for name,x,t in fixtures():
            if args.case!='all' and args.case!=name:continue
            if args.limit_n and len(x)>args.limit_n:x=x[:args.limit_n].copy();name+=f'_prefix{args.limit_n}'
            if args.mode=='stress' and len(x)>512:x=x[:512].copy();name+='_prefix512'
            oracle,ref=engine.run(x,t,'fp64');result['records'].append(dict(case=name,role='reference',**ref))
            if len(x)<=32:
                dx=x.astype(np.float64)[:,None,:]-x.astype(np.float64)[None,:,:]
                expected=np.flatnonzero(np.sum(dx*dx,axis=2)<=t).astype(np.uint64)
                assert np.array_equal(oracle,expected),(name,'CPU oracle')
            if args.mode=='bench':
                # Resolve every possible full-shape route target before timing.
                for target in FORMATS+['fp64']:
                    got,rec=engine.run(x,t,target);assert np.array_equal(got,oracle)
                    result['records'].append(dict(case=name,role='compiler_preparation',**rec))
            methods=FORMATS+['fp64','route']
            if args.order%2:methods=list(reversed(methods))
            for fmt in methods:
                reps=16 if args.mode=='stress' else (7 if args.mode=='bench' else 1)
                for rep in range(reps):
                    got,rec=engine.run(x.copy(),t,fmt)
                    assert np.array_equal(got,oracle),(name,fmt,len(got),len(oracle),'exact ID mismatch')
                    rec.update(case=name,rep=rep,warmup=args.mode=='bench' and rep<2,exact_ids=True)
                    result['records'].append(rec)
                    print(json.dumps(rec),flush=True)
    if args.mode=='stress':
        result['distinct_gpu_input_addresses']=len(engine.addresses)
        assert len(engine.addresses)>=16,'pointer churn was not demonstrated'
    ROOT.joinpath('results').mkdir(exist_ok=True)
    (ROOT/'results'/f'{args.label}.json').write_text(json.dumps(result,indent=2))
    engine.capture(ROOT/'results'/f'{args.label}.compiled.json')
    print(json.dumps({'pass':True,'label':args.label}),flush=True)

if __name__=='__main__':main()
