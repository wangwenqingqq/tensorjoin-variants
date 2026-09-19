"""Integer work queues and speculative scheduling; inherited numerical formula."""
import triton
import triton.language as tl
from gpu_kernels import upadd, dnadd, upmul
from g2_kernels import compact


@triton.jit
def plan_tiles(Q, NORMS, TR, TC, FLAGS, DFLAGS, JOBS, COUNTS, PAIRS,
               OFFSET, TOTAL: tl.constexpr, N: tl.constexpr,
               CHUNK: tl.constexpr, LIMIT: tl.constexpr,
               SAMPLE: tl.constexpr, REPAIR: tl.constexpr,
               FORCE: tl.constexpr):
    p = OFFSET + tl.program_id(0)
    rr = tl.load(TR+p); cc = tl.load(TC+p)
    if SAMPLE:
        ix = tl.arange(0,16)*4
        jx = tl.arange(0,16)*4
    else:
        ix = tl.arange(0,64)
        jx = tl.arange(0,64)
    i = rr*64+ix; j = cc*64+jx; k = tl.arange(0,64)
    a = tl.load(Q+i[:,None]*64+k[None,:], i[:,None]<N, 0)
    b = tl.load(Q+j[None,:]*64+k[:,None], j[None,:]<N, 0)
    dot = tl.dot(a,b,out_dtype=tl.float32)
    ni = tl.load(NORMS+i,i<N,0); nj = tl.load(NORMS+j,j<N,0)
    d = ni[:,None]+nj[None,:]-2.0*dot
    valid = (i[:,None]<N)&(j[None,:]<N)&(i[:,None]<=j[None,:])
    cand = valid&(d<=LIMIT)
    count = tl.sum(tl.sum(cand.to(tl.int32),1),0)
    keep = count>0
    if FORCE==1:
        keep = p<0
    elif FORCE==2:
        keep = p>=0
    elif FORCE==3:
        keep = (p%2)==0
    tl.store(FLAGS+p,keep.to(tl.uint8))
    tl.store(PAIRS+p,count)
    if REPAIR:
        draft = tl.load(DFLAGS+p)
        keep = keep & (draft==0)
    group = p//CHUNK
    pos = tl.atomic_add(COUNTS+group,1,mask=keep,sem='relaxed')
    tl.store(JOBS+group*CHUNK+pos,p,keep)


@triton.jit
def scan_queue(z,nlo,nhi,zu,eu,TR,TC,JOBS,COUNTS,out,amb,c,
               GROUP, N:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,
               CHUNK:tl.constexpr,GAMMA:tl.constexpr,T:tl.constexpr):
    lane=tl.program_id(0)
    if lane<tl.load(COUNTS+GROUP):
        p=tl.load(JOBS+GROUP*CHUNK+lane)
        i=tl.load(TR+p)*64+tl.arange(0,64)
        j=tl.load(TC+p)*64+tl.arange(0,64)
        dot=tl.zeros((64,64),tl.float32);kk=tl.arange(0,64)
        for start in range(0,D,64):
            k=start+kk
            a=tl.load(z+i[:,None]*D+k[None,:],mask=(i[:,None]<N)&(k[None,:]<D),other=0)
            b=tl.load(z+j[None,:]*D+k[:,None],mask=(k[:,None]<D)&(j[None,:]<N),other=0)
            dot=tl.dot(a,b,dot,out_dtype=tl.float32)
        il=tl.load(nlo+i,mask=i<N,other=0)[:,None];jl=tl.load(nlo+j,mask=j<N,other=0)[None,:]
        ih=tl.load(nhi+i,mask=i<N,other=0)[:,None];jh=tl.load(nhi+j,mask=j<N,other=0)[None,:]
        iz=tl.load(zu+i,mask=i<N,other=0)[:,None];jz=tl.load(zu+j,mask=j<N,other=0)[None,:]
        ie=tl.load(eu+i,mask=i<N,other=0)[:,None];je=tl.load(eu+j,mask=j<N,other=0)[None,:]
        b=upmul(upmul(tl.full((),GAMMA,tl.float32),iz),jz)
        b=upadd(b,upmul(ie,jz));b=upadd(b,upmul(iz,je));b=upadd(b,upmul(ie,je))
        b=upadd(b,tl.full((),2048.0*2.0**-126,tl.float32))
        r=upadd(upmul(2.0,b),upmul(tl.full((),2.0**-38,tl.float32),upadd(ih,jh)))
        r=upadd(r,tl.full((),1.0000001044244144e-12,tl.float32))
        neg=-2.0*dot
        lo=tl.maximum(dnadd(dnadd(dnadd(il,jl),neg),-r),0.0)
        hi=upadd(upadd(upadd(ih,jh),neg),r)
        valid=(i[:,None]<N)&(j[None,:]<N)&(i[:,None]<=j[None,:])
        yes=valid&(hi<=T);no=valid&(lo>T);unc=valid&~(yes|no)
        compact(i,j,yes,unc,out,amb,c,N,CAP)
