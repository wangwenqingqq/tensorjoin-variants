import triton
import triton.language as tl
from gpu_kernels import upadd, dnadd, upmul

@triton.jit
def filter_tiles(Q, NORMS, TR, TC, KIND, FINE, ROWS, COLS, COUNTS,
                 N:tl.constexpr, LIMIT:tl.constexpr, MODE:tl.constexpr):
    p=tl.program_id(0)
    ii=tl.arange(0,64);jj=tl.arange(0,64);k=tl.arange(0,64)
    i=tl.load(TR+p)*64+ii;j=tl.load(TC+p)*64+jj
    a=tl.load(Q+i[:,None]*64+k[None,:],i[:,None]<N,0)
    b=tl.load(Q+j[None,:]*64+k[:,None],j[None,:]<N,0)
    dot=tl.dot(a,b,out_dtype=tl.float32)
    ni=tl.load(NORMS+i,i<N,0);nj=tl.load(NORMS+j,j<N,0)
    d=ni[:,None]+nj[None,:]-2.0*dot
    valid=(i[:,None]<N)&(j[None,:]<N)&(i[:,None]<=j[None,:])
    cand=valid&(d<=LIMIT)
    count=tl.sum(tl.sum(cand.to(tl.int32),1),0)
    tl.store(COUNTS+p,count)
    if MODE==1:
        sub=tl.reshape(cand,(4,16,4,16))
        flags=tl.max(tl.max(sub.to(tl.int32),3),1)
        loc=tl.arange(0,4)[:,None]*4+tl.arange(0,4)[None,:]
        tl.store(FINE+p*16+loc,flags.to(tl.uint8))
    if MODE==2:
        ar=tl.max(cand.to(tl.int32),1);ac=tl.max(cand.to(tl.int32),0)
        nr=tl.sum(ar,0);nc=tl.sum(ac,0)
        row=(nr>0)&(nr<=16)&((nr<=nc)|(nc>16))
        col=(nc>0)&(nc<=16)&~row
        kind=tl.where(count==0,0,tl.where(row,2,tl.where(col,3,1)))
        rp=tl.cumsum(ar,0)-1;cp=tl.cumsum(ac,0)-1
        # Arrays are initialized to N by the caller; unused packed lanes stay invalid.
        tl.store(ROWS+p*16+rp,i,(ar>0)&row)
        tl.store(COLS+p*16+cp,j,(ac>0)&col)
        tl.store(KIND+p,kind)
    else:
        tl.store(KIND+p,(count>0).to(tl.uint8))

@triton.jit
def scan_rect(z,nlo,nhi,zu,eu,TR,TC,JOBS,ROWS,COLS,out,amb,c,
              N:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,GAMMA:tl.constexpr,
              T:tl.constexpr,M:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    job=tl.load(JOBS+tl.program_id(0))
    if MODE==1:
        p=job//16;s=job%16
        i=tl.load(TR+p)*64+(s//4)*16+tl.arange(0,M)
        j=tl.load(TC+p)*64+(s%4)*16+tl.arange(0,B)
    elif MODE==2:
        p=job;i=tl.load(ROWS+p*16+tl.arange(0,M))
        j=tl.load(TC+p)*64+tl.arange(0,B)
    else:
        p=job;i=tl.load(TR+p)*64+tl.arange(0,M)
        j=tl.load(COLS+p*16+tl.arange(0,B))
    dot=tl.zeros((M,B),tl.float32);kk=tl.arange(0,64)
    for start in range(0,D,64):
        k=start+kk
        a=tl.load(z+i[:,None]*D+k[None,:],(i[:,None]<N)&(k[None,:]<D),0)
        b=tl.load(z+j[None,:]*D+k[:,None],(j[None,:]<N)&(k[:,None]<D),0)
        dot=tl.dot(a,b,dot,out_dtype=tl.float32)
    il=tl.load(nlo+i,i<N,0)[:,None];jl=tl.load(nlo+j,j<N,0)[None,:]
    ih=tl.load(nhi+i,i<N,0)[:,None];jh=tl.load(nhi+j,j<N,0)[None,:]
    iz=tl.load(zu+i,i<N,0)[:,None];jz=tl.load(zu+j,j<N,0)[None,:]
    ie=tl.load(eu+i,i<N,0)[:,None];je=tl.load(eu+j,j<N,0)[None,:]
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
    ids=i[:,None].to(tl.int64)*N+j[None,:].to(tl.int64)
    lanes=tl.zeros((M,B),tl.int32)
    pos=tl.atomic_add(c+lanes,1,mask=yes,sem='relaxed')
    save=yes&(pos<CAP);tl.store(out+pos,ids,save)
    tl.atomic_add(c+2+lanes,1,mask=yes&~save,sem='relaxed')
    pos=tl.atomic_add(c+1+lanes,1,mask=unc,sem='relaxed')
    save=unc&(pos<CAP);tl.store(amb+pos,ids,save)
    tl.atomic_add(c+2+lanes,1,mask=unc&~save,sem='relaxed')
