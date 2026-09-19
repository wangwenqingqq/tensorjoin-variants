"""Matched 64x64 precision/fusion perturbation; no autotuned configurations."""
import triton
import triton.language as tl
from gpu_kernels import upadd, dnadd, upmul

@triton.jit
def compact(i,j,yes,unc,out,amb,c,N:tl.constexpr,CAP:tl.constexpr):
    ids=i[:,None].to(tl.int64)*N+j[None,:].to(tl.int64)
    lanes=tl.zeros((64,64),tl.int32)
    p=tl.atomic_add(c+lanes,1,mask=yes,sem='relaxed')
    save=yes&(p<CAP);tl.store(out+p,ids,mask=save)
    tl.atomic_add(c+2+lanes,1,mask=yes&~save,sem='relaxed')
    p=tl.atomic_add(c+1+lanes,1,mask=unc,sem='relaxed')
    save=unc&(p<CAP);tl.store(amb+p,ids,mask=save)
    tl.atomic_add(c+2+lanes,1,mask=unc&~save,sem='relaxed')

@triton.jit
def scan8(q,qt,s,h,e,tr,tc,score,out,amb,c,ld,ud,
          epslo,epshi,exprrel,exprabs,finalrel,finalabs,
          N:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,MODE:tl.constexpr,DUMP:tl.constexpr):
    p=tl.program_id(0);i=tl.load(tr+p)*64+tl.arange(0,64);j=tl.load(tc+p)*64+tl.arange(0,64)
    pos=p*4096+tl.arange(0,64)[:,None]*64+tl.arange(0,64)[None,:]
    if MODE==1:
        acc=tl.load(score+pos)
    else:
        acc=tl.zeros((64,64),tl.int32)
        kk=tl.arange(0,64)
        for start in range(0,D,64):
            k=start+kk
            a=tl.load(q+i[:,None]*D+k[None,:],mask=(i[:,None]<N)&(k[None,:]<D),other=0)
            b=tl.load(qt+k[:,None]*N+j[None,:],mask=(k[:,None]<D)&(j[None,:]<N),other=0)
            acc=tl.dot(a,b,acc,out_dtype=tl.int32)
        if MODE==0 or DUMP:tl.store(score+pos,acc)
    if MODE!=0:
        si=tl.load(s+i,mask=i<N,other=0);sj=tl.load(s+j,mask=j<N,other=0)
        ni=tl.load(h+i,mask=i<N,other=0);nj=tl.load(h+j,mask=j<N,other=0)
        ei=tl.load(e+i,mask=i<N,other=0);ej=tl.load(e+j,mask=j<N,other=0)
        dot=acc.to(tl.float32)*si[:,None]*sj[None,:]
        d2=ni[:,None]+nj[None,:]-2.0*dot
        mag=tl.abs(ni[:,None])+tl.abs(nj[None,:])+2.0*tl.abs(dot)
        radius=exprrel*mag+exprabs
        rlo=tl.sqrt(tl.maximum(d2-radius,0.0));rhi=tl.sqrt(tl.maximum(d2+radius,0.0))
        er=ei[:,None]+ej[None,:];fm=tl.maximum(rhi+er,1.0);fr=finalrel*fm+finalabs
        lo=tl.maximum(rlo-er-fr,0.0);hi=rhi+er+fr
        if DUMP:tl.store(ld+pos,lo);tl.store(ud+pos,hi)
        valid=(i[:,None]<N)&(j[None,:]<N)&(i[:,None]<=j[None,:])
        yes=valid&(hi<=epslo);no=valid&(lo>epshi);unc=valid&~(yes|no)
        compact(i,j,yes,unc,out,amb,c,N,CAP)

@triton.jit
def scan16(z,nlo,nhi,zu,eu,tr,tc,score,out,amb,c,ld,ud,
           N:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,GAMMA:tl.constexpr,T:tl.constexpr,
           MODE:tl.constexpr,DUMP:tl.constexpr):
    p=tl.program_id(0);i=tl.load(tr+p)*64+tl.arange(0,64);j=tl.load(tc+p)*64+tl.arange(0,64)
    pos=p*4096+tl.arange(0,64)[:,None]*64+tl.arange(0,64)[None,:]
    if MODE==1:
        dot=tl.load(score+pos)
    else:
        dot=tl.zeros((64,64),tl.float32);kk=tl.arange(0,64)
        for start in range(0,D,64):
            k=start+kk
            a=tl.load(z+i[:,None]*D+k[None,:],mask=(i[:,None]<N)&(k[None,:]<D),other=0)
            b=tl.load(z+j[None,:]*D+k[:,None],mask=(k[:,None]<D)&(j[None,:]<N),other=0)
            dot=tl.dot(a,b,dot,out_dtype=tl.float32)
        if MODE==0 or DUMP:tl.store(score+pos,dot)
    if MODE!=0:
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
        if DUMP:tl.store(ld+pos,lo);tl.store(ud+pos,hi)
        valid=(i[:,None]<N)&(j[None,:]<N)&(i[:,None]<=j[None,:])
        yes=valid&(hi<=T);no=valid&(lo>T);unc=valid&~(yes|no)
        compact(i,j,yes,unc,out,amb,c,N,CAP)
