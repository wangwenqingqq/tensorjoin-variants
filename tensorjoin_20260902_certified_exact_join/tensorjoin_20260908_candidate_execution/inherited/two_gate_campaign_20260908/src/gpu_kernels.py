"""Fused reconstruction metadata and directed-FP32 panel classification."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def upadd(a, b):
    return tl.inline_asm_elementwise('add.rp.f32 $0, $1, $2;', constraints='=f,f,f', args=[a,b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def dnadd(a, b):
    return tl.inline_asm_elementwise('add.rm.f32 $0, $1, $2;', constraints='=f,f,f', args=[a,b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def upmul(a, b):
    return tl.inline_asm_elementwise('mul.rp.f32 $0, $1, $2;', constraints='=f,f,f', args=[a,b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def cvup(a):
    return tl.inline_asm_elementwise('cvt.rp.f32.f64 $0, $1;', constraints='=f,d', args=[a], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def cvdown(a):
    return tl.inline_asm_elementwise('cvt.rm.f32.f64 $0, $1;', constraints='=f,d', args=[a], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def half_metadata(x, z, norm_lo, norm_hi, z_upper, residual_upper, invalid,
                  D:tl.constexpr, G:tl.constexpr):
    row=tl.program_id(0);k=tl.arange(0,D)
    a=tl.load(x+row*D+k)
    h=a.to(tl.float16).to(tl.float32)
    h=tl.where(tl.abs(h)<2.0**-14,0.0,h)
    tl.store(z+row*D+k,h.to(tl.float16))
    aa=a.to(tl.float64);hh=h.to(tl.float64);r=aa-hh
    nx=tl.sum(aa*aa,0);nz=tl.sum(hh*hh,0);nr=tl.sum(r*r,0)
    gamma=tl.full((),G,tl.float64);tiny=tl.full((),1024.0*2.0**-1022,tl.float64)
    rx=gamma*nx+tiny;rz=gamma*nz+tiny;rr=gamma*nr+tiny
    padding=tl.full((),1.0+8.0*2.0**-52,tl.float64)
    zu=libdevice.nextafter(libdevice.sqrt_rn(nz+rz)*padding,tl.full((),float('inf'),tl.float64))
    eu=libdevice.nextafter(libdevice.sqrt_rn(nr+rr)*padding,tl.full((),float('inf'),tl.float64))
    tl.store(norm_lo+row,tl.maximum(cvdown(nx-rx),0.0))
    tl.store(norm_hi+row,cvup(nx+rx))
    tl.store(z_upper+row,cvup(zu));tl.store(residual_upper+row,cvup(eu))
    bad=tl.sum((~((a==a)&(tl.abs(a)<=1.0))).to(tl.int32),0)>0
    tl.atomic_or(invalid,1,mask=bad,sem='relaxed')


@triton.jit(do_not_specialize=['row_start','column_start'])
def half_classify(dots, norm_lo, norm_hi, z_upper, residual_upper,
                  accepted, ambiguous, counts, lower_dump, upper_dump,
                  row_start, column_start,
                  N:tl.constexpr, ROWS:tl.constexpr, COLS:tl.constexpr,
                  CAP:tl.constexpr, GAMMA:tl.constexpr, T:tl.constexpr,
                  BLOCK:tl.constexpr, DUMP:tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    inside=o<ROWS*COLS;i=row_start+o//COLS;j=column_start+o%COLS
    valid=inside&(i<=j)&(i<N)&(j<N)
    p=tl.load(dots+o,mask=inside,other=0)
    il=tl.load(norm_lo+i,mask=inside,other=0);jl=tl.load(norm_lo+j,mask=inside,other=0)
    ih=tl.load(norm_hi+i,mask=inside,other=0);jh=tl.load(norm_hi+j,mask=inside,other=0)
    iz=tl.load(z_upper+i,mask=inside,other=0);jz=tl.load(z_upper+j,mask=inside,other=0)
    ie=tl.load(residual_upper+i,mask=inside,other=0);je=tl.load(residual_upper+j,mask=inside,other=0)
    b=upmul(upmul(tl.full((),GAMMA,tl.float32),iz),jz)
    b=upadd(b,upmul(ie,jz));b=upadd(b,upmul(iz,je));b=upadd(b,upmul(ie,je))
    b=upadd(b,tl.full((),2048.0*2.0**-126,tl.float32))
    r=upadd(upmul(2.0,b),upmul(tl.full((),2.0**-38,tl.float32),upadd(ih,jh)))
    r=upadd(r,tl.full((),1.0000001044244144e-12,tl.float32))
    negative_dot=-2.0*p
    lo=tl.maximum(dnadd(dnadd(dnadd(il,jl),negative_dot),-r),0.0)
    hi=upadd(upadd(upadd(ih,jh),negative_dot),r)
    if DUMP:
        tl.store(lower_dump+o,lo,mask=inside);tl.store(upper_dump+o,hi,mask=inside)
    yes=valid&(hi<=T);no=valid&(lo>T);uncertain=valid&~(yes|no)
    y=yes.to(tl.int32);u=uncertain.to(tl.int32)
    ny=tl.sum(y,0);nu=tl.sum(u,0);nn=tl.sum(no.to(tl.int32),0)
    py=tl.atomic_add(counts,ny,mask=ny>0,sem='relaxed')+tl.cumsum(y,0)-1
    pu=tl.atomic_add(counts+1,nu,mask=nu>0,sem='relaxed')+tl.cumsum(u,0)-1
    tl.atomic_add(counts+3,nn,mask=nn>0,sem='relaxed')
    ids=i.to(tl.int64)*N+j.to(tl.int64)
    tl.store(accepted+py,ids,mask=yes&(py<CAP));tl.store(ambiguous+pu,ids,mask=uncertain&(pu<CAP))
    overflow=tl.sum((yes&(py>=CAP)).to(tl.int32),0)+tl.sum((uncertain&(pu>=CAP)).to(tl.int32),0)
    tl.atomic_add(counts+2,overflow,mask=overflow>0,sem='relaxed')
