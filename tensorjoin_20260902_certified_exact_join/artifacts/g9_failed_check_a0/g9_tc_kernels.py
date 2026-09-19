"""Exact signed-limb TC products and directed certificates for G9 only."""

import triton
import triton.language as tl
from g8_layout_kernels import interval_state


@triton.jit
def add_down(a, b):
    return tl.inline_asm_elementwise("add.rm.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def add_up(a, b):
    return tl.inline_asm_elementwise("add.rp.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def sub_down(a, b):
    return tl.inline_asm_elementwise("sub.rm.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def sub_up(a, b):
    return tl.inline_asm_elementwise("sub.rp.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def mul_down(a, b):
    return tl.inline_asm_elementwise("mul.rm.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def mul_up(a, b):
    return tl.inline_asm_elementwise("mul.rp.f64 $0, $1, $2;", "=d,d,d", [a,b], tl.float64, True, 1)


@triton.jit
def limb_dot(H, L, ROWS, COLS, SLOTS, DOT, D: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.load(ROWS + pid * 16 + tl.arange(0,16))
    j = tl.load(COLS + pid * 16 + tl.arange(0,16))
    hh = tl.full((16,16), 0, tl.int32)
    hl = tl.full((16,16), 0, tl.int32)
    lh = tl.full((16,16), 0, tl.int32)
    ll = tl.full((16,16), 0, tl.int32)
    for start in range(tl.cdiv(D,K)):
        k = start*K + tl.arange(0,K)
        ah = tl.load(H + i[:,None]*D + k[None,:], k[None,:]<D, 0)
        al = tl.load(L + i[:,None]*D + k[None,:], k[None,:]<D, 0)
        bh = tl.load(H + j[None,:]*D + k[:,None], k[:,None]<D, 0)
        bl = tl.load(L + j[None,:]*D + k[:,None], k[:,None]<D, 0)
        hh = tl.dot(ah,bh,hh)
        hl = tl.dot(ah,bl,hl)
        lh = tl.dot(al,bh,lh)
        ll = tl.dot(al,bl,ll)
    dot = hh.to(tl.int64)*16384 + (hl.to(tl.int64)+lh.to(tl.int64))*128 + ll.to(tl.int64)
    cells = tl.arange(0,16)[:,None]*16 + tl.arange(0,16)[None,:]
    slots = tl.load(SLOTS + pid*256 + cells)
    tl.store(DOT+slots, dot, slots>=0)


@triton.jit
def tc_certificate(IDS, DOT, SCALE, NORM2, HUP, RUP, OUT, LOW, HIGH,
                   Q:tl.constexpr, N:tl.constexpr, T:tl.constexpr, B:tl.constexpr):
    k=tl.program_id(0)*B+tl.arange(0,B)
    ids=tl.load(IDS+k,k<Q,0)
    i,j=ids//N,ids%N
    si,sj=tl.load(SCALE+i),tl.load(SCALE+j)
    ni,nj=tl.load(NORM2+i),tl.load(NORM2+j)
    dot=tl.load(DOT+k,k<Q,0).to(tl.float64)
    cross=dot*si*sj
    lz=tl.maximum(sub_down(add_down(ni,nj),2.0*cross),0.0)
    uz=tl.maximum(sub_up(add_up(ni,nj),2.0*cross),0.0)
    e=add_up(tl.load(RUP+i),tl.load(RUP+j))
    h=add_up(tl.load(HUP+i),tl.load(HUP+j))
    error=add_up(mul_up(2.0*h,e),mul_up(e,e))
    lower=tl.maximum(sub_down(lz,error),0.0)
    upper=add_up(uz,error)
    state=tl.where(upper<=T,1,tl.where(lower>T,0,2)).to(tl.uint8)
    tl.store(OUT+k,state,k<Q)
    tl.store(LOW+k,lower,k<Q)
    tl.store(HIGH+k,upper,k<Q)


@triton.jit
def fp32_subset(X, IDS, SLOTS, OUT, M:tl.constexpr, N:tl.constexpr,
                 D:tl.constexpr, LO:tl.constexpr, HI:tl.constexpr,
                 B:tl.constexpr, K:tl.constexpr):
    k=tl.program_id(0)*B+tl.arange(0,B)
    slots=tl.load(SLOTS+k,k<M,0)
    ids=tl.load(IDS+slots)
    i,j=ids//N,ids%N
    dist=tl.full((B,),0,tl.float32)
    mag=tl.full((B,),0,tl.float32)
    different=tl.full((B,),0,tl.int32)
    for start in range(tl.cdiv(D,K)):
        f=start*K+tl.arange(0,K)
        x=tl.load(X+i[:,None]*D+f[None,:],(k[:,None]<M)&(f[None,:]<D),0)
        y=tl.load(X+j[:,None]*D+f[None,:],(k[:,None]<M)&(f[None,:]<D),0)
        d=x-y
        a=tl.abs(x)+tl.abs(y)
        dist+=tl.sum(d*d,1)
        mag+=tl.sum(a*a,1)
        different+=tl.sum((x!=y).to(tl.int32),1)
    state=interval_state(dist,mag,different,LO,HI)
    tl.store(OUT+slots,state,k<M)


@triton.jit
def fp64_subset(X, IDS, SLOTS, OUT, M:tl.constexpr, N:tl.constexpr,
                 D:tl.constexpr, T:tl.constexpr, K:tl.constexpr):
    p=tl.program_id(0)
    slot=tl.load(SLOTS+p)
    ids=tl.load(IDS+slot)
    i,j=ids//N,ids%N
    dist=tl.full((),0,tl.float64)
    for start in range(tl.cdiv(D,K)):
        f=start*K+tl.arange(0,K)
        x=tl.load(X+i*D+f,f<D,0).to(tl.float64)
        y=tl.load(X+j*D+f,f<D,0).to(tl.float64)
        d=x-y
        dist+=tl.sum(d*d,0)
    lo=mul_down(dist,1.0-2.0**-40)
    hi=mul_up(dist,1.0+2.0**-40)
    state=tl.where(hi<=T,1,tl.where(lo>T,0,2)).to(tl.uint8)
    tl.store(OUT+slot,state)
