"""Exact admission checks and deterministic metadata; CPU costs are excluded."""

import math
import time
import numpy as np


def sqrt_upper(num,den):
    value=math.sqrt(num/den)
    while True:
        a,b=value.as_integer_ratio()
        if a*a*den>=num*b*b:
            return value
        value=np.nextafter(value,math.inf).item()


def encode(x):
    started=time.perf_counter()
    x=np.asarray(x,dtype=np.float32)
    assert x.ndim==2 and x.shape[1]==512 and np.isfinite(x).all()
    assert np.max(np.abs(x))<=1
    scaled=x.astype(np.float64)*2.0**40
    assert np.max(np.abs(scaled))<2**53
    xi=scaled.astype(np.int64)
    assert np.array_equal(xi.astype(np.float64),scaled), 'input outside admitted grid'
    mx=np.max(np.abs(x.astype(np.float64)),axis=1)
    exponents=np.ceil(np.log2(np.maximum(mx,2.0**-40)/16319)).astype(np.int64)
    s=np.exp2(exponents.astype(np.float64))
    q=np.rint(x.astype(np.float64)/s[:,None]).astype(np.int64)
    assert np.max(np.abs(q))<=16319
    h=(q+64)//128
    l=q-128*h
    assert h.min()>=-127 and h.max()<=127 and l.min()>=-64 and l.max()<=63
    assert np.array_equal(128*h+l,q)
    z=q.astype(np.float64)*s[:,None]
    assert np.array_equal(z.astype(np.float32).astype(np.float64),z)
    zi=(z*2.0**40).astype(np.int64)
    assert np.array_equal(zi.astype(np.float64),z*2.0**40)
    residual=xi-zi
    maxres=int(np.max(np.abs(residual)))
    assert 512*maxres*maxres<2**63
    r2=np.sum(residual*residual,axis=1,dtype=np.int64)
    qn=np.sum(q*q,axis=1,dtype=np.int64)
    norm2=qn.astype(np.float64)*s*s
    assert np.all((norm2==0)|(norm2>=np.finfo(np.float64).tiny))
    assert np.isfinite(norm2).all()
    # Each scaled integer norm is exactly representable; verify rational identity.
    for v,qn_i,ei in zip(norm2,qn,exponents):
        a,b=float(v).as_integer_ratio()
        e=2*int(ei)
        assert (a==int(qn_i)*b*(1<<e)) if e>=0 else (a*(1<<(-e))==int(qn_i)*b)
    hup=np.array([sqrt_upper(*float(v).as_integer_ratio()) for v in norm2],dtype=np.float64)
    rup=np.array([sqrt_upper(int(v),1<<80) for v in r2],dtype=np.float64)
    return dict(h=h.astype(np.int8),l=l.astype(np.int8),q=q,s=s,norm2=norm2,
                hup=hup,rup=rup,xi=xi,host_seconds=time.perf_counter()-started,
                audit=dict(q_max=int(np.max(np.abs(q))),h_min=int(h.min()),h_max=int(h.max()),
                    l_min=int(l.min()),l_max=int(l.max()),scale_exponent_min=int(exponents.min()),
                    scale_exponent_max=int(exponents.max()),max_residual_grid_integer=maxres,
                    max_residual_norm=float(rup.max()),exact_grid_and_norm_checks=True))


def pack(ids,n,diagonal):
    started=time.perf_counter()
    count=len(ids)
    i,j=ids//n,ids%n
    if diagonal:
        groups=(count+15)//16
        rows=np.zeros(groups*16,dtype=np.int32)
        cols=np.zeros(groups*16,dtype=np.int32)
        rows[:count],cols[:count]=i,j
        slots=np.full((groups,256),-1,dtype=np.int32)
        p=np.arange(count)
        slots[p//16,(p%16)*17]=p
        rows,cols=rows.reshape(groups,16),cols.reshape(groups,16)
    else:
        assert n%16==0
        tile=(i//16)*(n//16)+j//16
        unique,inv=np.unique(tile,return_inverse=True)
        groups=len(unique)
        rows=((unique//(n//16))[:,None]*16+np.arange(16)).astype(np.int32)
        cols=((unique%(n//16))[:,None]*16+np.arange(16)).astype(np.int32)
        slots=np.full((groups,256),-1,dtype=np.int32)
        slots[inv,(i%16)*16+j%16]=np.arange(count)
    assert np.array_equal(np.sort(slots[slots>=0]),np.arange(count))
    return dict(rows=rows,cols=cols,slots=slots,groups=groups,
                padded_products=groups*256,padding=groups*256/count,
                host_seconds=time.perf_counter()-started)


def dot_reference(q,ids,n):
    result=np.empty(len(ids),dtype=np.int64)
    for b in range(0,len(ids),512):
        p=ids[b:b+512]
        result[b:b+len(p)]=np.sum(q[p//n]*q[p%n],axis=1,dtype=np.int64)
    return result


def exact_truth_grid(xi,ids,n,threshold):
    """Exact dyadic CPU check for small high-entropy / boundary fixtures."""
    a,b=float(threshold).as_integer_ratio()
    out=[]
    for p in ids:
        delta=xi[int(p)//n]-xi[int(p)%n]
        value=sum(int(v)*int(v) for v in delta)
        out.append(int(value*b <= a*(1<<80)))
    return np.array(out,dtype=np.uint8)
