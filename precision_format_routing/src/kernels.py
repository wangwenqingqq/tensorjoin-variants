"""Owned format producers and a common tiled dot path for the bounded screen."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from retained import cvup, cvdown


@triton.jit
def metadata(x, q, scales, norm_lo, norm_hi, z_upper, residual_upper, invalid,
             D:tl.constexpr, FORMAT:tl.constexpr):
    row=tl.program_id(0); k=tl.arange(0,D)
    a=tl.load(x+row*D+k); maximum=tl.max(tl.abs(a),0)
    if FORMAT == 'int8':
        scale=libdevice.div_rn(maximum,127.0)
        scale=tl.where(maximum<2.0**-100,1.0,scale)
        code=tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(a,scale)),-127.0),127.0).to(tl.int8)
        h=code.to(tl.float32); tl.store(q+row*D+k,code)
    elif FORMAT == 'fp16' or FORMAT == 'fp32':
        scale=tl.full((),1.0,tl.float32)
        if FORMAT == 'fp16':
            h=a.to(tl.float16).to(tl.float32)
            h=tl.where(tl.abs(h)<2.0**-14,0.0,h)
        else:
            h=tl.where(tl.abs(a)<2.0**-126,0.0,a)
        tl.store(q+row*D+k,h)
    else:
        if FORMAT == 'e3m4':
            MAX:tl.constexpr=30.0
        elif FORMAT == 'e4m3':
            MAX:tl.constexpr=448.0
        else:
            MAX:tl.constexpr=57344.0
        ratio=tl.maximum(maximum/MAX,2.0**-100)
        exp=((ratio.to(tl.uint32,bitcast=True)>>23)&255).to(tl.int32)-127
        scale=((exp+127).to(tl.uint32)<<23).to(tl.float32,bitcast=True)
        scale=tl.where(maximum>scale*MAX,scale*2.0,scale)
        scale=tl.where(maximum<2.0**-100,1.0,scale)
        v=libdevice.div_rn(a,scale)
        if FORMAT == 'e3m4':
            av=tl.abs(v)
            e=((av.to(tl.uint32,bitcast=True)>>23)&255).to(tl.int32)-127
            e=tl.maximum(e,-2)
            step=((e-4+127).to(tl.uint32)<<23).to(tl.float32,bitcast=True)
            h=tl.minimum(libdevice.rint(av/step)*step,30.0)
            he=((h.to(tl.uint32,bitcast=True)>>23)&255).to(tl.int32)-127
            he=tl.maximum(he,-2)
            inv=((127-he).to(tl.uint32)<<23).to(tl.float32,bitcast=True)
            mag=tl.where(h<0.25,libdevice.rint(h*64).to(tl.int32),
                         ((he+3)<<4)+libdevice.rint((h*inv-1.0)*16.0).to(tl.int32))
            sign=(v.to(tl.uint32,bitcast=True)>>24)&128
            code=(mag.to(tl.uint32)|sign).to(tl.uint8)
            h=tl.where(sign!=0,-h,h); tl.store(q+row*D+k,code)
        elif FORMAT == 'e4m3':
            encoded=v.to(tl.float8e4nv); h=encoded.to(tl.float32)
            tl.store(q+row*D+k,encoded.to(tl.uint8,bitcast=True))
        else:
            encoded=v.to(tl.float8e5); h=encoded.to(tl.float32)
            tl.store(q+row*D+k,encoded.to(tl.uint8,bitcast=True))
    tl.store(scales+row,scale)
    aa=a.to(tl.float64); hh=h.to(tl.float64)*scale.to(tl.float64); r=aa-hh
    nx=tl.sum(aa*aa,0); nz=tl.sum(hh*hh,0); nr=tl.sum(r*r,0)
    gamma=tl.full((),(512*2.0**-52)/(1-512*2.0**-52),tl.float64)
    tiny=tl.full((),1024.0*2.0**-1022,tl.float64)
    rx=gamma*nx+tiny; rz=gamma*nz+tiny; rr=gamma*nr+tiny
    padding=tl.full((),1.0+8.0*2.0**-52,tl.float64)
    zu=libdevice.nextafter(libdevice.sqrt_rn(nz+rz)*padding,tl.full((),float('inf'),tl.float64))
    eu=libdevice.nextafter(libdevice.sqrt_rn(nr+rr)*padding,tl.full((),float('inf'),tl.float64))
    tl.store(norm_lo+row,tl.maximum(cvdown(nx-rx),0.0)); tl.store(norm_hi+row,cvup(nx+rx))
    tl.store(z_upper+row,cvup(zu)); tl.store(residual_upper+row,cvup(eu))
    bad=tl.sum((~((a==a)&(tl.abs(a)<=1.0))).to(tl.int32),0)>0
    tl.atomic_or(invalid,1,mask=bad,sem='relaxed')


@triton.jit
def dot(q, scales, output, N:tl.constexpr, D:tl.constexpr,
        FORMAT:tl.constexpr, FORMAT_B:tl.constexpr,
        BM:tl.constexpr=32, BN:tl.constexpr=32, BK:tl.constexpr=64):
    i=tl.program_id(0)*BM+tl.arange(0,BM)
    j=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    if FORMAT == 'int8':
        acc=tl.zeros((BM,BN),tl.int32)
    else:
        acc=tl.zeros((BM,BN),tl.float32)
    for base in range(0,D,BK):
        a=tl.load(q+i[:,None]*D+(base+k[None,:]),mask=i[:,None]<N,other=0)
        b=tl.load(q+j[None,:]*D+(base+k[:,None]),mask=j[None,:]<N,other=0)
        if FORMAT == 'e3m4' or FORMAT == 'e4m3':
            a=a.to(tl.float8e4nv,bitcast=True)
        elif FORMAT == 'e5m2':
            a=a.to(tl.float8e5,bitcast=True)
        if FORMAT_B == 'e3m4' or FORMAT_B == 'e4m3':
            b=b.to(tl.float8e4nv,bitcast=True)
        elif FORMAT_B == 'e5m2':
            b=b.to(tl.float8e5,bitcast=True)
        if FORMAT == 'int8':
            acc += tl.dot(a,b,out_dtype=tl.int32)
        else:
            acc += tl.dot(a,b,input_precision='ieee',out_dtype=tl.float32,max_num_imprecise_acc=0)
    si=tl.load(scales+i,mask=i<N,other=1.0)
    sj=tl.load(scales+j,mask=j<N,other=1.0)
    value=acc.to(tl.float32)*si[:,None]*sj[None,:]
    tl.store(output+i[:,None]*N+j[None,:],value,mask=(i[:,None]<N)&(j[None,:]<N))


# Retained terminal arithmetic, with an explicitly binary64 threshold ABI.
@triton.jit
def terminal_fp64(
    query,
    base,
    fp64_ids,
    result_ids,
    counters,
    threshold_d2: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(fp64_ids + pair_index)
    query_row = pair_id // N_
    base_row = pair_id - query_row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_values = tl.load(query + query_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        base_values = tl.load(base + base_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        delta = query_values - base_values
        distance_d2 += tl.sum(delta * delta, axis=0)
    threshold64 = tl.full((), threshold_d2, tl.float64)
    inside = distance_d2 <= threshold64
    counter_lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(counters + counter_lane, 1, mask=inside, sem="relaxed")
    can_store = inside & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )

