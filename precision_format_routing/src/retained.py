"""Unmodified numerical kernels retained from the certified TensorJoin source."""
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


@triton.jit
def certified_fp32_filter_i64(
    vectors,
    ambiguous_ids,
    result_ids,
    fp64_ids,
    counters,
    threshold_lower,
    threshold_upper,
    distance_relative_radius,
    input_magnitude_radius,
    final_relative_radius,
    absolute_radius,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(ambiguous_ids + pair_index)
    row = pair_id // N_
    column = pair_id - row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float32)
    input_magnitude_d2 = tl.zeros((1,), dtype=tl.float32)
    different_coordinates = tl.zeros((1,), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        x = tl.load(vectors + row * K + k, mask=k < K, other=0.0)
        y = tl.load(vectors + column * K + k, mask=k < K, other=0.0)
        delta = x - y
        input_magnitude = tl.abs(x) + tl.abs(y)
        distance_d2 += tl.sum(delta * delta, axis=0)
        input_magnitude_d2 += tl.sum(input_magnitude * input_magnitude, axis=0)
        different_coordinates += tl.sum((x != y).to(tl.int32), axis=0)

    radius = (
        distance_relative_radius * tl.abs(distance_d2)
        + input_magnitude_radius * tl.abs(input_magnitude_d2)
        + absolute_radius
    )
    final_magnitude = tl.abs(distance_d2) + radius
    radius += final_relative_radius * final_magnitude + absolute_radius
    lower = tl.maximum(distance_d2 - radius, 0.0)
    upper = distance_d2 + radius
    bitwise_equal = different_coordinates == 0
    accept = bitwise_equal | (upper <= threshold_lower)
    reject = (~bitwise_equal) & (lower > threshold_upper)
    needs_fp64 = ~(accept | reject)
    counter_lane = tl.zeros((1,), dtype=tl.int32)

    position = tl.atomic_add(counters + counter_lane, 1, mask=accept, sem="relaxed")
    can_store = accept & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=accept & ~can_store,
        sem="relaxed",
    )
    fp64_position = tl.atomic_add(
        counters + 3 + counter_lane, 1, mask=needs_fp64, sem="relaxed"
    )
    fp64_can_store = needs_fp64 & (fp64_position < CAPACITY)
    tl.store(fp64_ids + fp64_position, pair_id, mask=fp64_can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=needs_fp64 & ~fp64_can_store,
        sem="relaxed",
    )
    tl.atomic_add(counters + 4 + counter_lane, 1, mask=reject, sem="relaxed")
    tl.atomic_add(counters + 5 + counter_lane, 1, mask=bitwise_equal, sem="relaxed")


@triton.jit
def refine_ambiguous_fp64_i64(
    query,
    base,
    fp64_ids,
    result_ids,
    counters,
    threshold_d2,
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
    inside = distance_d2 <= threshold_d2
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


