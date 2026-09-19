// One synchronous warp MMA per record; raw-code broadcast probe, not GEMM.
#include <stdint.h>
#ifndef ATYPE
#define ATYPE "e4m3"
#endif
#ifndef BTYPE
#define BTYPE "e4m3"
#endif

extern "C" __global__ void probe(const uint32_t* in, float* out) {
    const int q = blockIdx.x;
    const uint32_t a = in[q * 4], b = in[q * 4 + 1];
    float d0, d1, d2, d3;
#ifdef SCALED
    const uint32_t sa = in[q * 4 + 2], sb = in[q * 4 + 3];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4."
        "block_scale.scale_vec::1X.f32." ATYPE "." BTYPE ".f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%9,%9,%9,%9}, "
        "%6, {%8,%8}, %7, {%8,%8};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a), "r"(b), "r"(sa), "r"(sb), "h"(uint16_t(0)), "f"(0.0f));
#else
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32."
        ATYPE "." BTYPE ".f32 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {0.0,0.0,0.0,0.0};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a), "r"(b));
#endif
    const int p = q * 128 + threadIdx.x * 4;
    out[p] = d0; out[p + 1] = d1; out[p + 2] = d2; out[p + 3] = d3;
}
