#pragma once
#include <cuda_runtime.h>
#include <cstdint>
struct G18Decision { unsigned stage,dims; float prefix; double terminal; };
// Stage 0: safe reject; 1: safe accept; 2/3: terminal accept/reject.
__device__ __forceinline__ bool g18Accept(unsigned stage) { return stage==1 || stage==2; }
__device__ __forceinline__ G18Decision g18Decide(const float* q,const float* c,unsigned stride,
    unsigned dim,float low,float high,double threshold,const uint32_t* inverse) {
    float sum=0;unsigned d=0;
    for(;d+3<dim;d+=4) {
        float delta=__fsub_rn(q[d],c[d*stride]);sum=__fmaf_rn(delta,delta,sum);
        delta=__fsub_rn(q[d+1],c[(d+1)*stride]);sum=__fmaf_rn(delta,delta,sum);
        delta=__fsub_rn(q[d+2],c[(d+2)*stride]);sum=__fmaf_rn(delta,delta,sum);
        delta=__fsub_rn(q[d+3],c[(d+3)*stride]);sum=__fmaf_rn(delta,delta,sum);
        if(sum>high)return {0,d+4,sum,0};
    }
    for(;d<dim;++d) {
        float delta=__fsub_rn(q[d],c[d*stride]);sum=__fmaf_rn(delta,delta,sum);
        if(sum>high)return {0,d+1,sum,0};
    }
    if(sum<=low)return {1,dim,sum,0};
    double reference=0;
    for(unsigned original=0;original<dim;++original) {
        unsigned j=inverse[original];
        double delta=__dsub_rn(double(q[j]),double(c[j*stride]));
        reference=__dadd_rn(reference,__dmul_rn(delta,delta));
    }
    return {unsigned(reference<=threshold?2:3),dim,sum,reference};
}
