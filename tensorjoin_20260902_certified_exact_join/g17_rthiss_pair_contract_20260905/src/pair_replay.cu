#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
__inline__ __device__ bool pointWithinEps(const float* query, const float* candidate, uint32_t pointsInBatch, uint32_t DIM, float EPSILON_SQ)
{
    uint32_t d = 0;
    float distance = 0.0f;

    // candidate coordinate for dim d is candidate[d * pointsInBatch]
    for (; d + 3 < DIM; d += 4)
    {
        float dx = query[d + 0] - candidate[(d + 0) * pointsInBatch];
        distance = fmaf(dx, dx, distance);
        dx = query[d + 1] - candidate[(d + 1) * pointsInBatch];
        distance = fmaf(dx, dx, distance);
        dx = query[d + 2] - candidate[(d + 2) * pointsInBatch];
        distance = fmaf(dx, dx, distance);
        dx = query[d + 3] - candidate[(d + 3) * pointsInBatch];
        distance = fmaf(dx, dx, distance);
        if (distance > EPSILON_SQ) return false;
    }

    for (; d < DIM; ++d)
    {
        float dx = query[d] - candidate[d * pointsInBatch];
        distance = fmaf(dx, dx, distance);
        if (distance > EPSILON_SQ) return false;
    }

    return true;
}




struct Result { float native_sum; double fp64_sum; int native_decision; int reference_decision; };
__global__ void replay(const float* x,const uint32_t* inverse,Result* result,float nativeT,double referenceT) {
    unsigned pair=threadIdx.x;if(pair>=4)return;
    const float* a=x+pair*1024;const float* b=a+512;
    bool native=pointWithinEps(a,b,1,512,nativeT);
    float native_sum=0;
    for(unsigned d=0;d<512;++d) {float delta=a[d]-b[d];native_sum=fmaf(delta,delta,native_sum);}
    double sum=0;
    for(unsigned d=0;d<512;++d) {unsigned j=inverse[d];double delta=__dsub_rn(double(a[j]),double(b[j]));sum=__dadd_rn(sum,__dmul_rn(delta,delta));}
    result[pair]={native_sum,sum,int(native),int(sum<=referenceT)};
}
int main(int argc,char** argv) {
    if(argc!=5)return 3;
    std::vector<float>x(4*1024);std::vector<uint32_t> inverse(512);
    std::ifstream a(argv[1],std::ios::binary),b(argv[2],std::ios::binary);
    a.read(reinterpret_cast<char*>(x.data()),x.size()*4);b.read(reinterpret_cast<char*>(inverse.data()),512*4);
    if(!a||!b)return 4;
    float* dx;uint32_t* di;Result* dr;
    if(cudaMalloc(&dx,x.size()*4)!=cudaSuccess||cudaMalloc(&di,512*4)!=cudaSuccess||cudaMalloc(&dr,4*sizeof(Result))!=cudaSuccess)return 5;
    cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(di,inverse.data(),512*4,cudaMemcpyHostToDevice);
    replay<<<1,32>>>(dx,di,dr,atof(argv[3]),atof(argv[4]));
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 6;
    Result r[4];if(cudaMemcpy(r,dr,sizeof(r),cudaMemcpyDeviceToHost)!=cudaSuccess)return 7;
    printf("[");for(int i=0;i<4;++i)printf("%s{\"native_sum\":%.17g,\"fp64_sum\":%.17g,\"native\":%d,\"reference\":%d}",i?",":"",double(r[i].native_sum),r[i].fp64_sum,r[i].native_decision,r[i].reference_decision);printf("]\n");
    cudaFree(dx);cudaFree(di);cudaFree(dr);return 0;
}
