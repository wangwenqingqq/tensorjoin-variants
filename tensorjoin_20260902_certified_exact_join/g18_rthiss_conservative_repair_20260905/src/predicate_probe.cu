#include "g18_predicate.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(8);} } while(0)
struct Result { double reference,terminal;float prefix;uint32_t dims,stage,agree; };
static_assert(sizeof(Result)==32);
__global__ void probe(const float* q,const float* b,const float* low,const float* high,const double* t,
    const uint32_t* inverse,Result* r,unsigned n,unsigned stride) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const float* a=q+size_t(i)*512;const float* c=b+size_t(i)*512*stride;
 G18Decision v=g18Decide(a,c,stride,512,low[i],high[i],t[i],inverse);
 double ref=0;
 for(unsigned j=0;j<512;++j){unsigned dim=inverse[j];double delta=__dsub_rn(double(a[dim]),double(c[dim*stride]));ref=__dadd_rn(ref,__dmul_rn(delta,delta));}
 r[i]={ref,v.terminal,v.prefix,v.dims,v.stage,unsigned(g18Accept(v.stage)==(ref<=t[i]))};
}
template<class T> std::vector<T> read(std::string path,size_t count){std::vector<T>x(count);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(x.data()),count*sizeof(T));if(!f)std::exit(5);return x;}
template<class T> T* upload(const std::vector<T>&x){T*p;CK(cudaMalloc(&p,x.size()*sizeof(T)));CK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){
 if(argc!=5)return 3;std::string root=argv[1];unsigned n=std::stoul(argv[2]),iterations=std::stoul(argv[3]);
 auto q=read<float>(root+"/q.f32",size_t(n)*512),b=read<float>(root+"/b.f32",size_t(n)*512);
 auto low=read<float>(root+"/low.f32",n),high=read<float>(root+"/high.f32",n);auto t=read<double>(root+"/threshold.f64",n);auto inv=read<uint32_t>(root+"/inverse.u32",512);
 std::vector<float> strided(b.size()*3,-0.8125f);for(size_t i=0;i<b.size();++i)strided[i*3]=b[i];
 float* dq[2]={upload(q),upload(q)};float* db[2]={upload(b),upload(strided)};
 float* dl=upload(low);float* dh=upload(high);double* dt=upload(t);uint32_t* di=upload(inv);
 Result* dr[2];for(auto&v:dr)CK(cudaMalloc(&v,n*sizeof(Result)));
 std::vector<Result> expected(n),current(n);
 for(unsigned run=0;run<iterations;++run){unsigned slot=run%2;CK(cudaMemset(dr[slot],0xa5,n*sizeof(Result)));
  probe<<<(n+127)/128,128>>>(dq[slot],db[slot],dl,dh,dt,di,dr[slot],n,slot?3:1);
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(current.data(),dr[slot],n*sizeof(Result),cudaMemcpyDeviceToHost));
  for(auto&v:current)if(!v.agree){fprintf(stderr,"decision mismatch iteration %u\n",run);return 9;}
  if(!run)expected=current;else if(memcmp(expected.data(),current.data(),n*sizeof(Result))){fprintf(stderr,"repeat mismatch %u\n",run);return 10;}
 }
 std::ofstream f(argv[4],std::ios::binary);f.write(reinterpret_cast<const char*>(expected.data()),expected.size()*sizeof(Result));if(!f)return 6;
 for(auto p:dq)CK(cudaFree(p));for(auto p:db)CK(cudaFree(p));for(auto p:dr)CK(cudaFree(p));CK(cudaFree(dl));CK(cudaFree(dh));CK(cudaFree(dt));CK(cudaFree(di));
 printf("G18_PROBE_COMPLETE n=%u iterations=%u alternating_strides=1,3\n",n,iterations);return 0;
}
