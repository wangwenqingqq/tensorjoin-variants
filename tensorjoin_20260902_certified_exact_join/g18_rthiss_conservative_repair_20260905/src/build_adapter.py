"""Create a separate conservative-predicate adapter without editing native G17."""
import difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;old=P/'g17_rthiss_pair_contract_20260905/adapter_a0';out=H/'adapter_a0'
assert not out.exists();out.mkdir()
for p in old.iterdir():
 if p.is_file():shutil.copyfile(p,out/p.name)
(out/'OWL').symlink_to('../../adapters/rthiss_g7_a0/OWL',target_is_directory=True)
shutil.copyfile(H/'src/g18_predicate.cuh',out/'g18_predicate.cuh')
(out/'g18_host.h').write_text('#pragma once\n#include <cstdint>\nvoid g18Configure(const uint32_t*,double,float,float);\nvoid g18ReadStats(unsigned long long*);\n')
p=out/'utility.cu';s=p.read_text();s=s.replace('#include "g17_export.h"','#include "g17_export.h"\n#include "g18_predicate.cuh"\n#include "g18_host.h"',1)
a=s.index('__inline__ __device__ bool pointWithinEps(const float* query, const float* candidate, uint32_t pointsInBatch');b=s.index('// Kernel',a)
replacement='''__constant__ uint32_t g18_inverse[512];
__constant__ float g18_low,g18_high;
__constant__ double g18_threshold;
__device__ unsigned long long g18_stats[6];
void g18Configure(const uint32_t* inverse,double threshold,float low,float high) {
    cudaErrorCheck(cudaMemcpyToSymbol(g18_inverse,inverse,512*sizeof(uint32_t)),"G18 inverse upload");
    cudaErrorCheck(cudaMemcpyToSymbol(g18_threshold,&threshold,sizeof(double)),"G18 threshold upload");
    cudaErrorCheck(cudaMemcpyToSymbol(g18_low,&low,sizeof(float)),"G18 low upload");
    cudaErrorCheck(cudaMemcpyToSymbol(g18_high,&high,sizeof(float)),"G18 high upload");
    unsigned long long zeros[6]={};
    cudaErrorCheck(cudaMemcpyToSymbol(g18_stats,zeros,sizeof(zeros)),"G18 counter reset");
}
void g18ReadStats(unsigned long long* counters) {
    cudaErrorCheck(cudaMemcpyFromSymbol(counters,g18_stats,6*sizeof(unsigned long long)),"G18 counter read");
}
__inline__ __device__ bool pointWithinEps(const float* query,const float* candidate,uint32_t pointsInBatch,uint32_t DIM,float EPSILON_SQ) {
    G18Decision r=g18Decide(query,candidate,pointsInBatch,DIM,g18_low,g18_high,g18_threshold,g18_inverse);
    atomicAdd(&g18_stats[r.stage],1ull);
    atomicAdd(&g18_stats[4],static_cast<unsigned long long>(r.dims));
    if(r.stage>=2)atomicAdd(&g18_stats[5],static_cast<unsigned long long>(DIM));
    return g18Accept(r.stage);
}

'''
s=s[:a]+replacement+s[b:];p.write_text(s)
p=out/'hostCode.cpp';s=p.read_text().replace('#include "g17_export.h"','#include "g17_export.h"\n#include "g18_host.h"',1)
needle='dataset = reorderByDimensions(dataset, N, DIM, reorderMode);';assert s.count(needle)==1
s=s.replace(needle,needle+'''
    g17::require(std::getenv("G18_T") && std::getenv("G18_LOW") && std::getenv("G18_HIGH"),"G18 cutoffs required");
    double g18T=std::stod(std::getenv("G18_T"));
    float g18Low=std::strtof(std::getenv("G18_LOW"),nullptr),g18High=std::strtof(std::getenv("G18_HIGH"),nullptr);
    g17::require(std::isfinite(g18T) && g18T>=0 && g18T<=2048 && std::isfinite(g18Low) && std::isfinite(g18High) && g18Low<=g18High,"G18 invalid threshold");
    g17::require(g17::dimension_map.size()==512,"G18 dimension map missing");
    std::vector<uint32_t> g18Inverse(512);
    for(uint32_t j=0;j<512;++j)g18Inverse[g17::dimension_map[j]]=j;
    g18Configure(g18Inverse.data(),g18T,g18Low,g18High);
''',1)
needle='g17::finish(sampledCount);';assert s.count(needle)==1
s=s.replace(needle,'''unsigned long long g18Stats[6];g18ReadStats(g18Stats);
    g17::save("g18_stats.u64",std::vector<unsigned long long>(g18Stats,g18Stats+6));
    '''+needle,1);p.write_text(s)
p=out/'g17_export.h';s=p.read_text();assert 'source_predicate_unchanged\\\":true' in s
s=s.replace('source_predicate_unchanged\\\":true','source_predicate_unchanged\\\":false,\\\"g18_repaired_refinement\\\":true');p.write_text(s)
p=out/'CMakeLists.txt';s=p.read_text();s+='\n# G18 predicate contract: gradual underflow; explicit rounding intrinsics.\ntarget_compile_options(RT-HiSS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)\n';p.write_text(s)
changes={}
for name in ['utility.cu','hostCode.cpp','g17_export.h','CMakeLists.txt']:
 a=(old/name).read_text();b=(out/name).read_text();(H/'artifacts'/f'{name}.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='g17/'+name,tofile='g18/'+name)))
 changes[name]={'before_sha256':hashlib.sha256(a.encode()).hexdigest(),'after_sha256':hashlib.sha256(b.encode()).hexdigest()}
# Outside the replaced predicate, every original CUDA kernel is identical.
a=(old/'utility.cu').read_text();b=(out/'utility.cu').read_text();key='__global__ void identifyNeighborsGridPrimitiveSharedQueryGlobal('
assert a[a.index(key):]==b[b.index(key):]
changes['cuda_kernel_bodies_after_predicate_unchanged']=True
(H/'artifacts/adapter_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
