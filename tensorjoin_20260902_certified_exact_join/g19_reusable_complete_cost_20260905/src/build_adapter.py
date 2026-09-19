"""Generate a separate reusable RT-HiSS artifact; never modify G18."""
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

H=Path(__file__).resolve().parents[1];P=H.parent
old=P/'g18_rthiss_conservative_repair_20260905/adapter_a0';new=H/'adapter_a0'
assert not new.exists()
shutil.copytree(old,new,symlinks=True)

def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new,1)

def quiet(text):
    # Remove complete standalone diagnostic print statements, preserving CUDA
    # checks and branches. Calls in commented lines are not matched.
    pattern=re.compile(r'^[ \t]*(?:printf|fprintf|puts)\s*\(',re.M)
    while (match:=pattern.search(text)):
        i=match.end();depth=1;quote=None;escaped=False
        while depth:
            ch=text[i];i+=1
            if quote:
                if escaped:escaped=False
                elif ch=='\\':escaped=True
                elif ch==quote:quote=None
            elif ch in '\"\'':quote=ch
            elif ch=='(':depth+=1
            elif ch==')':depth-=1
        while text[i].isspace():i+=1
        assert text[i]==';';i+=1
        text=text[:match.start()]+'    (void)0;'+text[i:]
    text=re.sub(r'^[ \t]*std::cout\s*<<.*?;', '    (void)0;',text,flags=re.M|re.S)
    return text

host=(old/'hostCode.cpp').read_text()
head=host[:host.index('int main(')]
body=host[host.index('    if (MAX_KD_LEVELS <= 0)'):host.index('    unsigned long long g18Stats[6]')]
body=replace(body,'    OWLContext context;\n    OWLModule module;','    OWLContext context=g19_engine.context;\n    OWLModule module=g19_engine.module;')
body=replace(body,'    setupModuleAndContext(deviceCode_ptx, context, module);','    // Data-independent context/module/programs were created by g19Create.')
start=body.index('    g17::require(std::getenv("G18_T")')
end=body.index('    g17::require(g17::dimension_map.size()',start)
body=body[:start]+body[end:]
body=replace(body,'g18Configure(g18Inverse.data(),g18T,g18Low,g18High);','g18Configure(g18Inverse.data(),threshold,low,high);')
body=replace(body,'    OWLGeom userSpheresGeom;\n    OWLRayGen rayGen;\n    OWLParams subPartition_lp;',
             '    OWLGeom userSpheresGeom=g19_engine.geom;\n    OWLRayGen rayGen=g19_engine.raygen;\n    OWLParams subPartition_lp=g19_engine.params;')
body=replace(body,'    setupVarsForShaderPrograms(context, module, userSpheresGeom, rayGen, subPartition_lp);','    // Fixed shader parameter objects are reused; buffers are bound anew.')
body=replace(body,'\tstatic double workGenTime = 0;','\tdouble workGenTime = 0;')
body=quiet(body)
engine=r'''
struct G19Engine {
    OWLContext context=nullptr; OWLModule module=nullptr;
    OWLGeom geom=nullptr; OWLRayGen raygen=nullptr; OWLParams params=nullptr;
    bool running=false;
};
static G19Engine g19_engine;
static std::string g19_error;
extern "C" const char* g19Error(){return g19_error.c_str();}
extern "C" int g19Diagnostics(){return G19_DIAGNOSTICS;}
extern "C" int g19Create() {
    try {
        g17::require(!g19_engine.context,"G19 singleton engine already exists");
        setupModuleAndContext(deviceCode_ptx,g19_engine.context,g19_engine.module);
        setupVarsForShaderPrograms(g19_engine.context,g19_engine.module,g19_engine.geom,g19_engine.raygen,g19_engine.params);
        owlBuildPrograms(g19_engine.context);owlBuildPipeline(g19_engine.context);
        cudaErrorCheck(cudaDeviceSynchronize(),"G19 engine initialization");
        return 0;
    }catch(const std::exception& e){g19_error=e.what();return 1;}
}
extern "C" int g19Destroy() {
    try {
        g17::require(g19_engine.context && !g19_engine.running,"G19 destroy invalid state");
        cudaErrorCheck(cudaDeviceSynchronize(),"G19 engine final synchronization");
        owlContextDestroy(g19_engine.context);g19_engine=G19Engine{};
        g17::release_input_state();return 0;
    }catch(const std::exception& e){g19_error=e.what();return 1;}
}
extern "C" const uint64_t* g19ResultData(void* result) {
    return static_cast<std::vector<uint64_t>*>(result)->data();
}
extern "C" uint64_t g19ResultCount(void* result) {
    return static_cast<std::vector<uint64_t>*>(result)->size();
}
extern "C" void g19ResultRelease(void* result) {delete static_cast<std::vector<uint64_t>*>(result);}

extern "C" int g19Run(const float* input,uint32_t N,double threshold,float EPSILON,
                      float low,float high,const char* directory,void** result) {
    try {
        g17::require(g19_engine.context && !g19_engine.running,"G19 unavailable/concurrent engine");
        g17::require(input && result && std::isfinite(threshold) && threshold>=0 && threshold<=2048
            && std::isfinite(EPSILON) && EPSILON>0 && std::isfinite(low) && std::isfinite(high) && low<=high,"G19 invalid call");
        g19_engine.running=true;*result=nullptr;
        const unsigned DIM=DATASET_DIM;
        ReorderMode reorderMode=REORDER_HIGHEST_VARIANCE;
        THRESHOLD=0.01*EPSILON;MAX_KD_LEVELS=MAX_KD_LEVELS_OPT;
        cudaOnlyTime=timeToCopyResults=timeToCompress=kdTreeConstructionTime=0;
        rCall=urCall=0;numPointComparisons=0;
        float* dataset=g17::begin_input(input,N,EPSILON,directory);
'''
tail=r'''
#if G19_DIAGNOSTICS
        unsigned long long counters[6];g18ReadStats(counters);
        g17::save("g18_stats.u64",std::vector<unsigned long long>(counters,counters+6));
#endif
        g17::finish(sampledCount);
        // All consumers are complete before releasing input-specific references.
        cudaErrorCheck(cudaDeviceSynchronize(),"G19 operation cleanup boundary");
        owlRayGenSetGroup(rayGen,"handle",nullptr);
        owlGeomSetBuffer(userSpheresGeom,"prims",nullptr);
        for(const char* name : {"candidatePoints","spheres","totalCandidateCount","queryIndices",
             "intersectionCount","primitiveIntersectionCount","primitivePosTowrite"})
            owlParamsSetBuffer(subPartition_lp,name,nullptr);
        owlGroupRelease(handle);owlGroupRelease(spheresGroups);
        for(OWLBuffer buffer : {queryIndices,OWL_datasetReordered,owl_groupCountBuffer,
             dataPointsSpheresBuffer,sampledPointsToSearchSpheresBuffer,owl_groupedPointsBuffer,
             sampledCandidateCount,owl_intersectionCountBuffer,owl_primitiveIntersectionCount,
             owl_primitivePosToWrite,owl_candidatePointsBuffer,owl_totalCandidateCount})
            owlBufferRelease(buffer);
        cudaErrorCheck(cudaFree(d_primitiveOrder),"G19 primitive order cleanup");
#if !USE_PAGEABLE_MEMORY
        cudaErrorCheck(cudaFreeHost(h_pinnedResultMask),"G19 pinned staging cleanup");
#else
        free(h_pinnedResultMask);
#endif
        free(dataset);
        *result=new std::vector<uint64_t>(std::move(g17::output));
        g17::release_input_state();g19_engine.running=false;return 0;
    }catch(const std::exception& e){g19_error=e.what();return 1;}
}
'''
(new/'hostCode.cpp').write_text(head+engine+body+tail)
(new/'g17_export.h').write_text((H/'src/g19_export.h').read_text())
seq=(old/'utility_seq.cpp').read_text()
seq=replace(seq,'    owlInstanceGroupSetVisibilityMasks(handle, visibilityMasks);',
            '    owlInstanceGroupSetVisibilityMasks(handle, visibilityMasks);\n    delete[] visibilityMasks;')
seq=replace(seq,'    owlBuildPrograms(context);\n    owlBuildPipeline(context);\n    owlBuildSBT(context);',
            '    // Programs/pipeline are fixed for the engine; geometry/SBT is per call.\n    owlBuildSBT(context);')
seq=replace(seq,'    static double dataBufferTime = 0;','    double dataBufferTime = 0;')
seq=replace(seq,'    double dataBufferTimeEnd = omp_get_wtime();','    free(datasetReordered);\n    double dataBufferTimeEnd = omp_get_wtime();')
(new/'utility_seq.cpp').write_text(quiet(seq))
cu=(old/'utility.cu').read_text()
cu=replace(cu,'__device__ unsigned long long g18_stats[6];','#if G19_DIAGNOSTICS\n__device__ unsigned long long g18_stats[6];\n#endif')
cu=replace(cu,'    unsigned long long zeros[6]={};\n    cudaErrorCheck(cudaMemcpyToSymbol(g18_stats,zeros,sizeof(zeros)),"G18 counter reset");',
            '#if G19_DIAGNOSTICS\n    unsigned long long zeros[6]={};\n    cudaErrorCheck(cudaMemcpyToSymbol(g18_stats,zeros,sizeof(zeros)),"G18 counter reset");\n#endif')
cu=replace(cu,'    cudaErrorCheck(cudaMemcpyFromSymbol(counters,g18_stats,6*sizeof(unsigned long long)),"G18 counter read");',
            '#if G19_DIAGNOSTICS\n    cudaErrorCheck(cudaMemcpyFromSymbol(counters,g18_stats,6*sizeof(unsigned long long)),"G18 counter read");\n#else\n    std::fill(counters,counters+6,0ull);\n#endif')
start='    atomicAdd(&g18_stats[r.stage],1ull);'
end='    if(r.stage>=2)atomicAdd(&g18_stats[5],static_cast<unsigned long long>(DIM));'
cu=replace(cu,start,'#if G19_DIAGNOSTICS\n'+start);cu=replace(cu,end,end+'\n#endif')
# This host allocation has no reader in the frozen compressed-output path.
cu=replace(cu,'    uint32_t*h_resultmask = (uint32_t*)malloc(sizeof(uint32_t)* numTotalResultWords);','    uint32_t*h_resultmask = nullptr; // No uncompressed host output is materialized.')
(new/'utility.cu').write_text(quiet(cu))
cm=(old/'CMakeLists.txt').read_text()
cm=replace(cm,'project(RT-HiSS)','project(RT-HiSS)\nset(CMAKE_POSITION_INDEPENDENT_CODE ON)\noption(G19_DIAGNOSTICS "Keep G18 diagnostic counters and exports" OFF)')
cm=replace(cm,'add_executable(RT-HiSS hostCode.cpp utility.cu utility_seq.cpp utility.h)',
            'add_library(RT-HiSS SHARED hostCode.cpp utility.cu utility_seq.cpp utility.h)')
cm=replace(cm,'        DATASET_DIM=${DIM}','        DATASET_DIM=${DIM}\n        G19_DIAGNOSTICS=$<BOOL:${G19_DIAGNOSTICS}>')
(new/'CMakeLists.txt').write_text(cm)
art=H/'artifacts';art.mkdir(exist_ok=True)
changes={}
for name in ['hostCode.cpp','utility_seq.cpp','utility.cu','g17_export.h','CMakeLists.txt']:
    a=(old/name).read_text();b=(new/name).read_text()
    (art/(name+'.diff')).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='G18/'+name,tofile='G19/'+name)))
    changes[name]=dict(before_sha256=hashlib.sha256(a.encode()).hexdigest(),after_sha256=hashlib.sha256(b.encode()).hexdigest())
(art/'adapter_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
print('GENERATED',new)
