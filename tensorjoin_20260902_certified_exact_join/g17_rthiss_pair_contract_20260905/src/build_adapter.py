"""Apply exact host-only export changes to a separate pinned-source copy."""
from pathlib import Path
import hashlib,json,difflib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parents[1]
OLD=ROOT/'adapters/rthiss_g7_a0';NEW=HERE/'adapter_a0'
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
changes={}
for name in ['hostCode.cpp','utility_seq.cpp','utility.cu']:
    original=(OLD/name).read_text();s=original
    s=replace(s,'#include "utility.h"','#include "utility.h"\n#include "g17_export.h"')
    if name=='hostCode.cpp':
        s=replace(s,'    char* fileName = argv[1];','    g17::require(argc>=3,"G17 input and epsilon required");\n    char* fileName = argv[1];')
        s=replace(s,'float* dataset = readData(fileName, &N, &DIM);','float* dataset = g17::read_input(fileName, &N, &DIM, EPSILON);')
        s=replace(s,'uint64_t * h_refinedCandidatePoints;\n    uint64_t * d_refinedCandidatePoints;', 'uint64_t * h_refinedCandidatePoints = nullptr;\n    uint64_t * d_refinedCandidatePoints = nullptr;')
        s=replace(s,'    uint32_t intersectionsPerBatch = totalNodeIntersections;', '    g17::require(totalNodeIntersections>0 && totalNodeIntersections<UINT32_MAX,"G17 intersection count not admitted");\n    uint32_t intersectionsPerBatch = totalNodeIntersections;')
        s=replace(s,'        // Need to do a RT run again to find primitive pos to write','        g17::require(curIntersectionCount<=intersectionsPerBatch,"G17 batch capacity exceeded");\n        // Need to do a RT run again to find primitive pos to write')
        s=replace(s,'    double allBatchEnd = omp_get_wtime();','    g17::finish(sampledCount);\n    double allBatchEnd = omp_get_wtime();')
    elif name=='utility_seq.cpp':
        s=replace(s,'#pragma omp parallel for\n    for (int i = 0; i < N; i++)','    for (const auto& entry:devDimPair) g17::dimension_map.push_back(entry.second);\n#pragma omp parallel for\n    for (int i = 0; i < N; i++)')
        s=replace(s,'            uint32_t curPoint = sphereData.pID;','            uint32_t curPoint = sphereData.pID;\n            g17::point_map[pointIdx]=curPoint;')
        s=replace(s,'    pointsToSearchSpheres = buildSpheresPrimBuffer(pointsToSearch, datasetReordered, DIM, EPSILON);', '    g17::verify_reordering(datasetReordered);\n    pointsToSearchSpheres = buildSpheresPrimBuffer(pointsToSearch, datasetReordered, DIM, EPSILON);')
    else:
        s=replace(s,'    const uint64_t numTotalResultWords = resultMaskPrefix[NUM_PRIMITIVES];','    const uint64_t numTotalResultWords = resultMaskPrefix[NUM_PRIMITIVES];\n    g17::require(numTotalResultWords>0 && numTotalResultWords*32<(uint64_t(1)<<32),"G17 mask capacity not admitted");')
        s=replace(s,'            cudaErrorCheck(cudaFree(d_counter), "cudaFree(d_counter)");','            unsigned long long exportedCount=0;\n            cudaErrorCheck(cudaMemcpy(&exportedCount,d_counter,sizeof(exportedCount),cudaMemcpyDeviceToHost),"G17 compressed count");\n            g17::require(exportedCount==*h_refinedCount,"G17 compression count mismatch");\n            cudaErrorCheck(cudaFree(d_counter), "cudaFree(d_counter)");')
        s=replace(s,'            fprintf(stderr, "\\nCopying uncompressed mask");','            g17::require(false,"G17 admits only default compressed output");\n            fprintf(stderr, "\\nCopying uncompressed mask");')
        s=replace(s,'            free(h_compressedResultMask);','            g17::export_batch(h_compressedResultMask,*h_refinedCount,d_resultMask,resultMaskPrefix,h_groupCount,NUM_PRIMITIVES,h_primitiveIntersectionCount,d_primitivePosToWrite,d_candidatePoints,N,d_groupedPoints);\n            free(h_compressedResultMask);')
        # Exact device-source invariant, from the first kernel section to the host wrapper.
        a=original.index('__inline__ __device__ bool pointWithinEps');b=original.index('// Host function: launch kernel with blocks split by query workload')
        newa=s.index('__inline__ __device__ bool pointWithinEps');newb=s.index('// Host function: launch kernel with blocks split by query workload')
        assert original[a:b]==s[newa:newb]
        changes['device_region_sha256']=hashlib.sha256(original[a:b].encode()).hexdigest()
    (NEW/name).write_text(s)
    (HERE/'artifacts'/f'{name}.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='G7/'+name,tofile='G17/'+name)))
    changes[name]=dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),adapter_sha256=hashlib.sha256(s.encode()).hexdigest())
(HERE/'artifacts/adapter_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
