#include "utility.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef SAFE_SHARED_MEMORY_RUNTIME

__constant__ int d_safeSharedMemory;

int getSafeSharedMemory()
{
  static int cached = -1;
  if (cached < 0) {
    int device = 0;
    cudaGetDevice(&device);
    int maxSharedMem = 0;
    cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cached = maxSharedMem - 640; 
    cudaMemcpyToSymbol(d_safeSharedMemory, &cached, sizeof(int));
  }
  return cached;
}
#endif

double cudaOnlyTime = 0;
double timeToCopyResults = 0;
double timeToCompress = 0;
double kdTreeConstructionTime = 0;

void parallel_memcpy(void* dst, const void* src, size_t n) {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nt = omp_get_num_threads();
        size_t chunk = n / nt;
        size_t start = id * chunk;
        size_t end   = (id == nt-1) ? n : start + chunk;

        memcpy((char*)dst + start, (char*)src + start, end - start);
    }
}


void test()
{
    const unsigned int numElements = 74145211;
    const size_t bytes = numElements * sizeof(unsigned int);

    unsigned int *h_ptr;
    unsigned int *d_ptr;

    printf("\n\nAllocating %.3lf GiB to copy\n", (double)bytes/1024/1024/1024);

    h_ptr = (unsigned int*)malloc(bytes);
    cudaErrorCheck(cudaMalloc(&d_ptr, bytes), "cudaMalloc(&d_ptr, bytes)");
    cudaErrorCheck(cudaMemset(d_ptr, 0, bytes), "cudaMemset(d_ptr, 0, bytes)");

    double s = omp_get_wtime();
    cudaErrorCheck(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost)");
    double e = omp_get_wtime();

    puts("\n======================================================================\n");
    printf("Total time to copy using pageable memory is %lf seconds", e-s);
     puts("\n======================================================================\n");

    unsigned int*h_pinned_ptr;

    double s1 = omp_get_wtime();
    cudaErrorCheck(cudaMallocHost(&h_pinned_ptr, bytes), "cudaMallocHost(&h_pinned_ptr, bytes)");
    double e1 = omp_get_wtime();

    printf("\n Time to allocate pinned memory is %.3lf", e1-s1);

    s = omp_get_wtime();
    cudaErrorCheck(cudaMemcpy(h_pinned_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(h_pinned_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost)");
    e = omp_get_wtime();

    printf("\n\nTime to copy using pinned memory is %lf seconds", e-s);
     puts("\n======================================================================\n");
    printf("Total time for pinned memory is %lf seconds", (e1-s1) + (e-s));
     puts("\n======================================================================\n");

    unsigned int*h_pinned_staged_ptr;
    unsigned int stagedElements = 1024*1024;
    size_t stagedBytes = sizeof(unsigned int) * stagedElements;

    printf("\n\n Allocating only %.3lf MiB of Pinned memory", (double)stagedBytes/1024/1024);
    
    s = omp_get_wtime();
    cudaErrorCheck(cudaMallocHost(&h_pinned_staged_ptr, stagedBytes), "cudaMallocHost(&h_pinned_staged_ptr, stagedBytes)");
    e = omp_get_wtime();
    
    printf("\n Time to allocate staged pinned memory is %.3lf", e-s);

    s1 = omp_get_wtime();

    size_t left = bytes;
    size_t copiedBytes = 0;
    while(left > 0)
    {   
        size_t curBytes = std::min(left, stagedBytes);
        // unsigned int numElements = curBytes/sizeof(unsigned int);
        size_t elemOffset = copiedBytes/sizeof(unsigned int);
        cudaErrorCheck(cudaMemcpy(h_pinned_staged_ptr, d_ptr + elemOffset, curBytes, cudaMemcpyDeviceToHost), "cudaMemcpy(h_pinned_staged_ptr, d_ptr + elemOffset, curBytes, cudaMemcpyDeviceToHost)");
        unsigned int *h_dest = h_ptr + elemOffset;
        unsigned int*h_src = h_pinned_staged_ptr;
        // #pragma omp parallel for
        // for(unsigned int i = 0 ; i < numElements ; i++)
        // {
        //     h_dest[i] = h_src[i];
        // }
        std::memcpy(h_dest, h_src, curBytes);
        left -= curBytes;
        copiedBytes += curBytes;
    }
    e1 = omp_get_wtime();

    printf("\n\nTime to copy using staged pinned memory is %lf seconds", e1-s1);
    puts("\n======================================================================\n");
    printf("Total time using staged pinned memory is %lf seconds", (e1-s1) + (e - s));
    puts("\n======================================================================\n");



    // const unsigned int THREADS_TO_COPY = 2;
    const unsigned int numIter = bytes/sizeof(unsigned int);
    const unsigned int numIterPerThread = (numIter + THREADS_TO_COPY - 1)/THREADS_TO_COPY;
    size_t stagedElementsAllThreads = stagedElements * THREADS_TO_COPY;
    size_t stagedBytesAllThreads = stagedElementsAllThreads * sizeof(unsigned int);

    unsigned int * h_pinned_allThreads;
    s1 = omp_get_wtime();
    cudaErrorCheck(cudaMallocHost(&h_pinned_allThreads, stagedBytesAllThreads), "cudaMallocHost(&h_pinned_allThreads, stagedBytesAllThreads)");
    e1 = omp_get_wtime();

    printf("\n Time to allocate all threads pinned memory is %.3lf", e1-s1);
    
    s = omp_get_wtime();
    omp_set_num_threads(THREADS_TO_COPY);
     double hostCopyTime = 0;
     int count = 0;
    #pragma omp parallel for schedule(static,1) reduction(+:hostCopyTime)
    for(unsigned int chunk = 0; chunk < THREADS_TO_COPY ; ++chunk)
    {
        unsigned int tid = omp_get_thread_num();
        printf("%d", tid);
        size_t startElem = static_cast<size_t>(chunk) * numIterPerThread;
        if (startElem >= numIter) {
            continue;
        }
        size_t elemsThisChunk = std::min(static_cast<size_t>(numIterPerThread),
                                         static_cast<size_t>(numIter) - startElem);
        size_t remainingBytes = elemsThisChunk * sizeof(unsigned int);
        unsigned int* threadStage = h_pinned_allThreads + static_cast<size_t>(tid) * stagedElements;
        size_t copiedElems = 0;
        while(remainingBytes > 0)
        {   
            size_t curBytes = std::min(remainingBytes, stagedBytes);
            unsigned int curElements = curBytes/sizeof(unsigned int);
            cudaErrorCheck(cudaMemcpy(threadStage, d_ptr + startElem + copiedElems, curBytes, cudaMemcpyDeviceToHost), "cudaMemcpy(threadStage, d_ptr + startElem + copiedElems, curBytes, cudaMemcpyDeviceToHost)");
            // for(unsigned int j = 0 ; j < curElements ; j++)
            // {
            //     h_ptr[startElem + copiedElems + j] = threadStage[j];
            // }
             double e1 = omp_get_wtime();
            std::memcpy(h_ptr+startElem+copiedElems, threadStage, curBytes);
             double e2 = omp_get_wtime();
             hostCopyTime += (e2-e1);
             #pragma omp atomic
             count++;
            remainingBytes -= curBytes;
            copiedElems += curElements;
        }
    }
    e = omp_get_wtime();
    printf("\n H2H copy is %lf seconds and count is %d", hostCopyTime, count);
    puts("\n======================================================================\n");
    printf("Total time using staged pinned memory using %u threads is %lf seconds", THREADS_TO_COPY, (e1-s1) + (e - s));
    puts("\n======================================================================\n");

    cudaErrorCheck(cudaFreeHost(h_pinned_allThreads), "cudaFreeHost(h_pinned_allThreads)");
    cudaErrorCheck(cudaFreeHost(h_pinned_staged_ptr), "cudaFreeHost(h_pinned_staged_ptr)");
    cudaErrorCheck(cudaFreeHost(h_pinned_ptr), "cudaFreeHost(h_pinned_ptr)");
    cudaErrorCheck(cudaFree(d_ptr), "cudaFree(d_ptr)");
    free(h_ptr);
}


template <typename T>
__host__ __inline__ void copyDeviceBufferToHostImpl(T* h_dst,
                                const T* d_src,
                                size_t numElements,
                                uint32_t* h_pinnedScratch)
{
    if (numElements == 0 || h_dst == nullptr || d_src == nullptr || h_pinnedScratch == nullptr)
    {
        return;
    }

    const size_t pinnedBytes = static_cast<size_t>(PINNED_MEMORY_SIZE);
    const size_t totalBytes = numElements * sizeof(T);
    const size_t threadsRequested = static_cast<size_t>(THREADS_TO_COPY);

    // Do not spawn more threads than can hold at least one element in the scratch.
    const size_t maxThreadsByScratch = pinnedBytes / sizeof(T);
    const size_t threadsToUse = std::max<size_t>(
        1,
        std::min({threadsRequested, maxThreadsByScratch == 0 ? static_cast<size_t>(1) : maxThreadsByScratch, numElements}));

    // Round per-thread staging down to a whole element to avoid misaligned strides.
    const size_t stagedBytesPerThreadRaw = pinnedBytes / threadsToUse;
    const size_t stagedBytesPerThread = (stagedBytesPerThreadRaw / sizeof(T)) * sizeof(T);
    if (stagedBytesPerThread < sizeof(T))
    {
        // Scratch too small
        cudaErrorCheck(cudaMemcpy(h_dst, d_src, totalBytes, cudaMemcpyDeviceToHost), "cudaMemcpy(h_dst, d_src, totalBytes, cudaMemcpyDeviceToHost)");
        return;
    }

    const size_t stagedElementsPerThread = stagedBytesPerThread / sizeof(T);
    const size_t numIterPerThread = (numElements + threadsToUse - 1) / threadsToUse;

    // double onlyCopyBatchTime = 0;
    // double hostCopyTime = 0;

    omp_set_num_threads(static_cast<int>(threadsToUse));
    double s = omp_get_wtime();
    #pragma omp parallel for schedule(static,1)
    for (size_t chunk = 0; chunk < threadsToUse; ++chunk)
    {
        const int tid = omp_get_thread_num();

        size_t startElem = static_cast<size_t>(chunk) * numIterPerThread;
        if (startElem >= numElements) {
            continue;
        }

        size_t elemsThisChunk = std::min(static_cast<size_t>(numIterPerThread),
                                         numElements - startElem);
        // stagedBytesPerThread is bytes per thread. We ned raw byte pointer
        uint8_t* threadStageBytes = reinterpret_cast<uint8_t*>(h_pinnedScratch) + static_cast<size_t>(tid) * stagedBytesPerThread;
        T* threadStage = reinterpret_cast<T*>(threadStageBytes);

        size_t copiedElems = 0;
        while (elemsThisChunk > 0)
        {
            size_t curElems = std::min(elemsThisChunk, stagedElementsPerThread);
            size_t curBytes = curElems * sizeof(T);

            // double s1 = omp_get_wtime();
            cudaErrorCheck(cudaMemcpy(threadStage, d_src + startElem + copiedElems, curBytes, cudaMemcpyDeviceToHost), "cudaMemcpy(threadStage, d_src + startElem + copiedElems, curBytes, cudaMemcpyDeviceToHost)");
            // double s2 = omp_get_wtime();
            // onlyCopyBatchTime += (s2 - s1);

            // double h1 = omp_get_wtime();
            std::memcpy(h_dst + startElem + copiedElems, threadStage, curBytes);
            // parallel_memcpy(h_dst + startElem + copiedElems, threadStage, curBytes);
            // double h2 = omp_get_wtime();
            // hostCopyTime += (h2 - h1);

            elemsThisChunk -= curElems;
            copiedElems += curElems;
        }
    }

    double e = omp_get_wtime();
}




// Supports both uint32_t and uint64_t transfers
void copyDeviceBufferToHost(uint32_t* h_dst,
                            const uint32_t* d_src,
                            size_t numWords,
                            uint32_t* h_pinnedScratch)
{
    copyDeviceBufferToHostImpl<uint32_t>(h_dst, d_src, numWords, h_pinnedScratch);
}

void copyDeviceBufferToHost(uint64_t* h_dst,
                            const uint64_t* d_src,
                            size_t numWords,
                            uint32_t* h_pinnedScratch)
{
    copyDeviceBufferToHostImpl<uint64_t>(h_dst, d_src, numWords, h_pinnedScratch);
}


__inline__ __host__ __device__ bool isWithinEpsilon(const float* point1, const float* point2, const float EPSILON,
                                                    const uint32_t DIM)
{
    double distance = 0;

    for (int i = 0; i < DIM; i++)
    {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    return (distance <= (EPSILON * EPSILON));
}


// Use this when coordinates belonging to the same point are close to each other
// __inline__ __device__ bool pointWithinEps(const float* query, const float* candidate, float EPSILON_SQ, uint32_t DIM)
// {
//     uint32_t d = 0;
//     float distance = 0.0f;
//     // if(DIM==4)
//     // {
//     //     printf("\nComparing query (%.0f, %.0f, %.0f, %.0f) with Candidate (%.0f, %.0f, %.0f, %.0f)", query[0], query[1], query[2], query[3], candidate[0], candidate[1], candidate[2], candidate[3]);
//     // }
//     for (; d + 3 < DIM; d += 4)
//     {
//         float dx = query[d + 0] - candidate[d + 0];
//         distance = fmaf(dx, dx, distance);
//         dx = query[d + 1] - candidate[d + 1];
//         distance = fmaf(dx, dx, distance);
//         dx = query[d + 2] - candidate[d + 2];
//         distance = fmaf(dx, dx, distance);
//         dx = query[d + 3] - candidate[d + 3];
//         distance = fmaf(dx, dx, distance);
//         if (distance > EPSILON_SQ) return false;
//     }
//     for (; d < DIM; ++d)
//     {
//         float dx = query[d] - candidate[d];
//         distance = fmaf(dx, dx, distance);
//         if (distance > EPSILON_SQ) return false;
//     }
//     return true;
// }


// Use this when all coordinates of the same dimensions are packed together
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



// Kernel
// All blocks do mostly same amount of work
// Primitive major 
// Query points -> Global memory
// Primitive points -> Shared memory
// Final version - best result
// Batching - YES
__global__ void identifyNeighborsGridPrimitiveSharedQueryGlobal(
    uint32_t* groupCount,
    uint32_t* groupedPoints,
    uint32_t* candidatePoints,
    uint64_t* d_refinedCandidatePoints, 
    uint64_t* h_refinedCandidatePoints,
    unsigned long long* refinedCount,
    uint32_t* primitivePosToWrite,
    uint32_t* primitiveIntersectionCount,
    uint32_t* blockToGlobalPrimMapper,         
    float* dataset,
    uint32_t NUM_PRIMITIVES,
    uint32_t DIM,
    float EPSILON_SQ,
    uint32_t* blockQueryIdxArray, 
    uint32_t queriesPerBlock,
    uint32_t* resultMaskPrefix,
    uint32_t*resultMask,
    uint64_t totalWords)           
{
    int primGlobalId = blockToGlobalPrimMapper[blockIdx.x];

    int blockQueryIdx = blockQueryIdxArray[blockIdx.x];

    const uint32_t queriesStart = primitivePosToWrite[primGlobalId];
    // Total number of queries for primitive
    const uint32_t totalQueries = primitiveIntersectionCount[primGlobalId];

    const uint32_t totalQueriesForPrimitive = primitiveIntersectionCount[primGlobalId];

    uint32_t queryBatchStart = blockQueryIdx * queriesPerBlock;
    uint32_t queryBatchEnd   = min(queryBatchStart + queriesPerBlock, totalQueries);
    uint32_t NUM_QUERIES     = (queryBatchEnd > queryBatchStart) ? (queryBatchEnd - queryBatchStart) : 0u;
    if (NUM_QUERIES == 0) return;

    const uint32_t primPointsStart = groupCount[primGlobalId];
    const uint32_t primPointsEnd   = groupCount[primGlobalId + 1];
    const uint32_t NUM_PRIM_POINTS = (primPointsEnd > primPointsStart) ? (primPointsEnd - primPointsStart) : 0u;
    if (NUM_PRIM_POINTS == 0) return;

    extern __shared__ float sharedMem[];
    float* primShared = sharedMem;

    unsigned int localHits = 0u;
    const uint32_t maxPrimBatch  = SHARED_MEM_PRIM_POINT_COUNT;

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison

    // Iterate primitive points in batches
    for (uint32_t primBatchStart = 0; primBatchStart < NUM_PRIM_POINTS; primBatchStart += maxPrimBatch)
    {
        uint32_t primBatchCount = min(maxPrimBatch, NUM_PRIM_POINTS - primBatchStart);
        uint32_t primTotalElems  = primBatchCount * DIM;

        // Load primitive batch into shared memory
        for (uint32_t e = threadIdx.x; e < primTotalElems; e += blockDim.x)
        {
            uint32_t pInBatch   = e / DIM;
            uint32_t dimNum     = e % DIM;
            uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pInBatch];

            primShared[dimNum*primBatchCount + pInBatch] = dataset[globalPtId * DIM + dimNum];
        }
        __syncthreads();

        const uint32_t totalComparisons = primBatchCount * NUM_QUERIES;
        for (uint32_t comp = threadIdx.x; comp < totalComparisons; comp += blockDim.x)
        {
            uint32_t qIdxInBatch = comp / primBatchCount;
            uint32_t pIdxInBatch = comp % primBatchCount;

            uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + qIdxInBatch];

            const float* qPtr = &dataset[queryIdx * DIM];
   
            const float*pPtr = &primShared[pIdxInBatch];

            if (pointWithinEps(qPtr, pPtr, primBatchCount, DIM, EPSILON_SQ))
            {
                
                #if !USE_CANDIDATE_POINT_COPY
                    ++localHits;
                    // Below snippet stores result mask
                    uint32_t pointInPrim = primBatchStart + pIdxInBatch;
                    uint32_t queryInPrim = queryBatchStart + qIdxInBatch;

                    // Compute linear index in the bitmask
                    uint64_t linearBitIndex = (uint64_t)pointInPrim * totalQueriesForPrimitive + queryInPrim;

                    uint32_t queryWordInPrim = linearBitIndex >> 5;
                    uint32_t queryBitPos     = linearBitIndex & 31;

                    uint32_t primWordStart = resultMaskPrefix[primGlobalId];

                    atomicOr(&resultMask[primWordStart + queryWordInPrim], 1u << queryBitPos);
                #else
                    uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pIdxInBatch];
                    uint64_t packedNeighbor = ((uint64_t)queryIdx << 32) | (uint32_t)globalPtId;
                    unsigned long long posToWrite = atomicAdd(refinedCount, 1);
                    d_refinedCandidatePoints[posToWrite] = packedNeighbor;
                #endif
            }
        }
        __syncthreads();
    }

    #if !USE_CANDIDATE_POINT_COPY
        if (localHits > 0u)
            atomicAdd(refinedCount, localHits);
    #endif
}

// Kernel
// All blocks do mostly same amount of work
// Primitive major 
// Query points -> Shared memory
// Primitive points -> Global memory
// Batching - YES
__global__ void identifyNeighborsGridPrimitiveGlobalQueryShared(
    uint32_t* groupCount,
    uint32_t* groupedPoints,
    uint32_t* candidatePoints,
    uint64_t* d_refinedCandidatePoints, 
    uint64_t* h_refinedCandidatePoints,
    unsigned long long* refinedCount,
    uint32_t* primitivePosToWrite,
    uint32_t* primitiveIntersectionCount,
    uint32_t* blockToGlobalPrimMapper,          
    float* dataset,
    uint32_t NUM_PRIMITIVES,
    uint32_t DIM,
    float EPSILON_SQ,
    uint32_t* blockQueryIdxArray, 
    uint32_t queriesPerBlock,
    uint32_t* resultMaskPrefix,
    uint32_t*resultMask,
    uint64_t totalWords)           
{
    int primGlobalId = blockToGlobalPrimMapper[blockIdx.x];

    int blockQueryIdx = blockQueryIdxArray[blockIdx.x];

    const uint32_t queriesStart = primitivePosToWrite[primGlobalId];
    // Total number of queries for primitive
    const uint32_t totalQueries = primitiveIntersectionCount[primGlobalId];

    const uint32_t totalQueriesForPrimitive = primitiveIntersectionCount[primGlobalId];

    uint32_t queryBatchStart = blockQueryIdx * queriesPerBlock;
    uint32_t queryBatchEnd   = min(queryBatchStart + queriesPerBlock, totalQueries);
    uint32_t NUM_QUERIES     = (queryBatchEnd > queryBatchStart) ? (queryBatchEnd - queryBatchStart) : 0u;
    if (NUM_QUERIES == 0) return;

    const uint32_t primPointsStart = groupCount[primGlobalId];
    const uint32_t primPointsEnd   = groupCount[primGlobalId + 1];
    const uint32_t NUM_PRIM_POINTS = (primPointsEnd > primPointsStart) ? (primPointsEnd - primPointsStart) : 0u;
    if (NUM_PRIM_POINTS == 0) return;

    extern __shared__ float sharedMem[];
    float* queryShared = sharedMem;

    unsigned int localHits = 0u;
    const uint32_t maxPrimBatch  = SHARED_MEM_PRIM_POINT_COUNT;
    const uint32_t maxQueryBatch = SHARED_MEM_PRIM_POINT_COUNT; // entire shared memory is for queries

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison

    // Iterate primitive points in batches
    for (uint32_t primBatchStart = 0; primBatchStart < NUM_PRIM_POINTS; primBatchStart += maxPrimBatch)
    {
        uint32_t primBatchCount = min(maxPrimBatch, NUM_PRIM_POINTS - primBatchStart);

        for (uint32_t queryOffset = 0; queryOffset < NUM_QUERIES; queryOffset += maxQueryBatch)
        {
            uint32_t queryBatchCount = min(maxQueryBatch, NUM_QUERIES - queryOffset);
            uint32_t queryTotalElems = queryBatchCount * DIM;

            // Load query batch into shared memory
            for (uint32_t e = threadIdx.x; e < queryTotalElems; e += blockDim.x)
            {
                uint32_t qInBatch = e / DIM;
                uint32_t dimNum   = e % DIM;
                uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + queryOffset + qInBatch];

                queryShared[qInBatch * DIM + dimNum] = dataset[queryIdx * DIM + dimNum];
            }
            __syncthreads();

            const uint32_t totalComparisons = primBatchCount * queryBatchCount;
            for (uint32_t comp = threadIdx.x; comp < totalComparisons; comp += blockDim.x)
            {
                uint32_t qIdxInBatch = comp / primBatchCount;
                uint32_t pIdxInBatch = comp % primBatchCount;

                uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pIdxInBatch];
                const float* qPtr = &queryShared[qIdxInBatch * DIM];
                const float* pPtr = &dataset[globalPtId * DIM];

                uint32_t pointInPrim = primBatchStart + pIdxInBatch;
                uint32_t queryInPrim = queryBatchStart + queryOffset + qIdxInBatch;
                uint64_t linearBitIndex = (uint64_t)pointInPrim * totalQueriesForPrimitive + queryInPrim;

                if (pointWithinEps(qPtr, pPtr, 1, DIM, EPSILON_SQ))
                {
                    #if !USE_CANDIDATE_POINT_COPY
                        ++localHits;
                        uint32_t queryWordInPrim = linearBitIndex >> 5;
                        uint32_t queryBitPos     = linearBitIndex & 31;

                        uint32_t primWordStart = resultMaskPrefix[primGlobalId];

                        atomicOr(&resultMask[primWordStart + queryWordInPrim], 1u << queryBitPos);
                    #else
                        uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + queryOffset + qIdxInBatch];
                        uint64_t packedNeighbor = ((uint64_t)queryIdx << 32) | (uint32_t)globalPtId;
                        unsigned long long posToWrite = atomicAdd(refinedCount, 1);
                        d_refinedCandidatePoints[posToWrite] = packedNeighbor;
                    #endif
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    #if !USE_CANDIDATE_POINT_COPY
        if (localHits > 0u)
            atomicAdd(refinedCount, localHits);
    #endif
}


// Kernel
// All blocks do mostly same amount of work
// Primitive major 
// Query points -> Shared memory
// Primitive points -> Shared memory
// Batching - YES
__global__ void identifyNeighborsGridPrimitiveSharedQueryShared(
    uint32_t* groupCount,
    uint32_t* groupedPoints,
    uint32_t* candidatePoints,
    uint64_t* d_refinedCandidatePoints, 
    uint64_t* h_refinedCandidatePoints,
    unsigned long long* refinedCount,
    uint32_t* primitivePosToWrite,
    uint32_t* primitiveIntersectionCount,
    uint32_t* blockToGlobalPrimMapper,          // sorted descending by number of queries
    float* dataset,
    uint32_t NUM_PRIMITIVES,
    uint32_t DIM,
    float EPSILON_SQ,
    uint32_t* blockQueryIdxArray, // per primitive work assignment
    uint32_t queriesPerBlock,// threshold per block
    uint32_t* resultMaskPrefix,
    uint32_t*resultMask,
    uint64_t totalWords)           
{
    int primGlobalId = blockToGlobalPrimMapper[blockIdx.x];

    int blockQueryIdx = blockQueryIdxArray[blockIdx.x];

    const uint32_t queriesStart = primitivePosToWrite[primGlobalId];
    // Total number of queries for primitive
    const uint32_t totalQueries = primitiveIntersectionCount[primGlobalId];

    const uint32_t totalQueriesForPrimitive = primitiveIntersectionCount[primGlobalId];

    uint32_t queryBatchStart = blockQueryIdx * queriesPerBlock;
    uint32_t queryBatchEnd   = min(queryBatchStart + queriesPerBlock, totalQueries);
    uint32_t NUM_QUERIES     = (queryBatchEnd > queryBatchStart) ? (queryBatchEnd - queryBatchStart) : 0u;
    if (NUM_QUERIES == 0) return;

    const uint32_t primPointsStart = groupCount[primGlobalId];
    const uint32_t primPointsEnd   = groupCount[primGlobalId + 1];
    const uint32_t NUM_PRIM_POINTS = (primPointsEnd > primPointsStart) ? (primPointsEnd - primPointsStart) : 0u;
    if (NUM_PRIM_POINTS == 0) return;

    extern __shared__ float sharedMem[];
    float* primShared = sharedMem;

    unsigned int localHits = 0u;
    const uint32_t maxPrimBatch = SHARED_MEM_PRIM_POINT_COUNT / 2;
    const uint32_t maxQueryBatch = SHARED_MEM_PRIM_POINT_COUNT / 2;

    float* queryShared = primShared + (size_t)maxPrimBatch * DIM; // place queries after prim region

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison

    // Iterate primitive points in batches
    for (uint32_t primBatchStart = 0; primBatchStart < NUM_PRIM_POINTS; primBatchStart += maxPrimBatch)
    {
        uint32_t primBatchCount = min(maxPrimBatch, NUM_PRIM_POINTS - primBatchStart);
        uint32_t primTotalElems  = primBatchCount * DIM;

        // Load primitive batch into shared memory
        for (uint32_t e = threadIdx.x; e < primTotalElems; e += blockDim.x)
        {
            uint32_t pInBatch   = e / DIM;
            uint32_t dimNum     = e % DIM;
            uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pInBatch];

            primShared[dimNum*primBatchCount + pInBatch] = dataset[globalPtId * DIM + dimNum];
        }
        __syncthreads();

        for (uint32_t queryOffset = 0; queryOffset < NUM_QUERIES; queryOffset += maxQueryBatch)
        {
            uint32_t queryBatchCount = min(maxQueryBatch, NUM_QUERIES - queryOffset);
            uint32_t queryTotalElems = queryBatchCount * DIM;

            // Load query batch into shared memory
            for (uint32_t e = threadIdx.x; e < queryTotalElems; e += blockDim.x)
            {
                uint32_t qInBatch = e / DIM;
                uint32_t dimNum   = e % DIM;
                uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + queryOffset + qInBatch];

                queryShared[qInBatch * DIM + dimNum] = dataset[queryIdx * DIM + dimNum];
            }
            __syncthreads();

            const uint32_t totalComparisons = primBatchCount * queryBatchCount;
            for (uint32_t comp = threadIdx.x; comp < totalComparisons; comp += blockDim.x)
            {
                uint32_t qIdxInBatch = comp / primBatchCount;
                uint32_t pIdxInBatch = comp % primBatchCount;

                // uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + queryOffset + qIdxInBatch];

                const float* qPtr = &queryShared[qIdxInBatch * DIM];
       
                const float*pPtr = &primShared[pIdxInBatch];

                uint32_t pointInPrim = primBatchStart + pIdxInBatch;
                uint32_t queryInPrim = queryBatchStart + queryOffset + qIdxInBatch;
                uint64_t linearBitIndex = (uint64_t)pointInPrim * totalQueriesForPrimitive + queryInPrim;

                if (pointWithinEps(qPtr, pPtr, primBatchCount, DIM, EPSILON_SQ))
                {
                   
                    #if !USE_CANDIDATE_POINT_COPY
                        ++localHits;
                        // Below snippet stores result mask
                        uint32_t queryWordInPrim = linearBitIndex >> 5;
                        uint32_t queryBitPos     = linearBitIndex & 31;

                        uint32_t primWordStart = resultMaskPrefix[primGlobalId];

                        atomicOr(&resultMask[primWordStart + queryWordInPrim], 1u << queryBitPos);
                    #else
                        uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + queryOffset + qIdxInBatch];
                        uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pIdxInBatch];
                        unsigned long long posToWrite = atomicAdd(refinedCount, 1);
                        uint64_t packedNeighbor = ((uint64_t)queryIdx << 32) | (uint32_t)globalPtId;
                        d_refinedCandidatePoints[posToWrite] = packedNeighbor;
                    #endif
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    #if !USE_CANDIDATE_POINT_COPY
        if (localHits > 0u)
            atomicAdd(refinedCount, localHits);
    #endif
}

// Kernel
// All blocks do mostly same amount of work
// Primitive major 
// Query points -> Global memory
// Primitive points -> Global memory
// Batching - YES
__global__ void identifyNeighborsGridQueryGlobalPrimitiveGlobal(
    uint32_t* groupCount,
    uint32_t* groupedPoints,
    uint32_t* candidatePoints,
    uint64_t* d_refinedCandidatePoints, 
    uint64_t* h_refinedCandidatePoints,
    unsigned long long* refinedCount,
    uint32_t* primitivePosToWrite,
    uint32_t* primitiveIntersectionCount,
    uint32_t* blockToGlobalPrimMapper,         
    float* dataset,
    uint32_t NUM_PRIMITIVES,
    uint32_t DIM,
    float EPSILON_SQ,
    uint32_t* blockQueryIdxArray, 
    uint32_t queriesPerBlock,
    uint32_t* resultMaskPrefix,
    uint32_t*resultMask,
    uint64_t totalWords)           
{
    int primGlobalId = blockToGlobalPrimMapper[blockIdx.x];

    int blockQueryIdx = blockQueryIdxArray[blockIdx.x];

    const uint32_t queriesStart = primitivePosToWrite[primGlobalId];
    // Total number of queries for primitive
    const uint32_t totalQueries = primitiveIntersectionCount[primGlobalId];

    const uint32_t totalQueriesForPrimitive = primitiveIntersectionCount[primGlobalId];

    uint32_t queryBatchStart = blockQueryIdx * queriesPerBlock;
    uint32_t queryBatchEnd   = min(queryBatchStart + queriesPerBlock, totalQueries);
    uint32_t NUM_QUERIES     = (queryBatchEnd > queryBatchStart) ? (queryBatchEnd - queryBatchStart) : 0u;
    if (NUM_QUERIES == 0) return;

    const uint32_t primPointsStart = groupCount[primGlobalId];
    const uint32_t primPointsEnd   = groupCount[primGlobalId + 1];
    const uint32_t NUM_PRIM_POINTS = (primPointsEnd > primPointsStart) ? (primPointsEnd - primPointsStart) : 0u;
    if (NUM_PRIM_POINTS == 0) return;

    unsigned int localHits = 0u;
    const uint32_t maxPrimBatch = SHARED_MEM_PRIM_POINT_COUNT;

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison

    // Iterate primitive points in batches
    for (uint32_t primBatchStart = 0; primBatchStart < NUM_PRIM_POINTS; primBatchStart += maxPrimBatch)
    {
        uint32_t primBatchCount = min(maxPrimBatch, NUM_PRIM_POINTS - primBatchStart);
        uint32_t primIdxBase    = primPointsStart + primBatchStart;

        const uint32_t totalComparisons = primBatchCount * NUM_QUERIES;
        for (uint32_t comp = threadIdx.x; comp < totalComparisons; comp += blockDim.x)
        {
            uint32_t qIdxInBatch = comp / primBatchCount;
            uint32_t pIdxInBatch = comp % primBatchCount;

            uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + qIdxInBatch];
            uint32_t globalPtId = groupedPoints[primIdxBase + pIdxInBatch];

            const float* qPtr = &dataset[queryIdx * DIM];
   
            const float*pPtr = &dataset[globalPtId * DIM];

            if (pointWithinEps(qPtr, pPtr, 1, DIM, EPSILON_SQ))
            {
                
                #if !USE_CANDIDATE_POINT_COPY
                    ++localHits;
                    // Below snippet stores result mask
                    uint32_t pointInPrim = primBatchStart + pIdxInBatch;
                    uint32_t queryInPrim = queryBatchStart + qIdxInBatch;

                    // Compute linear index in the bitmask
                    uint64_t linearBitIndex = (uint64_t)pointInPrim * totalQueriesForPrimitive + queryInPrim;

                    uint32_t queryWordInPrim = linearBitIndex >> 5;
                    uint32_t queryBitPos     = linearBitIndex & 31;

                    uint32_t primWordStart = resultMaskPrefix[primGlobalId];

                    atomicOr(&resultMask[primWordStart + queryWordInPrim], 1u << queryBitPos);
                #else
                    uint32_t globalPtId = groupedPoints[primPointsStart + primBatchStart + pIdxInBatch];
                    unsigned long long posToWrite = atomicAdd(refinedCount, 1);
                    uint64_t packedNeighbor = ((uint64_t)queryIdx << 32) | (uint32_t)globalPtId;
                    d_refinedCandidatePoints[posToWrite] = packedNeighbor;
                #endif
            }
        }
    }

    #if !USE_CANDIDATE_POINT_COPY
        if (localHits > 0u)
            atomicAdd(refinedCount, localHits);
    #endif
}

// Kernel
// All blocks do mostly same amount of work
// Primitive major 
// Query points -> Global memory
// Primitive points -> Global memory
// Batching - NO -> Using this as a baseline
__global__ void identifyNeighborsGridQueryGlobalPrimitiveGlobalNoBatching(
    uint32_t* groupCount,
    uint32_t* groupedPoints,
    uint32_t* candidatePoints,
    uint64_t* d_refinedCandidatePoints, 
    uint64_t* h_refinedCandidatePoints,
    unsigned long long* refinedCount,
    uint32_t* primitivePosToWrite,
    uint32_t* primitiveIntersectionCount,
    uint32_t* blockToGlobalPrimMapper,         
    float* dataset,
    uint32_t NUM_PRIMITIVES,
    uint32_t DIM,
    float EPSILON_SQ,
    uint32_t* blockQueryIdxArray, 
    uint32_t queriesPerBlock,
    uint32_t* resultMaskPrefix,
    uint32_t*resultMask,
    uint64_t totalWords)           
{
    int primGlobalId = blockToGlobalPrimMapper[blockIdx.x];

    int blockQueryIdx = blockQueryIdxArray[blockIdx.x];

    const uint32_t queriesStart = primitivePosToWrite[primGlobalId];
    // Total number of queries for primitive
    const uint32_t totalQueries = primitiveIntersectionCount[primGlobalId];

    const uint32_t totalQueriesForPrimitive = primitiveIntersectionCount[primGlobalId];

    uint32_t queryBatchStart = blockQueryIdx * queriesPerBlock;
    uint32_t queryBatchEnd   = min(queryBatchStart + queriesPerBlock, totalQueries);
    uint32_t NUM_QUERIES     = (queryBatchEnd > queryBatchStart) ? (queryBatchEnd - queryBatchStart) : 0u;
    if (NUM_QUERIES == 0) return;

    const uint32_t primPointsStart = groupCount[primGlobalId];
    const uint32_t primPointsEnd   = groupCount[primGlobalId + 1];
    const uint32_t NUM_PRIM_POINTS = (primPointsEnd > primPointsStart) ? (primPointsEnd - primPointsStart) : 0u;
    if (NUM_PRIM_POINTS == 0) return;

    unsigned int localHits = 0u;

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison

    // No batching: compare all primitive points with all queries in a single pass
    const uint32_t primBatchCount = NUM_PRIM_POINTS;
    const uint32_t totalComparisons = primBatchCount * NUM_QUERIES;
    for (uint32_t comp = threadIdx.x; comp < totalComparisons; comp += blockDim.x)
    {
        uint32_t qIdxInBatch = comp / primBatchCount;
        uint32_t pIdxInBatch = comp % primBatchCount;

        uint32_t queryIdx = candidatePoints[queriesStart + queryBatchStart + qIdxInBatch];
        uint32_t globalPtId = groupedPoints[primPointsStart + pIdxInBatch];

        const float* qPtr = &dataset[queryIdx * DIM];
   
        const float*pPtr = &dataset[globalPtId * DIM];

            if (pointWithinEps(qPtr, pPtr, 1, DIM, EPSILON_SQ))
            {
                #if !USE_CANDIDATE_POINT_COPY
                    ++localHits;
                    // Below snippet stores result mask
                    uint32_t pointInPrim = pIdxInBatch;
                    uint32_t queryInPrim = queryBatchStart + qIdxInBatch;

                    // Compute linear index in the bitmask
                    uint64_t linearBitIndex = (uint64_t)pointInPrim * totalQueriesForPrimitive + queryInPrim;

                    uint32_t queryWordInPrim = linearBitIndex >> 5;
                    uint32_t queryBitPos     = linearBitIndex & 31;

                    uint32_t primWordStart = resultMaskPrefix[primGlobalId];

                    atomicOr(&resultMask[primWordStart + queryWordInPrim], 1u << queryBitPos);
                #else
                    unsigned long long posToWrite = atomicAdd(refinedCount, 1);
                    uint64_t packedNeighbor = ((uint64_t)queryIdx << 32) | (uint32_t)globalPtId;
                    d_refinedCandidatePoints[posToWrite] = packedNeighbor;
                #endif
            }
        }
    #if !USE_CANDIDATE_POINT_COPY
        if (localHits > 0u)
            atomicAdd(refinedCount, localHits);
    #endif
}


__global__ void compressResultMask(
    const uint32_t *resultMask,
    uint32_t numOfWords,
    unsigned long long *counter,
    uint32_t *compressedResultMask)
{
    uint32_t tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= numOfWords)
        return;

    uint32_t word = resultMask[tId];

    uint32_t bitcount = __popc(word);
    if (bitcount == 0) return;

    uint64_t base = atomicAdd(counter, (unsigned long long)bitcount);

    for (uint32_t i = 0, written = 0; i < 32; i++) {
        if ((word >> i) & 1u) {
            uint32_t globalBitPos = tId * 32u + i;
            compressedResultMask[base + written] = globalBitPos;
            written++;
        }
    }
}


// Host function: launch kernel with blocks split by query workload
extern "C" void refineCandidatePointsGrid(
    uint32_t* d_groupCount, uint32_t*h_groupCount, uint32_t* d_groupedPoints,
    uint32_t* d_candidatePoints, uint64_t* d_refinedCandidatePoints, uint64_t* h_refinedCandidatePoints, unsigned long long* h_refinedCount, float* d_dataset,
    uint32_t* d_primitivePosToWrite, uint32_t* d_primitiveIntersectionCount,
    uint32_t* d_primitiveOrder, uint32_t*h_primitiveOrder, uint32_t *h_pinnedResultMask,
    uint32_t N, uint32_t NUM_PRIMITIVES, uint32_t DIM, float EPSILON_SQ)
{
    unsigned long long* d_refinedCount;
    cudaErrorCheck(cudaMalloc(&d_refinedCount, sizeof(unsigned long long)), "cudaMalloc(&d_refinedCount, sizeof(unsigned long long))");
    cudaErrorCheck(cudaMemcpy(d_refinedCount, h_refinedCount, sizeof(unsigned long long), cudaMemcpyHostToDevice), "cudaMemcpy(d_refinedCount, h_refinedCount, sizeof(unsigned long long), cudaMemcpyHostToDevice)");

    int blockSize = 1024;
    uint32_t queriesPerBlock = blockSize * NUM_CALCULATIONS_PER_THREAD;

    // Copy primitiveIntersectionCount and primitiveOrder to host
    std::vector<uint32_t> h_primitiveIntersectionCount(NUM_PRIMITIVES);
    cudaErrorCheck(cudaMemcpy(h_primitiveIntersectionCount.data(), d_primitiveIntersectionCount,
               NUM_PRIMITIVES * sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy(h_primitiveIntersectionCount.data(), d_primitiveIntersectionCount, NUM_PRIMITIVES * sizeof(uint32_t), cudaMemcpyDeviceToHost)");

    // Compute prefix sum of blocks per primitive
    std::vector<uint32_t> h_blockPrefixSum(NUM_PRIMITIVES);
    uint32_t sum = 0;
    std::vector<uint32_t> threadBlockToGlobalPrimMapper;
    std::vector<uint32_t> blockQueryIdx;

    cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    #if USE_FIXED_WORK_PER_BLOCK
    for (int i = 0; i < NUM_PRIMITIVES; ++i)
    {
        uint32_t primId = h_primitiveOrder[i];
        uint32_t numQueries = h_primitiveIntersectionCount[primId];
        uint32_t blocks = (numQueries + queriesPerBlock - 1) / queriesPerBlock;
        sum += blocks;
        h_blockPrefixSum[i] = sum;
        threadBlockToGlobalPrimMapper.insert(threadBlockToGlobalPrimMapper.end(), blocks, primId);
        for(uint32_t j = 0 ; j < blocks ; j++)
        {
            blockQueryIdx.push_back(j);
        }
    }
    uint32_t totalBlocks = sum;
    #else
    uint32_t maxQueries = 0;
    for (int i = 0; i < NUM_PRIMITIVES; ++i)
    {
        uint32_t primId = h_primitiveOrder[i];
        uint32_t numQueries = h_primitiveIntersectionCount[primId];
        maxQueries = std::max(maxQueries, numQueries);
        uint32_t blocks = 1;
        sum += blocks;
        h_blockPrefixSum[i] = sum;
        threadBlockToGlobalPrimMapper.insert(threadBlockToGlobalPrimMapper.end(), blocks, primId);
        for(uint32_t j = 0 ; j < blocks ; j++)
        {
            blockQueryIdx.push_back(j);
        }
    }
    queriesPerBlock = maxQueries;
    uint32_t totalBlocks = sum;
    
    #endif

    // Copy sorted primitive order and prefix sum to device
    uint32_t* d_blockPrefixSum;
    cudaErrorCheck(cudaMalloc(&d_blockPrefixSum, NUM_PRIMITIVES * sizeof(uint32_t)), "cudaMalloc(&d_blockPrefixSum, NUM_PRIMITIVES * sizeof(uint32_t))");

    cudaErrorCheck(cudaMemcpy(d_blockPrefixSum, h_blockPrefixSum.data(),
               NUM_PRIMITIVES * sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy(d_blockPrefixSum, h_blockPrefixSum.data(), NUM_PRIMITIVES * sizeof(uint32_t), cudaMemcpyHostToDevice)");

    //Copy info on block to primitive mapper
    assert(totalBlocks == threadBlockToGlobalPrimMapper.size());
    assert(totalBlocks == blockQueryIdx.size());

    uint32_t *d_threadBlockToGlobalPrimMapper;
    cudaErrorCheck(cudaMalloc(&d_threadBlockToGlobalPrimMapper, sizeof(uint32_t)*totalBlocks), "cudaMalloc(&d_threadBlockToGlobalPrimMapper, sizeof(uint32_t)*totalBlocks)");
    cudaErrorCheck(cudaMemcpy(d_threadBlockToGlobalPrimMapper, threadBlockToGlobalPrimMapper.data(), sizeof(uint32_t)*totalBlocks, cudaMemcpyHostToDevice), "cudaMemcpy(d_threadBlockToGlobalPrimMapper, threadBlockToGlobalPrimMapper.data(), sizeof(uint32_t)*totalBlocks, cudaMemcpyHostToDevice)");

    uint32_t*d_blockQueryIdx;
    cudaErrorCheck(cudaMalloc(&d_blockQueryIdx, sizeof(uint32_t)*totalBlocks), "cudaMalloc(&d_blockQueryIdx, sizeof(uint32_t)*totalBlocks)");
    cudaErrorCheck(cudaMemcpy(d_blockQueryIdx, blockQueryIdx.data(), sizeof(uint32_t)*totalBlocks, cudaMemcpyHostToDevice), "cudaMemcpy(d_blockQueryIdx, blockQueryIdx.data(), sizeof(uint32_t)*totalBlocks, cudaMemcpyHostToDevice)");

    printf("\n TOTAL BLOCKS are : %d", totalBlocks);

    // For results mask
    // NUM_PRIM_POINTS point to number of points in each primitive
    // totalQueries point to total number of queries for all these point within the primitive
    // Result mask is of size NUM_PRIM_POINTS * totalQueries
    // For each primitive point, we should have a mask of size totalQueries to reflect the result of point comparison
    // compute per-primitive words and allocate host/device buffers
    // std::vector<uint32_t> h_groupCount(NUM_PRIMITIVES + 1);
    // cudaMemcpy(h_groupCount.data(), d_groupCount, sizeof(uint32_t)*(NUM_PRIMITIVES+1), cudaMemcpyDeviceToHost);

    std::vector<uint32_t> resultMaskPrefix(NUM_PRIMITIVES + 1, 0);

    // cudaDeviceSynchronize();

    uint32_t totalQueries = 0;

    for (uint32_t primGlobalId = 0; primGlobalId < NUM_PRIMITIVES; ++primGlobalId)
    {
        uint32_t totalQueriesForPrimitive = h_primitiveIntersectionCount[primGlobalId];

        uint32_t numPrimPoints = h_groupCount[primGlobalId + 1] - h_groupCount[primGlobalId];

        uint64_t maxBitsNeeded = static_cast<uint64_t>(numPrimPoints) * totalQueriesForPrimitive;
        uint32_t numWords = static_cast<uint32_t>((maxBitsNeeded + 31u) / 32u);

        resultMaskPrefix[primGlobalId + 1] = resultMaskPrefix[primGlobalId] + numWords;

        totalQueries += totalQueriesForPrimitive;
    }

    // // Total words allocated for all primitives
    const uint64_t numTotalResultWords = resultMaskPrefix[NUM_PRIMITIVES];


    printf("\nTotal result words: %lu, Total queries: %u N: %u\n", numTotalResultWords, totalQueries, N);
    printf("\nMemory required: %.3f GiB\n", sizeof(uint32_t) * (float)numTotalResultWords / 1024 / 1024 / 1024);

    // Allocate on device
    uint32_t* d_resultMaskPrefix;
    cudaErrorCheck(cudaMalloc(&d_resultMaskPrefix, sizeof(uint32_t) * (NUM_PRIMITIVES + 1)), "cudaMalloc(&d_resultMaskPrefix, sizeof(uint32_t) * (NUM_PRIMITIVES + 1))");
    cudaErrorCheck(cudaMemcpy(d_resultMaskPrefix, resultMaskPrefix.data(),
            sizeof(uint32_t) * (NUM_PRIMITIVES + 1), cudaMemcpyHostToDevice), "cudaMemcpy(d_resultMaskPrefix, resultMaskPrefix.data(), sizeof(uint32_t) * (NUM_PRIMITIVES + 1), cudaMemcpyHostToDevice)");

    uint32_t* d_resultMask;
    cudaErrorCheck(cudaMalloc(&d_resultMask, sizeof(uint32_t) * numTotalResultWords), "cudaMalloc(&d_resultMask, sizeof(uint32_t) * numTotalResultWords)");
    cudaErrorCheck(cudaMemset(d_resultMask, 0, sizeof(uint32_t) * numTotalResultWords), "cudaMemset(d_resultMask, 0, sizeof(uint32_t) * numTotalResultWords)");

    uint32_t*h_resultmask = (uint32_t*)malloc(sizeof(uint32_t)* numTotalResultWords);

    // static double cudaOnlyTime = 0;

#ifdef SAFE_SHARED_MEMORY_RUNTIME
    (void)getSafeSharedMemory();
#endif

    double start = omp_get_wtime();

    #if USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING
            // Shared memory for primitive and query points
            size_t sharedMem = sizeof(float) * SHARED_MEM_PRIM_POINT_COUNT * DIM;
            identifyNeighborsGridPrimitiveSharedQueryShared<<<totalBlocks, blockSize, sharedMem>>>(
                d_groupCount, d_groupedPoints, d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints,
                d_refinedCount, d_primitivePosToWrite,
                d_primitiveIntersectionCount, d_threadBlockToGlobalPrimMapper,
                d_dataset, NUM_PRIMITIVES, DIM, EPSILON_SQ,
                d_blockQueryIdx, queriesPerBlock, d_resultMaskPrefix, d_resultMask, numTotalResultWords);
    #elif USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING
            // Shared memory for primitive points
            size_t sharedMem = sizeof(float) * SHARED_MEM_PRIM_POINT_COUNT * DIM;
            identifyNeighborsGridPrimitiveSharedQueryGlobal<<<totalBlocks, blockSize, sharedMem>>>(
                d_groupCount, d_groupedPoints, d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints,
                d_refinedCount, d_primitivePosToWrite,
                d_primitiveIntersectionCount, d_threadBlockToGlobalPrimMapper,
                d_dataset, NUM_PRIMITIVES, DIM, EPSILON_SQ,
                d_blockQueryIdx, queriesPerBlock, d_resultMaskPrefix, d_resultMask, numTotalResultWords);

    #elif USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING
            // Shared memory for query points
            size_t sharedMem = sizeof(float) * SHARED_MEM_PRIM_POINT_COUNT * DIM;
            identifyNeighborsGridPrimitiveGlobalQueryShared<<<totalBlocks, blockSize, sharedMem>>>(
                d_groupCount, d_groupedPoints, d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints,
                d_refinedCount, d_primitivePosToWrite,
                d_primitiveIntersectionCount, d_threadBlockToGlobalPrimMapper,
                d_dataset, NUM_PRIMITIVES, DIM, EPSILON_SQ,
                d_blockQueryIdx, queriesPerBlock, d_resultMaskPrefix, d_resultMask, numTotalResultWords);
    #elif USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING
            size_t sharedMem = 0;
            identifyNeighborsGridQueryGlobalPrimitiveGlobal<<<totalBlocks, blockSize, sharedMem>>>(
                d_groupCount, d_groupedPoints, d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints,
                d_refinedCount, d_primitivePosToWrite,
                d_primitiveIntersectionCount, d_threadBlockToGlobalPrimMapper,
                d_dataset, NUM_PRIMITIVES, DIM, EPSILON_SQ,
                d_blockQueryIdx, queriesPerBlock, d_resultMaskPrefix, d_resultMask, numTotalResultWords);

    #elif USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING
            // Baseline kernel without any batching
            size_t sharedMem = 0;
            identifyNeighborsGridQueryGlobalPrimitiveGlobalNoBatching<<<totalBlocks, blockSize, sharedMem>>>(
                d_groupCount, d_groupedPoints, d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints,
                d_refinedCount, d_primitivePosToWrite,
                d_primitiveIntersectionCount, d_threadBlockToGlobalPrimMapper,
                d_dataset, NUM_PRIMITIVES, DIM, EPSILON_SQ,
                d_blockQueryIdx, queriesPerBlock, d_resultMaskPrefix, d_resultMask, numTotalResultWords);
    #else
        fprintf(stderr, "\nNo kernel is selected for candidate refinement\n");
        exit(1);
    #endif

    // Catch an invalid launch configuration (bad grid/block/shared-mem, or a
    // device pointer left null by a failed cudaMalloc) before we block on it.
    cudaErrorCheck(cudaGetLastError(), "identifyNeighbors kernel launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "identifyNeighbors kernel sync");
    double end = omp_get_wtime();

    cudaOnlyTime += (end - start);
    printf("\nCum CUDA TIME IS %.3lf\n", cudaOnlyTime);
    cudaErrorCheck(cudaMemcpy(h_refinedCount, d_refinedCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy(h_refinedCount, d_refinedCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost)");
    cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    // static double timeToCopyResults = 0;

    #if !USE_CANDIDATE_POINT_COPY
        uint32_t enoughMemory = true;

        // if((sizeof(uint32_t) * N) < (sizeof(uint32_t)* (*h_refinedCount)))
        {
            size_t freeMem, totalMem;

            cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem), "cudaMemGetInfo(&freeMem, &totalMem)");
            if((sizeof(uint32_t)* (*h_refinedCount) > freeMem) )
            {
                fprintf(stderr, "Memory available is not enough to store results");
                fprintf(stderr, "Memory avail = %.2f GiB and memory required is %.2f GiB", (float)freeMem/1024/1024/1024, (sizeof(uint32_t)* (float)(*h_refinedCount)/1024/1024/1024));
                enoughMemory = false;
            }
        }


        uint32_t*d_compressedResultMask;

        if(enoughMemory && !USE_UNCOMPRESSED_MASK)
        {
            // Allocate memory for result
            cudaErrorCheck(cudaMalloc(&d_compressedResultMask, sizeof(uint32_t)*(*h_refinedCount)), "cudaMalloc(&d_compressedResultMask, sizeof(uint32_t)*(*h_refinedCount))");

            uint32_t numBlockToCompress = (numTotalResultWords + blockSize - 1) / blockSize;

            unsigned long long *d_counter;
            cudaErrorCheck(cudaMalloc(&d_counter, sizeof(unsigned long long)), "cudaMalloc(&d_counter, sizeof(unsigned long long))");
            cudaErrorCheck(cudaMemset(d_counter, 0, sizeof(unsigned long long)), "cudaMemset(d_counter, 0, sizeof(unsigned long long))");

            cudaErrorCheck(cudaGetLastError(), "cudaGetLastError()");

            //Kernel to scan 32 bit words and fetch global position
            // static double timeToCompress  = 0;

            double compressTimeStart = omp_get_wtime();
            compressResultMask<<<numBlockToCompress, blockSize, 0>>>(
            d_resultMask,
            numTotalResultWords,
            d_counter,
            d_compressedResultMask);

            cudaErrorCheck(cudaGetLastError(), "compressResultMask launch");
            cudaErrorCheck(cudaDeviceSynchronize(), "compressResultMask sync");

            double compressTimeEnd = omp_get_wtime();

            timeToCompress += (compressTimeEnd - compressTimeStart);
            printf("\n Cum time to compress is %.3f", timeToCompress);
            cudaErrorCheck(cudaFree(d_counter), "cudaFree(d_counter)");
        }

        
        // Copying result is the most expensive task here
        double resultCopyStart = omp_get_wtime();

        if(!enoughMemory || USE_UNCOMPRESSED_MASK)
        {
            // Without page locked memory
            fprintf(stderr, "\nCopying uncompressed mask");
        
            uint32_t* h_resultMask = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numTotalResultWords));


            #if !USE_PAGEABLE_MEMORY
            copyDeviceBufferToHost(h_resultMask,
                                d_resultMask,
                                numTotalResultWords,
                                h_pinnedResultMask);
            #else

            cudaErrorCheck(cudaMemcpy(h_resultMask, d_resultMask, sizeof(uint32_t) * numTotalResultWords, cudaMemcpyDeviceToHost), "cudaMemcpy(h_resultMask, d_resultMask, sizeof(uint32_t) * numTotalResultWords, cudaMemcpyDeviceToHost)");

            #endif
            free(h_resultMask);

        }else
        {
            fprintf(stderr, "\n Copying compressed mask");

            uint32_t* h_compressedResultMask = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * (*h_refinedCount)));

            #if !USE_PAGEABLE_MEMORY
            copyDeviceBufferToHost(h_compressedResultMask,
                                d_compressedResultMask,
                                *h_refinedCount,
                                h_pinnedResultMask);
            
            #else

            cudaErrorCheck(cudaMemcpy(h_compressedResultMask, d_compressedResultMask, sizeof(uint32_t) * (*h_refinedCount), cudaMemcpyDeviceToHost), "cudaMemcpy(h_compressedResultMask, d_compressedResultMask, sizeof(uint32_t) * (*h_refinedCount), cudaMemcpyDeviceToHost)");

            #endif

            cudaErrorCheck(cudaFree(d_compressedResultMask), "cudaFree(d_compressedResultMask)");
            free(h_compressedResultMask);
        }

        cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
        cudaErrorCheck(cudaGetLastError(), "cudaGetLastError()");

        double resultCopyEnd = omp_get_wtime();

        timeToCopyResults += (resultCopyEnd - resultCopyStart);
        printf("\nTime to copy back the results is %.3f", timeToCopyResults);
    #else
        double resultCopyStart = omp_get_wtime();

        #if !USE_PAGEABLE_MEMORY
       
            copyDeviceBufferToHost(h_refinedCandidatePoints,
                                    d_refinedCandidatePoints,
                                    *h_refinedCount,
                                    h_pinnedResultMask);
        #else
            cudaErrorCheck(cudaMemcpy(h_refinedCandidatePoints, d_refinedCandidatePoints, sizeof(uint64_t) * (*h_refinedCount), cudaMemcpyDeviceToHost), "cudaMemcpy(h_refinedCandidatePoints, d_refinedCandidatePoints, sizeof(uint64_t) * (*h_refinedCount), cudaMemcpyDeviceToHost)");
        #endif
        double resultCopyEnd = omp_get_wtime();
        timeToCopyResults += (resultCopyEnd - resultCopyStart);
        printf("\nTime to copy back the results is %.3f", timeToCopyResults);
    #endif
    // NOTE - We are freeing result mask here as the result size is huge for large datasets and we quickly run out of memory
    free(h_resultmask);
    
    // TODO - It is better to reuse these as well.
    cudaErrorCheck(cudaFree(d_resultMask), "cudaFree(d_resultMask)");
    cudaErrorCheck(cudaFree(d_refinedCount), "cudaFree(d_refinedCount)");
    cudaErrorCheck(cudaFree(d_blockPrefixSum), "cudaFree(d_blockPrefixSum)");
    cudaErrorCheck(cudaFree(d_threadBlockToGlobalPrimMapper), "cudaFree(d_threadBlockToGlobalPrimMapper)");
    cudaErrorCheck(cudaFree(d_blockQueryIdx), "cudaFree(d_blockQueryIdx)");
    cudaErrorCheck(cudaFree(d_resultMaskPrefix), "cudaFree(d_resultMaskPrefix)");
}


extern "C" void getMaxNeighborsFromSampledDataset(float* dataset, uint32_t N, uint32_t DIM, float EPSILON,
                                                  uint32_t* maxNeighborCount)
{
    const int sampleSize = getSampleSize(N);

    *maxNeighborCount = 0;

    printf("Sampling %d points for max neighbor Count\n", N / sampleSize);

    for (int i = 0; i < N; i = i + sampleSize)
    {
        const float* queryPoint = dataset + i * DIM;
        uint32_t curNeighborCount = 0;
        for (int j = 0; j < N; j++)
        {
            const float* candidatePoint = dataset + j * DIM;
            if (isWithinEpsilon(queryPoint, candidatePoint, EPSILON, DIM))
            {
                curNeighborCount++;
            }
        }
        *maxNeighborCount = max(*maxNeighborCount, curNeighborCount);
    }
}

__device__ void getBoundingSphere(const SphereGPU a, const SphereGPU b, SphereGPU* result)
{
    float3 diff;
    diff.x = a.center.x - b.center.x;
    diff.y = a.center.y - b.center.y;
    diff.z = a.center.z - b.center.z;

    float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    result->center.x = (a.center.x + b.center.x) * 0.5f;
    result->center.y = (a.center.y + b.center.y) * 0.5f;
    result->center.z = (a.center.z + b.center.z) * 0.5f;

    result->radius = (distance + a.radius + b.radius) * 0.5f;
}


__global__ void findBestMateKernel(BoundingSphereGroupGPU* groups, size_t N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float bestRadius = FLT_MAX;
    int bestMate = -1;
    float bestMateCost = FLT_MAX; // just store the bounding sphere radius

    float3 myCenter = groups[tid].center;
    float myRadius = groups[tid].radius;

    for (int j = 0; j < N; j++)
    {
        if (j == tid) continue;

        float3 otherCenter = groups[j].center;
        float otherRadius = groups[j].radius;

        float dx = myCenter.x - otherCenter.x;
        float dy = myCenter.y - otherCenter.y;
        float dz = myCenter.z - otherCenter.z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        float mergedRadius = (dist + myRadius + otherRadius) * 0.5f;

        if (mergedRadius < bestRadius)
        {
            bestRadius = mergedRadius;
            bestMate = j;
            bestMateCost = mergedRadius;
        }
    }

    groups[tid].bestMateId = bestMate;
    groups[tid].bestMateCost = bestMateCost;
}

__global__ void findBestMateKernelOneQuery(BoundingSphereGroupCPU* groups, myFloat3 myCenter, float myRadius, size_t N,
                                           float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    myFloat3 otherCenter = groups[tid].center;
    float otherRadius = groups[tid].radius;

    float dx = myCenter.x - otherCenter.x;
    float dy = myCenter.y - otherCenter.y;
    float dz = myCenter.z - otherCenter.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    float mergedRadius = (dist + myRadius + otherRadius) * 0.5f;

    res[tid] = mergedRadius;
}

extern "C" void findBestMateGPUOneQuery(BoundingSphereGroupCPU* gpuGroups, myFloat3 myCenter, float myRadius, size_t N,
                                        float* mergedRadius, int* bestMateId)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    BoundingSphereGroupCPU* d_groups;
    cudaErrorCheck(cudaMalloc(&d_groups, N * sizeof(BoundingSphereGroupCPU)), "cudaMalloc(&d_groups, N * sizeof(BoundingSphereGroupCPU))");
    cudaErrorCheck(cudaMemcpy(d_groups, gpuGroups,
               N * sizeof(BoundingSphereGroupCPU),
               cudaMemcpyHostToDevice), "cudaMemcpy(d_groups, gpuGroups, N * sizeof(BoundingSphereGroupCPU), cudaMemcpyHostToDevice)");

    float* d_res;
    cudaErrorCheck(cudaMalloc(&d_res, N * sizeof(float)), "cudaMalloc(&d_res, N * sizeof(float))");
    findBestMateKernelOneQuery<<<numBlocks, blockSize>>>(d_groups, myCenter, myRadius, N, d_res);
    cudaErrorCheck(cudaGetLastError(), "findBestMateKernelOneQuery launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "findBestMateKernelOneQuery sync");
    cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    float* h_res = (float*)malloc(sizeof(float) * N);

    cudaErrorCheck(cudaMemcpy(h_res, d_res,
               N * sizeof(float),
               cudaMemcpyDeviceToHost), "cudaMemcpy(h_res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost)");

    for (int i = 0; i < N; i++)
    {
        if (h_res[i] < (*mergedRadius))
        {
            *mergedRadius = h_res[i];
            *bestMateId = i;
        }
    }

    cudaErrorCheck(cudaFree(d_groups), "cudaFree(d_groups)");
    cudaErrorCheck(cudaFree(d_res), "cudaFree(d_res)");
}

extern "C" void findBestMateGPU(BoundingSphereGroupGPU* gpuGroups, size_t N)
{
    // --- Launch GPU kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    BoundingSphereGroupGPU* d_groups;
    cudaErrorCheck(cudaMalloc(&d_groups, N * sizeof(BoundingSphereGroupGPU)), "cudaMalloc(&d_groups, N * sizeof(BoundingSphereGroupGPU))");
    cudaErrorCheck(cudaMemcpy(d_groups, gpuGroups,
               N * sizeof(BoundingSphereGroupGPU),
               cudaMemcpyHostToDevice), "cudaMemcpy(d_groups, gpuGroups, N * sizeof(BoundingSphereGroupGPU), cudaMemcpyHostToDevice)");

    findBestMateKernel<<<numBlocks, blockSize>>>(d_groups, N);
    cudaErrorCheck(cudaGetLastError(), "findBestMateKernel launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "findBestMateKernel sync");

    cudaErrorCheck(cudaMemcpy(gpuGroups, d_groups,
               N * sizeof(BoundingSphereGroupGPU),
               cudaMemcpyDeviceToHost), "cudaMemcpy(gpuGroups, d_groups, N * sizeof(BoundingSphereGroupGPU), cudaMemcpyDeviceToHost)");

    cudaErrorCheck(cudaFree(d_groups), "cudaFree(d_groups)");
}


//SORTING STUFF
extern "C" void sortByWorkload(uint64_t* d_candidatePoints, uint32_t* d_groupCount, const uint32_t N)
{
    auto t_candidatePoints = thrust::device_pointer_cast(d_candidatePoints);
    auto t_groupCount = thrust::device_pointer_cast(d_groupCount);

    thrust::sort(
        thrust::device,
        t_candidatePoints,
        t_candidatePoints + N,
        [t_groupCount] __device__ (const uint64_t& a, const uint64_t& b)
        {
            auto primA = static_cast<uint32_t>(a & 0xFFFFFFFFull);
            auto primB = static_cast<uint32_t>(b & 0xFFFFFFFFull);

            uint32_t countA = t_groupCount[primA + 1] - t_groupCount[primA];
            uint32_t countB = t_groupCount[primB + 1] - t_groupCount[primB];

            return countA > countB;
        }
    );
}


struct CompareSphereByX
{
    __device__ bool operator()(const SphereData& a, const SphereData& b) const
    {
        return a.sphere.center.x < b.sphere.center.x;
    }
};

struct CompareSphereByY
{
    __device__ bool operator()(const SphereData& a, const SphereData& b) const
    {
        return a.sphere.center.y < b.sphere.center.y;
    }
};

struct CompareSphereByZ
{
    __device__ bool operator()(const SphereData& a, const SphereData& b) const
    {
        return a.sphere.center.z < b.sphere.center.z;
    }
};


// GPU sorted kd-tree. Returns leaf ranges (begin inclusive, end exclusive) after MAX_KD_LEVELS splits.
thrust::host_vector<KDPartitionRange> computeKDPartitions(thrust::device_vector<SphereData>& allPointSpheresData)
{
    const uint32_t totalPoints = static_cast<uint32_t>(allPointSpheresData.size());
    thrust::host_vector<KDPartitionRange> emptyResult;

    if (totalPoints == 0)
    {
        return emptyResult;
    }

    const int maxThreads = std::max(omp_get_max_threads(), 1);
    std::vector<cudaStream_t> kdStreams(static_cast<size_t>(maxThreads), nullptr);
    bool streamsAllocated = true;
    for (int i = 0; i < maxThreads; ++i)
    {
        cudaError_t status = cudaStreamCreate(&kdStreams[static_cast<size_t>(i)]);
        if (status != cudaSuccess)
        {
            fprintf(stderr, "Unable to create stream for parallel sort on Thrust: %s\n",
                    cudaGetErrorString(status));
            streamsAllocated = false;
            break;
        }
    }

    if (!streamsAllocated)
    {
        kdStreams.assign(1u, static_cast<cudaStream_t>(0));
    }

    const uint32_t maxLeaves = 1u << MAX_KD_LEVELS;

    std::vector<KDPartitionRange> completedPartitions;
    completedPartitions.reserve(maxLeaves);

    std::vector<KDPartitionRange> currentLevel;
    currentLevel.reserve(maxLeaves);
    currentLevel.push_back({0u, totalPoints});

    for (uint32_t level = 0; level < MAX_KD_LEVELS && !currentLevel.empty(); ++level)
    {
        std::vector<KDPartitionRange> nextLevel;
        nextLevel.reserve(currentLevel.size() * 2u);

        const uint32_t axis = level % 3u;
        const size_t currentCount = currentLevel.size();
        const int activeThreads = std::max(1, static_cast<int>(std::min(currentCount, kdStreams.size())));

        #pragma omp parallel for schedule(static) num_threads(activeThreads)
        for (int curRange = 0; curRange < static_cast<int>(currentCount); ++curRange)
        {
            const KDPartitionRange range = currentLevel[static_cast<size_t>(curRange)];
            const uint32_t rangeSize = range.end - range.begin;

            if (rangeSize <= 1u)
            {
                #pragma omp critical
                {
                    completedPartitions.push_back(range);
                }
                continue;
            }

            auto rangeBegin = allPointSpheresData.begin() + range.begin;
            auto rangeEnd = allPointSpheresData.begin() + range.end;

            const int threadId = omp_get_thread_num();
            const size_t streamIndex = static_cast<size_t>(threadId % kdStreams.size());
            cudaStream_t curStream = kdStreams[streamIndex];

            switch (axis)
            {
            case 0u:
                thrust::sort(thrust::cuda::par.on(curStream), rangeBegin, rangeEnd, CompareSphereByX());
                break;
            case 1u:
                thrust::sort(thrust::cuda::par.on(curStream), rangeBegin, rangeEnd, CompareSphereByY());
                break;
            default:
                thrust::sort(thrust::cuda::par.on(curStream), rangeBegin, rangeEnd, CompareSphereByZ());
                break;
            }

            const uint32_t mid = range.begin + rangeSize / 2u;
        
            #pragma omp critical
            {
                nextLevel.push_back({range.begin, mid});
                nextLevel.push_back({mid, range.end});
            }
        }

        for (cudaStream_t stream : kdStreams)
        {
            cudaErrorCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)");
        }

        currentLevel.swap(nextLevel);
    }

    completedPartitions.insert(completedPartitions.end(), currentLevel.begin(), currentLevel.end());

    if (streamsAllocated)
    {
        for (cudaStream_t stream : kdStreams)
        {
            cudaErrorCheck(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)");
        }
    }

    return thrust::host_vector<KDPartitionRange>(completedPartitions.begin(), completedPartitions.end());
}


extern "C" uint32_t constructKDGroupsGPU(SphereData* hostData,
                                         uint32_t count,
                                         KDPartitionRange** partitions)
{
    if (partitions == nullptr)
    {
        return 0u;
    }

    *partitions = nullptr;

    if (hostData == nullptr || count == 0u)
    {
        return 0u;
    }

    thrust::device_vector<SphereData> d_allPointSpheresData(hostData, hostData + count);

    thrust::host_vector<KDPartitionRange> hostPartitions = computeKDPartitions(d_allPointSpheresData);

    thrust::copy(d_allPointSpheresData.begin(), d_allPointSpheresData.end(), hostData);

    if (!hostPartitions.empty())
    {
        KDPartitionRange* hostRanges = new KDPartitionRange[hostPartitions.size()];
        std::copy(hostPartitions.begin(), hostPartitions.end(), hostRanges);
        *partitions = hostRanges;
    }

    return static_cast<uint32_t>(hostPartitions.size());
}

// From gds_join
void warmUpGPU()
{
    // initialize all ten integers of a device_vector to 1 
    thrust::device_vector<int> D(10, 1); 
    // set the first seven elements of a vector to 9 
    thrust::fill(D.begin(), D.begin() + 7, 9); 
    // initialize a host_vector with the first five elements of D 
    thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
    // set the elements of H to 0, 1, 2, 3, ... 
    thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
    thrust::copy(H.begin(), H.end(), D.begin()); 
    // print D 
    for(int i = 0; i < D.size(); i++) 
    std::cout << " D[" << i << "] = " << D[i]; 
    return;
}
