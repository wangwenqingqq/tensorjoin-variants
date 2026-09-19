#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <owl/owl.h>
#include "deviceCode.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "owl/owl_host.h"
#include "string.h"
#include "omp.h"
#include "utility.h"
#include "owl/Group.h"

#define INCREMENT_SIZE 100
#define CV_THRESHOLD 60
#define FIRST_INTERSECTION_TEST false
#define SANITY_CHECK true

extern "C" char deviceCode_ptx[];

unsigned int AABB_TEST = 1;
unsigned int SPHERICAL_TEST = 0;
unsigned int INTERSECTION_TEST = SPHERICAL_TEST;
unsigned int NUM_INTERSECTION_TEST_POINTS = 1;

std::unordered_map<void*, BufferInfo> memoryTracker;
size_t curAllocatedMemory;
size_t totalAvailableMemory;

uint32_t rCall = 0;
uint32_t urCall = 0;


int main(int argc, char** argv)
{
    char* fileName = argv[1];
    const float EPSILON = atof(argv[2]);

    ReorderMode reorderMode = REORDER_HIGHEST_VARIANCE;
    if (argc > 3)
    {
        if (strcmp(argv[3], "highest") == 0)      reorderMode = REORDER_HIGHEST_VARIANCE;
        else if (strcmp(argv[3], "lowest") == 0)  reorderMode = REORDER_LOWEST_VARIANCE;
        else if (strcmp(argv[3], "random") == 0)  reorderMode = REORDER_RANDOM;
        else
        {
            std::cerr << "\nUnknown reorder mode '" << argv[3]
                      << "'. Use one of: highest | lowest | random" << std::endl;
            exit(1);
        }
    }

    THRESHOLD = THRESHOLD * EPSILON;

    unsigned int N, DIM;

    float* dataset = readData(fileName, &N, &DIM);

    printf("\n Warming up gpu ...\n");
    warmUpGPU();

    assert(DIM == DATASET_DIM);

    double start = omp_get_wtime();

    if (MAX_KD_LEVELS <= 0)
    {
        MAX_KD_LEVELS = (N > 1u) ? static_cast<int>(std::log(static_cast<double>(N))) : 1;
        if (MAX_KD_LEVELS < 1) MAX_KD_LEVELS = 1;
    }

    if (NUM_DATA_PARTITIONS * 3 > DIM && DIM > 3)
    {
        std::cerr << "\nNumber of partitions times 3 should be less than DIM" << std::endl;
        exit(1);
    }
    FILE * file;

    #if PARAM_VERBOSE
        printf("\n Number of points in the dataset are: %u", N);
        printf("\n Number of dataset dimensions are %u", DIM);
        printf("\n Dataset file : %s", fileName);
        printf("\n=========================================================");
        printf("\n Param USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING: %d", USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING);
        printf("\n Param USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING: %d", USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING);
        printf("\n Param USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING: %d", USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING);
        printf("\n Param USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING: %d", USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING);
        printf("\n Param USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING: %d", USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING);
        printf("\n=========================================================");
        printf("\n Param MAX_KD_LEVELS: %d", MAX_KD_LEVELS);
        printf("\n Param THREADS_TO_COPY: %d", THREADS_TO_COPY);
        printf("\n Param USE_UNCOMPRESSED_MASK: %d", USE_UNCOMPRESSED_MASK);
        printf("\n Param USE_PAGEABLE_MEMORY: %d", USE_PAGEABLE_MEMORY);
        printf("\n Param USE_FIXED_WORK_PER_BLOCK: %d", USE_FIXED_WORK_PER_BLOCK);
        printf("\n Param SORT_BY_WORKLOAD: %d", SORT_BY_WORKLOAD);
        printf("\n Param USE_CANDIDATE_POINT_COPY: %d", USE_CANDIDATE_POINT_COPY);
        printf("\n Param PINNED_MEMORY_OPT: %d", PINNED_MEMORY_SIZE_OPT);
        printf("\n=========================================================\n");
    #endif

    OWLContext context;
    OWLModule module;

    // Setup module
    setupModuleAndContext(deviceCode_ptx, context, module);

    // Reorder dimensions by variance
    double timeReorderStart = omp_get_wtime();
    dataset = reorderByDimensions(dataset, N, DIM, reorderMode);
    double timeReorderEnd = omp_get_wtime();

    printf("\nTime to reorder is %.3f seconds",timeReorderEnd - timeReorderStart);

    double totalTimeStart = omp_get_wtime();

    // RT PIPELINE TO ESTIMATE CANDIDATES AND ALLOCATE BUFFER ACCORDINGLY
    OWLGeom userSpheresGeom;
    OWLRayGen rayGen;
    OWLParams subPartition_lp;

    // Setup params and build shader programs
    setupVarsForShaderPrograms(context, module, userSpheresGeom, rayGen, subPartition_lp);

    OWLBuffer dataPointsSpheresBuffer;
    OWLBuffer candidatePoints;
    OWLGroup handle;
    OWLGroup spheresGroups;


    double RTTime = 0;


    struct curQueryDataPointPartitions sampledQueryDataset = fetchSampledQueryAndDataSpace(N);

    OWLBuffer sampledPointsToSearchSpheresBuffer;
    OWLBuffer sampledTruthBuffer;

    OWLBuffer queryIndices = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), sampledQueryDataset.queries.size(),sampledQueryDataset.queries.data());

    std::vector<uint32_t> mortonCode = computeMortonCode(dataset, N, DIM);

    uint32_t  NUM_PRIMITIVES;

    OWLBuffer OWL_datasetReordered;
    OWLBuffer owl_groupCountBuffer;
    OWLBuffer owl_groupedPointsBuffer;
    uint32_t maxPointsWithinPrimitive = 0;
    std::vector<uint32_t> h_groupCount;

    setupDataSpaceAndRayGenSpheresBuffer(context, userSpheresGeom, rayGen,
                                         sampledQueryDataset.queries,
                                         sampledQueryDataset.dataPartitions,
                                         sampledPointsToSearchSpheresBuffer,
                                         dataPointsSpheresBuffer,
                                         handle,
                                         spheresGroups,
                                         dataset,
                                         OWL_datasetReordered,
                                         owl_groupCountBuffer,
                                         h_groupCount,
                                         owl_groupedPointsBuffer,
                                         mortonCode,
                                         queryIndices,
                                         & NUM_PRIMITIVES,
                                         &maxPointsWithinPrimitive,
                                         DIM,
                                         EPSILON, subPartition_lp);


    setupIAS(userSpheresGeom, handle, spheresGroups, context, rayGen);

    
    auto sampledCandidateCount = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), 1,
                                                nullptr);


    double intersectionCountTestStart = omp_get_wtime();
    printf("\n Performing intersection count test");

    owlParamsSet1ui(subPartition_lp, "intersectionCountTest", 1);

    std::vector<uint32_t> h_intersectionCount(sampledQueryDataset.queries.size(), 0);

    OWLBuffer owl_intersectionCountBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t),
                                                                  sampledQueryDataset.queries.size(),
                                                                  h_intersectionCount.data());

    owlParamsSetBuffer(subPartition_lp, "intersectionCount", owl_intersectionCountBuffer);

    // To track primitive count
    std::vector<uint32_t> h_primitiveIntersectionCount( NUM_PRIMITIVES, 0);

    auto owl_primitiveIntersectionCount = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t),  NUM_PRIMITIVES,
                                                                h_primitiveIntersectionCount.data());

    owlParamsSetBuffer(subPartition_lp, "primitiveIntersectionCount", owl_primitiveIntersectionCount);

    double intersectionStart = omp_get_wtime();

    owlAsyncLaunch2D(rayGen, static_cast<int>(sampledQueryDataset.queries.size() * NUM_SAMPLED_DATA_PARTITIONS), 1,
                     subPartition_lp);

    owlLaunchSync(subPartition_lp);

    double intersectionEnd = omp_get_wtime();

    printf("\nTime for intersection test is %.3f", (intersectionEnd - intersectionStart));

    cudaError_t initialTestError = cudaGetLastError();
    if (initialTestError != cudaSuccess)
    {
        printf("\n RT initial intersection test - CUDA error: %s\n",cudaGetErrorString(initialTestError));
    }

    printf("\nIntersection test done.");

    double intersectionCountTestEnd = omp_get_wtime();

    auto owl_primitivePosToWrite = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), ( NUM_PRIMITIVES+1), nullptr);

    owlParamsSetBuffer(subPartition_lp, "primitivePosTowrite", owl_primitivePosToWrite);

    uint32_t*d_primitivePosToWrite = (uint32_t*)owlBufferGetPointer(owl_primitivePosToWrite, 0);

    auto* d_intersectionCountBuffer = (uint32_t*)owlBufferGetPointer(owl_intersectionCountBuffer, 0);

    auto* d_primitiveIntersectionCount = (uint32_t*)owlBufferGetPointer(owl_primitiveIntersectionCount, 0);

    cudaErrorCheck(cudaMemcpy(h_intersectionCount.data(), d_intersectionCountBuffer, sizeof(uint32_t) * sampledQueryDataset.queries.size(),
               cudaMemcpyDeviceToHost), "cudaMemcpy(h_intersectionCount.data(), d_intersectionCountBuffer, sizeof(uint32_t) * sampledQueryDataset.queries.size(), cudaMemcpyDeviceToHost)");


    cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");


    cudaErrorCheck(cudaMemset((uint32_t*)owlBufferGetPointer(owl_primitiveIntersectionCount, 0), 0,
               sizeof(uint32_t) *  NUM_PRIMITIVES), "cudaMemset((uint32_t*)owlBufferGetPointer(owl_primitiveIntersectionCount, 0), 0, sizeof(uint32_t) * NUM_PRIMITIVES)");

    uint64_t totalNodeIntersections = 0;

    for (uint32_t n = 0; n < sampledQueryDataset.queries.size(); n++)
    {
        totalNodeIntersections += h_intersectionCount[n];
    }


    float requiredMemory = (float)sizeof(uint32_t) * totalNodeIntersections +  (float )totalNodeIntersections * (maxPointsWithinPrimitive) / 8;

    size_t freeMem, totalMem;

    cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem), "cudaMemGetInfo(&freeMem, &totalMem)");
    float availmemory = (float)freeMem;

    fprintf(stderr, "\nNode intersections are %lu", totalNodeIntersections);
    fprintf(stderr, "\nRequire %.2f GB memory for candidatePoints. Available %.2f GB global memory",
            requiredMemory / 1024 / 1024 / 1024,
            availmemory / 1024 / 1024 / 1024);
    
    uint32_t*h_pinnedResultMask;
    uint32_t intersectionsPerBatch = totalNodeIntersections;
    if (availmemory < requiredMemory)
    {
        fprintf(stderr, "\nINSUFFICIENT MEMORY... Using batching scheme");
        //TODO : WE are already working on conservative memory here. Do we need this again?
        float conservativeAvailMemory = availmemory * NODE_INTERSECTION_FRACTION_OF_MEMORY;

        // My idea is that everytime, the query ray intersects with a primitive, we need 64 bits to store node intersection data
        // and we need atleast (query x points within primitive) to store the result mask.
        intersectionsPerBatch = floor(conservativeAvailMemory / (sizeof(uint32_t)+ (float)maxPointsWithinPrimitive/8)) * INTERSECTIONS_FRACTION;
    }

    uint64_t * h_refinedCandidatePoints;
    uint64_t * d_refinedCandidatePoints;
    #if USE_CANDIDATE_POINT_COPY
        printf("\nAllocating memory for candidate points key value pairs");
        size_t freeMemory, totalMemory;
        cudaErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory), "cudaMemGetInfo(&freeMemory, &totalMemory)");
        
        size_t usableMemory = freeMemory;
        // This should work in all worst case scenarios
        intersectionsPerBatch = ((double)(usableMemory) / (double)(sizeof(uint64_t) * maxPointsWithinPrimitive)) * 0.8;

        // Making space for refine candidatePoints to store key value pairs
        // NOTE - This is required to store neighbors data
        h_refinedCandidatePoints = (uint64_t*)malloc(sizeof(uint64_t) * intersectionsPerBatch * maxPointsWithinPrimitive);
        // Intersections per batch tells us the total number of intersections for all primitives
        cudaMalloc(&d_refinedCandidatePoints, sizeof(uint64_t) * intersectionsPerBatch * maxPointsWithinPrimitive);

        cudaError_t errorAlloc = cudaGetLastError();
        // Check free memory and then find intersections per batch
        if(errorAlloc != cudaSuccess)
        {
            fprintf(stderr, "\nError after memory allocation - %s", cudaGetErrorString(errorAlloc));
            fprintf(stderr, " This is due to allocating more than available memory. Reduce the number of intersections processed per batch to fix this.");
            exit(1);
        }
    #endif

    // Allocating pinned memory for result mask to reuse
    size_t requiredMemoryForResultMaskPerBatch = intersectionsPerBatch * (float)maxPointsWithinPrimitive / 8;

    double pinnedStart = omp_get_wtime();

    #if !USE_PAGEABLE_MEMORY
    // Using only 8 MiB as pinned memory to reuse
    cudaErrorCheck(cudaMallocHost(&h_pinnedResultMask, PINNED_MEMORY_SIZE), "cudaMallocHost(&h_pinnedResultMask, PINNED_MEMORY_SIZE)");
    #else
    // This can be used to copy traditional key value copy as well
    h_pinnedResultMask = (uint32_t*)malloc(PINNED_MEMORY_SIZE);
    #endif

    double pinnedEnd = omp_get_wtime();

    printf("\n Time to allocate pinned memory %lff\n", pinnedEnd - pinnedStart);

    unsigned long long sampledCount = 0;
    uint32_t queriesSize = sampledQueryDataset.queries.size();
    uint32_t curProcessedQueries = 0;

    double refineTime = 0;

    uint32_t* d_groupCount = (uint32_t*)owlBufferGetPointer(owl_groupCountBuffer, 0);
    uint32_t* d_groupedPoints = (uint32_t*)owlBufferGetPointer(owl_groupedPointsBuffer, 0);
    // Reordered dataset
    float* d_dataset = (float*)owlBufferGetPointer(OWL_datasetReordered, 0);

    double allBatchTime = omp_get_wtime();
    uint32_t curBatch = 0;

    auto owl_candidatePointsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), intersectionsPerBatch,
                                                               nullptr);

                                               
    uint32_t* d_candidatePoints = (uint32_t*)owlBufferGetPointer(owl_candidatePointsBuffer, 0);

    fprintf(stderr, "\n\nMemory allocated for candidate points is %.3lf GiB\n\n", sizeof(uint32_t)*(double)intersectionsPerBatch/1024/1024/1024);

    owlParamsSetBuffer(subPartition_lp, "candidatePoints", owl_candidatePointsBuffer);

    auto owl_totalCandidateCount = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), 1, nullptr);

    owlParamsSetBuffer(subPartition_lp, "totalCandidateCount", owl_totalCandidateCount);

    uint32_t* d_primitiveOrder;
    cudaErrorCheck(cudaMalloc(&d_primitiveOrder, sizeof(uint32_t)*( NUM_PRIMITIVES)), "cudaMalloc(&d_primitiveOrder, sizeof(uint32_t)*( NUM_PRIMITIVES))");

	static double workGenTime = 0;
	
    while (curProcessedQueries != queriesSize)
    {
        double workGenStart = omp_get_wtime();
        fprintf(stderr, "\nProcessing batch number - %u\n", curBatch);

        curBatch += 1;
        uint32_t curIntersectionCount = 0;

        std::vector<uint32_t> curQueries;
        while (curProcessedQueries < queriesSize && curIntersectionCount + h_intersectionCount[curProcessedQueries] <=
            intersectionsPerBatch)
        {
            curQueries.push_back(curProcessedQueries);
            curIntersectionCount += h_intersectionCount[curProcessedQueries];
            curProcessedQueries += 1;
        }

        if (curQueries.empty() && curProcessedQueries < queriesSize)
        {
            curQueries.push_back(curProcessedQueries);
            curIntersectionCount += h_intersectionCount[curProcessedQueries];
            curProcessedQueries += 1;
        }
        // Need to do a RT run again to find primitive pos to write
        cudaErrorCheck(cudaMemset(d_primitiveIntersectionCount, 0, sizeof(uint32_t) *  NUM_PRIMITIVES), "cudaMemset(d_primitiveIntersectionCount, 0, sizeof(uint32_t) * NUM_PRIMITIVES)");
        cudaErrorCheck(cudaMemset(d_intersectionCountBuffer, 0, sizeof(uint32_t) * sampledQueryDataset.queries.size()), "cudaMemset(d_intersectionCountBuffer, 0, sizeof(uint32_t) * sampledQueryDataset.queries.size())");

        owlParamsSet1ui(subPartition_lp, "intersectionCountTest", 1);

        cudaErrorCheck(cudaMemcpy((void*)owlBufferGetPointer(queryIndices,0), curQueries.data(), sizeof(uint32_t)*curQueries.size(), cudaMemcpyHostToDevice), "cudaMemcpy((void*)owlBufferGetPointer(queryIndices,0), curQueries.data(), sizeof(uint32_t)*curQueries.size(), cudaMemcpyHostToDevice)");

        owlParamsSet1ui(subPartition_lp, "maxCandidatePointsPerQueryPoint", curIntersectionCount);

        printf("\n max intersections are %u whereas current count is %u" , intersectionsPerBatch, curIntersectionCount);

        cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

        double RTInitialRunStart = omp_get_wtime();

        owlAsyncLaunch2D(rayGen, static_cast<int>(curQueries.size() * NUM_SAMPLED_DATA_PARTITIONS), 1,
                         subPartition_lp);

        owlLaunchSync(subPartition_lp);

		double RTInitialRunEnd = omp_get_wtime();
		
        cudaErrorCheck(cudaMemcpy(h_primitiveIntersectionCount.data(), d_primitiveIntersectionCount,
          sizeof(uint32_t) *  NUM_PRIMITIVES, cudaMemcpyDeviceToHost), "cudaMemcpy(h_primitiveIntersectionCount.data(), d_primitiveIntersectionCount, sizeof(uint32_t) * NUM_PRIMITIVES, cudaMemcpyDeviceToHost)");

        // Tracking order in which query primitives should be written to memory
        std::vector<uint32_t> h_primitives( NUM_PRIMITIVES, 0);
        // Adding + 1 to create a prefix sum array which can be used later while refining
        std::vector<uint32_t> h_primitivesPosToWrite( NUM_PRIMITIVES + 1, 0);

        for (uint32_t p = 0; p <  NUM_PRIMITIVES; p++)
        {
            h_primitives[p] = p;
        }

        #if SORT_BY_WORKLOAD
        thrust::sort_by_key(h_primitiveIntersectionCount.begin(), h_primitiveIntersectionCount.end(), h_primitives.begin(),
                            thrust::greater<uint32_t>());
        #endif

        uint32_t currentPrimOffset = 0;
        
        for (uint32_t p = 0; p <  NUM_PRIMITIVES; p++)
        {
            h_primitivesPosToWrite[h_primitives[p]] = currentPrimOffset;
            currentPrimOffset += h_primitiveIntersectionCount[p];
            // Metrics
            // numPointComparisons += (h_primitiveIntersectionCount[p]);
        }

        h_primitivesPosToWrite[ NUM_PRIMITIVES] = currentPrimOffset;

        cudaErrorCheck(cudaMemcpy(d_primitivePosToWrite, h_primitivesPosToWrite.data(), sizeof(uint32_t)*( NUM_PRIMITIVES + 1), cudaMemcpyHostToDevice), "cudaMemcpy(d_primitivePosToWrite, h_primitivesPosToWrite.data(), sizeof(uint32_t)*( NUM_PRIMITIVES + 1), cudaMemcpyHostToDevice)");

        cudaErrorCheck(cudaMemset(d_primitiveIntersectionCount, 0, sizeof(uint32_t) *  NUM_PRIMITIVES), "cudaMemset(d_primitiveIntersectionCount, 0, sizeof(uint32_t) * NUM_PRIMITIVES)");

        // End of rerun of RT
        owlParamsSet1ui(subPartition_lp, "intersectionCountTest", 0);

        cudaErrorCheck(cudaMemset((uint32_t*)owlBufferGetPointer(sampledCandidateCount, 0), 0, sizeof(uint32_t)), "cudaMemset((uint32_t*)owlBufferGetPointer(sampledCandidateCount, 0), 0, sizeof(uint32_t))");

        uint32_t h_totalCandidateCount = 0;

        owlBufferUpload(owl_totalCandidateCount, &h_totalCandidateCount);

        double queryLaunchStart = omp_get_wtime();

        double queryTimeStart = omp_get_wtime();
        owlAsyncLaunch2D(rayGen, static_cast<int>(curQueries.size() * NUM_SAMPLED_DATA_PARTITIONS), 1,
                         subPartition_lp);

        // Sync the same stream
        owlLaunchSync(subPartition_lp);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("\n RT Query - CUDA error: %s\n", cudaGetErrorString(error));
        }

        
        double queryTimeEnd = omp_get_wtime();

        RTTime += queryTimeEnd - queryTimeStart;
        RTTime += RTInitialRunEnd - RTInitialRunStart;

        unsigned long long h_refinedCount = 0;

        double workGenEnd = omp_get_wtime();

        workGenTime += workGenEnd - workGenStart;
        printf("\nCum Work Gen time is %.3f seconds", workGenTime);

        double refineStart = omp_get_wtime();

        cudaMemcpy(d_primitiveOrder, h_primitives.data(), sizeof(uint32_t)*( NUM_PRIMITIVES), cudaMemcpyHostToDevice);

        refineCandidatePointsGrid(d_groupCount, h_groupCount.data(), d_groupedPoints,
                                  d_candidatePoints, d_refinedCandidatePoints, h_refinedCandidatePoints, &h_refinedCount, d_dataset, d_primitivePosToWrite, d_primitiveIntersectionCount, d_primitiveOrder, h_primitives.data(), h_pinnedResultMask,
                                  curIntersectionCount,  NUM_PRIMITIVES, DIM, EPSILON * EPSILON);

        double refineEnd = omp_get_wtime();

        refineTime += refineEnd - refineStart;
        
        sampledCount += h_refinedCount;
    }

    double allBatchEnd = omp_get_wtime();
    double end = omp_get_wtime();
    printf("\nAll batches time is %.3f seconds", allBatchEnd-allBatchTime);

 
    puts("\n=============================================================");
    printf("\n Total number of queries are %lu", sampledQueryDataset.queries.size());
    printf("\n Time to perform only intersection test = %.3f seconds",
           (intersectionCountTestEnd - intersectionCountTestStart));
    puts("\n=============================================================");
    printf("\n TOTAL NUMBER OF IDENTIFIED NEIGHBORS ARE %llu", sampledCount);
    // printf("\n NUM POINT COMPARIONS is %llu", numPointComparisons);
    printf("\n Refine time is %.3f seconds", refineTime);
    printf("\n RT time is %.3f seconds", RTTime);
    // printf("\n CUDA ONLY TIME is %.3f", cudaOnlyTime);
    printf("\nTOTAL TIME (REFINE + QUERY) is %.3f seconds", (refineTime) + (RTTime));
    printf("\n\nTOTAL TIME INCLUDINF EVERYTHING - %.3f seconds\n\n", end-start);

    const char* reorderModeName = (reorderMode == REORDER_RANDOM)          ? "random"
                                  : (reorderMode == REORDER_LOWEST_VARIANCE) ? "lowest"
                                                                             : "highest";

    FILE* resultsFile = fopen("./results.txt", "a");
    if (resultsFile)
    {
        fprintf(resultsFile, "\n**********************START OF RESULT**********************\n");
        fprintf(resultsFile, "\n=============================================================\n");
        fprintf(resultsFile, "Dataset file: %s\n", fileName);
        fprintf(resultsFile, "Dataset points: %u\n", N);
        fprintf(resultsFile, "Dataset dimensions: %u\n", DIM);
        fprintf(resultsFile, "Epsilon: %.6f\n", EPSILON);
        fprintf(resultsFile, "Reorder mode: %s\n", reorderModeName);
        fprintf(resultsFile, "USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING: %d\n", USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING);
        fprintf(resultsFile, "USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING: %d\n", USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING);
        fprintf(resultsFile, "USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING: %d\n", USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING);
        fprintf(resultsFile, "USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING: %d\n", USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING);
        fprintf(resultsFile, "USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING: %d\n", USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING);
        fprintf(resultsFile, "MAX_KD_LEVELS: %d\n", MAX_KD_LEVELS);
        fprintf(resultsFile, "THREADS_TO_COPY: %d\n", THREADS_TO_COPY);
        fprintf(resultsFile, "USE_UNCOMPRESSED_MASK: %d\n", USE_UNCOMPRESSED_MASK);
        fprintf(resultsFile, "USE_PAGEABLE_MEMORY: %d\n", USE_PAGEABLE_MEMORY);
        fprintf(resultsFile, "USE_FIXED_WORK_PER_BLOCK: %d\n", USE_FIXED_WORK_PER_BLOCK);
        fprintf(resultsFile, "SORT_BY_WORKLOAD: %d\n", SORT_BY_WORKLOAD);
        fprintf(resultsFile, "USE_CANDIDATE_POINT_COPY: %d\n", USE_CANDIDATE_POINT_COPY);
        fprintf(resultsFile, "PINNED_MEMORY_OPT: %d\n", PINNED_MEMORY_SIZE_OPT);
        fprintf(resultsFile, "-------------------------------------------------------------\n");
        fprintf(resultsFile, "Total number of queries: %zu\n", sampledQueryDataset.queries.size());
        fprintf(resultsFile, "Time to perform only intersection test: %.3f seconds\n",
                (intersectionCountTestEnd - intersectionCountTestStart));
        fprintf(resultsFile, "Total identified neighbors: %llu\n", sampledCount);
        fprintf(resultsFile, "Time for KD tree construction: %.3f seconds\n", kdTreeConstructionTime);
        fprintf(resultsFile, "CUDA kernel only time: %.3lf\n", cudaOnlyTime);
        fprintf(resultsFile, "Time to copy results only: %.3lf\n", timeToCopyResults);
        fprintf(resultsFile, "Time to compress results: %.3lf\n", timeToCompress);
        // Work generation time is the amount of time spent on RT + generating device and host pointers and other prep
        fprintf(resultsFile, "Work Generation time: %.3lf\n", workGenTime);
        fprintf(resultsFile, "Refine time: %.3f seconds\n", refineTime);
        fprintf(resultsFile, "RT time: %.3f seconds\n", RTTime);
        fprintf(resultsFile, "TOTAL TIME (REFINE + QUERY): %.3f seconds\n", (refineTime) + (RTTime));
        fprintf(resultsFile, "TOTAL TIME INCLUDING EVERYTHING: %.3f seconds\n", end-start);
        fprintf(resultsFile, "\n");
        fprintf(resultsFile, "\n**********************END OF RESULT**********************\n");
        fclose(resultsFile);
    }
    else
    {
        printf("\nFailed to open results.txt for writing.\n");
    }

    return 0;
}
