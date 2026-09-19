#pragma once

#include <owl/owl.h>
#include <cstdint>
#include <cuda_runtime.h>
#include "stdio.h"
#include "deviceCode.h"
#include <queue>
#include  <iostream>
#include "omp.h"
#include <fstream>
#include <string.h>
#include <thread>
#include <chrono>
#include <set>
#include <unordered_set>
#include <mutex>
#include <thrust/sort.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "hilbert.hpp"
#include <random>
#include <thrust/device_ptr.h>
#include <unordered_map>


static void cudaErrorCheckImpl(cudaError_t e, const char* what, const char* file, int line) {
  if (e != cudaSuccess)
  {
    printf("CUDA error (%s) at %s:%d: %s\n", what, file, line, cudaGetErrorString(e));
    fflush(stdout);
    exit(EXIT_FAILURE);
  }
}

#define cudaErrorCheck(e, what) cudaErrorCheckImpl((e), (what), __FILE__, __LINE__)


#ifndef MAX_KD_LEVELS_OPT
#define MAX_KD_LEVELS_OPT (-1)
#endif

#ifndef THREADS_TO_COPY_OPT
#define THREADS_TO_COPY_OPT 8
#endif

#ifndef NUM_CALCULATIONS_PER_THREAD_OPT
#define NUM_CALCULATIONS_PER_THREAD_OPT 128
#endif

#ifndef USE_UNCOMPRESSED_MASK_OPT
#define USE_UNCOMPRESSED_MASK_OPT 0
#endif

#define INVALID_CANDIDATE_POINT UINT64_MAX
#define DIMS_PER_DATA_PARTITION 3
#define NUM_DATA_PARTITIONS 1

inline double FRACTION_OF_MEMORY = 1.15;
const inline double INTERSECTIONS_FRACTION = 0.85;
const uint32_t MAX_STREAMS = 1;
const uint32_t MAX_DATA_PARTITION_POINTS = 40000;
const uint32_t MAX_QUERY_POINTS = 25000;
const uint32_t MAX_SIZE = MAX_DATA_PARTITION_POINTS * MAX_QUERY_POINTS;
const uint32_t MIN_CLUSTER_POINTS = UINT32_MAX;
inline double THRESHOLD = 0.01;
const uint32_t MAX_CLUSTERED_QUERY_POINTS = 1000;
const inline double SAMPLING_FRACTION = 1;
const inline double ERROR_ESTIMATE = 1.1;
const inline double USABLE_MEMORY_ERROR_ESTIMATE = 0.9;

#define NUM_SAMPLED_DATA_PARTITIONS 1
#define EXTRA_BATCHES 0
#define NUM_BITS_PER_QUERY_CANDIDATE_COMB 4
#define NUM_SUB_TRUTH_BUFFER_PER_WORD 8 // 32/4
#define INVALID_TRUTH_BUFFER_VAL 0xF
#define MAX_VISIBILITY_MASKS 8
// Switch to use when users are interested in using results in memory
#define EXTRACT_NEIGHBORS true
#define PARAM_VERBOSE true
#define POINTS_PER_GROUP 3000
#define POINTS_IN_SMALL_GROUP 250
#define SMALL_GROUP_THRESHOLD 25
#define LARGE_GROUP_THRESHOLD 15
#define DEFAULT_THRESHOLD 0.5
#define MAX_BALL_LEVELS 2000
#define MIN_BALL_LEAF_SIZE 100

// Runtime kd-tree height. Initialized from the compile-time option; when that
// is the "auto" sentinel (<= 0) it is resolved to floor(ln(|D|)) in main()
// before the kd-tree is built.
inline int MAX_KD_LEVELS = MAX_KD_LEVELS_OPT;

#define THREADS_TO_COPY THREADS_TO_COPY_OPT
#define USE_UNCOMPRESSED_MASK USE_UNCOMPRESSED_MASK_OPT
#define USE_PAGEABLE_MEMORY USE_PAGEABLE_MEMORY_OPT
#define USE_FIXED_WORK_PER_BLOCK USE_FIXED_WORK_PER_BLOCK_OPT
#define SORT_BY_WORKLOAD USE_SORT_BY_WORKLOAD_OPT
#define USE_CANDIDATE_POINT_COPY USE_USE_CANDIDATE_POINT_COPY_OPT
#define PINNED_MEMORY_SIZE PINNED_MEMORY_SIZE_OPT 
#define NODE_INTERSECTION_FRACTION_OF_MEMORY 0.99

// Switches for different batching/ non-batching schemes
#define USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT
#define USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT
#define USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT 
#define USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT 
#define USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT 
const inline double BALL_MIN_FRACTION = 0.6;

// Shared memory limitations
#ifndef SAFE_SHARED_MEMORY
#define SAFE_SHARED_MEMORY_RUNTIME 1
#ifdef __CUDA_ARCH__
#define SAFE_SHARED_MEMORY d_safeSharedMemory
#else
int getSafeSharedMemory();
#define SAFE_SHARED_MEMORY getSafeSharedMemory()
#endif
#endif
#ifndef DATASET_DIM
#define DATASET_DIM 18 // dimension; override at build time with -DDIM=<n>
#endif
#define SHARED_MEM_PRIM_POINT_COUNT (SAFE_SHARED_MEMORY / (sizeof(float)*DATASET_DIM))
#define SHARED_MEM_QUERY_POINT_COUNT 2048
#define NUM_CALCULATIONS_PER_THREAD NUM_CALCULATIONS_PER_THREAD_OPT

inline unsigned long long numPointComparisons = 0;

extern double cudaOnlyTime;
extern double timeToCopyResults;
extern double timeToCompress;
extern double kdTreeConstructionTime;

enum class BufferType { OWL_ALLOCATED_BUFFER, DEVICE_BUFFER };

struct BufferInfo
{
    BufferType type;
    size_t size;
};

extern std::unordered_map<void*, BufferInfo> memoryTracker;
extern size_t curAllocatedMemory;
extern size_t totalAvailableMemory;


struct NodeIndices
{
    unsigned int pid; 
    NodeIndices* leftIdx; 
    NodeIndices* rightIdx; 

    explicit NodeIndices(const unsigned int id)
    {
        pid = id;
        leftIdx = nullptr;
        rightIdx = nullptr;
    }
};

struct keyValData
{
    float* point;
    unsigned int pid;
    unsigned int sortDim;

    keyValData(float* point, unsigned int pid, unsigned int sortDim)
    {
        this->point = point;
        this->pid = pid;
        this->sortDim = sortDim;
    }
};

struct queryDataPointPartitions
{
    std::vector<std::vector<uint32_t>> queries;
    std::vector<std::vector<uint32_t>> dataPartitions;

    queryDataPointPartitions(std::vector<std::vector<uint32_t>> queries,
                             std::vector<std::vector<uint32_t>>dataPartitions)
    {

        this->dataPartitions = dataPartitions;
        this->queries = queries;
    }
};

struct curQueryDataPointPartitions
{
    std::vector<uint32_t> queries;
    std::vector<uint32_t> dataPartitions;

    curQueryDataPointPartitions(std::vector<uint32_t> queries,
                             std::vector<uint32_t> dataPartitions)
    {

        this->dataPartitions = std::move(dataPartitions);
        this->queries = std::move(queries);
    }

    curQueryDataPointPartitions() = default;
};


struct fixedQueryVariableDataPointPartitions
{
    std::vector<uint32_t> queries;
    std::vector<std::vector<uint32_t>> dataPartitions;

    fixedQueryVariableDataPointPartitions(std::vector<uint32_t> queries,
                            std::vector<std::vector<uint32_t>> dataPartitions)
    {

        this->dataPartitions = dataPartitions;
        this->queries = queries;
    }

    fixedQueryVariableDataPointPartitions()
    = default;
};

struct SphereData
{
    Sphere sphere;
    uint32_t pID;

    __host__ __device__ SphereData(Sphere sphere, uint32_t pID)
    {
        this->sphere = sphere;
        this->pID = pID;
    }

    __host__ __device__ SphereData()
    {
        this->sphere = Sphere();
        this->pID = UINT_MAX;
    }

    __host__ __device__ SphereData(const SphereData&) = default;
    __host__ __device__ SphereData& operator=(const SphereData&) = default;
};

struct KDPartitionRange
{
    uint32_t begin;
    uint32_t end;
};


struct boundingSphereGroup
{
    Sphere sphere;
    std::vector<SphereData> sphereGroup;
};

struct boundingSphereGroupBestMate
{
    struct boundingSphereGroup sphereGroup;
    uint32_t bestMateId;

    boundingSphereGroupBestMate(boundingSphereGroup sphereGroup, uint32_t bestMateId)
    {
        this->bestMateId = bestMateId;
        this->sphereGroup = std::move(sphereGroup);
    }
};

struct queueStruct
{
    NodeIndices* parentNode;
    std::vector<keyValData> points;
    unsigned int depth;
    bool isLeft;

    queueStruct(NodeIndices* parentNode, std::vector<keyValData> points, unsigned int depth, bool isLeft)
    {
        this->parentNode = parentNode;
        this->points = points;
        this->depth = depth;
        this->isLeft = isLeft;
    }
};


struct SphereGPU
{
    float3 center;
    float radius;
};

struct SphereDataGPU
{
    SphereGPU sphere;
    uint32_t pID;
};

struct BoundingSphereGroupGPU
{
    float3 center;  
    float radius;   
    int bestMateId; 
    float bestMateCost;
};

struct myFloat3
{
    float x;
    float y;
    float z;

};
// I hate this
struct BoundingSphereGroupCPU
{
    myFloat3 center;
    float radius;
    int bestMateId;
    float bestMateCost;
};



extern "C" void refineCandidatePoints(uint32_t* queryPoints, uint64_t* d_candidatePoints,
                                      uint64_t** h_refinedCandidatePoints, uint32_t candidatePointsUnionTotalCount, uint32_t*dataPoints,  uint32_t**d_queryPoints_t, uint32_t**d_dataPoints_t,
                                      float* d_dataset, uint32_t N,uint32_t DATA_POINTS, float EPSILON, uint32_t DIM, cudaStream_t stream);

__global__ void identifyNeighbors(uint32_t* queryPoints, uint64_t* candidatePoints,
                                  uint32_t*dataPoints,
                                  float* dataset, uint32_t N, uint32_t MAX_CANDIDATES, float EPSILON,
                                  uint32_t DIM);

__inline__ __host__ __device__ bool isWithinEpsilon(const float* point1, const float* point2, const float EPSILON,
                                                    const uint32_t DIM);


// UTILITY_SEQ STUFF

std::vector<Sphere> buildSpheresPrimBuffer(std::vector<uint32_t> pointsToSearch, float* dataset, unsigned DIM,
                                           const float EPSILON);
std::vector<Sphere> buildCumDataPartitionSpheresPrimeBuffer(std::vector<uint32_t> pointsToSearch, float* dataset,
                                                            unsigned DIM, const float EPSILON,
                                                            const unsigned int NUM_DATA_PARTITIONS_);
int getSampleSize(uint32_t N);

enum ReorderMode
{
    REORDER_HIGHEST_VARIANCE = 0, 
    REORDER_LOWEST_VARIANCE  = 1, 
    REORDER_RANDOM           = 2  
};
#define REORDER_RANDOM_SEED 42u

float* reorderByDimensions(float* dataset, int N, int DIM, ReorderMode mode);
float* readData(char* fileName, unsigned int* N, unsigned int* dim);
bool compareKeyValDataStruct(keyValData a, keyValData b);
void sortKeyVal(std::vector<keyValData>& keyValDataset, unsigned int sortDim);
struct queryDataPointPartitions insertAllWithIdx(std::vector<keyValData>& keyValDataset, unsigned int NPOINTS,
                                                 unsigned int IDIM,
                                                 unsigned int numMaxTreeLevels,
                                                 std::vector<std::vector<unsigned int>>& partitions,
                                                 const float EPSILON);
void setupModuleAndContext(const char* deviceCode_ptx, OWLContext& context, OWLModule& module);
void setupIAS(OWLGeom& SpheresGeom, OWLGroup &handle, OWLGroup& spheresGroups, OWLContext context, OWLRayGen rayGen);
void setupVarsForShaderPrograms(OWLContext context, OWLModule module,  OWLGeom&userSpheresGeom, OWLRayGen& rayGen,
                                OWLParams& lp);
void setInterSectionTestAndDataPartitions(OWLParams& lp, unsigned int INTERSECTION_TEST,
                                          unsigned int NUM_DATA_PARTITIONS_);

void setupDataSpaceAndRayGenSpheresBuffer(OWLContext context, OWLGeom& SpheresGeom, OWLRayGen rayGen,
                                          std::vector<uint32_t> pointsToSearch, std::vector<uint32_t> dataPoints,
                                          OWLBuffer& pointsToSearchSpheresBuffer,
                                          OWLBuffer& dataPointsSpheresBuffer,
                                          OWLGroup& handle,
                                          OWLGroup& spheresGroups,
                                          float* dataset,
                                          OWLBuffer &OWL_datasetReordered,
                                          OWLBuffer &owl_groupCountBuffer,
                                          std::vector<uint32_t> &h_groupCount,
                                          OWLBuffer &owl_groupedPointsBuffer,
                                          std::vector<uint32_t> mortonCode,
                                          OWLBuffer &queryIndices,
                                          uint32_t*GROUPED_QUERY_COUNT,
                                          uint32_t*maxPointsWithinPrimitive,
                                          unsigned int DIM, float EPSILON, OWLParams& lp);

void updateLaunchParams(unsigned int NUM_POINTS_TO_SEARCH, unsigned int MAX_CANDIDATE_POINTS, OWLBuffer & d_totalCandidateCount,
                        OWLBuffer& candidatePointsCount, OWLBuffer& candidatePoints, uint32_t** candidateCount,
                        uint64_t** candidates, OWLContext context, OWLParams& lp);
void updateRayGenSphereBuffer(OWLContext context, std::vector<uint32_t> pointsToSearch,
                              OWLBuffer& pointsToSearchSpheresBuffer,
                              OWLBuffer& truthBuffer,
                              float* dataset, unsigned int DIM, float EPSILON,
                              OWLParams& lp, const unsigned int NUM_DATA_PARTITIONS_, const unsigned int N);

void setRayGenBuffer(OWLRayGen rayGen, OWLBuffer pointsToSearchSpheresBuffer);

void updateLaunchParamsWithExactBuffer(unsigned int NUM_POINTS_TO_SEARCH, const uint32_t* candidatePointsPerQuery,
                                       uint64_t totalCandidatePoints, OWLBuffer& candidatePointsCount,
                                       OWLBuffer& candidatePoints, OWLBuffer& candidatePointsOffset,
                                       uint32_t** candidateCount, uint32_t** candidates,
                                       uint32_t** candidatePointsOffsetPerPartition, OWLContext context, OWLParams& lp);
double getUsableMemory(int deviceID);

void updateQueriesInLaunchParams(OWLParams& lp, OWLBuffer& pointsToSearchSpheresBuffer,
                                 const uint32_t POINTS_TO_SEARCH);

void updateDataSpace(OWLContext context, OWLParams& lp, OWLRayGen rayGen, OWLGeom* SpheresGeom,
                     std::vector<OWLBuffer>& dataPointsSpheresBuffer, const unsigned int NUM_DATA_POINTS,
                     const unsigned int NUM_DATA_PARTITIONS_);


OWLBuffer registerMemory(OWLContext context,
                         OWLDataType type,
                         size_t count,
                         const void* data,
                         size_t allocatedMemory);

void registerMemory(void** d_ptr, size_t allocatedMemory);

void unRegisterMemory(void* d_ptr, BufferType bufferType);

void setupCUDAStream();

void releaseAll();


size_t estimateMemoryForEachPartitionForRT(const uint64_t* partitionCandidatePoints,
                                           const uint32_t* pointsToSearchPerPartition,
                                           std::vector<size_t>& memoryForEachPartitionForRT,
                                           uint32_t NUM_PARTITIONS, unsigned int N);

size_t estimateMemoryForEachPartitionForCuda(const uint32_t* pointsToSearchPerPartition,
                                             std::vector<size_t>& memoryForEachPartitionForCuda,
                                             uint32_t NUM_PARTITIONS);
size_t getCurrentAvailableMemory();

void waitForMemory(size_t requiredMem);

bool hasEnoughMemory(size_t requiredMem);

void resetTruthBuffer(OWLContext context, OWLParams& lp, OWLBuffer& truthBuffer, const unsigned int POINTS_TO_SEARCH,
                      const unsigned int N);


void partitionCandidatesToFitMemory(std::vector<std::vector<unsigned int>>& partitions,
                                    std::vector<uint32_t*>& candidatePointsForEachQueryPerPartition,
                                    std::vector<std::vector<uint32_t>>& candidatePointsForEachQueryPerPartitionVec,
                                    std::vector<uint64_t>& partitionCandidatePoints,
                                    std::vector<unsigned int>& PARTITION_SIZE,
                                    std::vector<OWLBuffer> pointsToSearchSpheresBuffer,
                                    std::vector<OWLBuffer> truthBuffer,
                                    uint32_t* NUM_PARTITIONS,
                                    size_t maxMemConservative,
                                    uint32_t NUM_DATA_PARTITIONS_);


struct fixedQueryVariableDataPointPartitions generateQueryDataPartitionsOnDemand(
    std::vector<keyValData>& sortedKeyValDataset,
    std::unordered_set<uint32_t> clusteredPoints,
    uint32_t* queryStartIndex,
    float* dataset,
    unsigned int NPOINTS,
    uint32_t DIM,
    const float EPSILON);

std::unordered_set<uint32_t> findClusteredPoints(std::vector<keyValData>& sortedKeyValDataset);


extern "C" void refineClusteredQueryPoints(uint32_t* d_queryPoints, uint32_t* queryPoints, float* d_dataset,
                                           uint32_t queryCount, uint32_t N,
                                           uint32_t DIM,
                                           float EPSILON, uint32_t* d_resCount, uint32_t*res, cudaStream_t stream);

__global__ void identifyClusteredNeighbors(uint32_t* d_queryPoints, float* d_dataset, uint32_t queryCount, uint32_t N,
                                           uint32_t DIM, float EPSILON, uint32_t* resCount);


struct curQueryDataPointPartitions fetchSampledQueryAndDataSpace(uint32_t N);

std::vector<uint32_t> fetchNextQueries(std::vector<keyValData>& sortedKeyValDataset, uint32_t startingIndex,
                                       uint32_t batchSize);

std::vector<uint32_t> sortQueryPrimitives(uint64_t* queryPrimitives, std::vector<uint32_t> mortonCode, uint32_t queryCount);

std::vector<uint32_t> fetchNextSortedQueries(std::vector<uint32_t> sortedQueries, uint32_t startingIndex,
                                             uint32_t batchSize);

std::vector<uint32_t> computeMortonCode(float*dataset, uint32_t N, uint32_t DIM);

Sphere constructBoundingSphere(std::vector<Sphere>& pointsGroup);

std::vector<boundingSphereGroup> constructDistinctNonOverlappingSpheres(std::vector<boundingSphereGroup>& sphereGroups);

extern "C" void findBestMateGPU(BoundingSphereGroupGPU* h_groups, size_t N);
__global__ void findBestMateKernel(BoundingSphereGroupGPU* groups, size_t N);
__device__ void getBoundingSphere(const SphereGPU a, const SphereGPU b, SphereGPU* result);

extern "C" void findBestMateGPUOneQuery(BoundingSphereGroupCPU* gpuGroups, myFloat3 myCenter, float myRadius, size_t N,
                                        float* mergedRadius, int* bestMateId);

__global__ void findBestMateKernelOneQuery(BoundingSphereGroupCPU* groups, myFloat3 myCenter, float myRadius, size_t N,
                                           float* res);

uint32_t EncodeMorton3(uint32_t x, uint32_t y, uint32_t z);

extern "C" void refineCandidatePointsGrid(uint32_t* d_groupCount, uint32_t*h_groupCount, uint32_t* d_groupedPoints,
                                       uint32_t* d_candidatePoints, uint64_t*d_refinedCandidatePoints, uint64_t*h_refinedCandidatePoints, unsigned long long* h_refinedCount, float* d_dataset, uint32_t*d_primitivePosToWrite, uint32_t*d_primitiveIntersectionCount, uint32_t*d_primitiveOrder, uint32_t*h_primitiveOrder,uint32_t *h_pinnedResult,                                       uint32_t N, uint32_t NUM_PRIMITIVES, uint32_t DIM, float EPSILON_SQ);



extern "C" void refineClusteredQueryPoints(uint32_t* d_queryPoints, uint32_t* queryPoints, float* d_dataset,
                                           uint32_t queryCount, uint32_t N,
                                           uint32_t DIM,
                                           float EPSILON, uint32_t* d_resCount, uint32_t* res, cudaStream_t stream);

extern "C" void sortByWorkload(uint64_t*d_candidatePoints, uint32_t* d_groupCount, const uint32_t N);

extern "C" uint32_t constructKDGroupsGPU(SphereData* hostData,
                                         uint32_t count,
                                         KDPartitionRange** partitions);


void warmUpGPU();


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
    uint64_t totalWords);

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
    uint64_t totalWords);

__global__ void identifyNeighborsGridPrimitiveSharedQueryShared(
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
    uint64_t totalWords);

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
    uint64_t totalWords);

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
    uint64_t totalWords);
