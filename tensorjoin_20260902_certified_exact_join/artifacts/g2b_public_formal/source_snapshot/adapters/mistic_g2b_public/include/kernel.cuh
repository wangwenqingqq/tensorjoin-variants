#pragma once

#include "include/params.cuh"




__global__
void GPUGenerateRanges(unsigned int * tree, //points to the tree constructed with buildTree()
                        unsigned int * lastTreeLayer,
                        unsigned int * numPoints, // the number of points in the dataset
                        unsigned int * pointBinNumbers, // the bin numbers of the points relative to the reference points
                        unsigned int * numLayers, // the number of layer the tree has
                        unsigned int * binSizes,	// the number of bins for each layer, or rather the width of the tree in bins for that layer
                        unsigned int * binAmounts, // the number of bins for each reference point, ranhge/epsilon
                        unsigned int * addIndexes, // where generateRanges will return the non-empty index locations in the tree's final layer
                        unsigned int * rangeIndexes, // the index locations that are adjacent to each non-empty index
                        unsigned int * rangeSizes, // the number of points in adjacent non-empty indexes for each non-empty index
                        unsigned int * numValidRanges, // the numnber of adjacent non-empty indexes for each non-empty index
                        unsigned long long * calcPerAdd, // the number of calculations that will be needed for each non-empty index
                        unsigned int * numPointsInAdd, // the number of points in each non-empty index
                        unsigned int * nonEmptyBins,
                        unsigned int * binNumbers,
                        unsigned int * tempAdd,
                        unsigned int * numSearches);

__device__
void GPUBinarySearch(	unsigned int searchIndex, // the bin to search in bin numbers
                    unsigned int  * tempAdd, //temporary address for searching
                    unsigned int * binNumbers, //array of bin numbrs 
                    unsigned int nonEmptyBins, //size of binNumebrs
                    unsigned int numLayers, //number of reference points
                    unsigned int * lastTreeLayer,
                    unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                    unsigned int * addIndexs, //location of nonempty bins in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches);


__device__
void GPUTreeTraversal(unsigned int tid,
                    unsigned int * tempAdd, //twmp array for the address being searched
                    unsigned int * tree, // the pointer to the tree
                    unsigned int * lastTreeLayer,
                    unsigned int * binSizes, // the width of the tree for each layer mesuared in number of bins
                    unsigned int * binAmounts, // the number of bins for each reference point
                    unsigned int * binNumbers, // the bin number for the home address
                    unsigned int numLayers, // the number of reference points/layers in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches);



__device__
int GPUDepthSearch( unsigned int * tree, //pointer to the tree built with buildTree()
                    unsigned int * binSizes,
                    unsigned int * binAmounts, // the number of bins for each reference point, i.e. range/epsilon
                    unsigned int numLayers, //the number of layers in the tree
                    unsigned int * searchBins);

__device__
int GPUBSearch(unsigned int tid,
            unsigned int * tempAdd, //address to search for
            unsigned int * binNumbers, //array of addresses
            unsigned int nonEmptyBins, //number of bins
            unsigned int numLayers);

__global__ 
void kernelUniqueKeys(unsigned int * pointIDKey,
                      unsigned long long int * N, 
                      unsigned int * uniqueKey, 
                      unsigned int * uniqueKeyPosition, 
                      unsigned int * cnt);



__host__ __device__ //may need to switch to inline
bool distanceCheck(double epsilon2, 
                   unsigned int dim,
                   double * data,
                   unsigned int p1, 
                   unsigned int p2, 
                   unsigned int numPoints);


__global__ 
void distanceCalculationsKernel(unsigned int *numPoints,
                                unsigned int * linearRangeID,
                                unsigned int * addAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long *numThreadsPerBatch,
                                unsigned long long * numThreadsPerAddress,
                                double * data, 
                                unsigned int * numValidRanges,
                                unsigned int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                unsigned int * addIndexRange,
                                unsigned long long *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b);

void distanceCalculationsKernel_CPU(unsigned int totalBlocks,
    unsigned int *numPoints,
    unsigned int * linearRangeID,
    unsigned int * addAssign,
    unsigned int * threadOffsets,
    double *epsilon2,
    unsigned int *dim,
    unsigned long long  *numThreadsPerBatch,
    unsigned long long  * numThreadsPerAddress,
    double * data,
    unsigned int *addIndexes,
    unsigned int * numValidRanges,
    unsigned int * rangeIndexes,
    unsigned int * rangeSizes,
    unsigned int * numPointsInAdd,
    unsigned int * addIndexRange,
    unsigned int * pointArray,
    unsigned long long  *keyValueIndex,
    std::vector<unsigned int> * hostPointA,
    std::vector<unsigned int> * hostPointB);

__global__ 
void nodeCalculationsKernel(unsigned int *numPoints,
                                unsigned int * pointOffsets,
                                unsigned int * nodeAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long  *numThreadsPerBatch,
                                unsigned long long  * numThreadsPerNode,
                                double * data, 
                                unsigned int * numNeighbors,
                                unsigned int * nodePoints,
                                unsigned int * neighbors,
                                unsigned int * neighborOffset,
                                unsigned long long  *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b);

void nodeCalculationsKernel_CPU(unsigned int numNodes,
                                unsigned int totalBlocks,
                                unsigned int *numPoints,
                                unsigned int * pointOffsets,
                                unsigned int * nodeAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long  *numThreadsPerBatch,
                                unsigned long long  * numThreadsPerNode,
                                double * data, 
                                unsigned int * numNeighbors,
                                unsigned int * nodePoints,
                                unsigned int * neighbors,
                                unsigned int * neighborOffset,
                                unsigned long long  *keyValueIndex);

__global__
void binningKernel(unsigned int * binNumbers, //array numPoints long
                    unsigned int * numPoints,
                    unsigned int * dim,
                    double * data, //all data
                    double * RP, //single rp
                    double * epsilon,
                    const unsigned int rPPerLayer);

__global__ 
void nodeByPoint( const unsigned int dim,
                    double * data, //
                    double * epsilon2,//
                    unsigned int * numPoints, //
                    unsigned int * batchPoints, //
                    unsigned int * nodeID, //
                    unsigned int * numNeighbors, //
                    unsigned int * numPointsNode, //
                    unsigned int * neighborNodes, //
                    unsigned int * neighborOffset, //
                    unsigned int * pointOffset, //
                    unsigned int * point_a, //
                    unsigned int * point_b, //
                    unsigned long long * keyValueIndex);

__global__ 
void nodeByPoint2( const unsigned int dim,
                    double * data, //
                    double * epsilon2,//
                    unsigned int * numPoints, //
                    unsigned int * batchPoints, //
                    unsigned int * nodeID, //
                    unsigned int * numNeighbors, //
                    unsigned int * numPointsNode, //
                    unsigned int * neighborNodes, //
                    unsigned int * neighborOffset, //
                    unsigned int * threadPoint,
                    unsigned int * pointOffset, //
                    unsigned int * point_a, //
                    unsigned int * point_b, //
                    unsigned long long * keyValueIndex);

__global__ 
void nodeByPoint3( const unsigned int dim,
                  double * data, //
                  double * epsilon2,//
                  unsigned int * numPoints, //
                  unsigned int * nodeID, //
                  unsigned int * numNeighbors, //
                  unsigned int * numPointsNode, //
                  unsigned int * neighborNodes, //
                  unsigned int * neighborOffset, //
                  unsigned int * pointOffset, //
                  unsigned int * point_a, //
                  unsigned int * point_b, //
                  unsigned long long * keyValueIndex,
                  unsigned int * pointIdent,
                  unsigned int * pointIndex);

__global__ 
void nodeByPoint4( const unsigned int dim,
                  double * data, //
                  double * epsilon2,//
                  unsigned int * numPoints, //
                  unsigned int * nodeID, //
                  unsigned int * numNeighbors, //
                  unsigned int * numPointsNode, //
                  unsigned int * neighborNodes, //
                  unsigned int * neighborOffset, //
                  unsigned int * pointOffset, //
                  unsigned int * point_a, //
                  unsigned int * point_b, //
                  unsigned long long * keyValueIndex,
                  unsigned int * pointIdent,
                  unsigned int * pointIndex);

__global__ 
void nodeByPoint5( const unsigned int dim,
                double * data, //
                const double epsilon2,//
                const unsigned int numPoints, //
                unsigned int * batchPoints, //
                unsigned int * nodeID, //
                unsigned int * numNeighbors, //
                unsigned int * numPointsNode, //
                unsigned int * neighborNodes, //
                unsigned int * neighborOffset, //
                unsigned int * pointOffset, //
                unsigned int * point_a, //
                unsigned int * point_b, //
                unsigned long long * keyValueIndex,
                unsigned int * tpp,
                unsigned int * pointsPerBatch);

__forceinline__ __host__ __device__ //may need to switch to inline (i did)
int cachedDistanceCheck(double epsilon2, double * data, double * p1, unsigned int p2, const unsigned int numPoints);

__global__ 
void searchKernelCOSS(const unsigned int batch_num,
                    double * A, // this is the imported data
					const unsigned int num_points, // total number of points
					unsigned int * point_a, // an array which will store the first point in a pair
					unsigned int * point_b, // an array vector that will store a second point in a pair
					unsigned int * address_array, // the array of all generated addresses
					unsigned long long int * key_value_index, //a simple counter to keep track of how many results in a batch
					unsigned int * point_array,//the ordered points
					const unsigned int array_counter, //the number of arrays
					const unsigned int rps, //the number of reference points
					const unsigned int dim, //the number of dimensions
					const double epsilon2, //the distance threshold
					unsigned int *point_address_array,
                    unsigned int * address_shared);

__global__ 
void searchKernelCOSStree(const unsigned int batch_num,
                    double * A, // this is the imported data
					const unsigned int num_points, // total number of points
					unsigned int * point_a, // an array which will store the first point in a pair
					unsigned int * point_b, // an array vector that will store a second point in a pair
					unsigned int * address_array, // the array of all generated addresses
					unsigned long long int * key_value_index, //a simple counter to keep track of how many results in a batch
					unsigned int * point_array,//the ordered points
					const unsigned int array_counter, //the number of arrays
					const unsigned int rps, //the number of reference points
					const unsigned int dim, //the number of dimensions
					const double epsilon2, //the distance threshold
					unsigned int *point_address_array,
                    unsigned int * address_shared,
                    unsigned int * tree,
                    unsigned int * binSizes,
                    unsigned int * binAmounts,
                    const unsigned int lastLayerOffset);


__global__
void dumbBrute(const unsigned int batch_num,
    double * data, // this is the imported data
    const unsigned int numPoints, // total number of points
    unsigned int * pointA, // an array which will store the first point in a pair
    unsigned int * pointB, // an array vector that will store a second point in a pair)
    const unsigned int dim, //the number of dimensions
    const double epsilon,
    unsigned long long int * key_value_index); //a simple counter to keep track of how many results in a batch
