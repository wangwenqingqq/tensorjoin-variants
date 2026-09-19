#pragma once

#include "include/params.cuh"
#include "include/utils.cuh"
#include "include/kernel.cuh"
#include "include/nodes.cuh"

void constructNeighborTable(unsigned int * pointInDistValue, 
    unsigned int * pointersToNeighbors, 
    unsigned long long int * cnt, 
    unsigned int * uniqueKeys, 
    unsigned int * uniqueKeyPosition, 
    unsigned int numUniqueKeys,
    struct neighborTable * tables);

//function to launch kcuda kernels and distance calculation kernels
struct neighborTable * launchKernel(unsigned int numLayers,// the number of layers in the tree
    double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
    unsigned int dim,//the dimensionality of the data
    unsigned int numPoints,//the number of points in the dataset
    double epsilon,//the distance threshold being searched
    unsigned int * addIndexes,//the non-empty index locations in the last layer of the tree
    unsigned int * addIndexRange,// the value of the non empty index locations  in the last layer of the tree, so the starting point number
    unsigned int * pointArray,// the array of point numbers ordered to match the sequence in the last array of the tree and the data
    unsigned int ** rangeIndexes,// the non-empty adjacent indexes for each non-empty index 
    unsigned int ** rangeSizes, // the size of the non-empty adjacent indexes for each non-empty index
    unsigned int * numValidRanges,// the number of adjacent non-empty indexes for each non-empty index
    unsigned int * numPointsInAdd,// the number of points in each non-empty index
    unsigned long long *calcPerAdd,// the number of calculations needed for each non-mepty index
    unsigned int nonEmptyBins,//the number of nonempty indexes
    unsigned long long sumCalcs,// the total number of calculations that will need to be made
    unsigned long long sumAdds,//the total number of addresses that will be compared to by other addresses for distance calcs
    unsigned int * linearRangeID,// an array for keeping trackj of starting points in the linear arrays
    unsigned int * linearRangeIndexes,// a linear version of rangeIndexes
    unsigned int * linearRangeSizes); // a linear version of rangeSizes

    struct neighborTable * launchGPUSearchKernel(unsigned int ** tree, // a pointer to the tree
        unsigned int numPoints, //the number of points in the data
        unsigned int ** pointBinNumbers,  //the bin numbers ofr each point to each reference point
        unsigned int numLayers, //the number of reference points
        unsigned int * binSizes, // the number of bins in each layer
        unsigned int * binAmounts, //the number of bins for each reference point
        double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
        unsigned int dim,//the dimensionality of the data
        double epsilon,//the distance threshold being searched
        unsigned int * pointArray);// the array of point numbers ordered to match the sequence in the last array of the tree and the data

struct neighborTable * nodeLauncher(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon);


struct neighborTable * nodeLauncher2(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon);
    
struct neighborTable * nodeLauncher3(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon);
        
struct neighborTable * nodeLauncher4(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon);
        
struct neighborTable * nodeLauncher5(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon);
    

    struct neighborTable * launchCOSS(unsigned int ** tree, // a pointer to the tree
        unsigned int numPoints, //the number of points in the data
        unsigned int ** pointBinNumbers,  //the bin numbers ofr each point to each reference point
        unsigned int numLayers, //the number of reference points
        unsigned int * binSizes, // the number of bins in each layer
        unsigned int * binAmounts, //the number of bins for each reference point
        double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
        unsigned int dim,//the dimensionality of the data
        double epsilon,//the distance threshold being searched
        unsigned int * pointArray);