#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"
#include "include/kernel.cuh"

unsigned int buildTree(unsigned int *** rbins,
	 				  double * data,
					  unsigned int dim,
					  unsigned long long numPoints, 
					  double epsilon, 
					  unsigned int maxBinAmount,  
					  unsigned int * pointArray, 
					  unsigned int *** rpointBinNumbers, 
					  unsigned int * binSizes, 
					  unsigned int * binAmounts,
					  unsigned int numRp);

// __host__ __device__
void treeTraversal(unsigned int * tempAdd, 
				   unsigned int ** tree, 
				   unsigned int * binSizes, 
				   unsigned int * binAmounts, 
				   unsigned int * binNumbers, 
				   unsigned int numLayers, 
				   unsigned long long * numCalcs, 
				   unsigned int * numberRanges, 
				   unsigned int ** rangeIndexes, 
				   unsigned int ** rangeSizes, 
				   unsigned int * numPointsInAdd, 
				   unsigned int numSearches);

// __host__ __device__
long int depthSearch(unsigned int ** tree, 
				unsigned int * binAmounts, 
				unsigned int numLayers, 
				unsigned int * searchBins);


unsigned int generateRanges(unsigned int ** tree, 
							unsigned int numPoints, 
							unsigned int ** pointBinNumbers, 
							unsigned int numLayers, 
							unsigned int * binSizes, 
							unsigned int * binAmounts, 
							unsigned int ** addIndexes, 
							unsigned int *** rangeIndexes, 
							unsigned int *** rangeSizes, 
							unsigned int ** numValidRanges, 
							unsigned long long ** calcPerAdd, 
							unsigned int ** numPointsInAdd );

// __host__ __device__
long int bSearch(unsigned int * tempAdd, //address to search for
			unsigned int ** binNumbers, //array of addresses
			unsigned int nonEmptyBins, //number of bins
			unsigned int numLayers); //numebr of layers or size of addresses

// __host__ __device__
inline long int compareBins(unsigned int * bin1,
					   unsigned int * bin2, 
					   unsigned int binSize);

// __host__ __device__
void binarySearch(	unsigned int searchIndex, // the bin to search in bin numebrs
					unsigned int  * tempAdd, //temporary address for searching
					unsigned int ** binNumbers, //array of bin numbrs 
					unsigned int nonEmptyBins, //size of binNumebrs
					unsigned int numLayers, //number of reference points
					unsigned int ** tree, //the tree structure
					unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
					unsigned int * addIndexes, //location of nonempty bins in the tree
					unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
					unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
					unsigned int ** rangeIndexes, // the array of non-empty adjacent index locations
					unsigned int ** rangeSizes, // the number of points in each of the adjacent non-empty indexes
					unsigned int * numPointsInAdd, //the number of points in the home address/iondex
					unsigned int numSearches);