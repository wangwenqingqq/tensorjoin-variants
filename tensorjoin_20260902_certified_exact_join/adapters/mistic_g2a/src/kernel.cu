#include "include/kernel.cuh"



__device__ int pow_int(int x, int y){
	int answ = 1;
	for(int i = 0; i < y; i++){
		answ *= x;
	}
	return answ;
}
__host__ __device__
bool add_comparison_ls(unsigned int *add1, unsigned int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) < *(add2+i+1))
    {
      return true;
    } else if (*(add1+i+1) > *(add2+i+1)){
			return false;
		}
  }
  return false;
}

//compares 2 addresses and returns true if 1 is equal or less than 2
__host__ __device__
bool add_comparison_eq_ls(unsigned int *add1, unsigned int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) > *(add2+i+1))
    {
      return false;
    }
  }
  return true;
}

//compares 2 addresses and returns true if  they are both equal
__host__ __device__
bool add_comparison_eq(unsigned int *add1, unsigned int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) != *(add2+i+1))
    {
      return false;
    }
  }
  return true;
}

__device__ __host__
int binary_search_basic(unsigned int *array, //points to an array
                        const unsigned int array_size, //points to an int
                        unsigned int *search, //points to an array
                        const int rps) //points to an int
{
	int first = 0;
	int last = array_size;
	int middle = (first+last)/2;
	const int strider = (rps+1);

	while (first <= last){

		if(add_comparison_ls(array+middle*strider, search, rps)){
			first = middle + 1;
			middle = first+(last-first)/2;
		} else if (add_comparison_eq(array+middle*strider, search, rps)){
			return middle;
		}else{
			last = middle - 1;
			middle = first+(last-first)/2;
		}
	}
	return -1;
}

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
                        unsigned int * numSearches){ 
    
    // guiding function for dynamic programing of either the binary searches or tree traversals

    //assign one thread to each address
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    // if the tid is greawter than the number of non empty bins then return
    if(tid >= *nonEmptyBins){
        return;
    }

    //searches can be perfomed with either dynamic programming or with the single thread sequentialy

    #if BINARYSEARCH
        // call the GPU binary sesarch with dynamic programing

        //call the GPU binary search for a single thread
        GPUBinarySearch( tid, // the bin to search in bin numebrs
                        tempAdd, //temporary address for searching
                        binNumbers, //array of bin numbrs 
                        *nonEmptyBins, //size of binNumebrs
                        *numLayers, //number of reference points
                        lastTreeLayer,
                        binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                        addIndexes, // location of nonempty bin in tree
                        calcPerAdd, // for keeping track of the number of distance calculations to be performed
                        numValidRanges, // the number of adjacent non-empty addresses/indexes
                        rangeIndexes, // this addresses/index's array to keep track of adjacent indexes
                        rangeSizes, // the number of points in the indexes in localRangeIndexes
                        numPointsInAdd, // the number of points in this nonempty address
                        *numSearches); // the number of searches that need to be performed, 3^r

    #else // do a tree traversal

        // call the GPU tree traversal with dynamic programing

        //call the GPU tree traversal with a single thread
        GPUTreeTraversal(tid,
                        tempAdd, // array of int for temp storage for searching
                        tree, // pointer to the tree made with buildTree()
                        lastTreeLayer,
                        binSizes, // the widths of each layer of the tree measured in bins
                        binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                        binNumbers, // the address/bin numbers of the current index/address
                        *numLayers, // the number of reference points or layers in the tree, same thing
                        calcPerAdd, // for keeping track of the number of distance calculations to be performed
                        numValidRanges, // the number of adjacent non-empty addresses/indexes
                        rangeIndexes, // this addresses/index's array to keep track of adjacent indexes
                        rangeSizes, // the number of points in the indexes in localRangeIndexes
                        numPointsInAdd, // the number of points in this nonempty address
                        *numSearches); // the number of searches that need to be performed, 3^r

    #endif


}











__device__
void GPUBinarySearch(	unsigned int searchIndex, // the bin to search in bin numbers
                    unsigned int  * tempAdd, //temporary address for searching
                    unsigned int * binNumbers, //array of bin numbrs 
                    unsigned int nonEmptyBins, //size of binNumebrs
                    unsigned int numLayers, //number of reference points
                    unsigned int * lastTreeLayer, //the last layer of the tree structure
                    unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                    unsigned int * addIndexs, //location of nonempty bins in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches){ //the number of searches that are being perfomred for each addresss


    //keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

    // keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

    //permute through bin variations (3^r) and run depth searches
    for(unsigned int i = 0; i < numSearches; i++){

        //modify temp add for the search based on our itteration i
        for(unsigned int j = 0; j < numLayers; j++){
            tempAdd[searchIndex*numLayers + j] = binNumbers[searchIndex*numLayers + j] + (i / (int)pow((double)3, (double)j) % 3)-1;
        }
        //perform the search and get the index location of the return 
        long int index = GPUBSearch(searchIndex, &tempAdd[searchIndex*numLayers], binNumbers, nonEmptyBins, numLayers);

        //check if the index location was non empty
        if(index >= 0){
            unsigned int newIndex = addIndexs[index];
            //store the non empty index location
            rangeIndexes[searchIndex*numSearches + localNumRanges] = newIndex;

            //calcualte the size of the index, i.e. the number of points in the index
            unsigned long long size = lastTreeLayer[newIndex+1] - lastTreeLayer[newIndex]; //may need to +- to index here!!!!!!!!!!!!!!!

            //store that in the sizes array
            rangeSizes[searchIndex*numSearches + localNumRanges] = size;

            // keep running total of the sizes for getting the number of calculations latter
            localNumCalcs += size;

            //keep track of the number of non-empty adjacent indexes
            localNumRanges++;
        }
    }

    // get the index of the home address
    long int homeIndex = addIndexs[GPUBSearch(searchIndex, &binNumbers[searchIndex*numLayers], binNumbers, nonEmptyBins, numLayers)];
    // find the number of points in the home address
    *numPointsInAdd = lastTreeLayer[homeIndex+1] - lastTreeLayer[homeIndex]; //may need to +- one to index here !!!!!!!!!
    // use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
    *numCalcs = localNumCalcs*(*numPointsInAdd);
    
    *numberRanges = localNumRanges;
}


__device__
void GPUTreeTraversal(unsigned int tid,
                    unsigned int * tempAdd, //twmp array for the address being searched
                    unsigned int * tree, // the pointer to the tree
                    unsigned int * lastTreeLayer, // pointer to the last layer of the tree
                    unsigned int * binSizes, // the width of the tree for each layer mesuared in number of bins
                    unsigned int * binAmounts, // the number of bins for each reference point
                    unsigned int * binNumbers, // the bin number for the home address
                    unsigned int numLayers, // the number of reference points/layers in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int  * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches){ //the number of searches that are being perfomred for each addresss

    //keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

    // keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

    //permute through bin variations (3^r) and run depth searches
    for(unsigned int i = 0; i < numSearches; i++){

        //modify temp add for the search based on our itteration i
        for(unsigned int j = 0; j < numLayers; j++){
            tempAdd[tid*numLayers+j] = binNumbers[tid*numLayers+j] + (i / (int)pow((double)3, (double)j) % 3)-1;
        }

        //perform the search and get the index location of the return 

        int index = GPUDepthSearch(tree, binSizes, binAmounts, numLayers, &tempAdd[tid*numLayers]);

        //check if the index location was non empty
        if(index >= 0){
            unsigned int newIndex = index;
            //store the non empty index location
            rangeIndexes[tid*numSearches + localNumRanges] = newIndex;

            //calcualte the size of the index, i.e. the number of points in the index
            unsigned long long size = lastTreeLayer[newIndex+1] - lastTreeLayer[newIndex]; //may need to +- to index here!!!!!!!!!!!!!!!

            //store that in the sizes array
            rangeSizes[tid*numSearches+localNumRanges] = size;

            // keep running total of the sizes for getting the number of calculations latter
            localNumCalcs += size;

            //keep track of the number of non-empty adjacent indexes
            localNumRanges++;
        }
    }

    // get the index of the home address
    int homeIndex = GPUDepthSearch(tree, binSizes, binAmounts, numLayers, &binNumbers[tid*numLayers]);

    // find the number of points in the home address
    *numPointsInAdd = lastTreeLayer[homeIndex+1] - lastTreeLayer[homeIndex]; //may need to +- one to index here !!!!!!!!!

    // use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
    *numCalcs = localNumCalcs*(*numPointsInAdd);

    *numberRanges = localNumRanges;

}


__device__
int GPUDepthSearch( unsigned int * tree, //pointer to the tree built with buildTree()
                    unsigned int * binSizes,
                    unsigned int * binAmounts, // the number of bins for each reference point, i.e. range/epsilon
                    unsigned int numLayers, //the number of layers in the tree
                    unsigned int * searchBins){ // the bin number that we are searching for

    // the offset is used for keeping track of the offset from the begining of each layer to the index
    unsigned int offset = 0;
    unsigned int layerOffset = 0;
    //go through each layer up to the last to determine if the index is non-empty and if it is then find the offset into the next layer
    for(unsigned int i = 0; i < numLayers - 1; i++){

        //check the current layer at the bin number + offset
        if (tree[layerOffset + offset + searchBins[i+1]] == 0){
            // if( i != 0) printf("%d, ", i);
            return -2;
        }

        // the next offset will be the previous layer index number * the number of bins for the reference point in the next layer
        offset = (tree[layerOffset + searchBins[i+1]+offset]-1)*binAmounts[i+1];

        layerOffset += binSizes[i];
    }

    //the index will be the last layers bin number plus the offset for the last layer
    int index = searchBins[ numLayers-1+1] + offset;
    // printf("%d, ",index);
    //if last layer has points then return the index value
    if(tree[layerOffset + index] < tree[layerOffset+index+1]){
        return index;
        // printf("%d, ",index);
    }else{
        return -1;
    }

}


__device__
int GPUBSearch(unsigned int tid,
            unsigned int * tempAdd, //address to search for
            unsigned int * binNumbers, //array of addresses
            unsigned int nonEmptyBins, //number of bins
            unsigned int numLayers) //number of layers or size of addresses
            {

    // initial conditions of the search
    unsigned int left = 0;
    unsigned int right = nonEmptyBins-1;

    //while loop for halving search each itterations
    while(left <= right){
        //calculate the middle
        unsigned int mid = (left + right)/2;
        // -1 for smaller, 1 for larger, 0 for equal
        int loc = 0; //compareBins( binNumbers[mid], tempAdd, numLayers);

        for(unsigned int i = 0; i < numLayers; i++){
            if(binNumbers[mid*numLayers + i] < tempAdd[ i]){
                left = mid + 1;
                break;
            }
            if(binNumbers[mid*numLayers+i] > tempAdd[ i]){
                right = mid - 1;
                break;
            }
        }

        //if we found the index
        if( loc == 0) return mid;

    }

    return -1;

}



//unique key array on the GPU
__global__ 
void kernelUniqueKeys(unsigned int * pointIDKey,
                      unsigned long long int * N, 
                      unsigned int * uniqueKey, 
                      unsigned int * uniqueKeyPosition, 
                      unsigned int * cnt){
	unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

	if (tid >= *N){
		return;
	}

	if (tid == 0)
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[0];
		uniqueKeyPosition[idx] = 0;
		return;
	
	}
	
	//All other threads, compare to previous value to the array and add
	
	if (pointIDKey[tid-1] != pointIDKey[tid])
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[tid];
		uniqueKeyPosition[idx] = tid;
	}
	
}





__global__ 
void distanceCalculationsKernel(unsigned int *numPoints,
                                unsigned int * linearRangeID,
                                unsigned int * addAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long  *numThreadsPerBatch,
                                unsigned long long  * numThreadsPerAddress,
                                double * data, 
                                unsigned int * numValidRanges,
                                unsigned int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                unsigned int * addIndexRange,
                                unsigned long long  *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }

    // unsigned int currentAdd = addAssign[tid]; 
    // unsigned int threadOffset = threadOffsets[tid];
    // unsigned int startingRangeID = linearRangeID[currentAdd];

    for(unsigned int i = 0; i < numValidRanges[addAssign[tid]]; i++){
        // unsigned long long int numCalcs = (unsigned long long int)rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];
        for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)rangeSizes[linearRangeID[addAssign[tid]] + i] * numPointsInAdd[addAssign[tid]]; j += numThreadsPerAddress[addAssign[tid]]){

            unsigned int p1 = addIndexRange[addAssign[tid]] + j / rangeSizes[linearRangeID[addAssign[tid]] + i];
            unsigned int p2 = rangeIndexes[linearRangeID[addAssign[tid]] + i] + j % rangeSizes[linearRangeID[addAssign[tid]]+ i];

            // double sum = 0;
            // for(unsigned int i = 0; i < *dim; i++){
            //     #if DATANORM
            //     sum += pow(data[i*(*numPoints) + p1] - data[i*(*numPoints) + p2], 2);
            //     #else
            //     sum += pow(data[p1*(*dim)+i]-data[p2*(*dim)+i],2);
            //     #endif
            //     if(sum > *epsilon2) break;
            // }

            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
            // if (sum <= *epsilon2){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}

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
                                unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }
    // unsigned int assignedNode = nodeAssign[tid];
    // unsigned int prevJ = threadOffsets[tid];
    // unsigned int pBase = pointOffsets[ nodeAssign[tid]];

    // double pData[DIM];

    for(unsigned int i = 0; i < numNeighbors[ nodeAssign[tid]]; i++){
        unsigned int neighborIndex = neighbors[neighborOffset[nodeAssign[tid]] + i];
        unsigned int neighborPoints = nodePoints[neighborIndex];
        // unsigned long long numCals = (unsigned long long int)nodePoints[ nodeAssign[tid]]* nodePoints[ neighbors[neighborOffset[ nodeAssign[tid]] + i]];
        // unsigned int p1 = pBase + prevJ / nodePoints[ neighborIndex];
        // for(unsigned int k = 0; k < DIM; i++){
        //     pData[k] = data[k*(*numPoints) + pBase];
        // }
        // cache
        for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)nodePoints[ nodeAssign[tid]]* neighborPoints; j += numThreadsPerNode[ nodeAssign[tid]]){

            // if(j / neighborPoints != prevJ){
            //     prevJ = j / neighborPoints;
            //     p1 = pBase + j / neighborPoints;
            //     //update cache
            //     for(unsigned int k = 0; k < DIM; k++){
            //         pData[k] = data[k*(*numPoints) + pBase];
            //     }
            // }


            unsigned int p1 = pointOffsets[ nodeAssign[tid]] + j / neighborPoints;

            unsigned int p2 = pointOffsets[ neighborIndex] + j % neighborPoints;

            // if (cachedDistanceCheck((*epsilon2), DIM, data, pData, p2, (*numPoints))){
            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
            // if (sum <= *epsilon2){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}

void nodeCalculationsKernel_CPU( unsigned int numNodes,
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
                            unsigned long long  *keyValueIndex){

    #pragma omp parallel for
    for(unsigned int h = 0; h < BLOCK_SIZE*totalBlocks; h++)
    {
        unsigned int tid = h;
        if(tid < *numThreadsPerBatch){
            // if(nodeAssign[tid]>numNodes) printf("ERROR0: %u / %u", nodeAssign[tid],numNodes);
            for(unsigned int i = 0; i < numNeighbors[nodeAssign[tid]]; i++){
                unsigned int neighborIndex = neighbors[neighborOffset[nodeAssign[tid]]+i];
                for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)nodePoints[nodeAssign[tid]]* nodePoints[neighborIndex]; j += numThreadsPerNode[nodeAssign[tid]]){
    
                    unsigned int p1 = pointOffsets[nodeAssign[tid]] + j / nodePoints[neighborIndex];
                    unsigned int p2 = pointOffsets[neighbors[neighborOffset[nodeAssign[tid]] + i]] + j % nodePoints[neighborIndex];
    
                    // if(p1 > *numPoints) printf("ERROR1: %u, %u, %u, %u\n", p1, nodeAssign[tid], neighborIndex, pointOffsets[nodeAssign[tid]]);
                    // if(p2 > *numPoints) printf("ERROR2: %u, %u, %u, %u\n", p2, nodeAssign[tid], neighborIndex, pointOffsets[nodeAssign[tid]]);
    
                    if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
    
                        #pragma omp critical
                        {
                            keyValueIndex++;
                        }
                    }
                }
            }
        }
    }
}



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
                                    std::vector<unsigned int> * hostPointB){


    #pragma omp parallel for
    for(unsigned int h = 0; h < BLOCK_SIZE*totalBlocks; h++)
    {

        unsigned int tid = h;

        if(tid < *numThreadsPerBatch ){
            
            //the current index/address that we are searching for
            unsigned int currentAdd = addAssign[tid]; 

            //the offset of the thread based on the address we are currently in
            unsigned int threadOffset = threadOffsets[tid];

            // the strating location for this index into the linear arrays
            unsigned int startingRangeID = linearRangeID[currentAdd];
        
            //go through each adjacent index and calcualte
            for(unsigned int i = 0; i < numValidRanges[currentAdd]; i++){

                //the number of calcs for this address is the nuymber of points in it * number of points in current address
                unsigned long long int numCalcs = (unsigned long long int)rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];

                //each thread starts at its offset then increments by the number of threads for the address untill past the numebr of calcs
                for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){
        
                    // the first point will be from the home address
                    // the starting point from addIndexRange plus the floor of the itteration over the number of points in the adjacent address
                    // once all of the threads have made the calcs for a point they all move to the next point in the home address
                    unsigned int p1 = addIndexRange[currentAdd] + j / rangeSizes[startingRangeID + i];
                    unsigned int p2 = rangeIndexes[startingRangeID + i] + j % rangeSizes[startingRangeID+ i];

                    if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
                        //  store point

                        #pragma omp critical
                        {
                            *keyValueIndex += 1;
                            
                            // unsigned long long int index = *keyValueIndex;
                            (*hostPointA).push_back(p1); //stores the first point Number
                            (*hostPointB).push_back(p2); // this stores the coresponding point number to form a pair
                        }
                    }
                }
            }
        }
    }
}

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
                  unsigned long long * keyValueIndex){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    unsigned int point = (*batchPoints) + tid/TPP;

    if(point >= (*numPoints) ){
        return;
    }
 
    unsigned int node = nodeID[point];

    // double pData[DIM];

    // for(unsigned int i = 0; i < DIM; i++){
    //     pData[i] = data[i*(*numPoints) + point];
    // }

    //for every neighboring node
    for(unsigned int i = 0; i < numNeighbors[node]; i++){
        
        unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

        for(unsigned int j = tid%TPP; j < numPointsNode[neighborNodeIndex]; j+=TPP){
            // unsigned int comparePoint = pointOffset[neighborNodeIndex]+j;
            
            // double sum = 0;
            // for(unsigned int k = 0; k < DIM; i++){
            //     sum += (pData[k] - data[k*(*numPoints) + comparePoint])*(pData[k] - data[k*(*numPoints) + comparePoint]);
            // }
            // if(sum <= (*epsilon2)){
            if (distanceCheck((*epsilon2), dim, data, point, pointOffset[neighborNodeIndex]+j, (*numPoints))){
            // if (cachedDistanceCheck((*epsilon2), DIM, data, pData, comparePoint, (*numPoints))){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = point; //stores the first point Number
                point_b[index] = pointOffset[neighborNodeIndex]+j; // this stores the coresponding point number to form a pair
            }
        }
    }

}

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
                  unsigned long long * keyValueIndex){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;


    if((*batchPoints) + tid/TPP >= (*numPoints) ){
        return;
    }

    unsigned int point = threadPoint[(*batchPoints) + tid/TPP];


    unsigned int node = nodeID[point];

    // double pData[DIM];

    // for(unsigned int i = 0; i < DIM; i++){
    //     pData[i] = data[i*(*numPoints) + point];
    // }

    //for every neighboring node
    for(unsigned int i = 0; i < numNeighbors[node]; i++){
        
        unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

        for(unsigned int j = tid%TPP; j < numPointsNode[neighborNodeIndex]; j+=TPP){
            // unsigned int comparePoint = pointOffset[neighborNodeIndex]+j;
            
            // double sum = 0;
            // for(unsigned int k = 0; k < DIM; i++){
            //     sum += (pData[k] - data[k*(*numPoints) + comparePoint])*(pData[k] - data[k*(*numPoints) + comparePoint]);
            // }
            // if(sum <= (*epsilon2)){
            if (distanceCheck((*epsilon2), dim, data, point, pointOffset[neighborNodeIndex]+j, (*numPoints))){
            // if (cachedDistanceCheck((*epsilon2), DIM, data, pData, comparePoint, (*numPoints))){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = point; //stores the first point Number
                point_b[index] = pointOffset[neighborNodeIndex]+j; // this stores the coresponding point number to form a pair
            }
        }
    }

}

__global__ 
void nodeByPoint3( const unsigned int dim,
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
                  unsigned long long * keyValueIndex){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;


    if((*batchPoints) + tid/TPP >= (*numPoints) ){
        return;
    }

    unsigned int point = threadPoint[(*batchPoints) + tid/TPP];


    unsigned int node = nodeID[point];

    // double pData[DIM];

    // for(unsigned int i = 0; i < DIM; i++){
    //     pData[i] = data[i*(*numPoints) + point];
    // }

    //for every neighboring node
    for(unsigned int i = 0; i < numNeighbors[node]; i++){
        
        unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

        for(unsigned int j = tid%TPP; j < numPointsNode[neighborNodeIndex]; j+=TPP){
            // unsigned int comparePoint = pointOffset[neighborNodeIndex]+j;
            
            // double sum = 0;
            // for(unsigned int k = 0; k < DIM; i++){
            //     sum += (pData[k] - data[k*(*numPoints) + comparePoint])*(pData[k] - data[k*(*numPoints) + comparePoint]);
            // }
            // if(sum <= (*epsilon2)){
            if (distanceCheck((*epsilon2), dim, data, point, pointOffset[neighborNodeIndex]+j, (*numPoints))){
            // if (cachedDistanceCheck((*epsilon2), DIM, data, pData, comparePoint, (*numPoints))){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = point; //stores the first point Number
                point_b[index] = pointOffset[neighborNodeIndex]+j; // this stores the coresponding point number to form a pair
            }
        }
    }

}


__forceinline__ __host__ __device__ //may need to switch to inline (i did)
int cachedDistanceCheck(double epsilon2, double * data, double * p1, unsigned int p2, const unsigned int numPoints){

    double runningDist[ILP];

    #pragma unroll
	for(int j=0; j<ILP; j++){
		runningDist[j]=0;
    }

    // int counter = 0;
    for(int l=0; l < DIM; l+=ILP) {
        #pragma unroll
        for(int j=0; j<ILP && l+j < DIM; j++) {
            runningDist[j] += (p1[l+j]- data[(l+j)*numPoints + p2])*(p1[l+j] - data[(l+j)*numPoints + p2]);
            // counter++;
        }

        #pragma unroll
        for(int j=1; j<ILP; j++) {
            runningDist[0] += runningDist[j];
            runningDist[j]=0;
        }

        if (runningDist[0] > epsilon2) {
                // return counter;
                return 0;
            }
    }

    // return counter;
    return 1;
}


__host__ __device__ //may need to switch to inline (i did)
inline bool distanceCheck(double epsilon2, unsigned int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints){

    double sum = 0;
    for(unsigned int i = 0; i < dim; i++){
        sum += (data[i*numPoints + p1] - data[i*numPoints + p2])*(data[i*numPoints + p1] - data[i*numPoints + p2]);
        if(sum > epsilon2) return false;
    }

    return true;
}

__global__
void binningKernel(unsigned int * binNumbers, //array numPoints long
                    unsigned int * numPoints,
                    unsigned int * dim,
                    double * data, //all data
                    double * RP, //single rp
                    double * epsilon,
                    const unsigned int rPPerLayer){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numPoints){
        return;
    }

    for(unsigned int i = 0; i < rPPerLayer; i++){

        double distance = 0;
        for(unsigned int j = 0; j < *dim; j++){
            // distance += pow(data[tid*(*dim) + j]-RP[j + (*dim)*i],2);
            distance += (data[tid*(*dim) + j]-RP[j + (*dim)*i]) * (data[tid*(*dim) + j]-RP[j + (*dim)*i]);
        }
    
        
        binNumbers[tid+(*numPoints)*i] = floor(sqrt(distance) / (*epsilon));
    
    }

    return;
}


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
                  unsigned int * pointIndex){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    
    while (true){

        if (*keyValueIndex > (BUFFERSIZE * 0.80) ){
            return;
        }

        unsigned int point;
        // if(tid % TPP == 0){
            point = atomicAdd(pointIndex, 1);
            // pointIdent[tid / TPP] = point;
        // }

        // __syncthreads();

        // if(tid % TPP != 0){
        //     point = pointIdent[tid/TPP];

        // }

        if(point >= (*numPoints)){
            return;
        }



    
        unsigned int node = nodeID[point];

        // double pData[DIM];

        // for(unsigned int i = 0; i < DIM; i++){
        //     pData[i] = data[i*(*numPoints) + point];
        // }

        //for every neighboring node
        for(unsigned int i = 0; i < numNeighbors[node]; i++){
            
            unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

            for(unsigned int j = tid%TPP; j < numPointsNode[neighborNodeIndex]; j+=TPP){
                // unsigned int comparePoint = pointOffset[neighborNodeIndex]+j;
                
                // double sum = 0;
                // for(unsigned int k = 0; k < DIM; i++){
                //     sum += (pData[k] - data[k*(*numPoints) + comparePoint])*(pData[k] - data[k*(*numPoints) + comparePoint]);
                // }
                // if(sum <= (*epsilon2)){
                if (distanceCheck((*epsilon2), dim, data, point, pointOffset[neighborNodeIndex]+j, (*numPoints))){
                // if (cachedDistanceCheck((*epsilon2), DIM, data, pData, comparePoint, (*numPoints))){
                    //  store point
                    unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                    point_a[index] = point; //stores the first point Number
                    point_b[index] = pointOffset[neighborNodeIndex]+j; // this stores the coresponding point number to form a pair
                }
            }
        }
    }

}


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
                  unsigned int * pointsPerBatch){

    const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;


    if((*batchPoints) + tid/(*tpp) >= numPoints ||  tid/(*tpp) >= (*pointsPerBatch)){
        return;
    }

    const unsigned int point = (*batchPoints) + tid/(*tpp);

    const unsigned int node = nodeID[point];

    ///////////////////////////////////////////
    double pData[DIM];

    for(unsigned int i = 0; i < DIM; i++){
        pData[i] = data[i*numPoints + point];
    }

    ////////////////////////////////////////

    //for every neighboring node
    for(unsigned int i = 0; i < numNeighbors[node]; i++){
        
        const unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

        for(unsigned int j = tid%(*tpp); j < numPointsNode[neighborNodeIndex]; j+=(*tpp)){

            // if (distanceCheck(epsilon2, dim, data, point, pointOffset[neighborNodeIndex]+j, numPoints)){
            if(cachedDistanceCheck(epsilon2, data, pData, pointOffset[neighborNodeIndex]+j, numPoints)){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = point; //stores the first point Number
                point_b[index] = pointOffset[neighborNodeIndex]+j; // this stores the coresponding point number to form a pair
            }
        }
    }
}


// extern __shared__ unsigned int address_shared[];
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
                    unsigned int * address_shared)

{
  //the thread id is the id in the block plus the max id of the last batch
    const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    const char stride = rps+1;
    const int point_location = (tid+(BLOCK_SIZE*KERNEL_BLOCKS*TPP)*(batch_num))/(TPP);
    double cachedPoint[DIM];
    for(unsigned int i = 0; i < DIM; i++){
        cachedPoint[i] = A[num_points*i + point_location];
    }

	// an exit clause if the number of threads wanted does not line up with block sizes
	if ( blockIdx.x*blockDim.x+threadIdx.x >= BLOCK_SIZE*TPP*KERNEL_BLOCKS || point_location >= num_points)
	{
		return;
	}


	//find the point number and the address number
    const int address_num = point_address_array[(tid+(BLOCK_SIZE*KERNEL_BLOCKS*TPP)*(batch_num))/(TPP)];
	

    //number possible combos is rps - the first odd address
	for (int i = 0; i < pow_int(3, rps); i++) // this itterates through every possible address combo
    {	
		

		for(int j = 0; j < rps; j++){

			int temp = (i / pow_int(3, j) ) % 3;
			if (temp == 2){
				temp = -1;
			}
			// *(address_shared+threadIdx.x*stride+j+1) =  temp + address_array[(address_num)*stride+j+1];
            address_shared[tid*stride+j+1] = temp + address_array[(address_num)*stride+j+1];
		}

		int address_location = -1;
		// address_location = binary_search_basic(address_array, array_counter, &address_shared[threadIdx.x*stride], rps);
        address_location = binary_search_basic(address_array, array_counter, &address_shared[tid*stride], rps);


		if( address_location < 0)
		{
			continue;
		}

        // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results


		//getting the ranges of the points
		const int start = address_array[address_location*stride]; //range_array[2*address_array[address_location*stride]];//inclusive
		const int end = address_location == array_counter-1 ? num_points:address_array[(address_location+1)*stride]; //range_array[2*address_array[address_location*stride]+1];//exclusive

		for(int j = start+(tid % (TPP)); j < end; j+=(TPP))
		{

			// if(distance <= (epsilon2)) //if sqrt of the distance is <= epsilon
            // if (distanceCheck(epsilon2, dim, A, point_location, j, num_points))
            if(cachedDistanceCheck(epsilon2, A, cachedPoint, j, num_points))
			{
				unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results
				point_a[index] = point_location; //stores the first point Number
				point_b[index] = j; // this store the cooresponding point number to form a pair
		
			}
		}
			// }
		// }
	}
}

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
                    const unsigned int lastLayerOffset)

{
  //the thread id is the id in the block plus the max id of the last batch
    const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    const char stride = rps+1;
    const int point_location = (tid+(BLOCK_SIZE*KERNEL_BLOCKS*TPP)*(batch_num))/(TPP);
    double cachedPoint[DIM];
    for(unsigned int i = 0; i < DIM; i++){
        cachedPoint[i] = A[num_points*i + point_location];
    }

	// an exit clause if the number of threads wanted does not line up with block sizes
	if ( blockIdx.x*blockDim.x+threadIdx.x >= BLOCK_SIZE*TPP*KERNEL_BLOCKS || point_location >= num_points)
	{
		return;
	}


	//find the point number and the address number
    const int address_num = point_address_array[(tid+(BLOCK_SIZE*KERNEL_BLOCKS*TPP)*(batch_num))/(TPP)];
	

    //number possible combos is rps - the first odd address
	for (int i = 0; i < pow_int(3, rps); i++) // this itterates through every possible address combo
    {	
		

		for(int j = 0; j < rps; j++){

			int temp = (i / pow_int(3, j) ) % 3;
			if (temp == 2){
				temp = -1;
			}
			// *(address_shared+threadIdx.x*stride+j+1) =  temp + address_array[(address_num)*stride+j+1];
            address_shared[tid*stride+j+1] = temp + address_array[(address_num)*stride+j+1];
		}

		int address_location = -1;
		// address_location = binary_search_basic(address_array, array_counter, &address_shared[threadIdx.x*stride], rps);
        #if BINARYSEARCH == 1
        address_location = binary_search_basic(address_array, array_counter, &address_shared[tid*stride], rps);
        #elif BINARYSEARCH == 2
        address_location = GPUDepthSearch(tree, binSizes, binAmounts, rps, &address_shared[tid*stride]);
        #endif

		if( address_location < 0)
		{
			continue;
		}

        // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results


		//getting the ranges of the points
        #if BINARYSEARCH == 1
		const int start = address_array[address_location*stride]; //range_array[2*address_array[address_location*stride]];//inclusive
		const int end = address_location == array_counter-1 ? num_points:address_array[(address_location+1)*stride]; //range_array[2*address_array[address_location*stride]+1];//exclusive
        #else
        const int start = tree[lastLayerOffset+address_location];
        const int end = tree[lastLayerOffset+address_location+1];
        #endif

        // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1);
		for(int j = start+(tid % (TPP)); j < end; j+=(TPP))
		{

            // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results


			// // if(distance <= (epsilon2)) //if sqrt of the distance is <= epsilon
            // // if (distanceCheck(epsilon2, dim, A, point_location, j, num_points))
            // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)cachedDistanceCheck(epsilon2, A, cachedPoint, j, num_points)); // atomic add to count results
            // unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results

            if(cachedDistanceCheck(epsilon2, A, cachedPoint, j, num_points))
			{
				unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1); // atomic add to count results
				point_a[index] = point_location; //stores the first point Number
				point_b[index] = j; // this store the cooresponding point number to form a pair
		
			}
		}
			// }
		// }
	}
}

__global__
void dumbBrute(const unsigned int batch_num,
    double * data, // this is the imported data
    const unsigned int numPoints, // total number of points
    unsigned int * pointA, // an array which will store the first point in a pair
    unsigned int * pointB, // an array vector that will store a second point in a pair)
    const unsigned int dim, //the number of dimensions
    const double epsilon,
    unsigned long long int * key_value_index) //a simple counter to keep track of how many results in a batch
{
    const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    const int point = (tid+(BLOCK_SIZE*KERNEL_BLOCKS*TPP)*(batch_num))/(TPP);
    for(unsigned int i = 0; i < numPoints; i++){
        if (distanceCheck(epsilon, dim, data, point, i, numPoints)){
                unsigned long long int index = atomicAdd(key_value_index,(unsigned long long int)1);
                pointA[index] = point; //stores the first point Number
                pointB[index] = i; // this stores the coresponding point number to form a pair
            }
    }
}