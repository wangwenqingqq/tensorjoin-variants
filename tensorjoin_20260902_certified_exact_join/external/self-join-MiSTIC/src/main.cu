#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>


#include "include/tree.cuh"
#include "include/nodes.cuh"
#include "include/utils.cuh"
#include "include/launcher.cuh"


int main(int argc, char*argv[]){
    
	cudaSetDevice(CUDA_DEVICE); // 0 is bad on the node

    //reading in command line arguments
	char *filename = argv[1]; // first argument is the file with the dataset as a .bin
	unsigned int dim = atoi(argv[2]); // second argument is the dimensionality of the data, i.e. number of columns
	unsigned int numRP = atoi(argv[3]);

	unsigned int concurent_streams = 2; // number of cuda streams, should only ever need to be 2 but can be set to a parameter
	double epsilon;
	sscanf(argv[4], "%lf", &epsilon); // third argumernt is the distance threshold being searched

	double time0 = omp_get_wtime(); //start initial timer

	// std::string sFileName(filename);
	// char * ext = strrchr(filename, '.');
	
	double* A;
	unsigned int numPoints;
	// if(&ext[1] == "bin" ){

		std::ifstream file(	filename, std::ios::in | std::ios::binary);
		file.seekg(0, std::ios::end); 
		size_t size = file.tellg();  
		file.seekg(0, std::ios::beg); 
		char * read_buffer = (char*)malloc(sizeof(char)*size);
		file.read(read_buffer, size*sizeof(double));
		file.close();
		A = (double*)read_buffer;
		numPoints = size/sizeof(double)/dim;
		
	// } else {
		// FILE *fptr; 
		// fptr = fopen(filename, "r");
		// if (fptr == NULL)
		// {
		// 	printf("No such File\n");
		// 	exit(0);
		// }
		// double check = 0;
		
		// std::vector<double> data1;

		// while( fscanf(fptr, "%lf, ", &check) == 1 || fscanf(fptr, "%lf ", &check) == 1)
		// {
		// 	data1.push_back(check);
		// }
		// fclose(fptr);
		// numPoints = data1.size() / dim;
		// A = &data1[0];
	// }

	double time00 = omp_get_wtime();
	printf("\nTime to read in file: %f\n", time00-time0);

	// double* A = (double*)read_buffer;//reinterpret as doubles

	// unsigned int numPoints = size/sizeof(double)/dim; // calculate number of points based on the siez of the input

	// can set a subset of the data for easier debugging
	//////////////
	// numPoints = 10000;
	////////////
	if(ERRORPRINT) fprintf(stderr,"\n%f %u ", epsilon, numPoints);
	// if(TESTING_SEARCH) fprintf(stderr,"\nRP, %d,", numRP);
	if(TESTING_SEARCH) fprintf(stderr," E, %f,", epsilon);

	
	printf("\nNumber points: %d ", numPoints);
	printf("\nNumber Dimensions: %d ", dim);
	printf("\nNumber Reference Points: %d ", numRP);
	printf("\nNumber Concurent Streams: %d", concurent_streams);
	printf("\nBlock Size: %d, Kernel Blocks: %d",BLOCK_SIZE,KERNEL_BLOCKS);
	if(BINARYSEARCH == 0) printf("\nUsing tree traversals"); 
	if(BINARYSEARCH == 1) printf("\nUsing using binary searches");
	if(BINARYSEARCH == 2) printf("\nUsing dynamic searching");
	printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);


	//if using a small datset for debugging, also run brute force so we can double check results
	// if(numPoints <= 100000) 	brute_force( numPoints, dim, epsilon, A);

	double time1 = omp_get_wtime();

	//dimensionOrder holds the order of dimensions sorted by thier varience
	unsigned int *dimensionOrder = (unsigned int*)malloc(sizeof(unsigned int)*dim);

	//dimOrderedData holds the dataset after it has been reordered based on dimensional varience
	double * dimOrderedData = (double*)malloc(sizeof(double)*numPoints*dim);

    dimensionOrder = stddev(A, dim, numPoints); //find the order of dimensions by varience

	// reorder the origional data into the dimesionaly ordered data
	// this makes earlier columns of the data have higher varience than later columns
	// the points maintain thier order relative to other points
	// this can increase short circuiting because higher variences are calculated earlier
    #pragma omp parallel for
    for(unsigned int i = 0; i < numPoints; i++){
        for(unsigned int j = 0; j < dim; j++){
            dimOrderedData[i*dim + j] = A[i*dim + dimensionOrder[j]];
        }
    }

	// allocate and set an array to keep the order of the points
	// this allows us to refer to the row of the intial data when returning pairs
	unsigned int * pointArray = (unsigned int*)malloc(sizeof(unsigned int)*numPoints);
	for (unsigned int i = 0; i < numPoints; i++){
		pointArray[i] = i;
	}

	#if NODES

	double time2 = omp_get_wtime();

	#if KT == 1
	nodeLauncher(dimOrderedData,
					dim,
					numPoints,
					0, //numRP
					pointArray,
					epsilon);
	#endif

	#if KT == 2
	nodeLauncher2(dimOrderedData,
		dim,
		numPoints,
		0, //numRP
		pointArray,
		epsilon);
	#endif

	#if KT == 3
	nodeLauncher3(dimOrderedData,
		dim,
		numPoints,
		0, //numRP
		pointArray,
		epsilon);
	#endif

	#if KT == 4
	nodeLauncher4(dimOrderedData,
		dim,
		numPoints,
		0, //numRP
		pointArray,
		epsilon);
	#endif

	#if KT == 5
	nodeLauncher5(dimOrderedData,
		dim,
		numPoints,
		0, //numRP
		pointArray,
		epsilon);
	#endif

	double time3 = omp_get_wtime();
	printf("Node Laucnher time: %f\n", time3 - time2);
	if(ERRORPRINT) fprintf(stderr,"%f ",time3-time2);

	#else

	// poinmt bin numbers holds the bins relative to reference points that each point is in
	unsigned int ** pointBinNumbers;

	// binSizes is the number of bins for each layer of the tree , which includes the spread from the previous layer
	unsigned int * binSizes = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);

	//bin amounts is the number of bins for that reference point, i.e. the range of points / epsilon
	unsigned int * binAmounts = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);

	//maxBinAmount limits the number of bins that can be in a layer to reduce space complexity, not usually an issue
	unsigned int maxBinAmount = MAX_BIN;

	//this will be the tree structure of pointers to layers
	unsigned int ** tree;

	//build the tree into tree and returns the number of layers that was selected for the tree
	unsigned int numLayers = buildTree(
					&tree, // outputs into this pointer
					dimOrderedData, //uses the data after ordered for dimensions
					dim, // the dimensionality of the data
					numPoints, // the number of points in the dataset
					epsilon, // the distance threshold being searched
					maxBinAmount, // the limmiter on tree width in number of bins
					pointArray, // the ordered points, this will be rearanged when building the tree
					&pointBinNumbers, // this will hold of the bin number for each point relative to each reference point
					binSizes, // the width for each layer of the tree which is built in the fuinction
					binAmounts,//the number of bins for each reference point as in the range / epsilon
					numRP); 


	// allocate a data array for used with distance calcs
	// the data is moved around so that point in bin are near eachother in the array
	// the location is based  on the point array that was altered during tree construction
	// data can be organized 2 ways:
	// 1. if DATANORM = true
	//    the data is organized so that the the dimensions of each point are next to eachother
	//	  this allows for coalesced memory accsess on the gpu to increase perfomance
	//
	// 2. if DATANORM is false
	//	  this is the standard stride that was used after dimensional ordering
    double * data = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
	for(unsigned int i = 0; i < numPoints; i++){
		for(unsigned int j = 0; j < dim; j++){
			#if DATANORM
			data[i+numPoints*j] = dimOrderedData[pointArray[i]*dim+j];
			#else
			data[i*dim+j] = dimOrderedData[pointArray[i]*dim+j];
			#endif
		}
	}


	//git bin info
	if(ERRORPRINT) fprintf(stderr,"%u ",numLayers);
	if(ERRORPRINT && BINMETRICS){
	unsigned int nonEmptyBins = 0;
	std::vector<unsigned int> binCounts;
	unsigned int maxBinSize = 0;
	unsigned int minBinSize = numPoints;
	for(unsigned int i = 0; i < binSizes[numLayers-1]-1; i++){ 
        
		// if the tree value on the last layer is less than the next then it has points in it
		if(tree[numLayers-1][i] < tree[numLayers-1][i+1]){
            nonEmptyBins++;
			unsigned int size = tree[numLayers-1][i+1] - tree[numLayers-1][i];
			binCounts.push_back(size);
			if(size < minBinSize){
				minBinSize = size;
			}
			if(size > maxBinSize){
				maxBinSize = size;
			}

        }
    }

	double averageBinCount = numPoints*1.0 / nonEmptyBins;

	double stdDevBins = 0;
	double sumSqError = 0;
	for(unsigned int i = 0; i < nonEmptyBins; i++){
		stdDevBins += (binCounts[i]*1.0-averageBinCount)*(binCounts[i]*1.0-averageBinCount);
	}
	sumSqError = stdDevBins;
	stdDevBins /= numPoints;
	stdDevBins = sqrt(stdDevBins);

	if(ERRORPRINT && BINMETRICS) fprintf(stderr,"%u %u %f %f ", minBinSize, maxBinSize, sumSqError, stdDevBins);
	}


	// checking that the last bin size is not negative or zero and that the tree has every data point in it
	printf("Last Layer Bin Count: %d\nTree Check: %d\n",binSizes[numLayers-1], tree[numLayers-1][binSizes[numLayers-1]-1]);

	double time2 = omp_get_wtime();
	// if(TESTING_SEARCH) fprintf(stderr," TT, %f,", time2-time1);
	if(ERRORPRINT) fprintf(stderr,"%f ",time2-time1);

    printf("Time to build tree: %f\n", time2-time1);

	#if COSS
	struct neighborTable * table = launchCOSS(tree,
		numPoints,
		pointBinNumbers,
		numLayers,
		binSizes,
		binAmounts,
		data,
		dim,
		epsilon,
		pointArray);
	#endif

	// #if GPUSEARCH

	// struct neighborTable * table1 = launchGPUSearchKernel(tree,
	// 													numPoints,
	// 													pointBinNumbers,
	// 													numLayers,
	// 													binSizes,
	// 													binAmounts,
	// 													data,
	// 													dim,
	// 													epsilon,
	// 													pointArray);

	// #else
	// // addIndexes holds the return from generating ranges which contains the non-empty index locations in the last layer of tree
    // unsigned int * addIndexes;

	// // rangeIndexes holds the return from generating ranges that correspond to the non-empty indexes 
	// // and has the adjacent non-empty index locations
    // unsigned int ** rangeIndexes;

	// // range sizes has the number of points in the adjacent ranges in rangeIndexes
    // unsigned int ** rangeSizes;

	// // the number of adjacent indexes for each noin-empty index
    // unsigned int * numValidRanges;

	// //the number of distance calculations that will be needed for each non-empty index
    // unsigned long long * calcPerAdd;

	// // the number of points in each non-empty index
	// unsigned int *numPointsInAdd;


	// //generate the ranges and perform the searches
    // unsigned int nonEmptyBins = generateRanges(tree, //the tree arrays pointer
	// 								  numPoints, //the number of points in the dataset
	// 								  pointBinNumbers, // the bin numbers for each point
	// 								  numLayers, // the number of layers of the tree
	// 								  binSizes, // the width of each layer of the tree
	// 								  binAmounts, // the number of bins for each reference point
	// 								  &addIndexes, // for returning the addIndex info
	// 								  &rangeIndexes, //for returning the range index info
	// 								  &rangeSizes,// for returning the range size info
	// 								  &numValidRanges, //for returning the valid range count
	// 								  &calcPerAdd, // for returning the number of clacs for each address
	// 								  &numPointsInAdd); //for returning the number of points in each non-empty index

	// // keep track of the number of total calcs needed
    // unsigned long long sumCalcs = 0;

	// //keep track of the number of address that were found in searching for each address
    // unsigned long long sumAdds = 0;

	// //itterating through just to find sum values
    // for(unsigned int i = 0; i < nonEmptyBins; i++){
    //     sumCalcs += calcPerAdd[i];
    //     sumAdds += numValidRanges[i];
    // }

	// // add index range has the values at each nonempty index of the last layer of the tree
	// unsigned int * addIndexRange = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);
	// for(unsigned int i = 0; i < nonEmptyBins; i++){
	// 	addIndexRange[i] = tree[numLayers-1][addIndexes[i]];
	// }

	// // an array for keeping track of where the linear indexs start for each elemetn in the 2d one
	// unsigned int * linearRangeID = (unsigned int*)malloc(sizeof(unsigned int) * nonEmptyBins);

	// // a linear range index made from the 2d range index array to keep track of adjacent non-empty indexes for each non-empty index
	// unsigned int * linearRangeIndexes = (unsigned int*)malloc(sizeof(unsigned int)*sumAdds);

	// //a linear range sizes made from the range siezes array to keep track of the number of points in adjacent indexes
	// unsigned int * linearRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*sumAdds);

	// //running total for keeping track of starts for each index in the linear arrays
	// unsigned int runningTotal = 0;
	// //linearizeing 2d arrays to 1d arrays for GPU work
	// for(unsigned int i = 0; i < nonEmptyBins; i++){

	// 	//keeping track of start locations in the linear arrays
	// 	linearRangeID[i] = runningTotal;
		

	// 	//populating the linear arrays from the 2d ones
	// 	for(unsigned int j = 0; j < numValidRanges[i]; j++){
	// 		linearRangeIndexes[runningTotal + j] = tree[numLayers-1][rangeIndexes[i][j]];
	// 		linearRangeSizes[runningTotal + j] = rangeSizes[i][j];
	// 	}

	// 	//increment the running total by the number of ranges for the current index
	// 	runningTotal += numValidRanges[i];
	// }

	// if(TESTING_SEARCH) fprintf(stderr," NB, %d, NC, %llu, NA, %llu", nonEmptyBins, sumCalcs, sumAdds );

    // printf("Number non-empty bins: %d\nNumber of calcs: %llu\nNumber Address for calcs: %llu\n", nonEmptyBins, sumCalcs, sumAdds);


	// double time3 = omp_get_wtime();


	// if((log2(nonEmptyBins) < numLayers && BINARYSEARCH == 2) || BINARYSEARCH == true){
	// 	printf("Tree BINARY search time: %f\n", time3-time2);
	// 	if(TESTING_SEARCH) fprintf(stderr," B, %f,", time3-time2);
	// }else{
	// 	printf("Tree TRAVERSAL search time: %f\n", time3-time2);
	// 	if(TESTING_SEARCH) fprintf(stderr," T, %f,", time3-time2);
	// }


	// #if !TESTING_SEARCH

	

	// struct neighborTable * table1 =  launchKernel(numLayers, // the number of layers in the tree
	// 			data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
	// 			dim, //the dimensionality of the data
	// 			numPoints, //the number of points in the dataset
	// 			epsilon, //the distance threshold being searched
	// 			addIndexes, //the non-empty index locations in the last layer of the tree
	// 		    addIndexRange, // the value of the non empty index locations  in the last layer of the tree, so the starting point number
	// 			pointArray, // the array of point numbers ordered to match the sequence in the last array of the tree and the data
	// 			rangeIndexes, // the non-empty adjacent indexes for each non-empty index 
	// 			rangeSizes, // the size of the non-empty adjacent indexes for each non-empty index
	// 			numValidRanges, // the number of adjacent non-empty indexes for each non-empty index
	// 			numPointsInAdd, // the number of points in each non-empty index
	// 			calcPerAdd, // the number of calculations needed for each non-mepty index
	// 			nonEmptyBins, //the number of nonempty indexes
	// 			sumCalcs, // the total number of calculations that will need to be made
	// 			sumAdds, //the total number of addresses that will be compared to by other addresses for distance calcs
	// 			linearRangeID, // an array for keeping trackj of starting points in the linear arrays
	// 			linearRangeIndexes, // a linear version of rangeIndexes
	// 			linearRangeSizes); // a linear version of 
				
	// double time4 = omp_get_wtime();
    // printf("Kernel time: %f\n", time4-time3);
	// #endif


	// #endif

	#endif

	printf("Total Time: %f\n",omp_get_wtime()-time1); //note that this does not include time to read in the data from disk to main memory
	if(ERRORPRINT) fprintf(stderr,"%f ",omp_get_wtime()-time1);








// just freeing memory here
////////////////////////////////////////////////////////////////////

	// for(unsigned int i = 0; i < nonEmptyBins; i++){
	// 	free(rangeIndexes[i]);
	// 	free(rangeSizes[i]);
	// }
	// for(unsigned int i = 0; i < numLayers; i++){
	// 	free(tree[i]);
	// }
	// free(tree);
	// for(unsigned int i = 0; i < numPoints; i++){
	// 	free(pointBinNumbers[i]);
	// }

	// free(pointBinNumbers);
	// free(numValidRanges);
	// free(calcPerAdd);
	// free(addIndexes);
	// free(rangeIndexes);
	// free(rangeSizes);
	// free(A);
	// free(binSizes);
	// free(binAmounts);
	// free(pointArray);
	// free(data);
	// free(dimensionOrder);
	// free(dimOrderedData);
	// free(addIndexRange);

    return 0;

}