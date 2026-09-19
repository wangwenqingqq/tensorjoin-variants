#include "include/tree.cuh"

unsigned int buildTree(unsigned int *** rbins, //this will be where the tree itself is returned
						double * data, //this is the input data
						unsigned int dim, // the dimensionality of the data
						unsigned long long numPoints, // the number of points in the dataset
						double epsilon, // the distance threshold being searched
						unsigned int maxBinAmount, // the maximum nnumber of bins for one layer of the tree
						unsigned int * pointArray, // where the ordered point number will be returned
						unsigned int *** rpointBinNumbers, // the bin numbers for each point
						unsigned int * binSizes, // the number bins for that layer of the tree
						unsigned int * binAmounts,
						unsigned int numRP){ // the range / epsilon for that rp of that layer of the tree

	const unsigned int maxRP = MAXRP; // setting the max number of reference points
	const unsigned int numRPperLayer = RPPERLAYER; //could use log2(numPoints) // setting how many reference points are checked for each layer

	// printf("Selecting %d Rp from a pool of %d\n", numRPperLayer, (int)sqrt(numPoints));
	//an array to hold the bins of the tree that will be passed back to the calling function in rbins
	unsigned int ** bins = (unsigned int **)malloc(sizeof(unsigned int*)*maxRP);

	//an array to keep track of the number of non-empty bins in each layer
	unsigned int * binNonEmpty = (unsigned int*)malloc(sizeof(unsigned int)*maxRP);

	//a 2d array that keeps track of the offset of each point for placement in the next layer
	unsigned int ** pointBinOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*maxRP);
	for(int i = 0; i < maxRP; i++){
		pointBinOffsets[i] = (unsigned int*)malloc(sizeof(unsigned int)*numPoints);
	}

	//a 2d array that keeps track of the bin numbers for each point relative to the reference points
	unsigned int ** pointBinNumbers = (unsigned int**)malloc(sizeof(unsigned int*)*numPoints);
	for(unsigned int i = 0; i < numPoints; i++){ 
		pointBinNumbers[i] = (unsigned int*)malloc(sizeof(unsigned int)*maxRP);
	}

	//an array to hold the sum of squares error for calculating which reference point to use for each layer
	double * sumSqrsTemp = (double*)malloc(sizeof(double)*(maxRP+numRPperLayer));

	//an array to hold the final sum sqrs for each layer, helps determine the number of reference points to use
	double * sumSqrsLayers = (double*)malloc(sizeof(double)*maxRP);

	//array to track average number of points per bin
	double * averageNonEmptyBinCountTemp = (double *)malloc(sizeof(double)*(maxRP+numRPperLayer));
	double * averageNonEmptyBinCountLayers = (double *)malloc(sizeof(double)*maxRP);


	// variable for exiting the while loop, evaluates to false when sumsqrs is beyond a threshold
	bool check = true;

	// keeping track of which layer of the tree is being built
	unsigned int currentLayer = 0;

	struct DevicePointers devicePointers;

    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    double * d_epsilon;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon, &epsilon, sizeof(double), cudaMemcpyHostToDevice));

    
    devicePointers.d_data = d_data;
    devicePointers.d_dim = d_dim;
    devicePointers.d_numPoints = d_numPoints;
    devicePointers.d_epsilon = d_epsilon;

	// printf("entering buyild loop\n");
	
	// cinstruct the layers in this loop, terminates when we reach the max layer count or when loss function calls for return
	while(check && currentLayer < maxRP){
		
		// initial sumsqrs for layer is 0
		sumSqrsLayers[currentLayer] = 0;

		// set the temp sum sqrs for possible reference points to 0
		for(unsigned int i = 0; i < numRPperLayer; i++){
			sumSqrsTemp[i] = 0;
		}

		// make the array of reference points
		double * RPArray = createRPArray(data, numRPperLayer, dim, numPoints);
		
		unsigned int * allBinNumber = (unsigned int * )malloc(sizeof(unsigned int)*numPoints*(maxRP+numRPperLayer));

        //create bin number arrays on device
        unsigned int  * d_binNumber;
        assert(cudaSuccess == cudaMalloc((void**)&d_binNumber, sizeof(unsigned int)*numPoints*numRPperLayer));

        double * d_RP;
        assert(cudaSuccess == cudaMalloc((void**)&d_RP, sizeof(double)*dim*numRPperLayer));
        assert(cudaSuccess == cudaMemcpy(d_RP, RPArray, sizeof(double)*dim*numRPperLayer, cudaMemcpyHostToDevice));

        free(RPArray);

        cudaStream_t stream;
        cudaError_t stream_check = cudaStreamCreate(&stream);
        assert(cudaSuccess == stream_check);

        unsigned int totalBlocks = ceil(numPoints*1.0/1024);

        binningKernel<<<totalBlocks, 1024, 0, stream>>>(d_binNumber,
														devicePointers.d_numPoints,
														devicePointers.d_dim,
														devicePointers.d_data,
														d_RP,
														devicePointers.d_epsilon,
														numRPperLayer);

        cudaStreamSynchronize(stream);

        assert(cudaSuccess == cudaMemcpyAsync(allBinNumber, d_binNumber, sizeof(unsigned int)*numPoints*numRPperLayer, cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

		// printf("Finished binning\n");
        cudaFree(d_binNumber);
        cudaFree(d_RP);



		// this is the distance matrix for the points to the reference points in RPArray
		// double * distMat = (double * )malloc(sizeof(double)*numPoints*numRPperLayer);

		// a 2d array that keeps track of the bins for the current layer for each possible reference point
		unsigned int ** layerBins = (unsigned int**)malloc(sizeof(unsigned int*)*(maxRP+numRPperLayer));

		// an array for the number of bins in the current layer of the tree
		unsigned int * layerBinCount = (unsigned int*)malloc(sizeof(unsigned int)*(maxRP+numRPperLayer));

		// the number of non empty bins in the current layer of the tree
		unsigned int * layerBinNonEmpty = (unsigned int*)malloc((maxRP+numRPperLayer)*sizeof(unsigned int));

		// the offset of each point into the current layer
		unsigned int ** layerBinOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*(maxRP+numRPperLayer));

		// allocateing the 2d array structure for layer bin offsets
		for(unsigned int i = 0; i < (maxRP+numRPperLayer); i ++){
			layerBinOffsets[i] = (unsigned int*)malloc(sizeof(unsigned )*numPoints);
		}

		// the nuber of bins from the reference point to points
		unsigned int * layerNumBins = (unsigned int*)malloc(sizeof(unsigned int)*(maxRP+numRPperLayer));

		//the number of bins to skip form the start of the reference point
		unsigned int * skipBins = (unsigned int*)malloc(sizeof(unsigned int)*(maxRP+numRPperLayer));

		// printf("refChecking: \n");

		// entering into a loop to create a bunch of different possible layers for the current layer. then the best one is chosen
		#pragma omp parallel for num_threads(numRPperLayer)
		for(unsigned int i = 0; i < numRPperLayer; i++){

			//set the current number of non empty bins to 0
			layerBinNonEmpty[i] = 0;

			//fill out the distance matrix
			// double maxDistance = 0;  // for saving the max distance and used for determining layer size
			// double minDistance = euclideanDistance(&data[0*dim], dim , &RPArray[i*dim]); // the min distance starts as just a point

			// going through each point to build the distance matrix
			// #pragma omp parallel for
			unsigned int maxBin = allBinNumber[i*numPoints];
			unsigned int minBin = maxBin;
			for(unsigned int j = 0; j < numPoints; j++){
				// distMat[i*numPoints + j] = euclideanDistance(&data[j*dim], dim , &RPArray[i*dim]);

				// checking for the max distance
				if(allBinNumber[i*numPoints+j] > maxBin){
					maxBin = allBinNumber[i*numPoints+j];
				}

				//checking for the min distance
				if(allBinNumber[i*numPoints+j] < minBin){
					minBin = allBinNumber[i*numPoints+j];
				}
			}


			// printf("got max and min bins\n");

			// a modifier for padding the bin sizes
			unsigned int modifier = 2;

			//the number of bins to skip from the beigning to save on space
			skipBins[i] = minBin - modifier ;

			//the number of bins for a given reference point, i.e. the farthest bin number minus the closest bin number
			layerNumBins[i] = maxBin + modifier - skipBins[i];
			// layerNumBins[i] = ceil(maxDistance / epsilon) + 1;

			//if the current layer is the first layer, then the number of total bins for that layer will be just the number of bins for the reference point
			if(currentLayer == 0){ 
				layerBinCount[i] = layerNumBins[i];
			} else {
				// the number of bins in the layer is the number of bins for a reference point times the number of non empty bins in the previous layer
				layerBinCount[i] = binNonEmpty[currentLayer - 1] * layerNumBins[i];
			}
			
			// now that we know the size of the layer we can allocate the array to hold the bin values
			layerBins[i] = (unsigned int*)malloc(layerBinCount[i] * sizeof(unsigned int));

			// the bin values will all start at 0
			for(unsigned int j = 0; j < layerBinCount[i]; j++){
				layerBins[i][j] = 0;
			}

			// the first layer is unique because there are no offsets from the previous layer
			if(currentLayer == 0){
				// itterate through each point in the data set to assign point to bins
				for(unsigned int j = 0; j < numPoints; j++){
					// the bin number will be the the floor of this distance from the point to the reference point - the numebr of bins we skipped on the start
					unsigned int binNumber = allBinNumber[i*numPoints+j] - skipBins[i];

					// if the bin is empty, then increase the non empty bin count by 1
					if(layerBins[i][binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					//increment the value of the bin to count the number of points in that bin
					layerBins[i][binNumber]++; 

					// the offset from the first layer will be the bin number
					layerBinOffsets[i][j] = binNumber; 
				}
			} else {
				// assign every point to a bin for layers past the first one
				for(unsigned int j = 0; j < numPoints; j++){

					// the offset will be the bin number of the previous layer, which will be the number of non empty bins before it, times the number of bins for the reference point in the current layer
					unsigned int offset = (bins[currentLayer -1][pointBinOffsets[currentLayer-1][j]]-1)*layerNumBins[i];

					// the bin number will be the the floor of this distance from the point to the reference point - the number of bins we skipped on the start
					unsigned int binNumber = allBinNumber[i*numPoints+j] - skipBins[i];

					// if(offset+binNumber > layerBinCount[i]) {
					// 	// checkers = false;
					// 	printf("offset+binNumber is the problem with offset = %d, binnumber = %d, and layerbincount = %d", offset,binNumber,layerBinCount[i]);
					// }
					
					// if the bin was empty, then increment the number of non empty bins
					if(layerBins[i][offset+binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					// increment the value of the bin to count the number of points in that bin
					layerBins[i][offset+binNumber]++;

					// the offset for the point will be the previous offset + the bin number
					layerBinOffsets[i][j] = offset + binNumber;

				}

			}

			// printf("assigned bins and counted\n");

			// calculate the sum of squares based on the number of points in each bin. 
			// the lower the sum of squares, the more even the distribution of points
			averageNonEmptyBinCountTemp[i] = numPoints / layerBinNonEmpty[i];
			for(unsigned int j = 0; j < layerBinCount[i]; j++){
				sumSqrsTemp[i] += (layerBins[i][j] - averageNonEmptyBinCountTemp[i]) * (layerBins[i][j] - averageNonEmptyBinCountTemp[i]);
			}

			// //join to the bins
			// std::vector<std::vector<unsigned int>> bins;
			// for(unsigned int j = 0; j < ; j++ ){
				
			// }

		}

		//do coordinates now

		#pragma omp parallel for
		for(unsigned int i = numRPperLayer; i < maxRP+numRPperLayer; i++){
			layerBinNonEmpty[i] = 0;
			unsigned int modifier = 2;
			unsigned int currentDim = i - numRPperLayer;
			int maxBin = ceil((data[currentDim]+0.000001) / epsilon)+10;
			int minBin = maxBin-1;
			for(unsigned int j = 0; j < numPoints; j++){
				allBinNumber[i*numPoints+j] = ceil((data[j*dim+currentDim]+0.0000001) / epsilon)+10;
				if(allBinNumber[i*numPoints+j] > maxBin) maxBin = allBinNumber[i*numPoints+j];
				if(allBinNumber[i*numPoints+j] < minBin) minBin = allBinNumber[i*numPoints+j];
			}

			// printf("got max (%d) and min (%d) bins for %d\n", maxBin, minBin, i);

			

			// printf("got max and min bins\n");

			// a modifier for padding the bin sizes
			// unsigned int modifier = 2;

			//the number of bins to skip from the beigning to save on space

			skipBins[i] = minBin - modifier;

			// printf("got max (%d) and min (%d) bins for %d with %d skips\n", maxBin, minBin, i, skipBins[i]);

			//the number of bins for a given reference point, i.e. the farthest bin number minus the closest bin number
			layerNumBins[i] = maxBin + modifier - skipBins[i];
			// layerNumBins[i] = ceil(maxDistance / epsilon) + 1;

			//if the current layer is the first layer, then the number of total bins for that layer will be just the number of bins for the reference point
			if(currentLayer == 0){ 
				layerBinCount[i] = layerNumBins[i];
			} else {
				// the number of bins in the layer is the number of bins for a reference point times the number of non empty bins in the previous layer
				layerBinCount[i] = binNonEmpty[currentLayer - 1] * layerNumBins[i];
			}
			// printf("layerBinCount = %d\n", layerBinCount[i]);
			// now that we know the size of the layer we can allocate the array to hold the bin values
			layerBins[i] = (unsigned int*)malloc(layerBinCount[i] * sizeof(unsigned int));

			// the bin values will all start at 0
			for(unsigned int j = 0; j < layerBinCount[i]; j++){
				layerBins[i][j] = 0;
			}

			// the first layer is unique because there are no offsets from the previous layer
			if(currentLayer == 0){
				// itterate through each point in the data set to assign point to bins
				for(unsigned int j = 0; j < numPoints; j++){
					// the bin number will be the the floor of this distance from the point to the reference point - the numebr of bins we skipped on the start
					unsigned int binNumber = allBinNumber[i*numPoints+j] - skipBins[i];

					// if the bin is empty, then increase the non empty bin count by 1
					if(layerBins[i][binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					//increment the value of the bin to count the number of points in that bin
					layerBins[i][binNumber]++; 

					// the offset from the first layer will be the bin number
					layerBinOffsets[i][j] = binNumber; 
				}
			} else {
				// assign every point to a bin for layers past the first one
				for(unsigned int j = 0; j < numPoints; j++){

					// the offset will be the bin number of the previous layer, which will be the number of non empty bins before it, times the number of bins for the reference point in the current layer
					unsigned int offset = (bins[currentLayer -1][pointBinOffsets[currentLayer-1][j]]-1)*layerNumBins[i];

					// the bin number will be the the floor of this distance from the point to the reference point - the number of bins we skipped on the start
					unsigned int binNumber = allBinNumber[i*numPoints+j] - skipBins[i];

					// if(offset+binNumber > layerBinCount[i]) {
					// 	// checkers = false;
					// 	printf("offset+binNumber is the problem with offset = %d, binnumber = %d, and layerbincount = %d", offset,binNumber,layerBinCount[i]);
					// }
					
					// if the bin was empty, then increment the number of non empty bins
					if(layerBins[i][offset+binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					// increment the value of the bin to count the number of points in that bin
					layerBins[i][offset+binNumber]++;

					// the offset for the point will be the previous offset + the bin number
					layerBinOffsets[i][j] = offset + binNumber;

				}

			}

			// printf("assigned bins and counted lbne= %d\n",layerBinNonEmpty[i]);

			// calculate the sum of squares based on the number of points in each bin. 
			// the lower the sum of squares, the more even the distribution of points
			averageNonEmptyBinCountTemp[i] = numPoints / layerBinNonEmpty[i];
			if(averageNonEmptyBinCountTemp[i] >= numPoints / 3.0){
				sumSqrsTemp[i] = numPoints * numPoints;
			}
			for(unsigned int j = 0; j < layerBinCount[i]; j++){
				sumSqrsTemp[i] += (layerBins[i][j] - averageNonEmptyBinCountTemp[i]) * (layerBins[i][j] - averageNonEmptyBinCountTemp[i]);
				// sumSqrsTemp[i] += (layerBins[i][j]) * (layerBins[i][j]);

			}

		}

		//pick the one with lowest sum sqrs? sure why not //update: this was a bad idea
		unsigned int minSumIdx = 0; // this will be the reference point form all of the possible one that will be used for final layer construction
		for(unsigned int i = 1; i < numRPperLayer+maxRP; i++){
			#if MINSQRS
			if(sumSqrsTemp[minSumIdx] > sumSqrsTemp[i] ){
				minSumIdx = i; 
			}
			#else
			if(averageNonEmptyBinCountTemp[minSumIdx] > averageNonEmptyBinCountTemp[i] ){
				minSumIdx = i; 
			}
			#endif
		}

		// assign the sum sqrs for the layer as the one chosen
		sumSqrsLayers[currentLayer] = sumSqrsTemp[minSumIdx];
		averageNonEmptyBinCountLayers[currentLayer] = averageNonEmptyBinCountTemp[minSumIdx];

		// copy over the bin offsets based on the reference point that was chosen
		for(unsigned int i = 0; i < numPoints; i++){
			pointBinOffsets[currentLayer][i] = layerBinOffsets[minSumIdx][i];
		}

		// printf("calced metrics\n");
		
		// the size of the bins will be the size of the layer form the chosen one
		bins[currentLayer] = (unsigned int*)malloc(layerBinCount[minSumIdx]*sizeof(unsigned int));
		
		// tmnpcount will keep track of the number of previous non empty bins
		unsigned int tmpCount = 0;

		//go through each bin inn the layer
		for(unsigned int i = 0; i < layerBinCount[minSumIdx]; i++){

			//if the bin is not empty then increment our counter and assign that value to the bin
			if(layerBins[minSumIdx][i] != 0){
				tmpCount++;
				bins[currentLayer][i] = tmpCount;

			// if the bin is empty, then  leave the value as zero
			} else {
				bins[currentLayer][i] = 0;
			}
		}

		// the bin counts will be the number of bins in that layer
		binSizes[currentLayer] = layerBinCount[minSumIdx];

		// copy over the number of non empty bins for the layer
		binNonEmpty[currentLayer] = layerBinNonEmpty[minSumIdx];

		// copy the number of bins for the reference point
		binAmounts[currentLayer] = layerNumBins[minSumIdx];


		// assign the point bin numbers and save them
		for(unsigned int i = 0; i < numPoints; i++){
			// pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon)-skipBins[minSumIdx];
			pointBinNumbers[i][currentLayer] = allBinNumber[minSumIdx*numPoints+i]-skipBins[minSumIdx];
		}

		if(currentLayer==0){
			printf("Selected RPs: %d", minSumIdx);
		}else{
			printf(", %d", minSumIdx);
		}
		

		// check if the current layer is at least the min number of layers for the tree
		if(currentLayer == maxRP-1){
			// check if the sum of sqrs is still decreasing by adding layers or if we are at the max number of layers
		// if( currentLayer >= MINRP && (sumSqrsLayers[currentLayer-1]/sumSqrsLayers[currentLayer] < 3 || averageNonEmptyBinCountLayers[currentLayer] <= 10.0 || currentLayer == maxRP - 1)){
			
			//set check to false to exit the while loop
			check = false;

			// keep track of the running total of points in bins
			unsigned int runningTotal = 0;

			// go through each bin of the final layer
			for(unsigned int i = 0; i < binSizes[currentLayer]; i++){
				// the final layer of the tree will have the running total of points in the bins as values
				bins[currentLayer][i] = runningTotal;
				// update the running total number of points
				runningTotal += layerBins[minSumIdx][i];
				
			}

			
		}
		

		currentLayer++;
		// } 
		//free all the memory used for construction of this layer
		// free(distMat);
		for(unsigned int i = 0; i < numRPperLayer+maxRP; i++){
			free(layerBins[i]);
			free(layerBinOffsets[i]);
		}
		free(layerBins);
		free(layerBinOffsets);
		free(layerBinCount);
		free(layerBinNonEmpty);
		free(layerNumBins);
		free(skipBins);
		free(allBinNumber);
		// free(RPArray);

	}

	printf("\n");
	// the number of layers/ reference points is the current layer
	unsigned int snumRP = currentLayer;

	printf("Selected %d reference points\n", snumRP);

	for(unsigned int i = 0; i < snumRP; i++){
		if(TESTING_SEARCH) fprintf(stderr," L%d, %f,", i, sumSqrsLayers[i]);
		printf("Layer %d sumqrs: %f BinCount: %u AverageBinsCount: %f, nonEmpty: %u\n", i, sumSqrsLayers[i], binSizes[i], averageNonEmptyBinCountLayers[i], binNonEmpty[i]);
	}

    // sort the point arrays from bottom to top using a stable sort
	
	// use a thrust vector that will keep track of the point numbers
	thrust::host_vector<unsigned int*> pointVector(numPoints);

	//go through each point an load the point bin numbers
	for(unsigned int i = 0; i < numPoints; i++){
		// the point vector will contain the point number i.e. the row of the point in the original data, and then each bin of the point
		pointVector[i] = (unsigned int*)malloc(sizeof(unsigned int)*(snumRP+1));

		//copy over the bin numbers
		for(int j = 0; j < snumRP; j++){
			pointVector[i][j+1] = pointBinNumbers[i][j];
		}
	}

	// assign the point numbers which start as sequential
	for(unsigned int i = 0; i < numPoints; i++){
		pointVector[i][0] = pointArray[i];
	}

	// go through each layer of the tree backwards to sort the points
    for(unsigned int i = 0; i < snumRP; i++){

		// array to keep track of the bin values that are being used for this sort at the layer i
		unsigned int * oneBin = (unsigned int*)malloc(sizeof(unsigned int)*numPoints);

		// copy over the bin numebrs for sorting
		#pragma omp parallel for
		for(unsigned int j = 0; j < numPoints; j++ ){
			oneBin[j] = pointVector[j][snumRP-i];
		}

		// run the stable sort with the bin numbers at i as the key
		thrust::stable_sort_by_key(thrust::omp::par, oneBin, oneBin+numPoints, pointVector.begin());

		free(oneBin);
	}

	// copy over results after sorting
	#pragma omp parallel for
	for(unsigned int i = 0; i < numPoints; i++){
		// point array will have all of the point numbers is order
		pointArray[i] = pointVector[i][0];
		for(unsigned int j = 0; j < snumRP; j++){
			// the point bin numbers will have all of the bin numbers
			pointBinNumbers[i][j] = pointVector[i][j+1];
		}
		free(pointVector[i]);
	}

	// free data used only in tree cointsruction
	for(unsigned int i = 0; i < maxRP; i++){
		free(pointBinOffsets[i]);
	}
	free(pointBinOffsets);
	free(binNonEmpty);
	free(sumSqrsLayers);
	free(averageNonEmptyBinCountLayers);
	free(averageNonEmptyBinCountTemp);
	free(sumSqrsTemp);
	cudaFree(d_data);
	cudaFree(d_dim);
	cudaFree(d_numPoints);
	cudaFree(d_epsilon);


	// return the tree
	*rbins = bins;
	*rpointBinNumbers = pointBinNumbers;

	return(snumRP);
}


unsigned int generateRanges(unsigned int ** tree, //points to the tree constructed with buildTree()
							unsigned int numPoints, // the number of points in the dataset
							unsigned int ** pointBinNumbers, // the bin numbers of the points relative to the reference points
							unsigned int numLayers, // the number of layer the tree has
							unsigned int * binSizes,	// the number of bins for each layer, or rather the width of the tree in bins for that layer
							unsigned int * binAmounts, // the number of bins for each reference point, ranhge/epsilon
							unsigned int ** addIndexes, // where generateRanges will return the non-empty index locations in the tree's final layer
							unsigned int *** rangeIndexes, // the index locations that are adjacent to each non-empty index
							unsigned int *** rangeSizes, // the number of points in adjacent non-empty indexes for each non-empty index
							unsigned int ** numValidRanges, // the numnber of adjacent non-empty indexes for each non-empty index
							unsigned long long ** calcPerAdd, // the number of calculations that will be needed for each non-empty index
							unsigned int ** numPointsInAdd ){ // the number of points in each non-empty index

    // an array to temproarily hold the indexes of only non empty bins, over allocates to the maximum possible
    unsigned int*tempIndexes = (unsigned int*)malloc(sizeof(unsigned int)*binSizes[numLayers-1]);

	// the number of non empty bins in the final layer or number of addresses with points
    unsigned int nonEmptyBins = 0;

	// counting the number of non empty bins and keeping track of the indexes of those nonempty bins
    for(unsigned int i = 0; i < binSizes[numLayers-1]-1; i++){ 
        
		// if the tree value on the last layer is less than the next then it has points in it
		if(tree[numLayers-1][i] < tree[numLayers-1][i+1]){
			//keep track of that non empty index
            tempIndexes[nonEmptyBins] = i;
            nonEmptyBins++;
        }
    }

	// local ranges keeps track of the indexes that are searched for each address, and were distancs calcs are needed
	// i.e. for each nonempty last layer index, the indexes that are adjacent
    unsigned int ** localRangeIndexes = (unsigned int**)malloc(sizeof(unsigned int*)*nonEmptyBins);

	// local range sizes keeps track of how large each localRangeIndexes is, 
	// so the number of points in an adjacent nonempty indexe
	unsigned int ** localRangeSizes = (unsigned int **)malloc(sizeof(unsigned int*)*nonEmptyBins);

	// a temp variable to keep track of the num valid ranges for a single index 
    unsigned int * tempNumValidRanges = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins);

	// this keeps track of the number of distance calculations that will be needed for a non empty index/address
	unsigned long long * tempCalcPerAdd = (unsigned long long*)malloc(sizeof(unsigned long long)*nonEmptyBins);

	// an array for holding the non empty index locations in the final tree layer
    unsigned int * tempAddIndexes = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);

	// as array for keeping track of the number of points int the non empty  indexes, correlated with tempAddIndexes
	unsigned int * tempNumPointsInAdd = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);

	// copy the temp indexes which keeps track of the nonepty indexes into an array that is the correct size
    for(unsigned int i = 0; i < nonEmptyBins; i++){
        tempAddIndexes[i] = tempIndexes[i];
    }

	// free the overallocated array now that we have the data stored in tempAddIndexes
    free(tempIndexes);

	// the number of searches that are needed for a full search is the number of referecnes point cubed. This is always true, mostly.
	const unsigned int numSearches = pow(3,numLayers);

	
	//array to hold the point bin numbers
	unsigned int ** binNumbers = (unsigned int**)malloc(sizeof(unsigned int*)*nonEmptyBins);
	#pragma omp parallel for
	for(unsigned int i = 0; i < nonEmptyBins; i++){
		binNumbers[i] = (unsigned int*)malloc(sizeof(unsigned int)*numLayers);
		for(unsigned int j = 0; j < numLayers; j++){
			binNumbers[i][j] = pointBinNumbers[ tree[ numLayers-1 ][ tempAddIndexes[i] ]][j];
		}
	}



	bool binSearch = BINARYSEARCH;
	if(log2(nonEmptyBins) < numLayers & BINARYSEARCH == 2){
		binSearch = true;
	}

	#if TESTING_SEARCH
	double binarySearchTimes = 0;
	double treeSearchTimes = 0;
	#endif

	//go through each non empty bin and do all the needed searching and generate the arrays that are needed for the calculations kernels
	#pragma omp parallel for
    for(unsigned int i = 0; i < nonEmptyBins; i++){

		// this will record the number of calculations that need to be made by this index/address
		unsigned long long numCalcs;

		// the number of ranges, so the number of adjacent nonempty indexes 
		unsigned int numRanges;

		// a temporary address that will be modifed to perfom the searches
		unsigned int tempAdd[numLayers];

		// the number of points in this non empty index
		unsigned int localNumPointsInAdd;

		
		if(binSearch || TESTING_SEARCH){

			//set up array to binary search on
			binarySearch( i, // the bin to search in bin numebrs
				tempAdd, //temporary address for searching
				binNumbers, //array of bin numbrs 
				nonEmptyBins, //size of binNumebrs
				numLayers, //number of reference points
				tree, //the tree structure
				binAmounts, // the range of bins from a reference points, i.e. range / epsilon
				tempAddIndexes, // location of nonempty bin in tree
				&numCalcs, // for keeping track of the number of distance calculations to be performed
				&numRanges, // the number of adjacent non-empty addresses/indexes
				&localRangeIndexes[i], // this addresses/index's array to keep track of adjacent indexes
				&localRangeSizes[i], // the number of points in the indexes in localRangeIndexes
				&localNumPointsInAdd, // the number of points in this nonempty address
				numSearches); // the number of searches that need to be performed, 3^r
			


		}
		
		if (!binSearch) {

			treeTraversal(tempAdd, // array of int for temp storage for searching
				tree, // pointer to the tree made with buildTree()
				binSizes, // the widths of each layer of the tree measured in bins
				binAmounts, // the range of bins from a reference points, i.e. range / epsilon
				binNumbers[i], // the address/bin numbers of the current index/address
				numLayers, // the number of reference points or layers in the tree, same thing
				&numCalcs, // for keeping track of the number of distance calculations to be performed
				&numRanges, // the number of adjacent non-empty addresses/indexes
				&localRangeIndexes[i], // this addresses/index's array to keep track of adjacent indexes
				&localRangeSizes[i], // the number of points in the indexes in localRangeIndexes
				&localNumPointsInAdd, // the number of points in this nonempty address
				numSearches); // the number of searches that need to be performed, 3^r


		}

		// storing variables into arrays that coorespond to the non-empty indexes/addresses
		tempCalcPerAdd[i] = numCalcs;
		tempNumValidRanges[i] = numRanges;
		tempNumPointsInAdd[i] = localNumPointsInAdd;


    }


	// std::cerr << "Finished search" << std::endl;

	// assiging pointers for accsess outside of this functions scope
	*addIndexes = tempAddIndexes;
	*calcPerAdd = tempCalcPerAdd;
	*numValidRanges = tempNumValidRanges;
	*rangeSizes = localRangeSizes;
	*rangeIndexes = localRangeIndexes;
	*numPointsInAdd = tempNumPointsInAdd;

	//return the number of non-empty bins
	return nonEmptyBins;

}

// __host__ __device__ 
long int depthSearch(unsigned int ** tree, //pointer to the tree built with buildTree()
				unsigned int * binAmounts, // the number of bins for each reference point, i.e. range/epsilon
				unsigned int numLayers, //the number of layers in the tree
				unsigned int * searchBins){ // the bin number that we are searching for
	
	// the offset is used for keeping track of the offset from the begining of each layer to the index
	unsigned int offset = 0;
	
	//go through each layer up to the last to determine if the index is non-empty and if it is then find the offset into the next layer
	for(unsigned int i = 0; i < numLayers-1; i++){
		
		//check the current layer at the bin number + offset may or may not need -1 here
		if (tree[i][offset + searchBins[i]] == 0){
			return -2;
		}

		// the next offset will be the previous layer index number * the number of bins for the reference point in the next layer
		offset = (tree[i][searchBins[i]+offset]-1)*binAmounts[i+1];
	}

	//the index will be the last layers bin number plus the offset for the last layer
	long int index = searchBins[numLayers-1]+offset;

	//if last layer has points then return the index value
	if(tree[numLayers-1][index] < tree[numLayers-1][index+1]){
		return index;
	}else{
		return -1;
	}

}

// __host__ __device__
void treeTraversal(unsigned int * tempAdd, //twmp array for the address being searched
				   unsigned int ** tree, // the pointer to the tree
				   unsigned int * binSizes, // the width of the tree for each layer mesuared in number of bins
				   unsigned int * binAmounts, // the number of bins for each reference point
				   unsigned int * binNumbers, // the bin number for the home address
				   unsigned int numLayers, // the number of reference points/layers in the tree
				   unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
				   unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
				   unsigned int ** rangeIndexes, // the array of non-empty adjacent index locations
				   unsigned int ** rangeSizes, // the number of points in each of the adjacent non-empty indexes
				   unsigned int * numPointsInAdd, //the number of points in the home address/iondex
				   unsigned int numSearches){ //the number of searches that are being perfomred for each addresss

	//keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

	// keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

	//keep track of the locations of adjacent indexes that are not empty
	unsigned int * localRangeIndexes = (unsigned int*)malloc(sizeof(unsigned int)*numSearches);

	// the number of points in the adjacent non-empty indexes
	unsigned int * localRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*numSearches);

	//permute through bin variations (3^r) and run depth searches
	for(unsigned int i = 0; i < numSearches; i++){
		
		//modify temp add for the search based on our itteration i
		for(unsigned int j = 0; j < numLayers; j++){
			tempAdd[j] = binNumbers[j] + (i / (int)pow(3, j) % 3)-1;
		}
		
		//perform the search and get the index location of the return 

		int index = depthSearch(tree, binAmounts, numLayers, tempAdd);

		//check if the index location was non empty
		if(index >= 0){
			//store the non empty index location
			localRangeIndexes[localNumRanges] = index;

			//calcualte the size of the index, i.e. the number of points in the index
			unsigned long long size = tree[numLayers-1][index+1] - tree[numLayers-1][index]; //may need to +- to index here!!!!!!!!!!!!!!!

			//store that in the sizes array
			localRangeSizes[localNumRanges] = size;

			// keep running total of the sizes for getting the number of calculations latter
			localNumCalcs += size;

			//keep track of the number of non-empty adjacent indexes
			localNumRanges++;
		}
	}

	// get the index of the home address
	int homeIndex = depthSearch(tree, binAmounts, numLayers, binNumbers);
	// std::cerr << "index: "<< homeIndex << " above: "<<tree[numLayers-1][homeIndex+1]<<" below: "<< tree[numLayers-1][homeIndex] << " difference: "<<tree[numLayers-1][homeIndex+1] - tree[numLayers-1][homeIndex] << std::endl;

	// find the number of points in the home address
	unsigned long long numHomePoints = tree[numLayers-1][homeIndex+1] - tree[numLayers-1][homeIndex]; //may need to +- one to index here !!!!!!!!!

	if(numHomePoints == 0)printf("ERROR: no points found in address at index: %d\n", homeIndex);
	// use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
	*numCalcs = localNumCalcs*numHomePoints;

	// return the arrays with pointer magic
	*rangeIndexes = localRangeIndexes;
	*rangeSizes = localRangeSizes;

	*numberRanges = localNumRanges;

	*numPointsInAdd = numHomePoints;


}

// __host__ __device__
inline long int compareBins(unsigned int * bin1, unsigned int * bin2, unsigned int binSize){
	for(long int i = 0; i < binSize; i++){
		if(bin1[i] < bin2[i]){
			return -1;
		}
		if(bin1[i] > bin2[i]){
			return 1;
		}
	}
	return 0;
}
// __host__ __device__
long int bSearch(unsigned int * tempAdd, //address to search for
			unsigned int ** binNumbers, //array of addresses
			unsigned int nonEmptyBins, //number of bins
			unsigned int numLayers) //numebr of layers or size of addresses
			{

	// initial conditions of the search
	long int left = 0;
	long int right = nonEmptyBins-1;
	
	
	//while loop for halving search each itterations
	while(left <= right){
		//calculate the middle
		long int mid = (left + right)/2;
		// -1 for smaller, 1 for larger, 0 for equal
		long int loc = compareBins( binNumbers[mid], tempAdd, numLayers);
		//if we found the index
		if( loc == 0){
			return mid;
		//if the index was smaller
		}else if (loc == -1){
			left = mid + 1;
		//if the index was larger
		} else {
			right = mid - 1;
		}
	}

	return -1;

}

// __host__ __device__
void binarySearch(	unsigned int searchIndex, // the bin to search in bin numebrs
					unsigned int  * tempAdd, //temporary address for searching
					unsigned int ** binNumbers, //array of bin numbrs 
					unsigned int nonEmptyBins, //size of binNumebrs
					unsigned int numLayers, //number of reference points
					unsigned int ** tree, //the tree structure
					unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
					unsigned int * addIndexs, //location of nonempty bins in the tree
					unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
					unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
					unsigned int ** rangeIndexes, // the array of non-empty adjacent index locations
					unsigned int ** rangeSizes, // the number of points in each of the adjacent non-empty indexes
					unsigned int * numPointsInAdd, //the number of points in the home address/iondex
					unsigned int numSearches){ //the number of searches that are being perfomred for each addresss

	
	//keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

	// keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

	//keep track of the locations of adjacent indexes that are not empty
	unsigned int * localRangeIndexes = (unsigned int*)malloc(sizeof(unsigned int)*numSearches);

	// the number of points in the adjacent non-empty indexes
	unsigned int * localRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*numSearches);

	//permute through bin variations (3^r) and run depth searches
	for(unsigned int i = 0; i < numSearches; i++){
		
		//modify temp add for the search based on our itteration i
		for(unsigned int j = 0; j < numLayers; j++){
			tempAdd[j] = binNumbers[searchIndex][j] + (i / (int)pow(3, j) % 3)-1;
		}
		//perform the search and get the index location of the return 
		long int index = bSearch(tempAdd, binNumbers, nonEmptyBins, numLayers);

		//check if the index location was non empty
		if(index >= 0){
			index = addIndexs[index];
			//store the non empty index location
			localRangeIndexes[localNumRanges] = index;

			//calcualte the size of the index, i.e. the number of points in the index
			unsigned long long size = tree[numLayers-1][index+1] - tree[numLayers-1][index]; //may need to +- to index here!!!!!!!!!!!!!!!

			//store that in the sizes array
			localRangeSizes[localNumRanges] = size;

			// keep running total of the sizes for getting the number of calculations latter
			localNumCalcs += size;

			//keep track of the number of non-empty adjacent indexes
			localNumRanges++;
		}
	}

	// get the index of the home address
	long int homeIndex = addIndexs[bSearch(binNumbers[searchIndex], binNumbers, nonEmptyBins, numLayers)];
	// std::cerr << "index: "<< homeIndex << " above: "<<tree[numLayers-1][homeIndex+1]<<" below: "<< tree[numLayers-1][homeIndex] << " difference: "<<tree[numLayers-1][homeIndex+1] - tree[numLayers-1][homeIndex] << std::endl;
	// find the number of points in the home address
	unsigned long long numHomePoints = tree[numLayers-1][homeIndex+1] - tree[numLayers-1][homeIndex]; //may need to +- one to index here !!!!!!!!!

	if(numHomePoints == 0)printf("ERROR: no points found in address at index: %lu\n", homeIndex);
	// use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
	*numCalcs = localNumCalcs*numHomePoints;

	// return the arrays with pointer magic
	*rangeIndexes = localRangeIndexes;
	*rangeSizes = localRangeSizes;

	*numberRanges = localNumRanges;

	*numPointsInAdd = numHomePoints;

}