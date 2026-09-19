#include "include/launcher.cuh"


void constructNeighborTable(unsigned int * pointInDistValue, 
    unsigned int * pointersToNeighbors, 
    unsigned long long int * cnt, 
    unsigned int * uniqueKeys, 
    unsigned int * uniqueKeyPosition, 
    unsigned int numUniqueKeys,
    struct neighborTable * tables)
{

    #pragma omp parallel for
    for (unsigned int i=0; i < (*cnt); i++)
    {
        pointersToNeighbors[i] = pointInDistValue[i];
    }

    //////////////////////////////
    //NEW when Using unique on GPU
    //When neighbortable is initalized (memory reserved for vectors), we can write directly in the vector

    //if using unicomp we need to update different parts of the struct
    #pragma omp parallel for
    for (unsigned int i = 0; i < numUniqueKeys; i++) {

        unsigned int keyElem = uniqueKeys[i];
        //Update counter to write position in critical section
        omp_set_lock(&tables[keyElem].pointLock);

        unsigned int nextIdx = tables[keyElem].cntNDataArrays;
        tables[keyElem].cntNDataArrays++;
        omp_unset_lock(&tables[keyElem].pointLock);

        tables[keyElem].vectindexmin[nextIdx] = uniqueKeyPosition[i];
        tables[keyElem].vectdataPtr[nextIdx] = pointersToNeighbors;	

        //final value will be missing
        if (i == (numUniqueKeys - 1))
        {
            tables[keyElem].vectindexmax[nextIdx] = (*cnt)-1;
        }
        else
        {
           tables[keyElem].vectindexmax[nextIdx] = (uniqueKeyPosition[i+1]) - 1;
        }
    }

    return;
} 

struct neighborTable * launchGPUSearchKernel(unsigned int ** tree, // a pointer to the tree
                        unsigned int numPoints, //the number of points in the data
                        unsigned int ** pointBinNumbers,  //the bin numbers ofr each point to each reference point
                        unsigned int numLayers, //the number of reference points
                        unsigned int * binSizes, // the number of bins in each layer
                        unsigned int * binAmounts, //the number of bins for each reference point
                        double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
                        unsigned int dim,//the dimensionality of the data
                        double epsilon,//the distance threshold being searched
                        unsigned int * pointArray)// the array of point numbers ordered to match the sequence in the last array of the tree and the data
{


    double time2 = omp_get_wtime();

    // the number of searches that are needed for a full search is the number of referecnes point cubed. This is always true, mostly.
	const unsigned int numSearches = pow(3,numLayers);

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

    // an array for holding the non empty index locations in the final tree layer
    unsigned int * addIndexes = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);

	// copy the temp indexes which keeps track of the nonepty indexes into an array that is the correct size
    #pragma omp parallel for num_threads(8)
    for(unsigned int i = 0; i < nonEmptyBins; i++){
        addIndexes[i] = tempIndexes[i];
    }

	// free the overallocated array now that we have the data stored in tempAddIndexes
    free(tempIndexes);



    // create cuda memory for variables

    //device array to hold the address indexes locations of non-empty bins
    unsigned int * d_addIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(unsigned int )*nonEmptyBins, cudaMemcpyHostToDevice));
    
    //device memory for the range indexes that will be created in generate ranges
    // this is a 2d array of num non-empty bins x num searches
    unsigned int * d_rangeIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(unsigned int)*nonEmptyBins*numSearches));

    //device memory for the range sizes that will be created in generate ranges
    // this is a 2d array of num non-empty bins x num searches
    unsigned int * d_rangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*nonEmptyBins*numSearches));

    //device memory for the number of adgacent non-empty bins to each non-empty bin i.e. the depth of rangeIndexes and rangeSizes
    unsigned int * d_numValidRanges;
    assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(unsigned int)*nonEmptyBins));

    //deviucce memory to hold the number of calcs each address will need to compute
    unsigned long long * d_calcPerAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_calcPerAdd, sizeof(unsigned long long)*nonEmptyBins));

    //device memory to hold the number of popints in each non-empty bin
    unsigned int * d_numPointsInAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));

    //array to hold the point bin numbers
	unsigned int * binNumbers = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins*numLayers);
	#pragma omp parallel for
	for(unsigned int i = 0; i < nonEmptyBins; i++){
		for(unsigned int j = 0; j < numLayers; j++){
			binNumbers[i*numLayers+j] = pointBinNumbers[ tree[ numLayers-1 ][ addIndexes[i] ]][j];
		}
	}

    //device memory to hold the bin number arrays
    unsigned int * d_binNumbers;
    assert(cudaSuccess == cudaMalloc((void**)&d_binNumbers, sizeof(unsigned int)*nonEmptyBins*numLayers));
    assert(cudaSuccess ==  cudaMemcpy(d_binNumbers, binNumbers, sizeof(unsigned int )*nonEmptyBins*numLayers, cudaMemcpyHostToDevice));

    //device memory to hold the temorary address being searched
    unsigned int * d_tempAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_tempAdd, sizeof(unsigned int*)*nonEmptyBins*numLayers));

    //device memory to hold the number of searches that will be made for each non-empty bin
    unsigned int * d_numSearches;
    assert(cudaSuccess == cudaMalloc((void**)&d_numSearches, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numSearches, &numSearches, sizeof(unsigned int), cudaMemcpyHostToDevice));

    //device memory for the bin number of each point to each reference point
    unsigned int * d_pointBinNumbers;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointBinNumbers, sizeof(unsigned int)*numPoints*numLayers));
    for(unsigned int i = 0; i < numPoints; i++){
        assert(cudaSuccess ==  cudaMemcpy(d_pointBinNumbers+numPoints, pointBinNumbers[i], sizeof(unsigned int )*numLayers, cudaMemcpyHostToDevice));
    }

    // the number of bins for each layer
    unsigned int * d_binSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_binSizes, sizeof(unsigned int)*numLayers));
    assert(cudaSuccess ==  cudaMemcpy(d_binSizes, binSizes, sizeof(unsigned int)*numLayers, cudaMemcpyHostToDevice));

    //device memory for the number of bins to each reference point
    unsigned int * d_binAmounts;
    assert(cudaSuccess == cudaMalloc((void**)&d_binAmounts, sizeof(unsigned int)*numLayers));
    assert(cudaSuccess ==  cudaMemcpy(d_binAmounts, binAmounts, sizeof(unsigned int)*numLayers, cudaMemcpyHostToDevice));

    unsigned int totalBinNumber = 0;
    for(unsigned int i = 0; i < numLayers; i++){
        totalBinNumber += binSizes[i];
    }

    unsigned int binOffset = 0;
    unsigned int *Ltree = (unsigned int *)malloc(sizeof(unsigned int)*totalBinNumber);
    for(unsigned int i = 0; i < numLayers; i++){
        #pragma parallel for num_threads(8)
        for(unsigned int j = 0; j < binSizes[i]; j++){
            Ltree[binOffset+j] = tree[i][j];
        }
        binOffset += binSizes[i];
    }

    //device memory for the tree
    unsigned int * d_tree;
    assert(cudaSuccess == cudaMalloc((void**)&d_tree, sizeof(unsigned int)*totalBinNumber));
    assert(cudaSuccess ==  cudaMemcpy(d_tree, Ltree, sizeof(unsigned int )*totalBinNumber, cudaMemcpyHostToDevice));
    

    unsigned int * d_lastTreeLayer;
    assert(cudaSuccess == cudaMalloc((void**)&d_lastTreeLayer, sizeof(unsigned int)*binSizes[numLayers-1]));
    assert(cudaSuccess ==  cudaMemcpy(d_lastTreeLayer, tree[numLayers-1], sizeof(unsigned int )*binSizes[numLayers-1], cudaMemcpyHostToDevice));

    //device memory for the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    //device memory for the number of layers in the tree
    unsigned int * d_numLayers;
    assert(cudaSuccess == cudaMalloc((void**)&d_numLayers, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numLayers, &numLayers, sizeof(unsigned int), cudaMemcpyHostToDevice));

    //device memory for the number of layers in the tree
    unsigned int * d_nonEmptyBins;
    assert(cudaSuccess == cudaMalloc((void**)&d_nonEmptyBins, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_nonEmptyBins, &nonEmptyBins, sizeof(unsigned int), cudaMemcpyHostToDevice));
    



    const unsigned int numBlocks = ceil(1.0*nonEmptyBins/BLOCK_SIZE);

    GPUGenerateRanges<<<numBlocks, BLOCK_SIZE>>>(d_tree,
                                                d_lastTreeLayer,
                                                d_numPoints,
                                                d_pointBinNumbers,
                                                d_numLayers,
                                                d_binSizes,
                                                d_binAmounts,
                                                d_addIndexes,
                                                d_rangeIndexes,
                                                d_rangeSizes,
                                                d_numValidRanges,
                                                d_calcPerAdd,
                                                d_numPointsInAdd,
                                                d_nonEmptyBins,
                                                d_binNumbers,
                                                d_tempAdd,
                                                d_numSearches);
    
    cudaDeviceSynchronize();

    double time3 = omp_get_wtime();

	#if BINARYSEARCH
	printf("Tree BINARY search time: %f\n", time3-time2);
	#else
	printf("Tree TRAVERSAL search time: %f\n", time3-time2);
	#endif

    unsigned long long * calcPerAdd = (unsigned long long *)malloc(sizeof(unsigned long long)*nonEmptyBins);
    assert(cudaSuccess ==  cudaMemcpy(calcPerAdd, d_calcPerAdd, sizeof(unsigned long long)*nonEmptyBins, cudaMemcpyDeviceToHost));

    unsigned int * numValidRanges = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins);
    assert(cudaSuccess ==  cudaMemcpy(numValidRanges, d_numValidRanges, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyDeviceToHost));

    unsigned int * rangeSizes = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins*numSearches);
    assert(cudaSuccess ==  cudaMemcpy(rangeSizes, d_rangeSizes, sizeof(unsigned int)*nonEmptyBins*numSearches, cudaMemcpyDeviceToHost));

    unsigned int * rangeIndexes = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins*numSearches);
    assert(cudaSuccess ==  cudaMemcpy(rangeIndexes, d_rangeIndexes, sizeof(unsigned int)*nonEmptyBins*numSearches, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();



    // keep track of the number of total calcs needed
    unsigned long long sumCalcs = 0;

    //keep track of the number of address that were found in searching for each address
    unsigned long long sumAdds = 0;

    //itterating through just to find sum values
    for(unsigned int i = 0; i < nonEmptyBins; i++){
        sumCalcs += calcPerAdd[i];
        sumAdds += numValidRanges[i];
    }

    printf("Total calcs: %llu, Total Adds: %llu\n", sumCalcs, sumAdds);

    // add index range has the values at each nonempty index of the last layer of the tree
    unsigned int * addIndexRange = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);
    for(unsigned int i = 0; i < nonEmptyBins; i++){
        addIndexRange[i] = tree[numLayers-1][addIndexes[i]];
    }

    // an array for keeping track of where the linear indexs start for each elemetn in the 2d one
	unsigned int * linearRangeID = (unsigned int*)malloc(sizeof(unsigned int) * nonEmptyBins);

	// a linear range index made from the 2d range index array to keep track of adjacent non-empty indexes for each non-empty index
	unsigned int * linearRangeIndexes = (unsigned int*)malloc(sizeof(unsigned int)*sumAdds);

	//a linear range sizes made from the range siezes array to keep track of the number of points in adjacent indexes
	unsigned int * linearRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*sumAdds);

	//running total for keeping track of starts for each index in the linear arrays
	unsigned int runningTotal = 0;
	//linearizeing 2d arrays to 1d arrays for GPU work
	for(unsigned int i = 0; i < nonEmptyBins; i++){

		//keeping track of start locations in the linear arrays
		linearRangeID[i] = runningTotal;

		//populating the linear arrays from the 2d ones
		for(unsigned int j = 0; j < numValidRanges[i]; j++){
			linearRangeIndexes[runningTotal + j] = tree[numLayers-1][rangeIndexes[i*numSearches+j]];
			linearRangeSizes[runningTotal + j] = rangeSizes[i*numSearches+j];
		}

		//increment the running total by the number of ranges for the current index
		runningTotal += numValidRanges[i];
	}

    printf("Number non-empty bins: %d\nNumber of calcs: %llu\nNumber Address for calcs: %llu\n", nonEmptyBins, sumCalcs, sumAdds);

    
    cudaFree(d_tree);
    cudaFree(d_pointBinNumbers);
    cudaFree(d_numLayers);
    cudaFree(d_binSizes);
    cudaFree(d_binAmounts);
    cudaFree(d_nonEmptyBins);
    cudaFree(d_rangeIndexes);
    cudaFree(d_rangeSizes);


    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    //set a value for the number of calculations made by each thread per kernel invocation
    unsigned long long calcsPerThread = CALCS_PER_THREAD; 

    //the number of thrreads assigned to each non-empty address
    unsigned long long * numThreadsPerAddress = (unsigned long long *)malloc(sizeof(unsigned long long )*nonEmptyBins);

    //keeping track of the number of batches that will be needed
    unsigned int numBatches = 0;

    // the number of threads that will be avaliable for each kernel invocation
    unsigned long long threadsPerBatch = (unsigned long long)KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        // the number of threads for the address is the ceiling of the number of calcs for that address over calcs per thread
        numThreadsPerAddress[i] = calcPerAdd[i] / calcsPerThread;
        if(calcPerAdd[i] % calcsPerThread != 0) numThreadsPerAddress[i]++;

        // if(calcPerAdd[i] == 0) printf("ERROR: add %d has 0 calcs\n", i);

        // check if the number of threads is higher than the number of threads for a batch
        if (sum + numThreadsPerAddress[i] < threadsPerBatch){
            // if the number of threads is less than keep adding addresses to the batch
            sum += numThreadsPerAddress[i];
        }else{
            // if we would exceed the number of threads for that batch, then dont add
            sum = numThreadsPerAddress[i];

            // check for an error
            // if(numThreadsPerAddress[i] > threadsPerBatch) printf("Warning: Address %d is too big. Needs: %d threads which is more than %d\n", i, numThreadsPerAddress[i], threadsPerBatch);

            //increment the number of batches needed
            numBatches++;
        }
    }

    //always need at least one batch
    numBatches++;

    //keeping track of the number of calculations for each batch
    unsigned long long * numCalcsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    //keeping track of the number of addresses that batch will compute
    unsigned int * numAddPerBatch = (unsigned int*)calloc(numBatches, sizeof(unsigned int));

    //keeping track of the number of threads that are in each batch
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    // setting starting batch for loop
    unsigned int currentBatch = 0;

    // go through each non-empty index
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        //error check
        if(currentBatch > numBatches) printf("ERROR 3: current batch %d is greater than num batches %d\n", currentBatch, numBatches);

        //check if the batch is new or if the number of threads per batch will exceed the max if added
        if(numThreadsPerBatch[currentBatch] == 0 || numThreadsPerBatch[currentBatch] + numThreadsPerAddress[i] < threadsPerBatch){
            //add the number of threads for index i to the number of threads for the batch
            numThreadsPerBatch[currentBatch] += numThreadsPerAddress[i];

            //increment the number of addresses in the current batch
            numAddPerBatch[currentBatch]++;

            // add the number of calculations for the address to the number of calcs for the batch
            numCalcsPerBatch[currentBatch] += calcPerAdd[i];
        } else { //if the number of threads for the batch will be too many, then need to add to the next batch instead
            currentBatch++;
            i = i - 1;
        }
    }


    // array to track which thread is assigned to witch address
    unsigned int ** addAssign = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    //the offset of the thread for calculations inside an address
    unsigned int ** threadOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    for(int i = 0; i < numBatches; i++){
        // array to track which thread is assigned to witch address
        addAssign[i] = (unsigned int * )malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);

        //the offset of the thread for calculations inside an address
        threadOffsets[i] = (unsigned int*)malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);
    }

    //setting the intital batch starting address
    unsigned int batchFirstAdd = 0;

    //keep track of the total numebr of threads
    unsigned long long totalThreads = 0;

    // array to keep track of where linear arrays start for threads based on the batch number
    unsigned int * batchThreadOffset = (unsigned int *)malloc(sizeof(unsigned int)*numBatches);

    // calculating the thread assignements
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int threadCount = 0;

        //compute which thread does wich add
        for(unsigned int j = 0; j < numAddPerBatch[i]; j++){

            //basic error check
            // if(numThreadsPerAddress[batchFirstAdd + j] == 0) {
            //     printf("ERROR: add %d has 0 threads\n", j + batchFirstAdd);
            // }

            //for each address in the batch, assigne threads to it
            for(unsigned int k = 0; k < numThreadsPerAddress[batchFirstAdd + j]; k++){

                //assign the thread to the current address
                addAssign[i][threadCount] = j + batchFirstAdd;

                //thread offset is set to the thread number for that address
                threadOffsets[i][threadCount] = k;

                //increment thread count for all threads in the batch
                threadCount++;
            }
        }

        //keep track of the thread numebr each batch starts at for use in linear arrays
        batchThreadOffset[i] = totalThreads;

        //keep total thread counts
        totalThreads += threadCount;


        //increment the first address in the batch for following batches
        batchFirstAdd += numAddPerBatch[i];
    }
    
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //device array to hold the number of threads in each address
    unsigned long long * d_numThreadsPerAddress;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned long long)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned long long )*nonEmptyBins, cudaMemcpyHostToDevice));

    // the device array to keep the values of the non-empty indexes in the final layer of the tree
    // unsigned int * d_addIndexes;
    // assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(unsigned int)*nonEmptyBins));
    // assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // //the number of adjacent non-empty indexes for each non-empty index
    // unsigned int * d_numValidRanges;
    // assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(unsigned int)*nonEmptyBins));
    // assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_linearRangeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_linearRangeID, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_linearRangeID, linearRangeID, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));


    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_LrangeIndexes; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_LrangeIndexes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_LrangeIndexes, linearRangeIndexes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_LrangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_LrangeSizes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_LrangeSizes, linearRangeSizes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    // unsigned int * d_numPointsInAdd;
    // assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));
    // assert(cudaSuccess ==  cudaMemcpy(d_numPointsInAdd, numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //copy over the array that tracks the values of the non-empty indexes in the last layer of the tree
    unsigned int * d_addIndexRange;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the ordered point array that corresponds with the point numbers and the dataset
    unsigned int * d_pointArray;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));

    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of threads for each batch
    unsigned long long * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    // unsigned int * d_numPoints;
    // assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    // assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // the offsets into addAssign and threadOffsets based on batch number
    unsigned int * d_batchThreadOffset;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchThreadOffset, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchThreadOffset, batchThreadOffset, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));


    // copy over the thread assignments for the current batch
    unsigned int * d_addAssign;
    assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_addAssign[batchThreadOffset[i]], addAssign[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    // copy over the offsets for each thread in the batch
    unsigned int * d_threadOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_threadOffsets[batchThreadOffset[i]], threadOffsets[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for sotring the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    for (unsigned int i = 0; i < numPoints; i++)
    {	
    struct neighborTable temp;
    tables[i] = temp;
    //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

    tables[i].cntNDataArrays = 1; 
    tables[i].vectindexmin.resize(numBatches+1);
    tables[i].vectindexmin[0] = i;
    tables[i].vectindexmax.resize(numBatches+1);
    tables[i].vectindexmax[0] = i;
    tables[i].vectdataPtr.resize(numBatches+1);
    tables[i].vectdataPtr[0] = pointArray;
    omp_init_lock(&tables[i].pointLock);
    }

    // vectors for tracking results in the CPU version of the Kernel
    std::vector<unsigned int> hostPointA;
    std::vector<unsigned int> hostPointB;

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
    }

    #if HOST
        unsigned int * numPointsInAdd = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins);
        assert(cudaSuccess == cudaMemcpy(numPointsInAdd, d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyDeviceToHost));
    #endif

    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic)
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int tid = omp_get_thread_num();
        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);

        printf("BatchNumber: %d/%d, Calcs: %llu, addresses: %d, threads: %llu, blocks:%d \n", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);

        //launch distance kernel
        #if HOST
            distanceCalculationsKernel_CPU(totalBlocks, 
                                    &numPoints,
                                    linearRangeID, 
                                    addAssign[i], 
                                    threadOffsets[i], 
                                    &epsilon2, 
                                    &dim, 
                                    &numThreadsPerBatch[i], 
                                    numThreadsPerAddress, 
                                    data, 
                                    addIndexes, 
                                    numValidRanges, 
                                    linearRangeIndexes, 
                                    linearRangeSizes, 
                                    numPointsInAdd, 
                                    addIndexRange, 
                                    pointArray, 
                                    &keyValueIndex[i],
                                    &hostPointA,
                                    &hostPointB);

        #else
            distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE, 0, stream[tid]>>>(d_numPoints,
                                                                                d_linearRangeID,
                                                                                &d_addAssign[batchThreadOffset[i]],
                                                                                &d_threadOffsets[batchThreadOffset[i]],
                                                                                d_epsilon2,
                                                                                d_dim,
                                                                                &d_numThreadsPerBatch[i],
                                                                                d_numThreadsPerAddress,
                                                                                d_data,
                                                                                d_numValidRanges,
                                                                                d_LrangeIndexes,
                                                                                d_LrangeSizes,
                                                                                d_numPointsInAdd,
                                                                                d_addIndexRange,
                                                                                &d_keyValueIndex[i],
                                                                                d_pointA[tid],
                                                                                d_pointB[tid]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                                                &d_keyValueIndex[i], 
                                                                d_uniqueKeys[tid], 
                                                                d_uniqueKeyPosition[tid], 
                                                                &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);


        #endif


    }

    #if HOST
        std::vector< std::pair<unsigned int,unsigned int>> pairs;

        for(unsigned int i = 0; i < hostPointA.size();i++){
            pairs.push_back(std::make_pair(hostPointA[i],hostPointB[i]));
        }

        std::sort(pairs.begin(), pairs.end(), compPair);

        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    #endif

    double time4 = omp_get_wtime();
    printf("Kernel time: %f\n", time4-time3);

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    #if HOST
        printf("Total results Set Size: %llu , unique pairs: %lu\n", totals, pairs.size());
    #else
        printf("Total results Set Size: %llu \n", totals);
    #endif


    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

    return tables;








}

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
    unsigned long long *calcPerAdd,// the number of calculations needed for each non-empty index
    unsigned int nonEmptyBins,//the number of nonempty indexes
    unsigned long long sumCalcs,// the total number of calculations that will need to be made
    unsigned long long sumAdds,//the total number of addresses that will be compared to by other addresses for distance calcs
    unsigned int * linearRangeID,// an array for keeping trackj of starting points in the linear arrays
    unsigned int * linearRangeIndexes,// a linear version of rangeIndexes
    unsigned int * linearRangeSizes)
    { // a linear version of rangeSizes


        cudaSetDevice(CUDA_DEVICE);

    // double launchstart = omp_get_wtime();
    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    unsigned long long threadsPerKernel = BLOCK_SIZE*KERNEL_BLOCKS;
    unsigned long long desiredNumBatches = 10;
    //set a value for the number of calculations made by each thread per kernel invocation
    unsigned long long calcsPerThread = sumCalcs/(desiredNumBatches*threadsPerKernel); 

    if (calcsPerThread > MAX_CALCS_PER_THREAD){
        calcsPerThread = MAX_CALCS_PER_THREAD;
    }

    if(calcsPerThread < MIN_CALCS_PER_THREAD){
        calcsPerThread = MIN_CALCS_PER_THREAD;
    }

    //the number of thrreads assigned to each non-empty address
    unsigned long long * numThreadsPerAddress = (unsigned long long *)malloc(sizeof(unsigned long long )*nonEmptyBins);

    //keeping track of the number of batches that will be needed
    unsigned int numBatches = 0;

    // the number of threads that will be avaliable for each kernel invocation
    unsigned long long threadsPerBatch = (unsigned long long)KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        // the number of threads for the address is the ceiling of the number of calcs for that address over calcs per thread
        numThreadsPerAddress[i] = calcPerAdd[i] / calcsPerThread;
        if(calcPerAdd[i] % calcsPerThread != 0) numThreadsPerAddress[i]++;

        if(calcPerAdd[i] == 0) printf("ERROR: add %d has 0 calcs\n", i);

        // check if the number of threads is higher than the number of threads for a batch
        if (sum + numThreadsPerAddress[i] < threadsPerBatch){
            // if the number of threads is less than keep adding addresses to the batch
            sum += numThreadsPerAddress[i];
        }else{
            // if we would exceed the number of threads for that batch, then dont add
            sum = numThreadsPerAddress[i];

            // check for an error
            // if(numThreadsPerAddress[i] > threadsPerBatch) printf("Warning: Address %d is too big. Needs: %d threads which is more than %d\n", i, numThreadsPerAddress[i], threadsPerBatch);

            //increment the number of batches needed
            numBatches++;
        }
    }

    //always need at least one batch
    numBatches++;

    printf("Total Number of Batches: %d , with calcs per threads: %llu\n", numBatches, calcsPerThread);

    //keeping track of the number of calculations for each batch
    unsigned long long * numCalcsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    //keeping track of the number of addresses that batch will compute
    unsigned int * numAddPerBatch = (unsigned int*)calloc(numBatches, sizeof(unsigned int));

    //keeping track of the number of threads that are in each batch
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    // setting starting batch for loop
    unsigned int currentBatch = 0;

    // go through each non-empty index
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        //error check
        if(currentBatch > numBatches) printf("ERROR 3: current batch %d is greater than num batches %d\n", currentBatch, numBatches);

        //check if the batch is new or if the number of threads per batch will exceed the max if added
        if(numThreadsPerBatch[currentBatch] == 0 || numThreadsPerBatch[currentBatch] + numThreadsPerAddress[i] < threadsPerBatch){
            //add the number of threads for index i to the number of threads for the batch
            numThreadsPerBatch[currentBatch] += numThreadsPerAddress[i];

            //increment the number of addresses in the current batch
            numAddPerBatch[currentBatch]++;

            // add the number of calculations for the address to the number of calcs for the batch
            numCalcsPerBatch[currentBatch] += calcPerAdd[i];
        } else { //if the number of threads for the batch will be too many, then need to add to the next batch instead
            currentBatch++;
            i = i - 1;
        }
    }


    // array to track which thread is assigned to witch address
    unsigned int ** addAssign = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    //the offset of the thread for calculations inside an address
    unsigned int ** threadOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    for(int i = 0; i < numBatches; i++){
        // array to track which thread is assigned to witch address
        addAssign[i] = (unsigned int * )malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);

        //the offset of the thread for calculations inside an address
        threadOffsets[i] = (unsigned int*)malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);
    }

    //setting the intital batch starting address
    unsigned int batchFirstAdd = 0;

    //keep track of the total numebr of threads
    unsigned long long totalThreads = 0;

    // array to keep track of where linear arrays start for threads based on the batch number
    unsigned int * batchThreadOffset = (unsigned int *)malloc(sizeof(unsigned int)*numBatches);

    // calculating the thread assignements
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int threadCount = 0;

        //compute which thread does wich add
        for(unsigned int j = 0; j < numAddPerBatch[i]; j++){

            //basic error check
            if(numThreadsPerAddress[batchFirstAdd + j] == 0) {
                printf("ERROR: add %d has 0 threads\n", j + batchFirstAdd);
            }

            //for each address in the batch, assigne threads to it
            for(unsigned int k = 0; k < numThreadsPerAddress[batchFirstAdd + j]; k++){

                //assign the thread to the current address
                addAssign[i][threadCount] = j + batchFirstAdd;

                //thread offset is set to the thread number for that address
                threadOffsets[i][threadCount] = k;

                //increment thread count for all threads in the batch
                threadCount++;
            }
        }

        //keep track of the thread numebr each batch starts at for use in linear arrays
        batchThreadOffset[i] = totalThreads;

        //keep total thread counts
        totalThreads += threadCount;


        //increment the first address in the batch for following batches
        batchFirstAdd += numAddPerBatch[i];
    }

    // double launchend = omp_get_wtime();

    // printf("Launch setup time: %f\n", launchend - launchstart);
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //device array to hold the number of threads in each address
    unsigned long long * d_numThreadsPerAddress;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned long long)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned long long )*nonEmptyBins, cudaMemcpyHostToDevice));

    // the device array to keep the values of the non-empty indexes in the final layer of the tree
    unsigned int * d_addIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numValidRanges;
    assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_linearRangeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_linearRangeID, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_linearRangeID, linearRangeID, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));


    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_rangeIndexes; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeIndexes, linearRangeIndexes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_rangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeSizes, linearRangeSizes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_numPointsInAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numPointsInAdd, numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //copy over the array that tracks the values of the non-empty indexes in the last layer of the tree
    unsigned int * d_addIndexRange;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the ordered point array that corresponds with the point numbers and the dataset
    unsigned int * d_pointArray;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));

    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of threads for each batch
    unsigned long long * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // the offsets into addAssign and threadOffsets based on batch number
    unsigned int * d_batchThreadOffset;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchThreadOffset, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchThreadOffset, batchThreadOffset, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));


    // copy over the thread assignments for the current batch
    unsigned int * d_addAssign;
    assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_addAssign[batchThreadOffset[i]], addAssign[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    // copy over the offsets for each thread in the batch
    unsigned int * d_threadOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_threadOffsets[batchThreadOffset[i]], threadOffsets[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for sotring the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    for (unsigned int i = 0; i < numPoints; i++)
    {	
    struct neighborTable temp;
    tables[i] = temp;
    //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

    tables[i].cntNDataArrays = 1; 
    tables[i].vectindexmin.resize(numBatches+1);
    tables[i].vectindexmin[0] = i;
    tables[i].vectindexmax.resize(numBatches+1);
    tables[i].vectindexmax[0] = i;
    tables[i].vectdataPtr.resize(numBatches+1);
    tables[i].vectdataPtr[0] = pointArray;
    omp_init_lock(&tables[i].pointLock);
    }

    // vectors for tracking results in the CPU version of the Kernel
    std::vector<unsigned int> hostPointA;
    std::vector<unsigned int> hostPointB;

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }

    // printf("Time to transfer: %f\n", omp_get_wtime()-launchend);

    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic)
    for(unsigned int i = 0; i < numBatches; i++){

        cudaSetDevice(CUDA_DEVICE);

        unsigned int tid = omp_get_thread_num();
        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);

        // printf("BatchNumber: %d/%d, Calcs: %llu, addresses: %d, threads: %u, blocks:%d \n", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);

        // double kernelStartTime = omp_get_wtime();

        //launch distance kernel
        #if HOST
            distanceCalculationsKernel_CPU(totalBlocks, 
                                    &numPoints,
                                    linearRangeID, 
                                    addAssign[i], 
                                    threadOffsets[i], 
                                    &epsilon2, 
                                    &dim, 
                                    &numThreadsPerBatch[i], 
                                    numThreadsPerAddress, 
                                    data, 
                                    addIndexes, 
                                    numValidRanges, 
                                    linearRangeIndexes, 
                                    linearRangeSizes, 
                                    numPointsInAdd, 
                                    addIndexRange, 
                                    pointArray, 
                                    &keyValueIndex[i],
                                    &hostPointA,
                                    &hostPointB);

        #else
            distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE, 0, stream[tid]>>>(d_numPoints,
                                                                                d_linearRangeID,
                                                                                &d_addAssign[batchThreadOffset[i]],
                                                                                &d_threadOffsets[batchThreadOffset[i]],
                                                                                d_epsilon2,
                                                                                d_dim,
                                                                                &d_numThreadsPerBatch[i],
                                                                                d_numThreadsPerAddress,
                                                                                d_data,
                                                                                d_numValidRanges,
                                                                                d_rangeIndexes,
                                                                                d_rangeSizes,
                                                                                d_numPointsInAdd,
                                                                                d_addIndexRange,
                                                                                &d_keyValueIndex[i],
                                                                                d_pointA[tid],
                                                                                d_pointB[tid]);

            cudaStreamSynchronize(stream[tid]);

            // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            // printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                                                &d_keyValueIndex[i], 
                                                                d_uniqueKeys[tid], 
                                                                d_uniqueKeyPosition[tid], 
                                                                &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);


        #endif


    }

    #if HOST
        std::vector< std::pair<unsigned int,unsigned int>> pairs;

        for(unsigned int i = 0; i < hostPointA.size();i++){
            pairs.push_back(std::make_pair(hostPointA[i],hostPointB[i]));
        }

        std::sort(pairs.begin(), pairs.end(), compPair);

        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    #endif



    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    #if HOST
        printf("Total results Set Size: %llu , unique pairs: %lu\n", totals, pairs.size());
    #else
        printf("Total results Set Size: %llu \n", totals);
    #endif

    // for(unsigned int i = 0; i < NUMSTREAMS; i++){
    //     printf("Total time in kernel or stream %d: %f\n", i, totalKernelTime[i]);
    // }
    


    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

    return tables;
}


struct neighborTable * nodeLauncher(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon){


    cudaSetDevice(CUDA_DEVICE);
    
    double time1 = omp_get_wtime();
    std::vector<struct Node> nodes;

    // build the data structure
    unsigned int numNodes = buildNodeNet(data,
            dim,
            numPoints,
            numRP,
            pointArray,
            epsilon,
            &nodes);


    cudaSetDevice(CUDA_DEVICE);

    double time2 = omp_get_wtime();
    printf("Node Construct time: %f\n", time2 - time1);
    fprintf(stderr, "%f ", time2-time1);

    // unsigned long long res = nodeForce(&nodes, epsilon, data, dim, numPoints);
    // printf("Res: %llu\n", res);

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

    double * normData = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
        for(unsigned int i = 0; i < numPoints; i++){
            for(unsigned int j = 0; j < dim; j++){
            #if DATANORM
                normData[i+numPoints*j] = data[pointArray[i]*dim+j];
            #else
                normData[i*dim+j] = data[pointArray[i]*dim+j];
            #endif
        }
    }

    // printf("P1: %d, P2: %d\n", pointArray[0], nodes[0].nodePoints[0]);
    //build array of point offsets
    unsigned int * pointOffsets = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    //build array of number of calcs needed
    unsigned long long * numCalcs = (unsigned long long *)malloc(sizeof(unsigned long long)*numNodes);
    //build array of number of neighbors
    unsigned int * numNeighbors = (unsigned int*)malloc(sizeof(unsigned int)*numNodes);
    //array to count total number of neighbors for linear id
    unsigned int * neighborOffset = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    // number of points in each node
    unsigned int * nodePoints = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);

    //counter for neighbor offsets
    unsigned int neighborOffsetCount = 0;
    // std::vector<unsigned int> tempNeighbors;

    for(unsigned int i = 0; i < numNodes; i++){
        pointOffsets[i] = nodes[i].pointOffset;
        numCalcs[i] = nodes[i].numCalcs;
        neighborOffset[i] = neighborOffsetCount;
        numNeighbors[i] = nodes[i].neighborIndex.size();
        neighborOffsetCount += nodes[i].neighborIndex.size();
        nodePoints[i] = nodes[i].numNodePoints;
        // tempNeighbors.insert(tempNeighbors.end(), nodes[i].neighborIndex.begin(),nodes[i].neighborIndex.end());
    
    }

    // printf("po:%u\n", pointOffsets[10]);
    unsigned int * neighbors = (unsigned int *)malloc(sizeof(unsigned int)*neighborOffsetCount);
    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < numNeighbors[i]; j++){
        neighbors[counter+j] = nodes[i].neighborIndex[j];   
        }
        counter += numNeighbors[i];
    }

    // printf("total num neighbors: %u\n", counter);

    unsigned long long sumCalcs = totalNodeCalcs(nodes, numNodes);
    // printf("sum calcs: %llu\n", sumCalcs);

    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    unsigned long long threadsPerKernel = BLOCK_SIZE*KERNEL_BLOCKS;
    unsigned long long desiredNumBatches = 10;
    //set a value for the number of calculations made by each thread per kernel invocation
    unsigned long long calcsPerThread = sumCalcs/(desiredNumBatches*threadsPerKernel); 

    if (calcsPerThread > MAX_CALCS_PER_THREAD){
        calcsPerThread = MAX_CALCS_PER_THREAD;
    }

    if(calcsPerThread < MIN_CALCS_PER_THREAD){
        calcsPerThread = MIN_CALCS_PER_THREAD;
    }

    //the number of thrreads assigned to each non-empty address
    unsigned long long * numThreadsPerNode = (unsigned long long *)malloc(sizeof(unsigned long long )*numNodes);

    //keeping track of the number of batches that will be needed
    unsigned int numBatches = 0;

    // the number of threads that will be avaliable for each kernel invocation
    unsigned long long threadsPerBatch = (unsigned long long)KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(unsigned int i = 0; i < numNodes; i++){

        // the number of threads for the address is the ceiling of the number of calcs for that address over calcs per thread
        numThreadsPerNode[i] = numCalcs[i] / calcsPerThread;
        if(numCalcs[i] % calcsPerThread != 0) numThreadsPerNode[i]++;

        if(numCalcs[i] == 0) printf("ERROR: add %d has 0 calcs\n", i);

        // check if the number of threads is higher than the number of threads for a batch
        if (sum + numThreadsPerNode[i] < threadsPerBatch){
            // if the number of threads is less than keep adding addresses to the batch
            sum += numThreadsPerNode[i];
        }else{
            // if we would exceed the number of threads for that batch, then dont add
            sum = numThreadsPerNode[i];

            // check for an error
            // if(numThreadsPerAddress[i] > threadsPerBatch) printf("Warning: Address %d is too big. Needs: %d threads which is more than %d\n", i, numThreadsPerAddress[i], threadsPerBatch);

            //increment the number of batches needed
            numBatches++;
        }
    }

    //always need at least one batch
    numBatches++;

    printf("Total Number of Batches: %d , with calcs per threads: %llu\n", numBatches, calcsPerThread);

    //keeping track of the number of calculations for each batch
    unsigned long long * numCalcsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    //keeping track of the number of addresses that batch will compute
    unsigned int * numNodePerBatch = (unsigned int*)calloc(numBatches, sizeof(unsigned int));

    //keeping track of the number of threads that are in each batch
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    // setting starting batch for loop
    unsigned int currentBatch = 0;

    unsigned long long maxThreadsPerBatch = 0;

    // go through each non-empty index
    for(unsigned int i = 0; i < numNodes; i++){

        //error check
        if(currentBatch > numBatches) printf("ERROR 3: current batch %d is greater than num batches %d\n", currentBatch, numBatches);

        //check if the batch is new or if the number of threads per batch will exceed the max if added
        if(numThreadsPerBatch[currentBatch] == 0 || numThreadsPerBatch[currentBatch] + numThreadsPerNode[i] < threadsPerBatch){
            //add the number of threads for index i to the number of threads for the batch
            numThreadsPerBatch[currentBatch] += numThreadsPerNode[i];

            //increment the number of addresses in the current batch
            numNodePerBatch[currentBatch]++;

            // add the number of calculations for the address to the number of calcs for the batch
            numCalcsPerBatch[currentBatch] += numCalcs[i];
        } else { //if the number of threads for the batch will be too many, then need to add to the next batch instead
            currentBatch++;
            i = i - 1;
        }

        if(numThreadsPerBatch[currentBatch] > maxThreadsPerBatch) maxThreadsPerBatch = numThreadsPerBatch[currentBatch];
    }

    // array to track which thread is assigned to which address
    unsigned int ** nodeAssign = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    //the offset of the thread for calculations inside an address
    unsigned int ** threadOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    for(int i = 0; i < numBatches; i++){
        // array to track which thread is assigned to witch address
        nodeAssign[i] = (unsigned int * )malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);

        //the offset of the thread for calculations inside an address
        threadOffsets[i] = (unsigned int*)malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);
    }

    //setting the intital batch starting address
    unsigned int batchFirstNode = 0;

    //keep track of the total numebr of threads
    unsigned long long totalThreads = 0;

    // array to keep track of where linear arrays start for threads based on the batch number
    unsigned int * batchThreadOffset = (unsigned int *)malloc(sizeof(unsigned int)*numBatches);

    // calculating the thread assignements
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int threadCount = 0;

        //compute which thread does wich add
        for(unsigned int j = 0; j < numNodePerBatch[i]; j++){

            //basic error check
            if(numThreadsPerNode[batchFirstNode + j] == 0) {
                printf("ERROR: add %d has 0 threads\n", j + batchFirstNode);
            }

                //for each address in the batch, assigne threads to it
                for(unsigned int k = 0; k < numThreadsPerNode[batchFirstNode + j]; k++){

                //assign the thread to the current address
                nodeAssign[i][threadCount] = j + batchFirstNode;

                //thread offset is set to the thread number for that address
                threadOffsets[i][threadCount] = k;

                //increment thread count for all threads in the batch
                threadCount++;
            }
        }

        //keep track of the thread numebr each batch starts at for use in linear arrays
        batchThreadOffset[i] = totalThreads;

        //keep total thread counts
        totalThreads += threadCount;


        //increment the first address in the batch for following batches
        batchFirstNode += numNodePerBatch[i];
    }


    // printf("Launch setup time: %f\n", launchend - launchstart);
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, normData, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //device array to hold the number of threads in each address
    unsigned long long * d_numThreadsPerNode;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerNode, sizeof(unsigned long long)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerNode, numThreadsPerNode, sizeof(unsigned long long )*numNodes, cudaMemcpyHostToDevice));

    // // the device array to keep the values of the non-empty indexes in the final layer of the tree
    // unsigned int * d_nodeIndexes;
    // assert(cudaSuccess == cudaMalloc((void**)&d_nodeIndexes, sizeof(unsigned int)*numNodes));
    // assert(cudaSuccess ==  cudaMemcpy(d_nodeIndexes, nodeIndexes, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numNeighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_numNeighbors, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numNeighbors, numNeighbors, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_pointOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOffsets, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOffsets, pointOffsets, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));


    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_neighborOffset; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_neighborOffset, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_neighborOffset, neighborOffset, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_nodePoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodePoints, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_nodePoints, nodePoints, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_neighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_neighbors, sizeof(unsigned int)*neighborOffsetCount));
    assert(cudaSuccess ==  cudaMemcpy(d_neighbors, neighbors, sizeof(unsigned int)*neighborOffsetCount, cudaMemcpyHostToDevice));

    //copy over the array that tracks the values of the non-empty indexes in the last layer of the tree
    // unsigned int * d_addIndexRange;
    // assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(unsigned int)*numNodes));
    // assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the ordered point array that corresponds with the point numbers and the dataset
    // unsigned int * d_pointArray;
    // assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(unsigned int)*numPoints));
    // assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));

    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of threads for each batch
    unsigned long long * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // the offsets into addAssign and threadOffsets based on batch number
    unsigned int * d_batchThreadOffset;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchThreadOffset, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchThreadOffset, batchThreadOffset, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));


    // copy over the thread assignments for the current batch
    unsigned int * d_nodeAssign;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodeAssign, sizeof(unsigned int)*maxThreadsPerBatch*NUMSTREAMS));
    // for(unsigned int i = 0; i < numBatches; i++){
    //     assert(cudaSuccess ==  cudaMemcpy(&d_nodeAssign[batchThreadOffset[i]], nodeAssign[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    // }

    // copy over the offsets for each thread in the batch
    unsigned int * d_threadOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(unsigned int)*maxThreadsPerBatch*NUMSTREAMS));
    // for(unsigned int i = 0; i < numBatches; i++){
    //     assert(cudaSuccess ==  cudaMemcpy(&d_threadOffsets[batchThreadOffset[i]], threadOffsets[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    // }

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for storing the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    
    #if !HOST
    for (unsigned int i = 0; i < numPoints; i++){	
        struct neighborTable temp;
        tables[i] = temp;
        //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
    #endif

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }

    // printf("Time to transfer: %f\n", omp_get_wtime()-launchend);

    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic) if(!HOST)
    for(unsigned int i = 0; i < numBatches; i++){

        #if HOST

            unsigned int tid = omp_get_thread_num();
            unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);

            nodeCalculationsKernel_CPU( numNodes,
                                        totalBlocks,
                                        &numPoints,
                                        pointOffsets,
                                        nodeAssign[batchThreadOffset[i]],
                                        threadOffsets[batchThreadOffset[i]],
                                        &epsilon2,
                                        &dim,
                                        &numThreadsPerBatch[i],
                                        numThreadsPerNode,
                                        data,
                                        numNeighbors,
                                        nodePoints,
                                        neighbors,
                                        neighborOffset,
                                        &keyValueIndex[i]);
            printf("finsihed batch #%d\n", i);
        #else
            cudaSetDevice(CUDA_DEVICE);

            unsigned int tid = omp_get_thread_num();
            unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);
           
            assert(cudaSuccess ==  cudaMemcpyAsync(&d_nodeAssign[tid*maxThreadsPerBatch], nodeAssign[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice, stream[tid]));

            assert(cudaSuccess ==  cudaMemcpyAsync(&d_threadOffsets[tid*maxThreadsPerBatch], threadOffsets[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice, stream[tid]));


            // printf("BatchNumber: %d/%d, Calcs: %llu, addresses: %d, threads: %u, blocks:%d \n", i+1, numBatches, numCalcsPerBatch[i], numNodePerBatch[i], numThreadsPerBatch[i], totalBlocks);

            // double kernelStartTime = omp_get_wtime();

            //launch distance kernel
            nodeCalculationsKernel<<<totalBlocks, BLOCK_SIZE, 0, stream[tid]>>>(d_numPoints,
                                                        d_pointOffsets,
                                                        &d_nodeAssign[tid*maxThreadsPerBatch],
                                                        &d_threadOffsets[tid*maxThreadsPerBatch],
                                                        d_epsilon2,
                                                        d_dim,
                                                        &d_numThreadsPerBatch[i],
                                                        d_numThreadsPerNode,
                                                        d_data,
                                                        d_numNeighbors,
                                                        d_nodePoints,
                                                        d_neighbors,
                                                        d_neighborOffset,
                                                        &d_keyValueIndex[i],
                                                        d_pointA[tid],
                                                        d_pointB[tid]);



            cudaStreamSynchronize(stream[tid]);

            // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            // printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                        &d_keyValueIndex[i], 
                                        d_uniqueKeys[tid], 
                                        d_uniqueKeyPosition[tid], 
                                        &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);

        #endif

    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    printf("Total results Set Size: %llu \n", totals);

    free(numCalcsPerBatch);
    free(numNodePerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerNode);

    return tables;
}



struct neighborTable * nodeLauncher2(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon){


    cudaSetDevice(CUDA_DEVICE);
    
    double time1 = omp_get_wtime();
    std::vector<struct Node> nodes;

    // build the data structure
    unsigned int numNodes = buildNodeNet(data,
            dim,
            numPoints,
            numRP,
            pointArray,
            epsilon,
            &nodes);


    cudaSetDevice(CUDA_DEVICE);

    double time2 = omp_get_wtime();
    printf("Node Construct time: %f\n", time2 - time1);
    fprintf(stderr, "%f ", time2-time1);

    // unsigned long long res = nodeForce(&nodes, epsilon, data, dim, numPoints);
    // printf("Res: %llu\n", res);

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

    double * normData = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
        for(unsigned int i = 0; i < numPoints; i++){
            for(unsigned int j = 0; j < dim; j++){
            #if DATANORM
                normData[i+numPoints*j] = data[pointArray[i]*dim+j];
            #else
                normData[i*dim+j] = data[pointArray[i]*dim+j];
            #endif
        }
    }

    // printf("P1: %d, P2: %d\n", pointArray[0], nodes[0].nodePoints[0]);
    //build array of point offsets
    unsigned int * pointOffsets = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    //build array of number of calcs needed
    unsigned long long * numCalcs = (unsigned long long *)malloc(sizeof(unsigned long long)*numNodes);
    //build array of number of neighbors
    unsigned int * numNeighbors = (unsigned int*)malloc(sizeof(unsigned int)*numNodes);
    //array to count total number of neighbors for linear id
    unsigned int * neighborOffset = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    // number of points in each node
    unsigned int * nodePoints = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);

    unsigned int * nodeID = (unsigned int *)malloc(sizeof(unsigned int)*numPoints); 


    //counter for neighbor offsets
    unsigned int neighborOffsetCount = 0;
    // std::vector<unsigned int> tempNeighbors;

    unsigned int previous = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        pointOffsets[i] = nodes[i].pointOffset;
        numCalcs[i] = nodes[i].numCalcs;
        neighborOffset[i] = neighborOffsetCount;
        numNeighbors[i] = nodes[i].neighborIndex.size();
        neighborOffsetCount += nodes[i].neighborIndex.size();
        nodePoints[i] = nodes[i].numNodePoints;
        // tempNeighbors.insert(tempNeighbors.end(), nodes[i].neighborIndex.begin(),nodes[i].neighborIndex.end());
        for(unsigned int j = previous; j < previous+nodePoints[i]; j++){
            nodeID[j] = i;
        }
        previous += nodePoints[i]; // maybe-1 here
    }


    // printf("po:%u\n", pointOffsets[10]);
    unsigned int * neighbors = (unsigned int *)malloc(sizeof(unsigned int)*neighborOffsetCount);

    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < numNeighbors[i]; j++){
        neighbors[counter+j] = nodes[i].neighborIndex[j];   
        }
        counter += numNeighbors[i];
    }

    // printf("total num neighbors: %u\n", counter);

    unsigned long long sumCalcs = totalNodeCalcs(nodes, numNodes);
    // printf("sum calcs: %llu\n", sumCalcs);

    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

 
    unsigned int numBatches = ceil(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE))*TPP;
    // unsigned int leftOverBatch = floor(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE / TPP));
    unsigned int * batchPoints = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    unsigned int batchOffset = 0;
    for(unsigned int i = 0; i < numBatches; i++){
        batchPoints[i] = batchOffset;
        batchOffset += KERNEL_BLOCKS*BLOCK_SIZE;
    }



    // printf("Launch setup time: %f\n", launchend - launchstart);
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, normData, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numNeighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_numNeighbors, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numNeighbors, numNeighbors, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_batchPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchPoints, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchPoints, batchPoints, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));
    

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_nodeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodeID, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_nodeID, nodeID, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));
    
    
    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_pointOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOffsets, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOffsets, pointOffsets, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_neighborOffset; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_neighborOffset, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_neighborOffset, neighborOffset, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_nodePoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodePoints, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_nodePoints, nodePoints, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_neighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_neighbors, sizeof(unsigned int)*neighborOffsetCount));
    assert(cudaSuccess ==  cudaMemcpy(d_neighbors, neighbors, sizeof(unsigned int)*neighborOffsetCount, cudaMemcpyHostToDevice));


    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));


    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for storing the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    
    #if !HOST
    for (unsigned int i = 0; i < numPoints; i++){	
        struct neighborTable temp;
        tables[i] = temp;
        //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
    #endif

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }

    // printf("Time to transfer: %f\n", omp_get_wtime()-launchend);
    printf("Batchs: %d\n",numBatches);
    const unsigned int cdim = dim;
    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic) if(!HOST)
    for(unsigned int i = 0; i < numBatches; i++){
            cudaSetDevice(CUDA_DEVICE);

            unsigned int tid = omp_get_thread_num();

            // printf("BatchNumber: %d/%d\n", i+1, numBatches);

            // double kernelStartTime = omp_get_wtime();

            //launch distance kernel
            nodeByPoint<<<KERNEL_BLOCKS*TPP, BLOCK_SIZE, 0, stream[tid]>>>( cdim, 
                                                                        d_data,
                                                                        d_epsilon2,
                                                                        d_numPoints,
                                                                        &d_batchPoints[i],
                                                                        d_nodeID,
                                                                        d_numNeighbors,
                                                                        d_nodePoints,
                                                                        d_neighbors,
                                                                        d_neighborOffset,
                                                                        d_pointOffsets,
                                                                        d_pointA[tid],
                                                                        d_pointB[tid],
                                                                        &d_keyValueIndex[i]);



            cudaStreamSynchronize(stream[tid]);

            // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            // printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                        &d_keyValueIndex[i], 
                                        d_uniqueKeys[tid], 
                                        d_uniqueKeyPosition[tid], 
                                        &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);

    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    printf("Total results Set Size: %llu \n", totals);

    return tables;
}


struct neighborTable * nodeLauncher3(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon){


    cudaSetDevice(CUDA_DEVICE);
    
    double time1 = omp_get_wtime();
    std::vector<struct Node> nodes;

    // build the data structure
    unsigned int numNodes = buildNodeNet(data,
            dim,
            numPoints,
            numRP,
            pointArray,
            epsilon,
            &nodes);


    cudaSetDevice(CUDA_DEVICE);

    double time2 = omp_get_wtime();
    printf("Node Construct time: %f\n", time2 - time1);
    fprintf(stderr, "%f ", time2-time1);

    // unsigned long long res = nodeForce(&nodes, epsilon, data, dim, numPoints);
    // printf("Res: %llu\n", res);

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

    double * normData = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
        for(unsigned int i = 0; i < numPoints; i++){
            for(unsigned int j = 0; j < dim; j++){
            #if DATANORM
                normData[i+numPoints*j] = data[pointArray[i]*dim+j];
            #else
                normData[i*dim+j] = data[pointArray[i]*dim+j];
            #endif
        }
    }

    // printf("P1: %d, P2: %d\n", pointArray[0], nodes[0].nodePoints[0]);
    //build array of point offsets
    unsigned int * pointOffsets = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    //build array of number of calcs needed
    unsigned long long * numCalcs = (unsigned long long *)malloc(sizeof(unsigned long long)*numNodes);
    //build array of number of neighbors
    unsigned int * numNeighbors = (unsigned int*)malloc(sizeof(unsigned int)*numNodes);
    //array to count total number of neighbors for linear id
    unsigned int * neighborOffset = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    // number of points in each node
    unsigned int * nodePoints = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);

    unsigned int * nodeID = (unsigned int *)malloc(sizeof(unsigned int)*numPoints); 


    //counter for neighbor offsets
    unsigned int neighborOffsetCount = 0;
    // std::vector<unsigned int> tempNeighbors;

    unsigned int previous = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        pointOffsets[i] = nodes[i].pointOffset;
        numCalcs[i] = nodes[i].numCalcs;
        neighborOffset[i] = neighborOffsetCount;
        numNeighbors[i] = nodes[i].neighborIndex.size();
        neighborOffsetCount += nodes[i].neighborIndex.size();
        nodePoints[i] = nodes[i].numNodePoints;
        // tempNeighbors.insert(tempNeighbors.end(), nodes[i].neighborIndex.begin(),nodes[i].neighborIndex.end());
        for(unsigned int j = previous; j < previous+nodePoints[i]; j++){
            nodeID[j] = i;
        }
        previous += nodePoints[i]; // maybe-1 here
    }


    // printf("po:%u\n", pointOffsets[10]);
    unsigned int * neighbors = (unsigned int *)malloc(sizeof(unsigned int)*neighborOffsetCount);

    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < numNeighbors[i]; j++){
        neighbors[counter+j] = nodes[i].neighborIndex[j];   
        }
        counter += numNeighbors[i];
    }

    // printf("total num neighbors: %u\n", counter);

    unsigned long long sumCalcs = totalNodeCalcs(nodes, numNodes);
    // printf("sum calcs: %llu\n", sumCalcs);

    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

 
    unsigned int numBatches = ceil(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE))*TPP;
    // unsigned int leftOverBatch = floor(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE / TPP));
    unsigned int * batchPoints = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    unsigned int batchOffset = 0;
    for(unsigned int i = 0; i < numBatches; i++){
        batchPoints[i] = batchOffset;
        batchOffset += KERNEL_BLOCKS*BLOCK_SIZE;
    }

    // build the point ordering
    unsigned int * pointOrder = (unsigned int * )malloc(sizeof(unsigned int)*numPoints);

    //take first 32 points from each node, repeat
    unsigned int pRound = 0;
    unsigned int currentNode = 0;
    unsigned int skip = 0;
    for(unsigned int i = 0; i < numPoints; i++){
        currentNode = ((i+skip) / ORDP) % numNodes;
        pRound = ((i+skip) / ORDP) / numNodes;
        if(nodePoints[currentNode] <= pRound*ORDP + (i+skip) % ORDP){
            i--;
            skip++; 
            continue;
        }
        
        pointOrder[i] = pointOffsets[currentNode] + pRound*ORDP + (i+skip) % ORDP;
        
    }



    unsigned int * d_pointOrder;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOrder, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOrder, pointOrder, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));


    // printf("Launch setup time: %f\n", launchend - launchstart);
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, normData, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numNeighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_numNeighbors, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numNeighbors, numNeighbors, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_batchPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchPoints, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchPoints, batchPoints, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));
    

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_nodeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodeID, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_nodeID, nodeID, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));
    
    
    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_pointOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOffsets, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOffsets, pointOffsets, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_neighborOffset; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_neighborOffset, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_neighborOffset, neighborOffset, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_nodePoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodePoints, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_nodePoints, nodePoints, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_neighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_neighbors, sizeof(unsigned int)*neighborOffsetCount));
    assert(cudaSuccess ==  cudaMemcpy(d_neighbors, neighbors, sizeof(unsigned int)*neighborOffsetCount, cudaMemcpyHostToDevice));


    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));


    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for storing the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    
    #if !HOST
    for (unsigned int i = 0; i < numPoints; i++){	
        struct neighborTable temp;
        tables[i] = temp;
        //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
    #endif

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }

    // printf("Time to transfer: %f\n", omp_get_wtime()-launchend);
    printf("Batchs: %d\n",numBatches);
    const unsigned int cdim = dim;
    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic) if(!HOST)
    for(unsigned int i = 0; i < numBatches; i++){
            cudaSetDevice(CUDA_DEVICE);

            unsigned int tid = omp_get_thread_num();

            // printf("BatchNumber: %d/%d\n", i+1, numBatches);

            // double kernelStartTime = omp_get_wtime();

            //launch distance kernel
            nodeByPoint2<<<KERNEL_BLOCKS*TPP, BLOCK_SIZE, 0, stream[tid]>>>( cdim, 
                                                                        d_data,
                                                                        d_epsilon2,
                                                                        d_numPoints,
                                                                        &d_batchPoints[i],
                                                                        d_nodeID,
                                                                        d_numNeighbors,
                                                                        d_nodePoints,
                                                                        d_neighbors,
                                                                        d_neighborOffset,
                                                                        d_pointOrder,
                                                                        d_pointOffsets,
                                                                        d_pointA[tid],
                                                                        d_pointB[tid],
                                                                        &d_keyValueIndex[i]);



            cudaStreamSynchronize(stream[tid]);

            // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            // printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                        &d_keyValueIndex[i], 
                                        d_uniqueKeys[tid], 
                                        d_uniqueKeyPosition[tid], 
                                        &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);

    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    printf("Total results Set Size: %llu \n", totals);

    return tables;
}



struct neighborTable * nodeLauncher4(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon){


    cudaSetDevice(CUDA_DEVICE);
    
    double time1 = omp_get_wtime();
    std::vector<struct Node> nodes;

    // build the data structure
    unsigned int numNodes = buildNodeNet(data,
            dim,
            numPoints,
            numRP,
            pointArray,
            epsilon,
            &nodes);


    cudaSetDevice(CUDA_DEVICE);

    double time2 = omp_get_wtime();
    printf("Node Construct time: %f\n", time2 - time1);
    fprintf(stderr, "%f ", time2-time1);

    // unsigned long long res = nodeForce(&nodes, epsilon, data, dim, numPoints);
    // printf("Res: %llu\n", res);

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

    double * normData = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
        for(unsigned int i = 0; i < numPoints; i++){
            for(unsigned int j = 0; j < dim; j++){
            #if DATANORM
                normData[i+numPoints*j] = data[pointArray[i]*dim+j];
            #else
                normData[i*dim+j] = data[pointArray[i]*dim+j];
            #endif
        }
    }

    // printf("P1: %d, P2: %d\n", pointArray[0], nodes[0].nodePoints[0]);
    //build array of point offsets
    unsigned int * pointOffsets = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    //build array of number of calcs needed
    unsigned long long * numCalcs = (unsigned long long *)malloc(sizeof(unsigned long long)*numNodes);
    //build array of number of neighbors
    unsigned int * numNeighbors = (unsigned int*)malloc(sizeof(unsigned int)*numNodes);
    //array to count total number of neighbors for linear id
    unsigned int * neighborOffset = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    // number of points in each node
    unsigned int * nodePoints = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);

    unsigned int * nodeID = (unsigned int *)malloc(sizeof(unsigned int)*numPoints); 


    //counter for neighbor offsets
    unsigned int neighborOffsetCount = 0;
    // std::vector<unsigned int> tempNeighbors;

    unsigned int previous = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        pointOffsets[i] = nodes[i].pointOffset;
        numCalcs[i] = nodes[i].numCalcs;
        neighborOffset[i] = neighborOffsetCount;
        numNeighbors[i] = nodes[i].neighborIndex.size();
        neighborOffsetCount += nodes[i].neighborIndex.size();
        nodePoints[i] = nodes[i].numNodePoints;
        // tempNeighbors.insert(tempNeighbors.end(), nodes[i].neighborIndex.begin(),nodes[i].neighborIndex.end());
        for(unsigned int j = previous; j < previous+nodePoints[i]; j++){
            nodeID[j] = i;
        }
        previous += nodePoints[i]; // maybe-1 here
    }


    // printf("po:%u\n", pointOffsets[10]);
    unsigned int * neighbors = (unsigned int *)malloc(sizeof(unsigned int)*neighborOffsetCount);

    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < numNeighbors[i]; j++){
        neighbors[counter+j] = nodes[i].neighborIndex[j];   
        }
        counter += numNeighbors[i];
    }

    // printf("total num neighbors: %u\n", counter);

    unsigned long long sumCalcs = totalNodeCalcs(nodes, numNodes);
    // printf("sum calcs: %llu\n", sumCalcs);

    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

 
    unsigned int numBatches = MAXBATCH; //ceil(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE))*TPP;
    // unsigned int leftOverBatch = floor(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE / TPP));
    // unsigned int * batchPoints = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    // unsigned int batchOffset = 0;
    // for(unsigned int i = 0; i < numBatches; i++){
    //     batchPoints[i] = batchOffset;
    //     batchOffset += KERNEL_BLOCKS*BLOCK_SIZE;
    // }



    // printf("Launch setup time: %f\n", launchend - launchstart);
    double launchend = omp_get_wtime();
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, normData, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numNeighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_numNeighbors, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numNeighbors, numNeighbors, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    // unsigned int * d_batchPoints;
    // assert(cudaSuccess == cudaMalloc((void**)&d_batchPoints, sizeof(unsigned int)*numBatches));
    // assert(cudaSuccess ==  cudaMemcpy(d_batchPoints, batchPoints, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));
    

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_nodeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodeID, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_nodeID, nodeID, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));
    
    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_pointOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOffsets, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOffsets, pointOffsets, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_neighborOffset; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_neighborOffset, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_neighborOffset, neighborOffset, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_nodePoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodePoints, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_nodePoints, nodePoints, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_neighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_neighbors, sizeof(unsigned int)*neighborOffsetCount));
    assert(cudaSuccess ==  cudaMemcpy(d_neighbors, neighbors, sizeof(unsigned int)*neighborOffsetCount, cudaMemcpyHostToDevice));


    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));


    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_pointIdent[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointIdent[i], sizeof(unsigned int)*BLOCK_SIZE*KERNEL_BLOCKS*TPP));
    }

    unsigned int pointIndex = 0;
    unsigned int * d_pointIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointIndex, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_pointIndex, &pointIndex, sizeof(unsigned int), cudaMemcpyHostToDevice));


    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for storing the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    
    #if !HOST
    for (unsigned int i = 0; i < numPoints; i++){	
        struct neighborTable temp;
        tables[i] = temp;
        // tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
    #endif

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }

    printf("Time to transfer: %f\n", omp_get_wtime()-launchend);
    printf("Batchs: %d\n",numBatches);
    const unsigned int cdim = dim;
    bool flag = false;
    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic) if(!HOST)
    for(unsigned int i = 0; i < numBatches; i++){
        
        if(flag) continue;
        
        unsigned int tid = omp_get_thread_num();
            
        assert(cudaSuccess ==  cudaMemcpyAsync(&pointIndex, d_pointIndex, sizeof(unsigned int ), cudaMemcpyDeviceToHost, stream[tid]));
            
        if(pointIndex >= numPoints) {
            flag = true;
            i+=MAXBATCH;
            continue;
        }

            
        cudaSetDevice(CUDA_DEVICE);

            

        // printf("BatchNumber: %d/%d\n", i+1, numBatches);

        // double kernelStartTime = omp_get_wtime();

        //launch distance kernel
        nodeByPoint4<<<KERNEL_BLOCKS*TPP, BLOCK_SIZE, 0, stream[tid]>>>( cdim, 
                                                                    d_data,
                                                                    d_epsilon2,
                                                                    d_numPoints,
                                                                    d_nodeID,
                                                                    d_numNeighbors,
                                                                    d_nodePoints,
                                                                    d_neighbors,
                                                                    d_neighborOffset,
                                                                    d_pointOffsets,
                                                                    d_pointA[tid],
                                                                    d_pointB[tid],
                                                                    &d_keyValueIndex[i],
                                                                    d_pointIdent[tid],
                                                                    d_pointIndex);



        cudaStreamSynchronize(stream[tid]);

        // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

        assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
        cudaStreamSynchronize(stream[tid]);

        printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


        if(keyValueIndex[i] > bufferSizes[tid]){
            //  printf("tid: %d first run\n", tid);
            cudaFreeHost(pointB[tid]);
            //printf("tid: %d freed memory\n", tid);
            assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
            //printf("tid: %d pinned memory\n", tid);
            bufferSizes[tid] = keyValueIndex[i];
        }

        // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
        GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);


        assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

        cudaStreamSynchronize(stream[tid]);

        unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
        kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                    &d_keyValueIndex[i], 
                                    d_uniqueKeys[tid], 
                                    d_uniqueKeyPosition[tid], 
                                    &d_uniqueCnt[i]);

        cudaStreamSynchronize(stream[tid]);

        assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

        cudaStreamSynchronize(stream[tid]);

        // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
        GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


        unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
        assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

        unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
        assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

        dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

        cudaStreamSynchronize(stream[tid]);

        constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

        free(uniqueKeys);
        free(uniqueKeyPosition);

    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    printf("Total results Set Size: %llu \n", totals);

    return tables;
}

struct neighborTable * nodeLauncher5(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon){


    cudaSetDevice(CUDA_DEVICE);
    
    double time1 = omp_get_wtime();
    std::vector<struct Node> nodes;

    // build the data structure
    unsigned int numNodes = buildNodeNet(data,
            dim,
            numPoints,
            numRP,
            pointArray,
            epsilon,
            &nodes);


    cudaSetDevice(CUDA_DEVICE);

    double time2 = omp_get_wtime();
    printf("Node Construct time: %f\n", time2 - time1);
    fprintf(stderr, "%f ", time2-time1);

    // unsigned long long res = nodeForce(&nodes, epsilon, data, dim, numPoints);
    // printf("Res: %llu\n", res);

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

    double * normData = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
        for(unsigned int i = 0; i < numPoints; i++){
            for(unsigned int j = 0; j < dim; j++){
            #if DATANORM
                normData[i+numPoints*j] = data[pointArray[i]*dim+j];
            #else
                normData[i*dim+j] = data[pointArray[i]*dim+j];
            #endif
        }
    }

    // printf("P1: %d, P2: %d\n", pointArray[0], nodes[0].nodePoints[0]);
    //build array of point offsets
    unsigned int * pointOffsets = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    //build array of number of calcs needed
    unsigned long long * numCalcs = (unsigned long long *)malloc(sizeof(unsigned long long)*numNodes);
    //build array of number of neighbors
    unsigned int * numNeighbors = (unsigned int*)malloc(sizeof(unsigned int)*numNodes);
    //array to count total number of neighbors for linear id
    unsigned int * neighborOffset = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);
    // number of points in each node
    unsigned int * nodePoints = (unsigned int *)malloc(sizeof(unsigned int)*numNodes);

    unsigned int * nodeID = (unsigned int *)malloc(sizeof(unsigned int)*numPoints); 


    //counter for neighbor offsets
    unsigned int neighborOffsetCount = 0;
    // std::vector<unsigned int> tempNeighbors;

    unsigned int previous = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        pointOffsets[i] = nodes[i].pointOffset;
        numCalcs[i] = nodes[i].numCalcs;
        neighborOffset[i] = neighborOffsetCount;
        numNeighbors[i] = nodes[i].neighborIndex.size();
        neighborOffsetCount += nodes[i].neighborIndex.size();
        nodePoints[i] = nodes[i].numNodePoints;
        // tempNeighbors.insert(tempNeighbors.end(), nodes[i].neighborIndex.begin(),nodes[i].neighborIndex.end());
        for(unsigned int j = previous; j < previous+nodePoints[i]; j++){
            nodeID[j] = i;
        }
        previous += nodePoints[i]; // maybe-1 here
    }


    // printf("po:%u\n", pointOffsets[10]);
    unsigned int * neighbors = (unsigned int *)malloc(sizeof(unsigned int)*neighborOffsetCount);

    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < numNeighbors[i]; j++){
        neighbors[counter+j] = nodes[i].neighborIndex[j];   
        }
        counter += numNeighbors[i];
    }

    // printf("total num neighbors: %u\n", counter);

    unsigned long long sumCalcs = totalNodeCalcs(nodes, numNodes);
    // printf("sum calcs: %llu\n", sumCalcs);

    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    //calculate the number of batches
    double varCalc = sumCalcs*1.0 / numPoints / numPoints; 
    unsigned int batchCount = 0;
    unsigned long long calcCounter = 0;
    unsigned int pointCounter = 0;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double factor1 = (1+varCalc*CMP)*2000/sqrt(numPoints);
    double factor2 = CALC_PER_BATCH * pow(10,10) * 1.0 * sqrt(pow(1+epsilon,20));
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<unsigned int> pointsPerBatchVec;
    std::vector<unsigned int> tppVec;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < nodePoints[i]; j++){
            calcCounter += (numCalcs[i]/nodePoints[i]);
            pointCounter++;
            if(calcCounter * factor1 >= factor2 || pointCounter >= KERNEL_BLOCKS*BLOCK_SIZE ){
                batchCount++;
                calcCounter = 0;
                pointsPerBatchVec.push_back(pointCounter);
                tppVec.push_back(KERNEL_BLOCKS*BLOCK_SIZE / pointCounter);
                pointCounter = 0;
            }
        }
    }

    batchCount++;
    pointsPerBatchVec.push_back(pointCounter);
    tppVec.push_back(KERNEL_BLOCKS*BLOCK_SIZE / pointCounter);

 
    unsigned int numBatches = batchCount;
    if(ERRORPRINT) fprintf(stderr,"%u %f ", numBatches, varCalc);

    // unsigned int leftOverBatch = floor(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE / TPP));
    unsigned int * batchPoints = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    unsigned int * pointsPerBatch = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    unsigned int * tpp = (unsigned int *)malloc(sizeof(unsigned int )*numBatches);
    unsigned int offsetCount = 0;
    for(unsigned int i = 0; i < numBatches; i++){
        batchPoints[i] = offsetCount;
        offsetCount += pointsPerBatchVec[i];
        pointsPerBatch[i] = pointsPerBatchVec[i];
        #if TPP == 1
        tpp[i] = 1;
        #else
        tpp[i] = tppVec[i];
        #endif
    }



    // printf("Launch setup time: %f\n", launchend - launchstart);
    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, normData, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numNeighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_numNeighbors, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_numNeighbors, numNeighbors, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_batchPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchPoints, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchPoints, batchPoints, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int * d_pointsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointsPerBatch, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_pointsPerBatch, pointsPerBatch, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int * d_tpp;
    assert(cudaSuccess == cudaMalloc((void**)&d_tpp, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_tpp, tpp, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));
    

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_nodeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodeID, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_nodeID, nodeID, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));
    
    
    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_pointOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointOffsets, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_pointOffsets, pointOffsets, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_neighborOffset; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_neighborOffset, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_neighborOffset, neighborOffset, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_nodePoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_nodePoints, sizeof(unsigned int)*numNodes));
    assert(cudaSuccess ==  cudaMemcpy(d_nodePoints, nodePoints, sizeof(unsigned int)*numNodes, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_neighbors;
    assert(cudaSuccess == cudaMalloc((void**)&d_neighbors, sizeof(unsigned int)*neighborOffsetCount));
    assert(cudaSuccess ==  cudaMemcpy(d_neighbors, neighbors, sizeof(unsigned int)*neighborOffsetCount, cudaMemcpyHostToDevice));


    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    const double d_epsilon2 = epsilon2;
    // assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    // assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    const unsigned int d_dim = dim;
    // assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    // assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    const unsigned int d_numPoints = numPoints;
    // assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    // assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));


    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for storing the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    
    #if !HOST
    for (unsigned int i = 0; i < numPoints; i++){	
        struct neighborTable temp;
        tables[i] = temp;
        //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
    #endif

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    // double totalKernelTime[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
        // totalKernelTime[NUMSTREAMS] = 0;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // return tables;

    // printf("Time to transfer: %f\n", omp_get_wtime()-launchend);
    printf("Batchs: %d\n",numBatches);

    double kstartTime = omp_get_wtime();
    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic) if(!HOST)
    for(unsigned int i = 0; i < numBatches; i++){
            cudaSetDevice(CUDA_DEVICE);

            unsigned int tid = omp_get_thread_num();

            // printf("BatchNumber: %d/%d\n", i+1, numBatches);

            // double kernelStartTime = omp_get_wtime();

            //launch distance kernel
            nodeByPoint5<<<KERNEL_BLOCKS, BLOCK_SIZE, 0, stream[tid]>>>(d_dim, 
                                                                        d_data,
                                                                        d_epsilon2,
                                                                        d_numPoints,
                                                                        &d_batchPoints[i],
                                                                        d_nodeID,
                                                                        d_numNeighbors,
                                                                        d_nodePoints,
                                                                        d_neighbors,
                                                                        d_neighborOffset,
                                                                        d_pointOffsets,
                                                                        d_pointA[tid],
                                                                        d_pointB[tid],
                                                                        &d_keyValueIndex[i],
                                                                        &d_tpp[i],
                                                                        &d_pointsPerBatch[i]);



            cudaStreamSynchronize(stream[tid]);

            // totalKernelTime[tid] += omp_get_wtime() - kernelStartTime;

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            // GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                        &d_keyValueIndex[i], 
                                        d_uniqueKeys[tid], 
                                        d_uniqueKeyPosition[tid], 
                                        &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            // GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);

            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);

    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }
    double kendTime = omp_get_wtime();
    printf("Total results Set Size: %llu \n", totals);
    printf("Kernel Time: %f\n", kendTime - kstartTime);
    std::cerr << kendTime-kstartTime << " " << totals << " "; 
    // std::cerr << totals << " " ;
    // fprintf(stderr,"%llu ",totals);


    cudaFree(d_data);
    cudaFree(d_numNeighbors);
    cudaFree(d_batchPoints);
    cudaFree(d_pointsPerBatch);
    cudaFree(d_tpp);
    cudaFree(d_nodeID);
    cudaFree(d_pointOffsets);
    cudaFree(d_neighbors);
    cudaFree(d_keyValueIndex);
    

    return tables;
}



struct neighborTable * launchCOSS(unsigned int ** tree, // a pointer to the tree
    unsigned int numPoints, //the number of points in the data
    unsigned int ** pointBinNumbers,  //the bin numbers ofr each point to each reference point
    unsigned int numLayers, //the number of reference points
    unsigned int * binSizes, // the number of bins in each layer
    unsigned int * binAmounts, //the number of bins for each reference point
    double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
    unsigned int dim,//the dimensionality of the data
    double epsilon,//the distance threshold being searched
    unsigned int * pointArray)// the array of point numbers ordered to match the sequence in the last array of the tree and the data
{

    printf("Starting Kernel Launcher\n");
    // the number of searches that are needed for a full search is the number of referecnes point cubed. This is always true, mostly.
    const unsigned int numSearches = pow(3,numLayers);

    const unsigned int numBatches = ceil(numPoints*1.0/(KERNEL_BLOCKS*BLOCK_SIZE));
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

    printf("Non Empty Bins in last layer: %u\n", nonEmptyBins);
    fprintf(stderr, "%u ", nonEmptyBins);

    // an array for holding the non empty index locations in the final tree layer
    unsigned int * addIndexes = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);

	// copy the temp indexes which keeps track of the nonepty indexes into an array that is the correct size
    #pragma omp parallel for num_threads(8)
    for(unsigned int i = 0; i < nonEmptyBins; i++){
        addIndexes[i] = tempIndexes[i];
    }

	// free the overallocated array now that we have the data stored in tempAddIndexes
    free(tempIndexes);

    //array to hold the point bin numbers
	unsigned int * binNumbers = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins*(numLayers+1));
    unsigned int * pointBinIndex = (unsigned int *)malloc(sizeof(unsigned int)*numPoints);

    
	for(unsigned int i = 0; i < nonEmptyBins; i++){
        binNumbers[i*(numLayers+1)] = tree[ numLayers-1 ][ addIndexes[i] ];
        for(unsigned int j = 0; j < numLayers; j++){
			binNumbers[i*(numLayers+1)+j+1] = pointBinNumbers[ tree[ numLayers-1 ][ addIndexes[i] ]][j];
		}
	}


    for(unsigned int i = 0; i < nonEmptyBins-1; i++){
        for(unsigned int j = binNumbers[i*(numLayers+1)]; j < binNumbers[(i+1)*(numLayers+1)]; j++){
            pointBinIndex[j] = i;
        }
    }

    for(unsigned int j = binNumbers[(nonEmptyBins-1)*(numLayers+1)]; j < numPoints; j++){
        pointBinIndex[j] = nonEmptyBins - 1;
    }


    unsigned int totalBinNumber = 0;
    for(unsigned int i = 0; i < numLayers; i++){
        totalBinNumber += binSizes[i];
    }
    // const unsigned int lastLayerOffset = totalBinNumber - binSizes[numLayers-1];

    printf("Total Bin Count in Tree: %u\n", totalBinNumber);

    unsigned int binOffset = 0;
    unsigned int *Ltree = (unsigned int *)malloc(sizeof(unsigned int)*totalBinNumber);
    for(unsigned int i = 0; i < numLayers; i++){
        for(unsigned int j = 0; j < binSizes[i]; j++){
            Ltree[binOffset+j] = tree[i][j];
        }
        binOffset += binSizes[i];
    }

    
    // printf(" last pointBinIndex: %u, should be : %u\n", pointBinIndex[numPoints-1], nonEmptyBins-1);

    printf("Starting CUDA Mem transfers\n");

    //device memory for the tree
    unsigned int * d_tree;
    assert(cudaSuccess == cudaMalloc((void**)&d_tree, sizeof(unsigned int)*totalBinNumber));
    assert(cudaSuccess ==  cudaMemcpy(d_tree, Ltree, sizeof(unsigned int )*totalBinNumber, cudaMemcpyHostToDevice));

    //device memory to hold the bin number arrays
    unsigned int * d_binNumbers;
    assert(cudaSuccess == cudaMalloc((void**)&d_binNumbers, sizeof(unsigned int)*nonEmptyBins*(numLayers+1)));
    assert(cudaSuccess ==  cudaMemcpy(d_binNumbers, binNumbers, sizeof(unsigned int )*nonEmptyBins*(numLayers+1), cudaMemcpyHostToDevice));

    unsigned int * d_pointBinIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointBinIndex, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointBinIndex, pointBinIndex, sizeof(unsigned int )*numPoints, cudaMemcpyHostToDevice));
 
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    unsigned int * d_pointArray;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));
    
    // the number of bins for each layer
    unsigned int * d_binSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_binSizes, sizeof(unsigned int)*numLayers));
    assert(cudaSuccess ==  cudaMemcpy(d_binSizes, binSizes, sizeof(unsigned int)*numLayers, cudaMemcpyHostToDevice));

    //device memory for the number of bins to each reference point
    unsigned int * d_binAmounts;
    assert(cudaSuccess == cudaMalloc((void**)&d_binAmounts, sizeof(unsigned int)*numLayers));
    assert(cudaSuccess ==  cudaMemcpy(d_binAmounts, binAmounts, sizeof(unsigned int)*numLayers, cudaMemcpyHostToDevice));
    
    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }
    
    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));
    
    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_addressStorageSpace[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_addressStorageSpace[i], sizeof(unsigned int)*(numLayers+1)*TPP*KERNEL_BLOCKS*BLOCK_SIZE));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    printf("Building Neighbor Tables\n");
    //struct for sotring the results
    // struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    struct neighborTable * tables = new neighborTable[numPoints];
    for (unsigned int i = 0; i < numPoints; i++)
        {	
        struct neighborTable temp; 
        tables[i] = temp;
        //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

        tables[i].cntNDataArrays = 1; 
        tables[i].vectindexmin.resize(numBatches+1);
        tables[i].vectindexmin[0] = i;
        tables[i].vectindexmax.resize(numBatches+1);
        tables[i].vectindexmax[0] = i;
        tables[i].vectdataPtr.resize(numBatches+1);
        tables[i].vectdataPtr[0] = pointArray;
        omp_init_lock(&tables[i].pointLock);
    }
     
    cudaDeviceSynchronize(); 

    printf("Creating CUDA Streams\n");

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
    }

    const unsigned int d_numPoints = numPoints;
    // const unsigned int d_arrayCounter = nonEmptyBins;
    // const unsigned int d_numLayers = numLayers;
    const unsigned int d_dim = dim;
    const double d_epsilon2 = epsilon*epsilon;


    printf("Starting COSS style Kernel\n");
    double time3 = omp_get_wtime();


    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic)
    for(unsigned int i = 0; i < numBatches; i++){

        const unsigned int totalBlocks = ceil(TPP*KERNEL_BLOCKS);
        
        unsigned int tid = omp_get_thread_num();

        const int d_batch_num = i;

        #if BRUTE
        dumbBrute<<<totalBlocks,BLOCK_SIZE, 0, stream[tid]>>>(
            d_batch_num,
            d_data,
            d_numPoints,
            d_pointA[tid],
			d_pointB[tid],
            d_dim,
			d_epsilon2,
			&d_keyValueIndex[i]);
        #else
        // searchKernelCOSStree<<<totalBlocks,BLOCK_SIZE, 0, stream[tid]>>>(
		// 	d_batch_num,
        //     d_data,
		// 	d_numPoints,
		// 	d_pointA[tid],
		// 	d_pointB[tid],
		// 	d_binNumbers,
		// 	&d_keyValueIndex[i],
		// 	d_pointArray,
		// 	d_arrayCounter,
		// 	d_numLayers,
		// 	d_dim,
		// 	d_epsilon2,
		// 	d_pointBinIndex,
        //     d_addressStorageSpace[tid],
        //     d_tree,
        //     d_binSizes,
        //     d_binAmounts,   
        //     lastLayerOffset);
        #endif

        cudaStreamSynchronize(stream[tid]);
        // assert(cudaSuccess == cudaGetLastError());

        assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
        cudaStreamSynchronize(stream[tid]);

        printf("Batch %d Results: %llu, total Blocks: %u, BlockSize: %u \n", i,keyValueIndex[i], totalBlocks, BLOCK_SIZE);


        if(keyValueIndex[i] > bufferSizes[tid]){
            //  printf("tid: %d first run\n", tid);
            cudaFreeHost(pointB[tid]);
            //printf("tid: %d freed memory\n", tid);
            assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
            //printf("tid: %d pinned memory\n", tid);
            bufferSizes[tid] = keyValueIndex[i];
        }

        // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
        GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

        assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

        cudaStreamSynchronize(stream[tid]);

        unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
        kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                                            &d_keyValueIndex[i], 
                                                            d_uniqueKeys[tid], 
                                                            d_uniqueKeyPosition[tid], 
                                                            &d_uniqueCnt[i]);

        cudaStreamSynchronize(stream[tid]);

        assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

        cudaStreamSynchronize(stream[tid]);

        // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
        GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


        unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
        assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

        unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
        assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

        dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

        cudaStreamSynchronize(stream[tid]);

        constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

        free(uniqueKeys);
        free(uniqueKeyPosition);

    }

    double time4 = omp_get_wtime();
    printf("Kernel time: %f\n", time4-time3);
    if(ERRORPRINT) fprintf(stderr,"%f ",time4-time3);


    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }
    if(ERRORPRINT) fprintf(stderr,"%llu ", totals);

    printf("Total results Set Size: %llu \n", totals);

    return tables;

}