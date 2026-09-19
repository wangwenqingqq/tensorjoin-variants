#include "include/utils.cuh"

//comparator function for sorting pairs and is used when checking results for duplicates
bool compPair(const std::pair<unsigned int, unsigned int> &x, const std::pair<unsigned int, unsigned int> &y){
    if(x.first < y.first){
        return true;
    }

    if(x.first == y.first && x.second < y.second){
        return true;
    }

    return false;

}

void GPU_SortbyKey( cudaStream_t stream, unsigned int * A, unsigned int size, unsigned int * B){
	thrust::sort_by_key(thrust::cuda::par.on(stream), A, A+size, B);
}


unsigned int * stddev( double * A, unsigned int dim, unsigned int num_points) {
	double mean, devmean;
	double *deviation = (double*)malloc(sizeof(double) * dim);
	unsigned int *dimension = (unsigned int*)malloc(sizeof(unsigned int) * dim);
	for(unsigned int i = 0; i < dim; i++) {
		dimension[i] = i;
	}
	for(unsigned int i = 0; i < dim; i++){
		mean = 0.0;
		for(unsigned int j = 0; j < num_points; j++){
			mean += A[dim*j+i];
		}
		mean /= num_points;
		devmean = 0.0;
		for(unsigned int j = 0; j < num_points; j++){
			devmean += pow(A[dim*j + i] - mean,2);
		}
		devmean /= num_points;
		deviation[i] = sqrt(devmean);
	}
	thrust::sort_by_key(deviation, &deviation[dim], dimension);
	double *deviationret = (double*)malloc(sizeof(double) * dim);
	unsigned int *dimensionret = (unsigned int*)malloc(sizeof(unsigned int) * dim);
	for(unsigned int i = 0; i < dim; i++){
		deviationret[i] = deviation[dim-1-i];
		dimensionret[i] = dimension[dim-1-i];
	}
	free(deviationret);
	free(deviation);
	free(dimension);
	return dimensionret;
}

double euclideanDistance(double * dataPoint, unsigned int dim, double * RP){
    // get the euclidean distance
    double distance = 0;
    for(unsigned int i = 0; i < dim; i++){
        double diff = (RP[i] - dataPoint[i]);
        distance += diff * diff;
    }
    distance = sqrt(distance);
    return distance;
}

double * createRPArray(double * data, unsigned int numRP, unsigned int dim, unsigned long long numPoints){

	unsigned int sample_size = numPoints*SAMPLE_PER;


	unsigned int test_rp = numRP * 2;
	
	// double * testRPArray = new double[TEST_RP*dim];
	double * testRPArray = (double*)malloc(sizeof(double)*test_rp*dim);
	
	for(unsigned int i = 0; i < test_rp*dim; i++){
		testRPArray[i] = (double)rand()/(double)RAND_MAX;
	}
	

	//get the distances
	double* distmat = (double*)malloc(sizeof(double)*test_rp*sample_size);

	#pragma omp parallel for
	for(unsigned int i = 0; i < sample_size; i++){
		for(unsigned int j = 0; j < test_rp; j++){
			distmat[i*test_rp+j] = euclideanDistance(&data[i*dim], dim, &testRPArray[j*dim]);
		}
	}

	//get std dev of dist mat
	unsigned int * order = stddev(distmat, test_rp, sample_size);

	//get first numRP rps
	double * RPArray = (double *)malloc(sizeof(double)*numRP*dim);

	// #pragma omp parallel for
	for(unsigned int i = 0; i < numRP/2; i++){
		for(unsigned int j = 0; j < dim; j++){
			RPArray[i*dim+j] = testRPArray[ order[i]*dim + j ];
		}
	}

	//first rp is the max corner
	for(unsigned int i = 0; i < dim; i++)
	{
		testRPArray[i] = 0;
	}
	for(int i = 0; i < sample_size; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if(testRPArray[j] < data[i*dim+j])
			{
				testRPArray[j] = data[i*dim+j];
			}
		}
	}

	for(int i = 1; i < test_rp; i++) // the first rp is set
	{
		
		unsigned int step = pow(2,dim) / test_rp;
		for(int j = 0; j < dim; j++)
		{
			testRPArray[i*dim + j] = (i*step / (int)pow(2, j) % 2)*testRPArray[j];
				// RP[i*dim+i*step + j] = RP[j+i*step];
		}

	}

	for(unsigned int i = 0; i < numRP/2 + numRP%2; i++){
		for(unsigned int j = 0; j < dim; j++){
			RPArray[i*dim+j + numRP/2*dim] = testRPArray[ i*dim+j ];
		}
	}


///////////////////////////////////////////////////////////////////////////////////////////////////
	// if(RAND) srand(omp_get_wtime());

	// #if RANDRP
	// 	double * testRPArray = (double*)malloc(sizeof(double)*numRP*dim);
			
	// 	for(unsigned int i = 0; i < numRP*dim; i++){
	// 			testRPArray[i] = (double)rand()/(double)RAND_MAX;
	// 	}
	// 	return testRPArray;
	// #else


	// unsigned int test_rp = numRP * 2;
	
	// // double * testRPArray = new double[TEST_RP*dim];
	// double * testRPArray = (double*)malloc(sizeof(double)*test_rp*dim);

	// if(BOXED_RP){

	// 	//first rp is the max corner
	// 	for(unsigned int i = 0; i < dim; i++)
	// 	{
	// 		testRPArray[i] = 0;
	// 	}
	// 	for(int i = 0; i < numPoints; i++)
	// 	{
	// 		for(int j = 0; j < dim; j++)
	// 		{
	// 			if(testRPArray[j] < data[i*dim+j])
	// 			{
	// 				testRPArray[j] = data[i*dim+j];
	// 			}
	// 		}
	// 	}

	// 	for(int i = 1; i < test_rp; i++) // the first rp is set
	// 	{
			
	// 		unsigned int step = pow(2,dim) / test_rp;
	// 		for(int j = 0; j < dim; j++)
	// 		{
	// 			testRPArray[i*dim + j] = (i*step / (int)pow(2, j) % 2)*testRPArray[j];
	// 				// RP[i*dim+i*step + j] = RP[j+i*step];
	// 		}

	// 	}



		
	// }else{
	// 	//randomly place the rps
	// 	// #pragma omp parallel for
	// 	for(unsigned int i = 0; i < test_rp*dim; i++){
	// 		testRPArray[i] = (double)rand()/(double)RAND_MAX;
	// 	}
	// }




	// //get the distances
	// // double *distmat = new double[TEST_RP*sample_size];
	// double* distmat = (double*)malloc(sizeof(double)*test_rp*sample_size);

	// #pragma omp parallel for
	// for(unsigned int i = 0; i < sample_size; i++){
	// 	for(unsigned int j = 0; j < test_rp; j++){
	// 		distmat[i*test_rp+j] = euclideanDistance(&data[i*dim], dim, &testRPArray[j*dim]);
	// 	}
	// }

	// //get std dev of dist mat
	// unsigned int * order = stddev(distmat, test_rp, sample_size);

	// //get first numRP rps
	// double * RPArray = (double *)malloc(sizeof(double)*numRP*dim);

	// // #pragma omp parallel for
	// for(unsigned int i = 0; i < numRP; i++){
	// 	for(unsigned int j = 0; j < dim; j++){
	// 		RPArray[i*dim+j] = testRPArray[ order[i]*dim + j ];
	// 	}
	// }

	// for(unsigned int i = 0; i < numRP - numRP/2; i++){
	// 	for(unsigned int j = 0; j < dim; j++){
	// 		RPArray[i*dim+j] = (double)rand()/(double)RAND_MAX;
	// 	}
	// 	// testRPArray[i] = (double)rand()/(double)RAND_MAX;
	// }

	// double * testRPArray2 = (double*)malloc(sizeof(double)*numRP/2*dim);

	// //first rp is the max corner
	// for(unsigned int i = 0; i < dim; i++)
	// {
	// 	testRPArray2[i] = 0;
	// }
	// for(int i = 0; i < numPoints; i++)
	// {
	// 	for(int j = 0; j < dim; j++)
	// 	{
	// 		if(testRPArray2[j] < data[i*dim+j])
	// 		{
	// 			testRPArray2[j] = data[i*dim+j];
	// 		}
	// 	}
	// }

	// for(int i = 1; i < numRP/2; i++) // the first rp is set
	// {
		
	// 	unsigned int step = pow(2,dim) / numRP/2;
	// 	for(int j = 0; j < dim; j++)
	// 	{
	// 		testRPArray2[i*dim + j] = (i*step / (int)pow(2, j) % 2)*testRPArray2[j];
	// 			// RP[i*dim+i*step + j] = RP[j+i*step];
	// 	}

	// }

	// #pragma omp parallel for
	// for(unsigned int i = 0; i < numRP/2 +1; i++){
	// 	for(unsigned int j = 0; j < dim; j++){
	// 		RPArray[i*dim+j] = testRPArray[ order[i]*dim + j ];
	// 	}
	// }
	// for(unsigned int i = numRP/2 ; i < numRP; i++){
	// 	for(unsigned int j = 0; j < dim; j++){
	// 		RPArray[i*dim+j] = testRPArray2[ (i - numRP/2)*dim + j];
	// 	}
	// }
//////////////////////////////////////////////////////////////////////////////////////////

	free(testRPArray);
	free(distmat);
	free(order);

    return RPArray;
	// #endif
}



unsigned int brute_force( unsigned int num_points, unsigned int dim, double epsilon, double *A){
	//brute force check
	unsigned int brute_count = 0;
	omp_lock_t brute;
	omp_init_lock(&brute);

	#pragma omp parallel for
	for(unsigned int i = 0; i < num_points; i++)
	{
		for (unsigned int j = 0; j < num_points; j++)
		{
		double distance = 0;
			for (unsigned int k = 0; k < dim; k++)
			{
				if(distance > epsilon*epsilon)
				{
					break;
				} else {
					double a1 = A[i*dim + k];
					double a2 = A[j*dim + k];
					distance += (a1-a2)*(a1-a2);
				}
				}
				if(distance <= epsilon*epsilon){
					omp_set_lock(&brute);
					brute_count++;
					omp_unset_lock(&brute);
				}
		}
	}
	printf("\nBrute force has %d pairs.\n\n\n", brute_count);
	return brute_count;
}