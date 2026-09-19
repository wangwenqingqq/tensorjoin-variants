#pragma once

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h> 
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
 
#include "include/params.cuh"

bool compPair(const std::pair<unsigned int, unsigned int> &x, const std::pair<unsigned int, unsigned int> &y);

typedef struct DevicePointers{
    double * d_epsilon;
    unsigned int * d_dim;
    unsigned int * d_numPoints;
    double * d_data;
}DevicePointers;


struct result{
    unsigned int pid;
    unsigned int numNeighbors;
    unsigned int *neighbors;
};


typedef struct neighborTable
{
	unsigned int cntNDataArrays;
	std::vector<unsigned int>vectindexmin;
	std::vector<unsigned int>vectindexmax;
	std::vector<unsigned int *>vectdataPtr;
	omp_lock_t pointLock; //one lock per point

}neighborTable;

void GPU_SortbyKey( cudaStream_t stream, unsigned int * A, unsigned size, unsigned int * B);

double euclideanDistance(double * dataPoint, unsigned int dim, double * RP);

double * createRPArray(double * data, unsigned int numRP, unsigned int dim, unsigned long long numPoints);

unsigned int * stddev( double * A, unsigned int dim, unsigned int num_points);
unsigned int brute_force( unsigned int num_points, unsigned int dim, double epsilon, double *A);

