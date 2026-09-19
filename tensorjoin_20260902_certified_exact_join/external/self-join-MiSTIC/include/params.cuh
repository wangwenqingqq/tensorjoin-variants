#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "omp.h"
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <stack>

#define COSS true

#define BINMETRICS false
#define BRUTE true
#define DATANORM true
#define HOST false
#define BINARYSEARCH 2 // 1 for binary 2 for depth
#define GPUSEARCH false
#define TESTING_SEARCH false
#define MINSQRS true
#define NODES true
#define ERRORPRINT true
#define NODETEST false
#define RANDRP false

#define SUBG false

#define DEVICE_BUILD true
#define MIN_CALCS_PER_NODE 100000
#define USE_MIN_CALC true
#define CUDA_DEVICE 1
#define CALC_MULTI 1.5
// #define MIN_NODE_SIZE 1000
#define MAX_CALCS_PER_NODE 32 //millions
#define TPP 8
#define BUFFERSIZE 1000000
#define MAXBATCH 100
#define ORDP 32
#define KT 5
#define CMP 14
#define MAXRP 6
#define ILP 4
// #define DIM 90
const unsigned int RPPERLAYER = 32;



// const unsigned int MAXRP = 30;

const unsigned int MINRP = 3;

const unsigned int NUMIDIM = 6;

const unsigned int NUMSTREAMS = 1;
const unsigned int BLOCK_SIZE = BS; 
const unsigned int KERNEL_BLOCKS = KB;// * 1024 / BLOCK_SIZE;
const int RANDOM = false;
const int BOXED_RP = false;
const unsigned int TEST_RP = 100;
const double SAMPLE_PER = 0.01;
// const unsigned int MIN_NODE_SIZE = 1000; //value of 1 shuts this off

const unsigned long long int CALC_PER_BATCH = 115;
const unsigned int CALCS_PER_THREAD = 30000000;
const unsigned int MAX_CALCS_PER_THREAD = 6000000;
const unsigned int MIN_CALCS_PER_THREAD = 10000;

const unsigned int MAX_BIN = 100000; 
const unsigned int MINBINSIZE = 10000;

// const unsigned long long int calcsPerSecond = 50000000000;
// const unsigned int nodesPerSecond = 2000;

const double LAYER_DIFF = 1.25;

const int RAND = false;
const unsigned long long initalPinnedResultsSize = 100000;
const unsigned long long int resultsSize = 2500000000;