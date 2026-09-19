
#include "include/nodes.cuh"

//function to build up net of nodes
unsigned int buildNodeNet(double * data,
                 unsigned int dim,
                 unsigned int numPoints,
                 unsigned int numRP,
                 unsigned int * pointArray,
                 double epsilon,
                 std::vector<struct Node> * outNodes){


    cudaSetDevice(CUDA_DEVICE);

    std::vector<std::vector<struct Node>> subGraph;
    std::vector<struct Node> newNodes;
    unsigned int numNodes;
    double calcTime;
    double nodePerSecond;
    unsigned int numSplits = 0;
    double previousCalcTime = 0;
    double predictedNodeTime = 0;
    unsigned long long int  calcsPerSecond;

    unsigned int tempRPCalc = ceil(pow(1+epsilon,20))*3 + 10;
    const unsigned int rPPerLayer = RPPERLAYER; //tempRPCalc < 100 ? tempRPCalc:100;
    std::cerr << rPPerLayer << " ";


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


    unsigned long long int previousDistCalcs = numPoints*numPoints;
    unsigned int numPreviousNodes = 0;

    // need to go through each reference point
    for(unsigned int i = 0; i < MAXRP; i++){ 

        // generate some reference points
        double * RPArray = createRPArray(data, rPPerLayer, dim, numPoints);
    
        std::vector<std::vector<struct Node>> layerNodes;
        layerNodes.resize(rPPerLayer+NUMIDIM);
        unsigned long long int lowestDistCalcs = ULLONG_MAX;
        unsigned int bestRP = 0;

        

        unsigned int * allBinNumber = (unsigned int * )malloc(sizeof(unsigned int)*numPoints*(rPPerLayer+NUMIDIM));

        //create bin number arrays on device
        unsigned int  * d_binNumber;
        assert(cudaSuccess == cudaMalloc((void**)&d_binNumber, sizeof(unsigned int)*numPoints*rPPerLayer));

        double * d_RP;
        assert(cudaSuccess == cudaMalloc((void**)&d_RP, sizeof(double)*dim*rPPerLayer));
        assert(cudaSuccess == cudaMemcpy(d_RP, RPArray, sizeof(double)*dim*rPPerLayer, cudaMemcpyHostToDevice));

        free(RPArray);

        cudaStream_t stream;
        cudaError_t stream_check = cudaStreamCreate(&stream);
        assert(cudaSuccess == stream_check);

        unsigned int totalBlocks = ceil(numPoints*1.0/1024);

        double cT1 = omp_get_wtime();
        binningKernel<<<totalBlocks, 1024, 0, stream>>>(d_binNumber,
                                                                devicePointers.d_numPoints,
                                                                devicePointers.d_dim,
                                                                devicePointers.d_data,
                                                                d_RP,
                                                                devicePointers.d_epsilon,
                                                                rPPerLayer);

        cudaStreamSynchronize(stream);

        double cT2 = omp_get_wtime();





        if(i==0){
            calcsPerSecond = (unsigned long long int) numPoints*rPPerLayer / (cT2-cT1) * sqrt(dim) / (floor(pow(1+epsilon, 20))*10); //(sqrt(numPoints*pow(dim,2))/(1000*dim));//CALC_MULTI;
        } else {
            calcsPerSecond += (unsigned long long int) numPoints*rPPerLayer / (cT2-cT1) * sqrt(dim) / (floor(pow(1+epsilon,20))*10); //(sqrt(numPoints*pow(dim,2))/(1000*dim));//CALC_MULTI;
            calcsPerSecond = calcsPerSecond * CALC_MULTI;
        }
        printf("Predicted calcsPerSecond: %llu\n", calcsPerSecond);
        

        assert(cudaSuccess == cudaMemcpyAsync(allBinNumber, d_binNumber, sizeof(unsigned int)*numPoints*rPPerLayer, cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

        cudaFree(d_binNumber);
        cudaFree(d_RP);


        // int randosum = 0;
        // create dimensional partitions
        for(unsigned int j = 0; j < NUMIDIM; j++){
            for(unsigned int k = 0; k < numPoints; k++){
                allBinNumber[numPoints*(rPPerLayer+j)+k] = ceil((data[k*dim+j]+0.0000001)/ epsilon);
                // randosum += allBinNumber[numPoints*(rPPerLayer+j)+k];

            }
            
        }
        // std::cout << NUMIDIM << std::endl;
        // std::cout << numPoints << std::endl;
        //    std::cout << randosum << std::endl;

   
        // for( int blob = 0; blob < rPPerLayer+NUMIDIM; blob++){
        //     for( int bob = 0; bob < numPoints; bob++){
        //         randosum += allBinNumber[numPoints*blob+bob];
        //     }
        // // }
        // randosum+=allBinNumber[numPoints*(rPPerLayer+NUMIDIM-1)+numPoints-1];

     

        std::vector<std::vector<struct Node>> tempGraph;

        // itterate through all of the subgraphs 
        unsigned int numSubs;
        if(subGraph.size() == 0){
            numSubs = 1;
        } else {
            numSubs = subGraph.size();
        }

        lowestDistCalcs = 0;
        // printf("Num subs to gen: %u\n", numSubs);
        for(unsigned int n = 0; n < numSubs ; n++){
            unsigned long long int subLowestDistCalcs = ULLONG_MAX;
            double lowestStdDev = 0;
            bool firstCheck = true;


            #pragma omp parallel for num_threads(rPPerLayer)
            for(unsigned int j = 0; j < rPPerLayer+NUMIDIM; j++){
                // need to compare num dist calcs for different potental RP
                std::vector<struct Node> tempNodes;// subGraph[n];
                unsigned int tempNumNodes = 0;
                double tempCalcTime;
                double tempNodePerSecond;
                
                if(i == 0){
                    unsigned int * tempPointArray = (unsigned int *)malloc(sizeof(unsigned int)*numPoints);
                    #pragma omp parallel for num_threads(8)
                    for(unsigned int k = 0; k < numPoints; k++){
                        tempPointArray[k] = pointArray[k];
                    } 
                    tempNumNodes = initNodes(data, dim, numPoints, epsilon, &allBinNumber[numPoints*j], tempPointArray, &layerNodes[j], devicePointers, &tempCalcTime);
                    free(tempPointArray);
                }
                else{
                    tempNodes = subGraph[n];
                    tempNumNodes = splitNodes(&allBinNumber[numPoints*j], tempNodes, tempNodes.size(), epsilon, data, dim, numPoints, &layerNodes[j], devicePointers, &tempNodePerSecond);
                    // printf("    subgraph %u with %u origional nodes; RP %u has %u nodes\n", n, tempNodes.size(), j, tempNumNodes );
                }

                
                // printf("check: %llu\n", tempNodes[0].numCalcs);
                
                unsigned long long numCalcs = totalNodeCalcs(layerNodes[j], tempNumNodes);
                double stdDev = 0;
                double averageNode = numPoints*1.0 / layerNodes[j].size();
                for(unsigned int k = 0; k < layerNodes[j].size(); k++){
                    stdDev += (layerNodes[j][k].numNodePoints - averageNode)*(layerNodes[j][k].numNodePoints - averageNode);
                }

                // unsigned long long sumSqrs = nodeSumSqrs(layerNodes[j], tempNumNodes);
                // printf("    Layer %d for RP %d has Nodes: %u with calcs: %llu and stdev: %f \n", i, j, tempNumNodes, numCalcs, stdDev);
            
                #pragma omp critical
                {
                    // if(numCalcs < subLowestDistCalcs || firstCheck){
                    if(stdDev < lowestStdDev || firstCheck){
                        firstCheck = false;
                        subLowestDistCalcs = numCalcs;
                        lowestStdDev = stdDev;
                        bestRP = j;
                        // layerNodes = tempNodes;
                        numNodes = tempNumNodes;
                        if(i==0){
                            calcTime = tempCalcTime;
                            nodePerSecond = 2000;
                        } 
                        if(i!=0) nodePerSecond = tempNodePerSecond;
                    }
                }
                
            }

            free(allBinNumber);


            printf("SubGraph %u Layer %d Selecting RP %d with Nodes: %u and calcs: %llu and stdev %f:: ", n, i, bestRP, numNodes, subLowestDistCalcs, lowestStdDev);
            
            #if SUBG
            std::vector<std::vector<struct Node>> layerSubGraphs = genSubGraphs(layerNodes[bestRP]);
            tempGraph.insert(tempGraph.end(), layerSubGraphs.begin(), layerSubGraphs.end());
            #else
            tempGraph.push_back(layerNodes[bestRP]);
            #endif
       
            lowestDistCalcs += subLowestDistCalcs;
        }

        numNodes = 0; 
        for(unsigned int j = 0; j < tempGraph.size(); j++){
            numNodes += tempGraph[j].size();
        }

        subGraph = tempGraph;


        // printf("Layer %d Selecting RP %d with Nodes: %u and calcs: %llu\n", i, bestRP, numNodes, lowestDistCalcs);

        
    
        double actualNodeTime = numNodes*1.0 / nodePerSecond + (cT2-cT1);

        printf("Predicted time: %f, Actual Time: %f\n", predictedNodeTime, actualNodeTime);

        // newNodes = layerNodes[bestRP];
        
        // subGraph = genSubGraphs(newNodes);

        //lowestDistCalcs*LAYER_DIFF > previousDistCalcs 

        double calcRatio = lowestDistCalcs*1.0/ previousDistCalcs;
        double nodeRatio = numNodes/ numPreviousNodes;
        calcTime = lowestDistCalcs*1.0 / calcsPerSecond;
        double timeReduction = previousCalcTime - calcTime;
        predictedNodeTime = numNodes*1.0 / nodePerSecond + (cT2-cT1);

        // double calcsPerSecondDyn = calcsPerSecond;//numPoints / calcTime;
        printf("Build Time: %f, Calc Time: %f, reduction %f\n############################################\n", predictedNodeTime, calcTime, timeReduction);
        // if(i > MINRP && ( newNodes.size()*1.0 / nodePerSecond *10> lowestDistCalcs*1.0 / calcsPerSecondDyn || i >= MAXRP)){ 
        //if(i >= MINRP && ( predictedNodeTime > timeReduction || i >= MAXRP)){ 
        // if(i >= MINRP &&  (1.0*lowestDistCalcs / numPoints / numPoints < 0.5 || calcRatio > 0.99) && ( predictedNodeTime > timeReduction || i >= MAXRP)){ 
        if( i >= MAXRP){
            // printf("\nPrevious Calcs: %llu, Current: %llu, ratio: %f\n", previousDistCalcs, lowestDistCalcs, calcRatio);
            // printf("Previous Nodes: %u, Current Nodes: %u, ratio: %f\n",numPreviousNodes, newNodes.size(), nodeRatio);
            numSplits = i+1;
            break;
        }else{
            numSplits = i+1;
            previousCalcTime = calcTime;
            previousDistCalcs = lowestDistCalcs;
            numPreviousNodes = numNodes;
        }
    }




    //linearize the sub graphs

    #if SUBG
    unsigned int nodeCounter = 0;
    for(unsigned int i = 0; i < subGraph.size(); i++){
        for(unsigned int j = 0; j < subGraph[i].size(); j++){
            newNodes.push_back(subGraph[i][j]);
            newNodes[nodeCounter].nodeIndex = nodeCounter;
            subGraph[i][j].nodeIndex = nodeCounter;
            nodeCounter++;
        }
    }

    nodeCounter = 0;
    unsigned int largestSub = subGraph[0].size();
    for(unsigned int i = 0; i < subGraph.size(); i++){
        if(subGraph[i].size() > largestSub) largestSub = subGraph[i].size();
        for(unsigned int j = 0; j < subGraph[i].size(); j++){
            for(unsigned int k = 0; k < subGraph[i][j].neighborIndex.size(); k++){
                newNodes[nodeCounter].neighborIndex[k] = subGraph[i][subGraph[i][j].neighborIndex[k]].nodeIndex;
            }
            nodeCounter++;
        }
    }

        // printf("check: %llu\n", newNodes[0].numCalcs);
        unsigned long long numCalcs = totalNodeCalcs(newNodes, newNodes.size());
        // unsigned long long sumSqrs = nodeSumSqrs(newNodes, newNodes.size());
    
        printf("Final graph has %u nodes, %u subgraphs, largest sub: %u, %llu calcs total\n", numNodes,subGraph.size(), largestSub, numCalcs);
    
        if(ERRORPRINT) fprintf(stderr,"%u %u %u %u %llu ", subGraph.size(), largestSub, numSplits, numNodes, numCalcs);

    #else
        newNodes = subGraph[0];

        unsigned long long numCalcs = totalNodeCalcs(newNodes, newNodes.size());
        // unsigned long long sumSqrs = nodeSumSqrs(newNodes, newNodes.size());

        printf("Final graph has %u nodes, %llu calcs total\n", numNodes, numCalcs);

        if(ERRORPRINT) fprintf(stderr,"%u %u %llu ", numSplits, numNodes, numCalcs);
    #endif

    // printf("check: %llu\n", newNodes[0].numCalcs);



    //rearange the pointArray
    unsigned int counter = 0;
    for(unsigned int i = 0; i < newNodes.size(); i++){
        //append own index to neighbors
        newNodes[i].pointOffset = counter;
        for(unsigned int j = 0; j < newNodes[i].numNodePoints; j++){
            pointArray[counter] = newNodes[i].nodePoints[j];
            counter++;
        }
    }

    *outNodes = newNodes;

    cudaFree(d_data);
    cudaFree(d_dim);
    cudaFree(d_numPoints);
    cudaFree(d_epsilon);

    return numNodes;

}

unsigned int initNodes(double * data,
                        unsigned int dim,
                        unsigned int numPoints,
                        double epsilon,
                        unsigned int * binNumber,
                        unsigned int * pointArray,
                        std::vector<struct Node> * nodes,
                        struct DevicePointers devicePointers,
                        double * calcTime){


    cudaSetDevice(CUDA_DEVICE);
    std::vector<struct Node> newNodes;

    thrust::sort_by_key(thrust::omp::par, binNumber, binNumber+numPoints, pointArray);



    //if all the points are in the same bin
    if(binNumber[0] == binNumber[numPoints-1]){

        newNodes.push_back(newNode(numPoints, pointArray, binNumber[0], 0));

        updateNodeCalcs(&newNodes, newNodes.size());
        //free temp memory
        // free(binNumber);

        *nodes = newNodes;
        //go to the next node
        return 1;
    }

    //go through and make nodes

    //variable to keep track of last bin end
    unsigned int tempBinPointer = 0;

    //variable to count new nodes
    unsigned int numNewNodes = 0;

    unsigned int bcounter = 0;
    //scan through and create a new node for each non-empty bin
    for(unsigned int i = 0; i < numPoints; i++){
        bcounter++;
        //check if need to make a new node
        if(i == numPoints-1 || binNumber[i] != binNumber[i+1]){

            //push back the new node onto the temporary vector of nodes
            newNodes.push_back( newNode(i-tempBinPointer+1, pointArray+tempBinPointer, binNumber[i], numNewNodes ) );
            tempBinPointer = i+1;
            numNewNodes++;
            bcounter = 0;
        }
    }

    //create the connections
            //now that the split nodes exist, modify neighbor values
    //special case for the first
    if(newNodes[0].binNumbers.back() == newNodes[1].binNumbers.back() - 1 ){ //already know at least 2 nodes in list
        newNodes[0].neighborIndex.push_back(1);
    } 

    // handle all middle nodes
    for(unsigned int i = 1; i < numNewNodes-1; i++){
        //check if lower bin is one away
        if(newNodes[i].binNumbers.back() == newNodes[i-1].binNumbers.back() + 1){
            newNodes[i].neighborIndex.push_back(i-1);
        }
        //check if upper bin is one away
        if(newNodes[i].binNumbers.back() == newNodes[i+1].binNumbers.back() - 1){
            newNodes[i].neighborIndex.push_back(i+1);
        }
    }

    //special case for last node
    if(newNodes[numNewNodes-1].binNumbers.back() == newNodes[numNewNodes-2].binNumbers.back() + 1 ){ //already know at least 2 nodes in list
        newNodes[numNewNodes-1].neighborIndex.push_back(numNewNodes-2);
    }


    updateNodeCalcs(&newNodes, newNodes.size());
    //assign the vector to the return
 
    // printf("check: %llu\n", newNodes[0].numCalcs);

    *nodes = newNodes;
    
    // free(binNumber);
  
    return numNewNodes;

}

//splits a node based on a reference point and return the number of new nodes
unsigned int splitNodes(unsigned int * allBinNumbers, //the reference point used for the split
                    std::vector<struct Node> nodes,// the array of nodes
                    unsigned int numNodes,//the number of nodes
                    double epsilon, //the distance threshold of the search
                    double * data, //the dataset
                    unsigned int dim,//the number of dimensions of the data
                    unsigned int numPoints,// number of points in the dataset
                    std::vector<struct Node> * newNodes,
                    struct DevicePointers devicePointers,
                    double * nodePerSecond){  // pointer for returning the new nodes
    
    cudaSetDevice(CUDA_DEVICE);
    double time1 = omp_get_wtime();

    // printf("Start split\n");
    //need to keep track of all of the new split nodes
    std::vector<std::vector<struct Node>> tempNewNodes;
    tempNewNodes.resize(numNodes);

    // printf("allocated vec for %d nodes\n", numNodes);
    // go through each node and split
    for(unsigned int i = 0; i < numNodes; i++){

        #if USE_MIN_CALC
        if(nodes[i].numCalcs <  MIN_CALCS_PER_NODE/*MIN_CALCS_SIZE */ ){
        #else
        if(nodes[i].numNodePoints <  MINBINSIZE/*MIN_NODE_SIZE */ ){

        #endif
            std::vector<struct Node> tempNodes;
            tempNodes.push_back(newNode(nodes[i].numNodePoints, &(nodes[i].nodePoints[0]), nodes[i], (unsigned int) -1, 0 ) );
            tempNodes[0].split = false;
            tempNewNodes[i] = tempNodes; 
            continue;
        }

        // printf("For node %d, starting binning\n", i);
        //temp array to hold each points new bin number
        unsigned int * binNumber = (unsigned int * )malloc(sizeof(unsigned int)*nodes[i].numNodePoints);


        // break nodes into new nodes
        // #pragma omp parallel for
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            //get distance of each point in the node to the reference point
            binNumber[j] = allBinNumbers[nodes[i].nodePoints[j]];

        }

        // printf("finished binning\n");


        // sort the node points based on their bin numbers
        thrust::sort_by_key(thrust::omp::par, &binNumber[0], &binNumber[nodes[i].numNodePoints], &nodes[i].nodePoints[0]);
        
        // printf("finished sorting\n");
        
        //temp vector to hold new nodes
        std::vector<struct Node> tempNodes;


        //if all the points are in the same bin
        if(binNumber[0] == binNumber[nodes[i].numNodePoints-1]){
            // printf("no splits\n");
            //add the bin number
            // nodes[i].binNumbers.push_back(binNumber[0]);
            tempNodes.push_back(newNode(nodes[i].numNodePoints, &(nodes[i].nodePoints[0]), nodes[i], binNumber[0], 0 ) );
            tempNewNodes[i] = tempNodes;
            //free temp memory
            free(binNumber);

            //go to the next node
            continue;
        }

        // printf("finished same bin Check\n");




        //variable to keep track of last bin end
        unsigned int tempBinPointer = 0;

        //variable to count new nodes
        unsigned int numNewNodes = 0;

        unsigned int bcounter = 0;

        //scan through and create a new node for each non-empty bin
        // printf("NumNodePoints for node %u: %u\n", i,  nodes[i].numNodePoints);
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            bcounter++;
            
            //check if need to make a new node
            if(j == nodes[i].numNodePoints-1 || binNumber[j] != binNumber[j+1]){
                
                tempNodes.push_back( newNode(j-tempBinPointer+1, &(nodes[i].nodePoints[0]) + tempBinPointer, nodes[i], binNumber[j], numNewNodes ) );
                tempBinPointer = j+1;
                numNewNodes++;
                bcounter = 0;
            }
        }

        // printf("finished creating new nodes for non empty bins: new nodes : %u\n", numNewNodes);

        //now that the split nodes exist, modify neighbor values
        //special case for the first
        if(tempNodes[0].binNumbers.back() == tempNodes[1].binNumbers.back() - 1 ){ //already know at least 2 nodes in list
            tempNodes[0].neighborIndex.push_back(1);
        }

        // handle all middle nodes
        for(unsigned int j = 1; j < numNewNodes-1; j++){
            //check if lower bin is one away
            if(tempNodes[j].binNumbers.back() == tempNodes[j-1].binNumbers.back() + 1){
                tempNodes[j].neighborIndex.push_back(j-1);
            }
            //check if upper bin is one away
            if(tempNodes[j].binNumbers.back() == tempNodes[j+1].binNumbers.back() - 1){
                tempNodes[j].neighborIndex.push_back(j+1);
            }
        }

        //special case for last node
        if(tempNodes[numNewNodes-1].binNumbers.back() == tempNodes[numNewNodes-2].binNumbers.back() + 1 ){ //already know at least 2 nodes in list
            tempNodes[numNewNodes-1].neighborIndex.push_back(numNewNodes-2);
        }

        
        //copy the temp nodes back into the larger vector of nodes
        tempNewNodes[i]  = tempNodes;
        free(binNumber);

    }
    // printf("finished intital splitting\n");

    ////////////////////////////////////////////////////////////////
    // All nodes in the temp nodes now point to their neighbors within the smaller array
    // the neighbor index values will have to be updated when merging with the other vectors
    /////////////////////////////////////////////////////////////

    //need to make linear index ids now
    //a variable for the index offsets
    unsigned int nodeIndexOffset = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){
            tempNewNodes[i][j].nodeIndex += nodeIndexOffset;
            for(unsigned int k = 0; k < tempNewNodes[i][j].neighborIndex.size(); k++){
                tempNewNodes[i][j].neighborIndex[k] += nodeIndexOffset;
            }
        }
        nodeIndexOffset += tempNewNodes[i].size();
    }

    updateNeighbors(nodes,&tempNewNodes);

    // printf("finsihed updating neighbors\n");


    //make a new linear array of nodes
    std::vector<struct Node> nodeVec;
    for(unsigned int i = 0; i < tempNewNodes.size(); i++){
        nodeVec.insert(nodeVec.end(), tempNewNodes[i].begin(), tempNewNodes[i].end());
        // for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){
        //     nodeVec.push_back(tempNewNodes[i][j]);
        // }
    }

    updateNodeCalcs(&nodeVec, nodeVec.size());

    // printf("NumNodes: %u, TotalCalcs: %llu\n", (unsigned int)nodeVec.size(), totalNodeCalcs(nodeVec, nodeVec.size()));

    *newNodes = nodeVec;


    double time2 = omp_get_wtime();

    *nodePerSecond = numNodes/(time2-time1);

    return (unsigned int)nodeVec.size();

}


struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
                    unsigned int * nodePoints, // the start of the points that will go into the node
                    struct Node parent, //the parent node
                    unsigned int binNumber,//the bin number of the node
                    unsigned int nodeNumber){ //the index number of the node

    struct Node newNode;
    newNode.nodeIndex = nodeNumber;
    newNode.numNodePoints = numNodePoints;
    // newNode.binNumbers = parent.binNumbers;
    for(unsigned int i = 0; i < parent.binNumbers.size();i++){
        newNode.binNumbers.push_back(parent.binNumbers[i]);
    }
    newNode.binNumbers.push_back(binNumber);
    newNode.neighborIndex.push_back(nodeNumber);
    // newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1] ); //double check this
    newNode.nodePoints.resize(numNodePoints);
    for(unsigned int i = 0; i < numNodePoints; i++){
        newNode.nodePoints[i] = nodePoints[i];
    }
    return newNode;
}; 

struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
                    unsigned int * nodePoints, // the start of the points that will go into the node
                    unsigned int binNumber,//the bin number of the node
                    unsigned int nodeNumber){ //the index number of the node

    struct Node newNode;
    newNode.nodeIndex = nodeNumber;
    newNode.numNodePoints = numNodePoints;
    newNode.binNumbers.push_back(binNumber);
    newNode.neighborIndex.push_back(nodeNumber);
    // newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1]); //double check this
    newNode.nodePoints.resize(numNodePoints);
    for(unsigned int i = 0; i < numNodePoints; i++){
        newNode.nodePoints[i] = nodePoints[i];
    }
    return newNode;
};

void updateNodeCalcs(std::vector<struct Node> * nodes,
                     unsigned int numNodes){

    bool verboseNodeInfo = false;
    #pragma omp parallel for
    for(unsigned int i = 0; i < numNodes; i++){
        if(verboseNodeInfo) printf("Node %d has:\n",i);
        unsigned long long int numNeighboringPoints = 0;
        if(verboseNodeInfo) printf("    %u points\n", (*nodes)[i].numNodePoints);
        if(verboseNodeInfo) printf("    %lu neighbors\n", (*nodes)[i].neighborIndex.size());
        for(unsigned int j = 0; j < (*nodes)[i].neighborIndex.size(); j++){
            if(verboseNodeInfo) printf("    neighbors bin: %d with numPoints: %u\n", (*nodes)[i].neighborIndex[j],(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints);
            numNeighboringPoints += (unsigned long long int)(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints;
        }
        (*nodes)[i].numCalcs = (unsigned long long int)numNeighboringPoints*(*nodes)[i].numNodePoints;
        if(verboseNodeInfo) printf("    %llu total calcs to make\n", (*nodes)[i].numCalcs);
    }

}

unsigned long long totalNodeCalcs(std::vector<struct Node> nodes, unsigned int numNodes){
    unsigned long long totalCalcs = 0;
    
    for(unsigned int i = 0; i < nodes.size(); i++){
        
        totalCalcs += nodes[i].numCalcs;
    }

    return totalCalcs;
}

unsigned long long nodeSumSqrs(std::vector<struct Node> nodes, unsigned int numNodes){
    unsigned long long sumSqrs = 0;
    
    for(unsigned int i = 0; i < nodes.size(); i++){
        sumSqrs += nodes[i].numNodePoints*nodes[i].numNodePoints;
    }

    return sumSqrs;
}


void updateNeighbors(std::vector<struct Node> nodes, std::vector<std::vector<struct Node>> * newNodes){
        // go through the old node list and compare bins to adjacent splits of old nodes
    #pragma omp parallel for //num_threads(64/rPPerLayer)
    for(unsigned int i = 0; i < nodes.size(); i++){ //for each old node
        for(unsigned int j = 0; j < (*newNodes)[i].size(); j++){ //go through each split off node
            // bin number that we are looking for adjacents to
            unsigned int nodeBinNumber = (*newNodes)[i][j].binNumbers.back();
            for(unsigned int k = 1; k < nodes[i].neighborIndex.size(); k++){ //and check the neighbors
                //neighbor to check
                unsigned int neighborNodesIndex = nodes[i].neighborIndex[k]; // this will also give the index of the vector of split nodes
                if(neighborNodesIndex > (*newNodes).size()-1) printf("ERROR: neighbor index: %u max: %u\n", neighborNodesIndex,(*newNodes).size()-1 );
                //go through each neighbors split nodes
                for(unsigned int l = 0; l < (*newNodes)[neighborNodesIndex].size(); l++){
                    if((*newNodes)[neighborNodesIndex][l].split == false ||
                       (*newNodes)[i][j].split == false) {
                            (*newNodes)[i][j].neighborIndex.push_back((*newNodes)[neighborNodesIndex][l].nodeIndex);
                    }else{
                        unsigned int checkBin = (*newNodes)[neighborNodesIndex][l].binNumbers.back();
                        if( checkBin + 1 == nodeBinNumber || 
                            checkBin - 1 == nodeBinNumber ||
                            checkBin == nodeBinNumber){
                                (*newNodes)[i][j].neighborIndex.push_back((*newNodes)[neighborNodesIndex][l].nodeIndex);
                        }
                    }
                }
            }
        }
    }
}

unsigned long long nodeForce(std::vector<struct Node> * nodes, double epsilon, double * data, unsigned int dim, unsigned int numPoints){

    std::vector<unsigned int> PA;
    std::vector<unsigned int> PB;
    
    bool check1 = true;
    bool check2 = true;
    bool verboseNodeInfo = true;
    // #pragma omp parallel for
    if(verboseNodeInfo)printf("\n************************************************\n");
    if(verboseNodeInfo)printf("Number of Nodes: %lu\n", (*nodes).size());
    for(unsigned int i = 0; i < (*nodes).size();i++){
        if(verboseNodeInfo) printf("Node: %u\n  numNeighbors: %lu\n", i, (*nodes)[i].neighborIndex.size());
        unsigned long long sum = 0;
        for(unsigned int j = 0; j < (*nodes)[i].neighborIndex.size();j++){
            if(verboseNodeInfo) printf("    neighbor at index: %u\n", (*nodes)[i].neighborIndex[j]);
            if(verboseNodeInfo) printf("        numPoints: %u\n", (*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints);
            if(verboseNodeInfo){
                printf("        Bin Numbers: |");
                for(unsigned int h = 0; h <  (*nodes)[(*nodes)[i].neighborIndex[j]].binNumbers.size(); h++){
                    printf(" %u |", (*nodes)[(*nodes)[i].neighborIndex[j]].binNumbers[h]);
                }
                printf("\n");
            }
            for(unsigned int k = 0; k < (*nodes)[i].numNodePoints; k++){
                for(unsigned int l = 0; l <(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints; l++){
                    unsigned int a = (*nodes)[i].nodePoints[k];
                    unsigned int b = (*nodes)[(*nodes)[i].neighborIndex[j]].nodePoints[l];
                    if(a == 50 && check1) {
                        printf("                 point 50 Node Number: %u\n", i);
                        check1 = false;
                    }
                    if(a == 53 && check2) {
                        printf("                 point 53 Node Number: %u\n", i);
                        check2 = false;
                    }
                    if(verboseNodeInfo) if(b == a && j != 0) printf("ERROR:%u\n",a);
                    double running = 0;
                    for(unsigned int d = 0; d < dim; d++){
                        running += pow(data[a*dim + d] - data[b*dim + d], 2);
                    }
                    if(running <= epsilon*epsilon){
                        sum++;
                        PA.push_back(a);
                        PB.push_back(b);
                    }
                }

            }
        }
        (*nodes)[i].numResults = sum;
    }

    unsigned long long total = 0;
    for(unsigned int i = 0; i < (*nodes).size(); i++){
        total += (*nodes)[i].numResults;
    }
    // if(verboseNodeInfo)printf("TOTAL PAIRS: %llu\n", total);

    std::vector<unsigned int> BA;
    std::vector<unsigned int> BB;

    unsigned int brute_count = 0;
	omp_lock_t brute;
	omp_init_lock(&brute);

	#pragma omp parallel for
	for(unsigned int i = 0; i < numPoints; i++)
	{
		for (unsigned int j = 0; j < numPoints; j++)
		{
		double distance = 0;
			for (unsigned int k = 0; k < dim; k++)
			{
				if(distance > epsilon*epsilon)
				{
					break;
				} else {
					double a1 = data[i*dim + k];
					double a2 = data[j*dim + k];
					distance += (a1-a2)*(a1-a2);
				}
				}
				if(distance <= epsilon*epsilon){
					omp_set_lock(&brute);
					brute_count++;
                    BA.push_back(i);
                    BB.push_back(j);
					omp_unset_lock(&brute);
				}
		}
	}

    // printf("BrUTe count: %u", brute_count);

    std::vector< std::pair<unsigned int,unsigned int>> pairsB;

    for(unsigned int i = 0; i < BA.size();i++){
        pairsB.push_back(std::make_pair(BA[i],BB[i]));
    }

    std::sort(pairsB.begin(), pairsB.end(), compPair);

    std::vector< std::pair<unsigned int,unsigned int>> pairs;

    for(unsigned int i = 0; i < PA.size();i++){
        pairs.push_back(std::make_pair(PA[i],PB[i]));
    }

    std::sort(pairs.begin(), pairs.end(), compPair);

    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    
    std::vector< std::pair<unsigned int,unsigned int>> missing;

    // std::set_difference(pairsB.begin(), pairsB.end(), pairs.begin(), pairs.end(), std::inserter(missing, missing.begin()));
    unsigned int counter = 0;
    for(unsigned int i = 0; i < pairsB.size();i++){
        if(pairsB[i].first == pairs[counter].first && pairsB[i].second == pairs[counter].second){
            counter++;
        }else{
            missing.push_back(pairsB[i]);
        }
    }

    printf("Missing %lu pairs:\n", missing.size());
    for(unsigned int i = 0; i < missing.size(); i++){
        printf("(%u,%u),",missing[i].first, missing[i].second);
    }

    printf("\nTotal NodeFORCE results Set Size: %llu , unique pairs: %lu\n", total, pairs.size());

    if(verboseNodeInfo)printf("\n************************************************\n");


    return total;
}

std::vector<std::vector<struct Node>> genSubGraphs(std::vector<struct Node> inNodes){


    //pull out all of the sub graphs
    // printf("Generating subs\n");
    std::vector<struct Node> nodes;
    nodes.insert(nodes.begin(), inNodes.begin(), inNodes.end());
    std::vector<std::vector<struct Node>> subGraphs;
    std::stack<unsigned int> stack;
    for(unsigned int i = 0; i < nodes.size(); i++){
        if(nodes[i].visited == false){
            std::vector<struct Node> newGraph;
            newGraph.push_back(nodes[i]);
            nodes[i].visited = true;
            stack.push(nodes[i].nodeIndex);

            while(stack.size() > 0){
                unsigned int current = stack.top(); 
                stack.pop();
                for(unsigned int j = 0; j < nodes[current].neighborIndex.size();j++){
                    if(nodes[nodes[current].neighborIndex[j]].visited == false){
                        stack.push(nodes[current].neighborIndex[j]);
                        newGraph.push_back(nodes[nodes[current].neighborIndex[j]]);
                        nodes[nodes[current].neighborIndex[j]].visited = true;
                    }
                }
                
            }

            subGraphs.push_back(newGraph);
        }
         
    }

    //fix neighbor pointers to be local to sub graphs
    // unsigned int offsetCounter = 0;
    for(unsigned int i = 0; i < subGraphs.size();i++){
        // printf("\n#####################\nOffset counter for sub %u: %u  :   : %u nodes in sub\n", i, offsetCounter, subGraphs[i].size());
        for(unsigned int j = 0; j < subGraphs[i].size(); j++){
            nodes[subGraphs[i][j].nodeIndex].nodeIndex = j;
            subGraphs[i][j].nodeIndex = j;
            // printf("sub/node: %u::%u node index: %u\n   bins:", i,j, subGraphs[i][j].nodeIndex);
            // for(unsigned int b = 0; b < subGraphs[i][j].binNumbers.size(); b++){
            //     printf("%u, ",subGraphs[i][j].binNumbers[b] );
            // }
            // printf("\n  -------- \n");

            // subGraphs[i][j].neighborIndex.clear();
        }

        // for(unsigned int j = 0; j < subGraphs[i].size(); j++){
        //     unsigned int nodeBinNumber = subGraphs[i][j].binNumbers.back();
        //     for(unsigned int k = 0; k < subGraphs[i].size(); k++){
        //         if(subGraphs[i][k].split == false ||
        //                 subGraphs[i][j].split == false) {
        //                     subGraphs[i][j].neighborIndex.push_back(subGraphs[i][k].nodeIndex);
        //         }else{
        //             unsigned int checkBin = subGraphs[i][k].binNumbers.back();
        //             if( checkBin + 1 == nodeBinNumber || 
        //                 checkBin - 1 == nodeBinNumber ||
        //                 checkBin == nodeBinNumber){
        //                     subGraphs[i][j].neighborIndex.push_back(subGraphs[i][k].nodeIndex);
        //             }
        //         }
        //     }
        for(unsigned int j = 0; j < subGraphs[i].size(); j++){
            for(unsigned int k = 0; k < subGraphs[i][j].neighborIndex.size(); k++){
                
                subGraphs[i][j].neighborIndex[k] = nodes[subGraphs[i][j].neighborIndex[k]].nodeIndex;

                // printf("    sub: %u, node: %u at index %u: %u\n     bins:", i, j,k, subGraphs[i][j].neighborIndex[k]);
                // for(unsigned int b = 0; b < subGraphs[i][j].binNumbers.size(); b++){
                //     printf("%u, ",subGraphs[i][ subGraphs[i][j].neighborIndex[k] ].binNumbers[b] );
                // }
                // printf("\n");
                
            }
        }
        
        
        // offsetCounter += subGraphs[i].size();
    }
    
    printf("Num SubGraphs: %u\n", subGraphs.size());

    return subGraphs;

}