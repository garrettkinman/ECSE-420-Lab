#ifndef CUDACC
#define CUDACC
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

/*
helper functions to read and write data
*/
int read_input_one_two_four(int** input1, char* filepath);
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath);
void write_data(int* data, int length, char* filepath);

__device__ int numNextLevelNodes = 0;
__device__ int nextLevelNodesQueue[5000000];

/*
helper device function to solve a given logic gate and inputs
*/
__device__ int gate_solver(int gate, int x1, int x2) {
    switch (gate) {
    case AND:
        return x1 && x2;
    case OR:
        return x1 || x2;
    case NAND:
        return !(x1 && x2);
    case NOR:
        return !(x1 || x2);
    case XOR:
        return (x1 || x2) && !(x1 && x2);
    case XNOR:
        return (x1 && x2) || (!x1 && !x2);
    }
}

__global__ void block_queuing_kernel(int numCurrLevelNodes, int* currLevelNodes, int* nodeNeighbors, int* nodePtrs, int* nodeVisited, int* nodeInput, int* nodeOutput, int* nodeGate, int queueSize){
    
    // initialize shared memory queue
    extern __shared__ int sharedBlockQueue[];
    __shared__ int sharedBlockQueueSize, blockGlobalQueueIdx;

    if (threadIdx.x == 0)
        sharedBlockQueueSize = 0; 

    __syncthreads();

    int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);

    // Loop over all nodes in current level
    for (int id = threadIndex; id < numCurrLevelNodes; id++) {
        int nodeIdx = currLevelNodes[id];      

        // Loop over all neighbors of the node
        for (int nId = nodePtrs[nodeIdx]; nId < nodePtrs[nodeIdx+1]; nId++) {          
            int neighborIdx = nodeNeighbors[nId];

            // If the neighbor has not been visited yet
            const int visited = atomicExch(&(nodeVisited[neighborIdx]), 1);
            if (!(visited)) {
                const int queueIdx = atomicAdd(&sharedBlockQueueSize, 1);

                nodeOutput[neighborIdx] = gate_solver(nodeGate[neighborIdx], nodeOutput[nodeIdx], nodeInput[neighborIdx]);

                // add to block queue if not full
                // else, add to global queue
                if (queueIdx < queueSize)
                    sharedBlockQueue[queueIdx] = neighborIdx;                  
                else {
                    sharedBlockQueueSize = queueSize;
                    const int GlIdx = atomicAdd(&numNextLevelNodes, 1);
                    nextLevelNodesQueue[GlIdx] = neighborIdx; 
                }
            }      
        }
    }
    
    __syncthreads();

    if (threadIdx.x == 0)
        blockGlobalQueueIdx = atomicAdd(&numNextLevelNodes, sharedBlockQueueSize);
  
    __syncthreads();

    // store block queue in global queue
    for (int i = threadIdx.x; i < sharedBlockQueueSize; i += blockDim.x)
        nextLevelNodesQueue[blockGlobalQueueIdx + i] = sharedBlockQueue[i];
}

int main(int argc, char *argv[]){

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~
    
    if (argc < 10) {
        printf("Require parameters in the following order: <numBlock> <blockSize> <sharedQueueSize> <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>.\n");
        exit(1);
    }

    int numNodePtrs;
    int numNodes;
    int *nodePtrs_h;
    int *nodeNeighbors_h;
    int *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;
    int numNextLevelNodes_h = 0;
    int *nodeGate_h;
    int *nodeInput_h;
    int *nodeOutput_h;

    const int blockSize = atoi(argv[1]);
    const int numBlocks = atoi(argv[2]);
    const int queueSize = atoi(argv[3]);

    char* input1 = argv[4];
    char* input2 = argv[5];
    char* input3 = argv[6];
    char* input4 = argv[7];

    char* nodeOutputFilename = argv[8];
    char* nextLevelNodesFilename = argv[9];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in inputs from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input2);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input3);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input4);

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: allocate for GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    // outputs
    int *nextLevelNodes_h = (int *)malloc(numNodes*sizeof(int));
    int *nextLevelNodes_cuda = (int *)malloc(numNodes*sizeof(int));
    cudaMalloc (&nextLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(nextLevelNodes_cuda, nextLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
  
    // copy to device
    int numNodesSize = numNodes * sizeof(int);
    int* currLevelNodes_cuda = (int*)malloc(numCurrLevelNodes * sizeof(int)) ; 
    cudaMalloc (&currLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(currLevelNodes_cuda, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeNeighbors_cuda = (int*)malloc(numTotalNeighbors_h * sizeof(int)) ; 
    cudaMalloc (&nodeNeighbors_cuda, numTotalNeighbors_h * sizeof(int));
    cudaMemcpy(nodeNeighbors_cuda, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);

    int* nodePtrs_cuda = (int*)malloc(numNodePtrs * sizeof(int)) ; 
    cudaMalloc (&nodePtrs_cuda, numNodePtrs * sizeof(int));
    cudaMemcpy(nodePtrs_cuda, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeVisited_cuda = (int*)malloc(numNodesSize) ; 
    cudaMalloc (&nodeVisited_cuda, numNodesSize);
    cudaMemcpy(nodeVisited_cuda, nodeVisited_h,numNodesSize, cudaMemcpyHostToDevice);

    int* nodeInput_cuda = (int*)malloc(numNodesSize) ; 
    cudaMalloc (&nodeInput_cuda, numNodesSize);
    cudaMemcpy(nodeInput_cuda, nodeInput_h, numNodesSize, cudaMemcpyHostToDevice);

    int* nodeOutput_cuda = (int*)malloc(numNodesSize) ; 
    cudaMalloc (&nodeOutput_cuda, numNodesSize);
    cudaMemcpy(nodeOutput_cuda, nodeOutput_h, numNodesSize, cudaMemcpyHostToDevice);

    int* nodeGate_cuda = (int*)malloc(numNodesSize) ; 
    cudaMalloc (&nodeGate_cuda, numNodesSize);
    cudaMemcpy(nodeGate_cuda, nodeGate_h, numNodesSize, cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 4: time parallel execution
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);

    block_queuing_kernel<<<numBlocks, blockSize, queueSize*sizeof(int)>>>(numCurrLevelNodes, currLevelNodes_cuda, nodeNeighbors_cuda, nodePtrs_cuda, nodeVisited_cuda, nodeInput_cuda, nodeOutput_cuda, nodeGate_cuda, queueSize);
    cudaDeviceSynchronize();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    float timeGPU;
    cudaEventElapsedTime(&timeGPU, startGPU, stopGPU);

    printf("Block Queue: %.6f ms\n", timeGPU);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 5: write to file and done!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int* outputBuffer;
    outputBuffer = (int*)malloc(numNodesSize);
    cudaMemcpy(outputBuffer, nodeOutput_cuda, numNodesSize, cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(&numNextLevelNodes_h, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(nextLevelNodes_h, nextLevelNodesQueue, numNextLevelNodes_h * sizeof(int), 0, cudaMemcpyDeviceToHost);

    write_data(outputBuffer, numNodes, nodeOutputFilename);
    write_data(nextLevelNodes_h, numNextLevelNodes_h, nextLevelNodesFilename);

    // ~~~~~~~~~~~~~~~~~~~~~
    // step 6: free at last!
    // ~~~~~~~~~~~~~~~~~~~~~

    free(outputBuffer);
    free(nextLevelNodes_h);
    cudaFree(nextLevelNodes_cuda);
    cudaFree(currLevelNodes_cuda);
    cudaFree(nodeNeighbors_cuda);
    cudaFree(nodePtrs_cuda);
    cudaFree(nodeVisited_cuda);
    cudaFree(nodeInput_cuda);
    cudaFree(nodeOutput_cuda);
    cudaFree(nodeGate_cuda);
}

int read_input_one_two_four(int** input1, char* filepath) {
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = (int*)malloc(len * sizeof(int));

    int temp1;

    while (fscanf(fp, "%d", &temp1) == 1) {
        (*input1)[counter] = temp1;
        counter++;
    }

    fclose(fp);
    return len;
}

int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath) {
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = (int*)malloc(len * sizeof(int));
    *input2 = (int*)malloc(len * sizeof(int));
    *input3 = (int*)malloc(len * sizeof(int));
    *input4 = (int*)malloc(len * sizeof(int));

    int temp1;
    int temp2;
    int temp3;
    int temp4;
    while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
        (*input1)[counter] = temp1;
        (*input2)[counter] = temp2;
        (*input3)[counter] = temp3;
        (*input4)[counter] = temp4;
        counter++;
    }

    fclose(fp);
    return len;
}

void write_data(int* data, int length, char* filepath) {
    FILE* fp = fopen(filepath, "w");
    fprintf(fp, "%d\n", length);

    for (int i = 0; i < length; i++) {
        fprintf(fp, "%d\n", (data[i]));
    }

    fclose(fp);
}