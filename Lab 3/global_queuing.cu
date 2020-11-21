#ifndef CUDACC
#define CUDACC
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

// Reads the third input file
int read_third_file(char* path_inputfile, int** first_input, int** second_input, int** third_input, int** fourth_input) {
    int size_file;
    int read_value_1; int read_value_2; int read_value_3; int read_value_4;
    FILE* read_this;
    
    //File Open
    if ((read_this = fopen(path_inputfile, "r")) == NULL){
        printf("Error: Cannot Open File!");
        exit(1);
    } 

    //Allocating memory for input variables
    fscanf(read_this, "%d", &size_file);
    *first_input = (int*) malloc(size_file * sizeof(int)); *second_input = (int*) malloc(size_file * sizeof(int));
    *third_input = (int*) malloc(size_file * sizeof(int)); *fourth_input = (int*) malloc(size_file * sizeof(int));

    //Iterating through all four read values
    for (int i = 0; i < size_file; i++) {
        fscanf(read_this, "%d, %d, %d, %d", &read_value_1, &read_value_2, &read_value_3, &read_value_4);
        //Assigning read values to corresponding input variables
        (*first_input)[i] = read_value_1; (*second_input)[i] = read_value_2;
        (*third_input)[i] = read_value_3; (*fourth_input)[i] = read_value_4;
    }

    fclose(read_this);
    return size_file;

}

// Reads in the passed input file
int read_input(int** input_vals, char* path_inputfile) {
    int size_file;
    int read_value;

    FILE* read_this;
    
    //File Open
    if ((read_this = fopen(path_inputfile, "r")) == NULL){
        printf("Error: Cannot Open File!");
        exit(1);
    } 

    fscanf(read_this, "%d", &size_file);
    *input_vals = (int*) malloc(size_file * sizeof(int));

    //Reading input file values
    for (int i = 0; i < size_file; i++) {
        fscanf(read_this, "%d", &read_value);
        (*input_vals)[i] = read_value;
    }

    fclose(read_this);
    return size_file;
}

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


__device__ int globalQueue[7000000];
__device__ int numNextLevelNodes = 0;

__global__ void global_queuing_kernel(int totalThreads, int countNodes, int* nodePtrs, int* currLevelNodes, int* nodeNeighbors, int* nodeVisited, int* nodeGate, int* nodeInput, int* nodeOutput) {
    int nodesPerThread = countNodes / totalThreads;
    int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
    int beginIdx = threadIndex * nodesPerThread;
    //Loop over all nodes in the current level
    for (int id = beginIdx; id < countNodes && id < beginIdx + nodesPerThread; id++) {
        int nodeIdx = currLevelNodes[id];
        //Loop over all neighbors of the node
        for (int secondId = nodePtrs[nodeIdx]; secondId < nodePtrs[nodeIdx+1]; secondId++) {   
            int neighborIdx = nodeNeighbors[secondId];
            //If the neighbor hasn’t been visited yet
            const int visited = atomicExch(&(nodeVisited[neighborIdx]),1);
            if (!visited) {
                nodeOutput[neighborIdx] = gate_solver(nodeGate[neighborIdx], nodeOutput[nodeIdx], nodeInput[neighborIdx]);
                //Add it to the global queue
                const int globalQueueIdx = atomicAdd(&numNextLevelNodes,1); 
                globalQueue[globalQueueIdx] = neighborIdx; 
            }    
        }
         __syncthreads();
    }
}



int main(int argc, char *argv[]){

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    // User Input
    char* input_1 = argv[1]; char* input_2 = argv[2]; char* input_3 = argv[3]; char* input_4 = argv[4];
    char* output_node = argv[5]; char* output_nln = argv[6];
    
    // Declaring Variables
    
    //Node variables
    int countNodes = 0; int* nodePtrs; int numNodePtrs = 0;
    int* nodeInput; int* nodeGate; int* nodeOutput;
    
    //Node Neighbours variables
    int* nodeNeighbors; int countNodeNeighbors = 0; int* nodeVisited; 
    
    //Node level variables
    int* currLevelNodes; int numCurrLevelNodes = 0;  //Current level
    int* nextLevelNodes; int numNextLevelNodes = 0; //Next level

    //Argument Check
    if (argc != 7) {return fprintf(stderr, "Not Enough or Too Many Arguments!\n");}
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in inputs from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    countNodes = read_third_file(input_3, &nodeVisited, &nodeGate, &nodeInput, &nodeOutput);
    numNodePtrs = read_input(&nodePtrs, input_1);
    countNodeNeighbors = read_input(&nodeNeighbors, input_2);
    
    numCurrLevelNodes = read_input(&currLevelNodes, input_4);
    nextLevelNodes = (int *) malloc(countNodes * sizeof(int));
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: allocate for GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    cudaMemcpyToSymbol(globalQueue,nextLevelNodes, countNodes * sizeof(int));
    
    int countNodesSize = countNodes * sizeof(int);
    int numBlocks = 35;
    int blockSize = 128;
    int* nodePtrs_cuda = (int*)malloc( numNodePtrs * sizeof(int)) ; 
    int* currLevelNodes_cuda = (int*)malloc( numCurrLevelNodes * sizeof(int)) ; 
    int* nodeNeighbors_cuda = (int*)malloc( countNodeNeighbors * sizeof(int)) ; 
    int* nodeVisited_cuda = (int*)malloc( countNodesSize) ; 
    int* nodeGate_cuda = (int*)malloc( countNodesSize) ; 
    int* nodeInput_cuda = (int*)malloc( countNodesSize) ; 
    int* nodeOutput_cuda = (int*)malloc(countNodesSize) ; 
    
    // Calling CUDA Functions

    // CUDA: Accessing nodePtrs_cuda
    cudaMalloc (&nodePtrs_cuda, numNodePtrs * sizeof(int));
    cudaMemcpy(nodePtrs_cuda, nodePtrs, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing currLevelNodes_cuda
    cudaMalloc (&currLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(currLevelNodes_cuda, currLevelNodes, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeNeighbors_cuda
    cudaMalloc (&nodeNeighbors_cuda, countNodeNeighbors * sizeof(int));
    cudaMemcpy(nodeNeighbors_cuda, nodeNeighbors, countNodeNeighbors * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeVisited_cuda
    cudaMalloc (&nodeVisited_cuda, countNodesSize);
    cudaMemcpy(nodeVisited_cuda, nodeVisited,countNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeGate_cuda
    cudaMalloc (&nodeGate_cuda, countNodesSize);
    cudaMemcpy(nodeGate_cuda, nodeGate, countNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeInput_cuda
    cudaMalloc (&nodeInput_cuda, countNodesSize);
    cudaMemcpy(nodeInput_cuda, nodeInput, countNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeOutput_cuda
    cudaMalloc (&nodeOutput_cuda, countNodesSize);
    cudaMemcpy(nodeOutput_cuda, nodeOutput, countNodesSize, cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 4: time parallel execution
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);


    // kernel call
    global_queuing_kernel <<< numBlocks, blockSize >>> (blockSize * numBlocks, countNodes, nodePtrs_cuda, currLevelNodes_cuda, nodeNeighbors_cuda, nodeVisited_cuda, nodeGate_cuda, nodeInput_cuda, nodeOutput_cuda);

    cudaDeviceSynchronize();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);


    float timeGPU;
    cudaEventElapsedTime(&timeGPU, startGPU, stopGPU);

    printf("Global Queue: %.6f ms\n", timeGPU);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);
    
    //cudaGetLastError();

    cudaMemcpyFromSymbol(&numNextLevelNodes, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(nextLevelNodes,globalQueue, countNodesSize);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 5: write to file and done!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int *output_buff;
    output_buff = (int*)malloc( countNodesSize); 
    cudaMemcpy(output_buff, nodeOutput_cuda, countNodesSize, cudaMemcpyDeviceToHost);

    // Opening output files
    FILE *outfile_node = fopen(output_node, "w"); FILE *outfile_nln = fopen(output_nln, "w");
    
    // File Check
    if(!outfile_node || !outfile_nln){
        return fprintf(stderr, "Invalid Output Files");
    } 

    //Writing values to output_nodeOutput
    fprintf(outfile_node, "%d\n", countNodes);
    for (int i = 0; i < countNodes; i++) { fprintf(outfile_node, "%d\n", output_buff[i]); }
    fclose(outfile_node);
    
    //Writing values to output_nextLevelNodes
    fprintf(outfile_nln, "%d\n", numNextLevelNodes);
    for (int i = 0; i < numNextLevelNodes; i++) { fprintf(outfile_nln, "%d\n", nextLevelNodes[i]); }
    fclose(outfile_nln);

    // ~~~~~~~~~~~~~~~~~~~~~
    // step 6: free at last!
    // ~~~~~~~~~~~~~~~~~~~~~
    free(nodeGate); free(nodeInput); free(nodeOutput);
    free(nodePtrs); free(currLevelNodes); free(nodeNeighbors); free(nodeVisited);
    cudaFree(currLevelNodes_cuda); cudaFree(nodeNeighbors_cuda); cudaFree(nodePtrs_cuda); 
    cudaFree(nodeVisited_cuda); cudaFree(nodeInput_cuda); cudaFree(nodeOutput_cuda);
    cudaFree(nodeGate_cuda);
}