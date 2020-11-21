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
int read_input(int** input_vals, char* path_inputfile);
int read_third_file(char* path_inputfile, int** first_input, int** second_input, int** third_input, int** fourth_input);
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

    // Iterate through each node in current level
    for (int id = threadIndex; id < numCurrLevelNodes; id++) {
        int nodeIdx = currLevelNodes[id];      
        // Iterate through each neighbor of the node
        for (int nId = nodePtrs[nodeIdx]; nId < nodePtrs[nodeIdx+1]; nId++) {          
            int neighborIdx = nodeNeighbors[nId];
            // If the neighbor is not visited
            const int visited = atomicExch(&(nodeVisited[neighborIdx]), 1);
            if (!(visited)) {
                const int queueIdx = atomicAdd(&sharedBlockQueueSize, 1);
                // Solve Gate
                nodeOutput[neighborIdx] = gate_solver(nodeGate[neighborIdx], nodeOutput[nodeIdx], nodeInput[neighborIdx]);
                // if not full add to block queue 
                if (queueIdx < queueSize){
                  sharedBlockQueue[queueIdx] = neighborIdx;
                }                    
                else { // else, add to global queue
                    sharedBlockQueueSize = queueSize;
                    const int GlIdx = atomicAdd(&numNextLevelNodes, 1);
                    nextLevelNodesQueue[GlIdx] = neighborIdx; 
                }
            }      
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0){
      blockGlobalQueueIdx = atomicAdd(&numNextLevelNodes, sharedBlockQueueSize);
    } 
    __syncthreads();

    // storing block queue in global queue
    for (int i = threadIdx.x; i < sharedBlockQueueSize; i += blockDim.x)
        nextLevelNodesQueue[blockGlobalQueueIdx + i] = sharedBlockQueue[i];
}

int main(int argc, char *argv[]){

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    //User Input 
    const int blockSize = atoi(argv[1]); const int numBlocks = atoi(argv[2]); const int queueSize = atoi(argv[3]);
    char* input_1 = argv[4]; char* input_2 = argv[5]; char* input_3 = argv[6]; char* input_4 = argv[7];
    char* output_node = argv[8]; char* output_nln = argv[9];
    
    // Node Variables
    int countNodes = 0; int numNodePtrs = 0; int *nodePtrs;
    int *nodeGate; int *nodeInput; int *nodeOutput;
    
    // Node Neighbours Variables
    int *nodeNeighbors; int countNodeNeighbors = 0; int *nodeVisited; 

    //Node level variables
    int *currLevelNodes; int numCurrLevelNodes = 0; //Current Level
    int *nextLevelNodes; int numNextLevelNodes = 0; //Next Level


    //Argument Check
    if (argc != 10) {return fprintf(stderr, "Not Enough or Too Many Arguments!\n");}
  
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in inputs from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    countNodes = read_third_file(input_3, &nodeVisited, &nodeGate, &nodeInput, &nodeOutput);
    numNodePtrs = read_input(&nodePtrs, input_1);
    countNodeNeighbors = read_input(&nodeNeighbors, input_2);
    
    numCurrLevelNodes = read_input(&currLevelNodes, input_4);

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: allocate for GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    nextLevelNodes = (int *)malloc(countNodes*sizeof(int));
    int *nextLevelNodes_cuda = (int *)malloc(countNodes*sizeof(int));
    int numNodesSize = countNodes * sizeof(int);
    int* currLevelNodes_cuda = (int*)malloc(numCurrLevelNodes * sizeof(int)); 
    int* nodeNeighbors_cuda = (int*)malloc(countNodeNeighbors * sizeof(int)); 
    int* nodePtrs_cuda = (int*)malloc(numNodePtrs * sizeof(int)) ; 
    int* nodeVisited_cuda = (int*)malloc(numNodesSize);
    int* nodeInput_cuda = (int*)malloc(numNodesSize); 
    int* nodeOutput_cuda = (int*)malloc(numNodesSize); 
    int* nodeGate_cuda = (int*)malloc(numNodesSize);
    
    // Calling CUDA Functions

    // CUDA: Accessing nextLevelNodes_cuda
    cudaMalloc (&nextLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(nextLevelNodes_cuda, nextLevelNodes, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing currLevelNodes_cuda
    cudaMalloc (&currLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(currLevelNodes_cuda, currLevelNodes, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeNeighbors_cuda
    cudaMalloc (&nodeNeighbors_cuda, countNodeNeighbors * sizeof(int));
    cudaMemcpy(nodeNeighbors_cuda, nodeNeighbors, countNodeNeighbors * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing nodePtrs_cuda
    cudaMalloc (&nodePtrs_cuda, numNodePtrs * sizeof(int));
    cudaMemcpy(nodePtrs_cuda, nodePtrs, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeVisited_cuda
    cudaMalloc (&nodeVisited_cuda, numNodesSize);
    cudaMemcpy(nodeVisited_cuda, nodeVisited,numNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeInput_cuda
    cudaMalloc (&nodeInput_cuda, numNodesSize);
    cudaMemcpy(nodeInput_cuda, nodeInput, numNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeOutput_cuda
    cudaMalloc (&nodeOutput_cuda, numNodesSize);
    cudaMemcpy(nodeOutput_cuda, nodeOutput, numNodesSize, cudaMemcpyHostToDevice);
    // CUDA: Accessing nodeGate_cuda
    cudaMalloc (&nodeGate_cuda, numNodesSize);
    cudaMemcpy(nodeGate_cuda, nodeGate, numNodesSize, cudaMemcpyHostToDevice);

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

    int* output_buff;
    output_buff = (int*)malloc(numNodesSize);
    cudaMemcpy(output_buff, nodeOutput_cuda, numNodesSize, cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(&numNextLevelNodes, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(nextLevelNodes, nextLevelNodesQueue, numNextLevelNodes * sizeof(int), 0, cudaMemcpyDeviceToHost);

    // Writing data to output files
    write_data(output_buff, countNodes, output_node); //ouput_node
    write_data(nextLevelNodes, numNextLevelNodes, output_nln); //output_NodeLevelNode

    // ~~~~~~~~~~~~~~~~~~~~~
    // step 6: free at last!
    // ~~~~~~~~~~~~~~~~~~~~~
    free(output_buff); free(nextLevelNodes);
    cudaFree(nextLevelNodes_cuda); cudaFree(currLevelNodes_cuda); cudaFree(nodeNeighbors_cuda); 
    cudaFree(nodePtrs_cuda); cudaFree(nodeVisited_cuda); cudaFree(nodeInput_cuda);
    cudaFree(nodeOutput_cuda); cudaFree(nodeGate_cuda);
}

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

void write_data(int* data, int length, char* filepath) {
    FILE* fp = fopen(filepath, "w");
    fprintf(fp, "%d\n", length);

    for (int i = 0; i < length; i++) {
        fprintf(fp, "%d\n", (data[i]));
    }

    fclose(fp);
}