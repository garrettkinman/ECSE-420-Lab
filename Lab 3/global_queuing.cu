#include <time.h>
#include <stdio.h>
#include <stdlib.h>

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


__device__ int globalQueue[7000000];
__device__ int numNextLevelNodes = 0;

__global__ void global_queuing_kernel(int totalThreads, int countNodes, int* nodePtrs, int* currLevelNodes, int* nodeNeighbors, int* nodeVisited, int* nodeGate, int* nodeInput, int* nodeOutput) {
    
    int nodesPerThread = countNodes / totalThreads;
    int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
    int beginIndex = threadIndex * nodesPerThread;

    //Loop over all nodes in the current level
    for (int index = beginIndex; index < countNodes && index < beginIndex + nodesPerThread; index++) {
        
        int nodeIndex = currLevelNodes[index];

        //Loop over all neighbors of the node
        for (int secondIndex = nodePtrs[nodeIndex]; secondIndex < nodePtrs[nodeIndex+1]; secondIndex++) {   
            
            int neighborIndex = nodeNeighbors[secondIndex];
            const int alreadyVisited = atomicExch(&(nodeVisited[neighborIndex]),1);
            
            //If the neighbor hasn’t been visited yet
            if (!alreadyVisited) {
                
                int result = 0;
                int nInputV = nodeInput[neighborIndex];
                int nOutputV = nodeOutput[nodeIndex];
                int nGateV = nodeGate[neighborIndex];
                
                switch (nGateV) {
                case 0:
                  if (nInputV == 1 && nOutputV == 1) {
                      result = 1;
                  }
                  else {
                      result = 0;
                  }
                  break;
                case 1:
                  if (nInputV == 0 && nOutputV == 0) {
                      result = 0;
                  }
                  else {
                      result = 1;
                  }
                  break;
                case 2:
                  if (nInputV == 1 && nOutputV == 1) {
                      result = 0;
                  } else {
                      result = 1;
                  }
                  break;
                case 3:
                  if (nInputV == 0 && nOutputV == 0) {
                      result = 1; 
                  } else {
                      result = 0;
                  }
                  break;
                case 4:
                  if (nInputV == nOutputV) {
                      result = 0;
                  } else {
                      result = 1;
                  }
                  break;
                case 5:
                  if (nInputV == nOutputV) {
                      result = 1;
                  } else {
                      result = 0;
                  }
                  break;         
                }  

                //Update node output
                nodeOutput[neighborIndex] = result;
                int globalQueueIndex = atomicAdd(&numNextLevelNodes,1); 
               
                //Add it to the global queue
                globalQueue[globalQueueIndex] = neighborIndex; 
            }    
        }
         __syncthreads();
    }
}

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error at runtime %s: %s\n", msg, cudaGetErrorString(err));
  }
  return err;
}

int main(int argc, char *argv[]){
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
    
    countNodes = read_third_file(input_3, &nodeVisited, &nodeGate, &nodeInput, &nodeOutput);
    numNodePtrs = read_input(&nodePtrs, input_1);
    countNodeNeighbors = read_input(&nodeNeighbors, input_2);
    
    numCurrLevelNodes = read_input(&currLevelNodes, input_4);
    nextLevelNodes = (int *) malloc(countNodes * sizeof(int));
    
    checkCudaErr(cudaMemcpyToSymbol(globalQueue,nextLevelNodes, countNodes * sizeof(int)), "Copying");
    
    int countNodesSize = countNodes * sizeof(int);
    int numBlocks = 35;
    int blockSize = 128;
    
    // Cuda variables
    int* nodePtrs_cuda = (int*)malloc( numNodePtrs * sizeof(int)) ; 
    cudaMalloc (&nodePtrs_cuda, numNodePtrs * sizeof(int));
    cudaMemcpy(nodePtrs_cuda, nodePtrs, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);

    int* currLevelNodes_cuda = (int*)malloc( numCurrLevelNodes * sizeof(int)) ; 
    cudaMalloc (&currLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(currLevelNodes_cuda, currLevelNodes, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeNeighbors_cuda = (int*)malloc( countNodeNeighbors * sizeof(int)) ; 
    cudaMalloc (&nodeNeighbors_cuda, countNodeNeighbors * sizeof(int));
    cudaMemcpy(nodeNeighbors_cuda, nodeNeighbors, countNodeNeighbors * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeVisited_cuda = (int*)malloc( countNodesSize) ; 
    cudaMalloc (&nodeVisited_cuda, countNodesSize);
    cudaMemcpy(nodeVisited_cuda, nodeVisited,countNodesSize, cudaMemcpyHostToDevice);

    int* nodeGate_cuda = (int*)malloc( countNodesSize) ; 
    cudaMalloc (&nodeGate_cuda, countNodesSize);
    cudaMemcpy(nodeGate_cuda, nodeGate, countNodesSize, cudaMemcpyHostToDevice);

    int* nodeInput_cuda = (int*)malloc( countNodesSize) ; 
    cudaMalloc (&nodeInput_cuda, countNodesSize);
    cudaMemcpy(nodeInput_cuda, nodeInput, countNodesSize, cudaMemcpyHostToDevice);

    int* nodeOutput_cuda = (int*)malloc(countNodesSize) ; 
    cudaMalloc (&nodeOutput_cuda, countNodesSize);
    cudaMemcpy(nodeOutput_cuda, nodeOutput, countNodesSize, cudaMemcpyHostToDevice);

    clock_t begin_timer = clock();

    // kernel call
    global_queuing_kernel <<< numBlocks, blockSize >>> (blockSize * numBlocks, countNodes, nodePtrs_cuda, currLevelNodes_cuda, nodeNeighbors_cuda, nodeVisited_cuda, nodeGate_cuda, nodeInput_cuda, nodeOutput_cuda);

    clock_t stop_timer = clock();

    checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
    checkCudaErr(cudaGetLastError(), "GPU");

    cudaMemcpyFromSymbol(&numNextLevelNodes, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
    checkCudaErr(cudaMemcpyFromSymbol(nextLevelNodes,globalQueue, countNodesSize), "Copying");

    int *outputBuffer;
    outputBuffer = (int*)malloc( countNodesSize); 
    checkCudaErr(cudaMemcpy(outputBuffer, nodeOutput_cuda, countNodesSize, cudaMemcpyDeviceToHost), "Copying");

    // Opening output files
    FILE *outfile_node = fopen(output_node, "w"); FILE *outfile_nln = fopen(output_nln, "w");
    
    // File Check
    if(!outfile_node || !outfile_nln){
        return fprintf(stderr, "Invalid Output Files");
    } 

    //Writing values to output_nodeOutput
    fprintf(outfile_node, "%d\n", countNodes);
    for (int i = 0; i < countNodes; i++) { fprintf(outfile_node, "%d\n", outputBuffer[i]); }
    fclose(outfile_node);
    
    //Writing values to output_nextLevelNodes
    fprintf(outfile_nln, "%d\n", numNextLevelNodes);
    for (int i = 0; i < numNextLevelNodes; i++) { fprintf(outfile_nln, "%d\n", nextLevelNodes[i]); }
    fclose(outfile_nln);

    //Printing Runtime
    printf("Runtime time: %f ms\n", (double) (stop_timer - begin_timer) / CLOCKS_PER_SEC * 1000);

    // free variables
    free(nodeGate); free(nodeInput); free(nodeOutput);
    free(nodePtrs); free(currLevelNodes); free(nodeNeighbors); free(nodeVisited);
     

    // free cuda variables
    cudaFree(currLevelNodes_cuda);
    cudaFree(nodeNeighbors_cuda);
    cudaFree(nodePtrs_cuda);
    cudaFree(nodeVisited_cuda);
    cudaFree(nodeInput_cuda);
    cudaFree(nodeOutput_cuda);
    cudaFree(nodeGate_cuda);
}