#include <stdio.h>
#include <stdlib.h>
#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define NXOR 5

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
  return err;
}

// Time in nanoseconds
long long getNanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int read_input_one_two_four(int **input1, char* filepath){
 FILE* fp = fopen(filepath, "r");
  if (fp == NULL){
    fprintf(stderr, "Couldn't open file for reading\n");
    exit(1);
  } 
  
  int counter = 0;
  int len;
  int length = fscanf(fp, "%d", &len);
  *input1 = ( int *)malloc(len * sizeof(int));

  int temp1;

  while (fscanf(fp, "%d", &temp1) == 1) {
    (*input1)[counter] = temp1;
    counter++;
  }

  fclose(fp);
  return len;
}

int read_input_three(int** input1, int** input2, int** input3, int** input4,char* filepath){
  FILE* fp = fopen(filepath, "r");
  if (fp == NULL){
    fprintf(stderr, "Couldn't open file for reading\n");
    exit(1);
  } 
  
  int counter = 0;
  int len;
  int length = fscanf(fp, "%d", &len);
  *input1 = ( int *)malloc(len * sizeof(int));
  *input2 = ( int *)malloc(len * sizeof(int));
  *input3 = ( int *)malloc(len * sizeof(int));
  *input4 = ( int *)malloc(len * sizeof(int));

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

__device__ int numNextLevelNodes = 0;
__device__ int globalNextLevelNodesQueue[5000000];

__global__ void block_queuing_kernel(int numCurrLevelNodes, int* currLevelNodes, int* nodeNeighbors, int* nodePtrs, int* nodeVisited, int* nodeInput, int* nodeOutput_C, int* nodeGate, int queueSize){
    
  // Initialize shared memory queue
  extern __shared__ int sharedBlockQueue[];
  __shared__ int sharedBlockQueueSize, blockGlobalQueueIdx;

  if (threadIdx.x == 0)  sharedBlockQueueSize = 0; 

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
        int result = 0;
        int nodeGateVal = nodeGate[neighborIdx];
        int nodeInputVal = nodeInput[neighborIdx];
        int nodeOutputVal = nodeOutput_C[nodeIdx];

        switch (nodeGateVal) {
          case 0:
            result = nodeInputVal & nodeOutputVal;
            break;
          case 1:
            result = nodeInputVal | nodeOutputVal;
            break;
          case 2:
            result = !(nodeInputVal & nodeOutputVal);
            break;
          case 3:
            result = !(nodeInputVal | nodeOutputVal);
            break;
          case 4:
            result = nodeInputVal ^ nodeOutputVal;
            break;
          case 5:
            result = !(nodeInputVal ^ nodeOutputVal);
            break;
        }
        
        // Update node output
        nodeOutput_C[neighborIdx] = result; 

        // Add to block queue if not full else add to global queue
        if (queueIdx < queueSize) sharedBlockQueue[queueIdx] = neighborIdx;                  
        else {
          sharedBlockQueueSize = queueSize;
          const int GlIdx = atomicAdd(&numNextLevelNodes, 1);
          globalNextLevelNodesQueue[GlIdx] = neighborIdx; 
        }
      }      
    }
  }
    
  __syncthreads();

  if (threadIdx.x == 0) blockGlobalQueueIdx = atomicAdd(&numNextLevelNodes, sharedBlockQueueSize);
  
  __syncthreads();

  // Store block queue in global queue
  for (int i = threadIdx.x; i < sharedBlockQueueSize; i += blockDim.x) globalNextLevelNodesQueue[blockGlobalQueueIdx + i] = sharedBlockQueue[i];
}

int main(int argc, char *argv[]){

  if( argc < 10) {
    printf("Require parameters in the following order: <numBlock> <blockSize> <sharedQueueSize> <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>.\n");
    exit(1);
  }


  // Variables
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


  // User Input
  const int blockSize = atoi(argv[1]);
  const int numBlocks = atoi(argv[2]);
  const int queueSize = atoi(argv[3]);

  char* rawInput1 = argv[4];
  char* rawInput2 = argv[5];
  char* rawInput3 = argv[6];
  char* rawInput4 = argv[7];

  char* nodeOutput_fileName = argv[8];
  char* nextLevelNodes_fileName = argv[9];

  numNodePtrs = read_input_one_two_four(&nodePtrs_h, rawInput1);
  numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, rawInput2);
  numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, rawInput3);
  numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, rawInput4);


  // Output structures
  int *nextLevelNodes_h = (int *)malloc(numNodes*sizeof(int));
  int *nextLevelNodes_C = (int *)malloc(numNodes*sizeof(int));
  cudaMalloc (&nextLevelNodes_C, numCurrLevelNodes * sizeof(int));
  cudaMemcpy(nextLevelNodes_C, nextLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
  

  // Copy to device
  int numNodesSize = numNodes * sizeof(int);
  int* currLevelNodes_C = (int*)malloc(numCurrLevelNodes * sizeof(int)) ; 
  cudaMalloc (&currLevelNodes_C, numCurrLevelNodes * sizeof(int));
  cudaMemcpy(currLevelNodes_C, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

  int* nodeNeighbors_C = (int*)malloc(numTotalNeighbors_h * sizeof(int)) ; 
  cudaMalloc (&nodeNeighbors_C, numTotalNeighbors_h * sizeof(int));
  cudaMemcpy(nodeNeighbors_C, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);

  int* nodePtrs_C = (int*)malloc(numNodePtrs * sizeof(int)) ; 
  cudaMalloc (&nodePtrs_C, numNodePtrs * sizeof(int));
  cudaMemcpy(nodePtrs_C, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);

  int* nodeVisited_C = (int*)malloc(numNodesSize) ; 
  cudaMalloc (&nodeVisited_C, numNodesSize);
  cudaMemcpy(nodeVisited_C, nodeVisited_h,numNodesSize, cudaMemcpyHostToDevice);

  int* nodeInput_C = (int*)malloc(numNodesSize) ; 
  cudaMalloc (&nodeInput_C, numNodesSize);
  cudaMemcpy(nodeInput_C, nodeInput_h, numNodesSize, cudaMemcpyHostToDevice);

  int* nodeOutput_C = (int*)malloc(numNodesSize) ; 
  cudaMalloc (&nodeOutput_C, numNodesSize);
  cudaMemcpy(nodeOutput_C, nodeOutput_h, numNodesSize, cudaMemcpyHostToDevice);

  int* nodeGate_C = (int*)malloc( numNodesSize) ; 
  cudaMalloc (&nodeGate_C, numNodesSize);
  cudaMemcpy(nodeGate_C, nodeGate_h, numNodesSize, cudaMemcpyHostToDevice);


  // Processing 
  long long startTime = getNanos();

  block_queuing_kernel <<< numBlocks, blockSize, queueSize*sizeof(int) >>> (numCurrLevelNodes, currLevelNodes_C, nodeNeighbors_C, nodePtrs_C, nodeVisited_C, nodeInput_C, nodeOutput_C, nodeGate_C, queueSize);
  checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
  checkCudaErr(cudaGetLastError(), "GPU");

  long long endTime = getNanos();
	long long averageNanos = (endTime - startTime);

	printf("Average ms: %.2f, blockSize: %d, numBlocks: %d, queueSize: %d \n", (double)averageNanos / 1000000, blockSize, numBlocks, queueSize);


  // Copy from device
  int* outputBuffer;
  outputBuffer = (int*) malloc(numNodesSize);
  checkCudaErr(cudaMemcpy(outputBuffer, nodeOutput_C, numNodesSize, cudaMemcpyDeviceToHost), "Copying");

  cudaMemcpyFromSymbol(&numNextLevelNodes_h, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
  checkCudaErr(cudaMemcpyFromSymbol(nextLevelNodes_h,globalNextLevelNodesQueue, numNextLevelNodes_h * sizeof(int), 0, cudaMemcpyDeviceToHost), "Copying");


  // Write output
  FILE *nodeOutputFile = fopen(nodeOutput_fileName, "w");
  int counter = 0;
  fprintf(nodeOutputFile,"%d\n",numNodes);

  while (counter < numNodes) {
    fprintf(nodeOutputFile,"%d\n",(outputBuffer[counter]));
    counter++;
  }

  fclose(nodeOutputFile);

  FILE *nextLevelOutputFile = fopen(nextLevelNodes_fileName, "w");
  counter = 0;
  fprintf(nextLevelOutputFile,"%d\n",numNextLevelNodes_h);

  while (counter < numNextLevelNodes_h) {
    fprintf(nextLevelOutputFile,"%d\n",(nextLevelNodes_h[counter]));
    counter++;
  }

  fclose(nextLevelOutputFile);


  // Free memory
  free(outputBuffer);
  free(nextLevelNodes_h);
  cudaFree(nextLevelNodes_C);
  cudaFree(currLevelNodes_C);
  cudaFree(nodeNeighbors_C);
  cudaFree(nodePtrs_C);
  cudaFree(nodeVisited_C);
  cudaFree(nodeInput_C);
  cudaFree(nodeOutput_C);
  cudaFree(nodeGate_C);
}