#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*
helper functions for reading in the files
*/
int read_input_one_two_four(int** input1, char* filepath);
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath);

/*
helper function for solving output of a gate
*/
int gate_solver(int gate, int x1, int x2);

void sequentialBFS(int numCurrLevelNodes, int* currLevelNodes, int* nodePtrs,
                    int* nodeNeighbors, int* nodeVisited, int* nodeOutput,
                    int* nodeGate, int* nodeInput, int* nextLevelNodes, int numNextLevelNodes) {
	// loop over all nodes in the current level
    for (int i = 0; i <= numCurrLevelNodes; i++) {
        int node = currLevelNodes[i];
        // loop over all neighbors of the node
        for (int j = nodePtrs[node]; j <= nodePtrs[node + 1]; j++) {
            int neighbor = nodeNeighbors[j];
            // if the neighbor hasn't been visited yet
            // mark it and add it to the queue
            if (!nodeVisited[neighbor]) {
                nodeVisited[neighbor] = 1;
                nodeOutput[neighbor] = gate_solver(nodeGate[neighbor], nodeOutput[node], nodeInput[neighbor]);
                nextLevelNodes[numNextLevelNodes] = neighbor;
                ++numNextLevelNodes;
            }
        }
            
    }

			
}

__global__ void globalQueueingBFS() {

}

__global__ void blockQueueingBFS() {

}

int main() {
    // variables for file names
    char* inputs[4] = { "input1.raw", "input2.raw", "input3.raw", "input4.raw" };
    char* nodeOutput = "nodeOutput.raw";
    char* nextLevelNodes = "nextLevelNodes.raw";
    
    // variables to hold state of the logic circuit
    int numNodePtrs;
    int numNodes;
    int* nodePtrs_h;
    int* nodeNeighbors_h;
    int* nodeVisited_h;
    int numTotalNeighbors_h;
    int* currLevelNodes_h;
    int numCurrLevelNodes;
    int numNextLevelNodes_h;
    int* nodeGate_h;
    int* nodeInput_h;
    int* nodeOutput_h;
    int* nextLevelNodes_h;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, inputs[0]);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, inputs[1]);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, inputs[2]);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, inputs[3]);
}

/*
helper function for reading in the files
*/
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

/*
helper function for reading in the files
*/
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

/*
helper function for solving output of a gate
*/
int gate_solver(int gate, int x1, int x2) {
    // TODO
}