#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

/*
helper functions for reading and writing the files
*/
int read_input_one_two_four(int** input1, char* filepath);
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath);
void writeData(int* data, char* filepath);

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
    char* inputFilenames[4] = { "input1.raw", "input2.raw", "input3.raw", "input4.raw" };
    char* nodeOutputFilename = "nodeOutput.raw";
    char* nextLevelNodesFilename = "nextLevelNodes.raw";
    
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

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, inputFilenames[0]);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, inputFilenames[1]);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, inputFilenames[2]);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, inputFilenames[3]);

    // TODO: simulate!

    return 0;
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
helper function for writing out the files
*/
void writeData(int* data, char* filepath) {
    FILE* fp = fopen(filepath, "w+");
    for (int i = 0; i < sizeof(data) / sizeof(data[0]); i++) {
        fprintf(fp, "%d\n", data[i]);
    }
    fclose(fp);
}

/*
helper function for solving output of a gate
*/
int gate_solver(int gate, int x1, int x2) {
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