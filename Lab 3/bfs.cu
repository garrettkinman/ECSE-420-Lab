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
helper functions for reading, writing, and comparing the files
*/
int read_input_one_two_four(int** input1, char* filepath);
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath);
void writeData(int* data, char* filepath);
void compareOutputFiles(char* file_name1, char* file_name2);
void compareNextLevelNodeFiles(char* file_name1, char* file_name2);

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
    sequentialBFS(numCurrLevelNodes, currLevelNodes_h, nodePtrs_h, nodeNeighbors_h, nodeVisited_h, nodeOutput_h, nodeGate_h, nodeInput_h, nextLevelNodes_h, numNextLevelNodes_h);

    writeData(nodeOutput_h, nodeOutputFilename);
    writeData(nextLevelNodes_h, nextLevelNodesFilename);

    compareOutputFiles(nodeOutputFilename, "sol_nodeOutput.raw");
    compareNextLevelNodeFiles(nextLevelNodesFilename, "sol_nextLevelNodes.raw");

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
helper function for comparing the output files
*/
void compareOutputFiles(char* file_name1, char* file_name2) {
    //get from https://www.tutorialspoint.com/c-program-to-compare-two-files-and-report-mismatches
    FILE* fp1 = fopen(file_name1, "r");
    FILE* fp2 = fopen(file_name2, "r");
    // fetching character of two file 
    // in two variable ch1 and ch2 
    char ch1 = getc(fp1);
    char ch2 = getc(fp2);

    // error keeps track of number of errors 
    // pos keeps track of position of errors 
    // line keeps track of error line 
    int error = 0, pos = 0, line = 1;

    // iterate loop till end of file 
    while (ch1 != EOF && ch2 != EOF)
    {
        pos++;

        // if both variable encounters new 
        // line then line variable is incremented 
        // and pos variable is set to 0 
        if (ch1 == '\n' && ch2 == '\n')
        {
            line++;
            pos = 0;
        }

        // if fetched data is not equal then 
        // error is incremented 
        if (ch1 != ch2)
        {
            error++;
            printf("Line Number : %d \tError"
                " Position : %d \n", line, pos);
        }

        // fetching character until end of file 
        ch1 = getc(fp1);
        ch2 = getc(fp2);
    }

    printf("Total Errors : %d\t", error);
}

void sort(int* pointer, int size) {
    //get from https://stackoverflow.com/questions/13012594/sorting-with-pointers-instead-of-indexes
    int* i, * j, temp;
    for (i = pointer; i < pointer + size; i++) {
        for (j = i + 1; j < pointer + size; j++) {
            if (*j < *i) {
                temp = *j;
                *j = *i;
                *i = temp;
            }
        }
    }
}

void compareNextLevelNodeFiles(char* file_name1, char* file_name2) {
    FILE* fp_1 = fopen(file_name1, "r");
    if (fp_1 == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    FILE* fp_2 = fopen(file_name2, "r");
    if (fp_2 == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len_1;
    int len_2;
    int length_file_1 = fscanf(fp_1, "%d", &len_1);
    int length_file_2 = fscanf(fp_2, "%d", &len_2);

    if (length_file_1 != length_file_2) {
        fprintf(stderr, "Wrong file length\n");
        exit(1);
    }
    int* input1 = (int*)malloc(len_1 * sizeof(int));
    int* input2 = (int*)malloc(len_2 * sizeof(int));

    int temp1;
    int temp2;

    while ((fscanf(fp_1, "%d", &temp1) == 1) && (fscanf(fp_2, "%d", &temp2) == 1)) {
        (input1)[counter] = temp1;
        (input2)[counter] = temp2;
        counter++;
    }

    sort(input1, len_1);
    sort(input2, len_2);

    for (int i = 0; i < len_1; i++) {
        if (input1[i] != input2[i]) {
            fprintf(stderr, "Something goes wrong\n");
            exit(1);
        }
    }

    fprintf(stderr, "No errors!\n");
    exit(1);
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