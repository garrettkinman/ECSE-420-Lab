#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int gate_solver(int nodeInput, int nodeOutput, int nodeGate) {
    
    if(nodeGate == 0){
         if (nodeInput == 1 && nodeOutput == 1){return 1;}
         else {return 0;} 
    }
    if(nodeGate == 1){
        if (nodeInput == 0 && nodeOutput == 0) {return 0;}
        else {return 1;}    
    }

    if(nodeGate == 2){
        if (nodeInput == 1 && nodeOutput == 1) {return 0;}
        else {return 1;}    
    }
    
    if(nodeGate == 3){
        if (nodeInput == 0 && nodeOutput == 0) {return 1;}
        else {return 0;}    
    }

    if(nodeGate == 4){
        if (nodeInput == nodeOutput) {return 0;}
        else {return 1;}    
    }

    if(nodeGate == 5){
        if (nodeInput == nodeOutput) {return 1;}
        else {return 0;}    
    }
     
}

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

int main(int argc, char* argv[]) {  
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
    int* currLevelNodes; int numCurrLevelNodes = 0; //Current level
    int* nextLevelNodes; int numNextLevelNodes = 0; //Next level

    //Argument Check
    if (argc != 7) {return fprintf(stderr, "Not Enough or Too Many Arguments!\n");}

    //Assigning values for Node variables
    countNodes = read_third_file(input_3, &nodeVisited, &nodeGate, &nodeInput, &nodeOutput);
    numNodePtrs = read_input(&nodePtrs, input_1);
    countNodeNeighbors = read_input(&nodeNeighbors, input_2);
    
    numCurrLevelNodes = read_input(&currLevelNodes, input_4);
    nextLevelNodes = (int*) malloc(countNodes * sizeof(int));

    //BFS Loop
    clock_t begin_timer = clock();
    // Loop over all nodes in the current level 
    for (int i = 0; i < numCurrLevelNodes; i++) {
        int node = currLevelNodes[i];
        
        // Loop over all neighbors of the node
        for (int j = nodePtrs[node]; j < nodePtrs[node+1]; j++) {
            int neighbor = nodeNeighbors[j];
            
            // If the neighbor hasn't been visited yet 
            if (!nodeVisited[neighbor]) {
                
                // Mark it and add it to the queue 
                nodeVisited[neighbor] = 1;
                nodeOutput[neighbor] = gate_solver(nodeInput[neighbor], nodeOutput[node], nodeGate[neighbor]);
                nextLevelNodes[numNextLevelNodes] = neighbor;
                ++numNextLevelNodes;
            }
        }
    }
    clock_t stop_timer = clock();
    
    // Opening output files
    FILE* outfile_node = fopen(output_node, "w"); FILE* outfile_nln = fopen(output_nln, "w");

    // File Check
    if(!outfile_node || !outfile_nln){
        return fprintf(stderr, "Invalid Output Files");
    } 

    //Writing values to output_nodeOutput
    fprintf(outfile_node, "%d\n", countNodes);
    for (int i = 0; i < countNodes; i++) { fprintf(outfile_node, "%d\n", nodeOutput[i]); }
    fclose(outfile_node);

    //Writing values to output_nextLevelNodes
    fprintf(outfile_nln, "%d\n", numNextLevelNodes);
    for (int i = 0; i < numNextLevelNodes; i++) { fprintf(outfile_nln, "%d\n", nextLevelNodes[i]); }
    fclose(outfile_nln);
    
    //Printing Runtime
    printf("Sequential time: %f ms\n", (double) (stop_timer - begin_timer) / CLOCKS_PER_SEC * 1000);

}

