#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

#define THREADS_PER_BLOCK 1024

typedef struct Gate {
    char type;
    char x1;
    char x2;
    char y;
} Gate;

__global__ void simulate_parallel(Gate* gates, unsigned int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    switch (gates[i].type) {
        case AND:
            gates[i].y = gates[i].x1 && gates[i].x2;
            break;
        case OR:
            gates[i].y = gates[i].x1 || gates[i].x2;
            break;
        case NAND:
            gates[i].y = !(gates[i].x1 && gates[i].x2);
            break;
        case NOR:
            gates[i].y = !(gates[i].x1 || gates[i].x2);
            break;
        case XOR:
            gates[i].y = (gates[i].x1 || gates[i].x2) && !(gates[i].x1 && gates[i].x2);
            break;
        case XNOR:
            gates[i].y = (gates[i].x1 && gates[i].x2) || (!gates[i].x1 && !gates[i].x2);
            break;
    }
}

// function that reads in csv of gate specs and updates the list of gates with x1, x2, and type
void read_csv_to_gates(char* filename, Gate* gates, unsigned int size) {
    FILE* file = fopen(filename, "r");
    int i = 0;
    char line[16];

    while (fgets(line, 16, file)) {
        if (i >= size) break;
        gates[i].x1 = line[0] - '0';
        gates[i].x2 = line[2] - '0';
        gates[i].type = line[4] - '0';
        i++;
    }

    fclose(file);
}

void write_gates_to_file(char* filename, Gate* gates, unsigned int size) {
    FILE* file = fopen(filename, "w+");

    for (int i = 0; i < size; i++) {
        fputc(gates[i].y + '0', file);
        fputc('\n', file);
    }

    fclose(file);
}

int main(int argc, char* argv[]) {

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    if (argc != 4) {
        printf("Error: Input arguments are of format:\n"
            "./sequential <input filename> <input file length> <output filename>");
        return -1;
    }

    unsigned int input_length = atoi(argv[2]);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in inputs from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Gate* gates = (Gate*)malloc(sizeof(Gate) * input_length);
    read_csv_to_gates(argv[1], gates, input_length);

    // ~~~~~~~~~~~~~~~~~~~
    // step 3: copy to GPU
    // ~~~~~~~~~~~~~~~~~~~

    Gate* gates_cuda;
    cudaMalloc((void**)&gates_cuda, sizeof(Gate) * input_length);
    cudaMemcpy(gates_cuda, gates, sizeof(Gate) * input_length, cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 4: time parallel simulation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    clock_t startCPU = clock();
    simulate_parallel<<<(input_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gates_cuda, input_length);
    printf("Parallel Explicit: %u\n", clock() - startCPU);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 5: write to file and done!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cudaMemcpy(gates, gates_cuda, sizeof(Gate) * input_length, cudaMemcpyDeviceToHost);
    write_gates_to_file(argv[3], gates, input_length);
    free(gates);
    cudaFree(gates_cuda);

    return 0;
}