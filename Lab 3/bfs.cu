#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void sequentialBFS() {

}

__global__ void globalQueueingBFS() {

}

__global__ void blockQueueingBFS() {

}

int main(int argc, char* argv[]) {
    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    // TODO: CLI args
    if (argc != 4) {
        printf("Error: Input arguments are of format:\n./bfs <input filename> <output filename> <# threads>");
        return -1;
    }

    unsigned int n_threads = atoi(argv[3]);

    if (n_threads < 1) {
        printf("Error: '%u' is an invalid number of threads.\nNumber of threads must be greater than zero.", n_threads);
        return -1;
    }
}