#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void poolParallel(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // TODO
}

void poolSequential(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // TODO
}

int main(int argc, char *argv[]) {
    // TODO:
    // 1) read in and validate arguments
    // 2) load in input png from file
    // 3) make variables available to both CPU and GPU
    // 4) specify launch config of kernel function
    // 5) call parallelized pool function, record performance
    // 6) call sequential pool function, record performance
    // 7) write output image from parallelized pool function to file

    return 0;
}