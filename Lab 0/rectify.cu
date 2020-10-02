#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void rectifyParallel(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size)
{
    // iterate through all the blocks, same threadIdx for each
    for (int i = threadIdx.x; i < size; i += num_thread) {
        if (og_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = og_img[i];
        }
    }
}

void rectifySequential(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // iterate through all elements of og_img
    for (int i = 0; i < sizeof(og_img) / sizeof(og_img[0]); i++) {
        if (og_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = og_img[i];
        }
    }
}

int main(int argc, char *argv[]) {
    // TODO:
    // 1) read in and validate arguments
    // 2) load in input png from file
    // 3) make variables available to both CPU and GPU
    // 4) specify launch config of kernel function
    // 5) call parallelized rectify function, record performance
    // 6) call sequential rectify function, record performance
    // 7) write output image from parallelized rectify function to file

    return 0;
}