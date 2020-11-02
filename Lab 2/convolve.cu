#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <time.h>

__global__ void convolveParallel(unsigned char* original_img, unsigned char* new_img, unsigned int num_threads, unsigned int img_size)
{
    // TODO
}

void convolveSequential(unsigned char* original_img, unsigned char* new_img, unsigned int img_size) {
    // TODO
}

int main(int argc, char* argv[]) {
    // TODO
}