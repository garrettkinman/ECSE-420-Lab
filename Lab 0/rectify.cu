#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <time.h>

__global__ void rectifyParallel(unsigned char* original_img, unsigned char* new_img, unsigned int num_threads, unsigned int img_size)
{
    // iterate through all the blocks, same threadIdx for each
    for (int i = threadIdx.x; i < img_size; i += num_threads) {
        if (original_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = original_img[i];
        }
    }
}

void rectifySequential(unsigned char* original_img, unsigned char* new_img, unsigned int img_size) {
    // iterate through all elements of og_img
    for (int i = 0; i < img_size; i++) {
        if (original_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = original_img[i];
        }
    }
}

int main(int argc, char *argv[]) {

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    if (argc != 4) {
        printf("Error: Input arguments are of format:\n./rectify <input filename> <output filename> <# threads>");
        return -1;
    }

    unsigned int num_threads = atoi(argv[3]);

    if (num_threads < 1) {
        printf("Error: '%u' is an invalid number of threads.\nNumber of threads must be greater than zero.", num_threads);
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in input image from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    unsigned char* original_img;
    unsigned char* new_img;
    unsigned int img_width, img_height;

    int error = lodepng_decode32_file(&original_img, &img_width, &img_height, argv[1]);
    if (error) {
        printf("Error %d: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: make variables available to both CPU and GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    unsigned char* original_img_cuda;
    unsigned char* new_img_cuda;

    // allocate for CPU
    unsigned int img_size = img_width * img_height * 4 * sizeof(unsigned char);
    new_img = (unsigned char*)malloc(img_size);

    // allocate for GPU
    cudaMalloc((void**)&original_img_cuda, img_size);
    cudaMalloc((void**)&new_img_cuda, img_size);
    cudaMemcpy(original_img_cuda, original_img, img_size, cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 4: call parallelized rectify function, record performance
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // start timing GPU
    clock_t startGPU = clock();

    rectifyParallel<<<1, num_threads>>>(original_img_cuda, new_img_cuda, num_threads, img_size);
    cudaDeviceSynchronize();

    // record performance
    printf("Parallel: %u\n", clock() - startGPU);

    cudaMemcpy(new_img, new_img_cuda, img_size, cudaMemcpyDeviceToHost);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 5: write output image from parallelized rectify function to file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    error = lodepng_encode32_file(argv[2], new_img, img_width, img_height);
    if (error) {
        printf("Error %d: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 6: call sequential rectify function, record performance
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // start timing CPU
    clock_t startCPU = clock();

    rectifySequential(original_img, new_img, img_size);

    // record performance
    printf("Sequential: %u\n", clock() - startCPU);

    // ~~~~~~~~~~~~~~~~~~~~~
    // step 7: free at last!
    // ~~~~~~~~~~~~~~~~~~~~~

    free(original_img);
    free(new_img);
    cudaFree(original_img_cuda);
    cudaFree(new_img_cuda);

    return 0;
}