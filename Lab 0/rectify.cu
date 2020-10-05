#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>

__global__ void rectifyParallel(unsigned char* original_img, unsigned char* new_img, unsigned int num_thread, unsigned int size)
{
    // iterate through all the blocks, same threadIdx for each
    for (int i = threadIdx.x; i < size; i += num_thread) {
        if (original_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = original_img[i];
        }
    }
}

void rectifySequential(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // iterate through all elements of og_img
    for (int i = 0; i < size; i++) {
        if (og_img[i] < 127) {
            new_img[i] = 127;
        }
        else {
            new_img[i] = og_img[i];
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

    int input_filename_len = strlen(argv[1]);
    int output_filename_len = strlen(argv[2]);

    // dynamically allocate strings of appropriate size to hold filenames
    char *input_filename = (char*)malloc(input_filename_len * sizeof(char));
    char *output_filename = (char*)malloc(output_filename_len * sizeof(char));

    strcpy(input_filename, argv[1]);
    strcpy(output_filename, argv[2]);

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
    unsigned int width_in, height_in;

    int error = lodepng_decode32_file(&original_img, &width_in, &height_in, input_filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: make variables available to both CPU and GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    unsigned char* original_img_cuda;
    unsigned char* new_img_cuda;

    // allocate for CPU
    unsigned int img_size = width_in * height_in * 4 * sizeof(unsigned char);
    new_img = (unsigned char*)malloc(img_size);

    // allocate for GPU
    cudaMalloc((void**)&original_img_cuda, img_size);
    cudaMalloc((void**)&new_img_cuda, img_size);
    cudaMemcpy(original_img, original_img, img_size, cudaMemcpyHostToDevice);




    // TODO:
    // 1) read in and validate arguments
    // 2) load in input png from file
    // 3) make variables available to both CPU and GPU
    // 4) specify launch config of kernel function
    // 5) call parallelized rectify function, record performance
    // 6) call sequential rectify function, record performance
    // 7) write output image from parallelized rectify function to file


    free(input_filename);
    free(output_filename);

    return 0;
}