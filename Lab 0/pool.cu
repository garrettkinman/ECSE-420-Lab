#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void getDimensions(int numThreads, int* width, int* height) {
    *height = 1;
    *width = 1;

    bool side = true;
    while (numThreads > 1) {
        if (side) {
            *width *= 2;
        }
        else {
            *height *= 2;
        }
        side = !side;
        numThreads /= 2;
    }
}

__global__ void poolParallel(unsigned char* image, unsigned char* pool_image, int pixelWidth, int pixelHeight, int width, int height) {
    int thread_index = threadIdx.x;

    // index pixel sections assigned to thread
    int row = thread_index / width;
    int col = thread_index % width;

    // width and height of pixel sections
    int sector_height = pixelHeight / height;
    int sector_width = pixelWidth * 4 / width;

    sector_height -= sector_height % 2;
    sector_width -= sector_width % 2;

    // values and indices stored on 2x2 grid
    int max_val;
    int t_left, t_right, b_left, b_right;
    int t_left_index, t_right_index, b_left_index, b_right_index;

    // global index of pixel
    int glob_col, glob_row;

    // index of pool_image array
    int index_pool, offset;

    for (int i = 0; i <= sector_height; i += 2) {
        index_pool = sector_width * col / 2 + (i / 2 + sector_height / 2 * row) * pixelWidth * 2;
        offset = index_pool % 4;
        index_pool -= offset;

        for (int j = 0 - offset; j <= sector_width + 4 - offset; j += 8) {
            // calculating pixel location
            glob_row = i + sector_height * row;
            glob_col = j + sector_width * col;

            // align indices
            glob_col -= glob_col % 4;

            // iterate through colors
            for (int color = 0; color < 4; color++) {
                t_left = 0;
                t_right = 0;
                b_left = 0;
                b_right = 0;

                // get index of corner 2x2 region
                t_left_index = glob_col + color + glob_row * pixelWidth * 4;
                t_right_index = glob_col + color + 4 + glob_row * pixelWidth * 4;
                b_left_index = glob_col + color + (glob_row + 1) * pixelWidth * 4;
                b_right_index = glob_col + color + 4 + (glob_row + 1) * pixelWidth * 4;

                // get value of corner
                if (t_left < pixelWidth * pixelHeight * 4) {
                    t_left = image[t_left_index];
                }

                if (glob_col + color + 4 < pixelWidth * 4) {
                    t_right = image[t_right_index];
                }

                if (b_left_index < pixelWidth * pixelHeight * 4) {
                    b_left = image[b_left_index];
                }

                if (glob_col + color + 4 < pixelWidth * 4 && b_right_index < pixelWidth * pixelHeight * 4) {
                    b_right = image[b_right_index];
                }

                // calculate pool max
                max_val = t_left;

                if (t_right > max_val) {
                    max_val = t_right;
                }

                if (b_left > max_val) {
                    max_val = b_left;
                }

                if (b_right > max_val) {
                    max_val = b_right;
                }

                if (index_pool < pixelWidth * pixelHeight) {
                    pool_image[index_pool++] = max_val;
                }
                else {
                    break;
                }
            }
        }
    }
    
}

void poolSequential(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // TODO
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Error: Input arguments are of format:\n./pool <input filename> <output filename> <# threads>");
        return -1;
    }

    unsigned int numThreads = atoi(argv[3]);

    if (numThreads < 1) {
        printf("Error: '%u' is an invalid number of threads.\nNumber of threads must be greater than zero.", numThreads);
        return -1;
    }

    int len_png;
    unsigned char* image, * new_image, * pool_image;
    unsigned height, width;
    unsigned error;

    int sectors_x, sectors_y;
    getDimensions(numThreads, &sectors_x, &sectors_y);

    // load PNG image
    error = lodepng_decode32_file(&image, &width, &height, argv[1]);

    // error check
    if (error) { exit(error); }

    // calculate length of loaded PNG image 
    len_png = 4 * height * width * sizeof(unsigned char);

    // allocated space for image in shareable memory
    cudaMallocManaged((void**)&new_image, len_png * sizeof(unsigned char));
    cudaMallocManaged((void**)&pool_image, len_png / 4 * sizeof(unsigned char));

    // initialize data array for image
    for (int i = 0; i < len_png; i++) {
        new_image[i] = image[i];
    }

    // start timing GPU
    clock_t startGPU = clock();

    // launch pool() kernel on GPU with numThreads threads
    poolParallel<<<1, numThreads >> > (new_image, pool_image, width, height, sectors_x, sectors_y);

    // wait for threads to end on GPU
    cudaDeviceSynchronize();

    // record performance
    printf("Parallel: %u\n", clock() - startGPU);

    // write resulting image to output file
    lodepng_encode32_file(argv[2], pool_image, width / 2, height / 2);

    // cleanup
    cudaFree(new_image);
    cudaFree(pool_image);
    free(image);

    return 0;
}