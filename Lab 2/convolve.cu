#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <time.h>
#include <math.h>

// row-major order
const float weights[9] = {1, 2, -1, 2, 0.25, -2, 1, -2, -1};

typedef struct Image {
    unsigned char* img;
    unsigned int size;
    unsigned int width;
    unsigned int height;
} Image;

unsigned char clampedWeightedSum(const float* weights, Image* original, int i, int j) {
    float sum = 0.0;
    sum += weights[0] * (float)original->img[i - (original->width * 4) - 4 + j];
    sum += weights[1] * (float)original->img[i - ((original->width) * 4) + j];
    sum += weights[2] * (float)original->img[i - (original->width * 4) + 4 + j];
    sum += weights[3] * (float)original->img[i - 4 + j];
    sum += weights[4] * (float)original->img[i + j];
    sum += weights[5] * (float)original->img[i + 4 + j];
    sum += weights[6] * (float)original->img[i + (original->width * 4) - 4 + j];
    sum += weights[7] * (float)original->img[i + (original->width * 4) + j];
    sum += weights[8] * (float)original->img[i + (original->width * 4) + 4 + j];
    sum = roundf(sum);
    if (sum > 255)
        sum = 255.0;
    else if (sum < 0)
        sum = 0.0;
    return (unsigned char) sum;
}

__device__ unsigned char clampedWeightedSumKernel(float* weights, Image* original, int i, int j) {
    // TODO
}

__global__ void convolveParallel(Image* original, Image* convolved) {
    // TODO
}

void convolveSequential(Image* original, Image* convolved) {
    // increment by 4 values, as each pixel has 4 channels: RGBA
    int increment = 4 * sizeof(unsigned char);
    // k is index in the output image
    int k = 0;
    // width is in bytes, original->width is in pixels
    int width = 4 * original->width * sizeof(unsigned char);
    for (int i = 0; i < original->size; i += increment) {
        // if outermost pixel of original image, correct k
        // where image is row-major (TODO: make sure it is actually row-major and not column-major)
        if (i % width == 0 || (i + increment) % width == 0 || i - width < 0 || i + width > original->size) {
            k -= increment;
        }
        else {
            // else, convolve with weight matrix (don't need to convolve alpha channel)
            for (int j = 0; j < 3 * sizeof(unsigned char); j += sizeof(unsigned char)) {
                convolved->img[k + j] = clampedWeightedSum(weights, original, i, j);
            }
        }

    }
}

int main(int argc, char* argv[]) {

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // step 1: parse arguments
    // ~~~~~~~~~~~~~~~~~~~~~~~

    if (argc != 4) {
        printf("Error: Input arguments are of format:\n./convolve <input filename> <output filename> <# threads>");
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

    Image original;
    Image convolved;

    int error = lodepng_decode32_file(&(original.img), &(original.width), &(original.height), argv[1]);
    if (error) {
        printf("Error %d: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: allocate for CPU and GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Image original_cuda;
    Image convolved_cuda;

    // allocate for CPU
    original.size = original.width * original.height * 4 * sizeof(unsigned char);
    convolved.size = (original.width - 2) * (original.height - 2) * 4 * sizeof(unsigned char);
    convolved.img = (unsigned char*)malloc(convolved.size);


    // allocate for GPU
    cudaMalloc((void**)&(original_cuda.img), original.size);
    cudaMalloc((void**)&(convolved_cuda.img), convolved.size);
    cudaMalloc((void**)&original_cuda, sizeof(original));
    cudaMalloc((void**)&convolved_cuda, sizeof(convolved));
    cudaMemcpy(&original_cuda, &original, original.size, cudaMemcpyHostToDevice);
    cudaMemcpy(original_cuda.img, original.img, original.size, cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 6: convolve sequentially, record performance
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    convolveSequential(&original, &convolved);
    error = lodepng_encode32_file(argv[2], convolved.img, convolved.width, convolved.height);
}