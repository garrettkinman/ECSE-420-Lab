#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "wm.h"

#include <stdio.h>
#include <time.h>
#include <math.h>

unsigned char clampedWeightedSum(unsigned char* img, float* weights, unsigned width, unsigned start, unsigned color) {
    // perform convolution
    float sum = 0.0;
    for (int i = 0; i < 9; i++) {
        int index = start + (i / 3) * width + (i % 3);
        sum += (int)img[4 * index + color] * weights[i];
    }

    // clamp to fit as unsigned char
    if (sum > 255) {
        sum = 255;
    }
    if (sum < 0) {
        sum = 0;
    }

    // round and cast
    return (unsigned char)sum;
}

__device__ unsigned char clampedWeightedSumKernel(unsigned char* img, float* weights, unsigned width, unsigned start, unsigned color) {
    // perform convolution
    float sum = 0.0;
    for (int i = 0; i < 9; i++) {
        int index = start + (i / 3) * width + (i % 3);
        sum += (int) img[4 * index + color] * weights[i];
    }

    // clamp to fit as unsigned char
    if (sum > 255) {
        sum = 255;
    }
    if (sum < 0) {
        sum = 0;
    }

    // round and cast
    return (unsigned char)sum;
}

__global__ void convolveParallel(unsigned char* original, unsigned char* convolved, float* weights, unsigned width, unsigned height, int n_threads, unsigned windows_per_thread) {
    int windows_done_so_far = windows_per_thread * threadIdx.x;

    for (int i = 0; i < windows_per_thread; i++) {
        int window_start = ((windows_done_so_far + i) / (width - 2)) * (width)+((windows_done_so_far + i) % (width - 2));
        for (int color = 0; color < 4; color++) {
            float result = 0.0;
            if (color == 3) {
                result = 255;
            }
            else {
                result = clampedWeightedSumKernel(original, weights, width, window_start, color);
            }
            convolved[(windows_done_so_far + i) * 4 + color] = result;
        }
    }
}

void convolveSequential(unsigned char* original, unsigned char* convolved, float* weights, unsigned width, unsigned height) {
    int n_windows = (width - 2) * (height - 2);
    for (int i = 0; i < n_windows; i++) {
        int window_start = (i / (width - 2)) * (width) + (i % (width - 2));
        for (int color = 0; color < 4; color++) {
            float result = 0.0;
            if (color == 3) {
                result = 255;
            }
            else {
                result = clampedWeightedSum(original, weights, width, window_start, color);
            }
            convolved[i * 4 + color] = result;
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

    unsigned int n_threads = atoi(argv[3]);

    if (n_threads < 1) {
        printf("Error: '%u' is an invalid number of threads.\nNumber of threads must be greater than zero.", n_threads);
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 2: read in input image from file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    unsigned char* original;
    unsigned char* convolved;
    unsigned width, height;

    int error = lodepng_decode32_file(&original, &width, &height, argv[1]);
    if (error) {
        printf("Error %d: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 3: allocate for CPU and GPU
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // allocate for CPU
    size_t original_size = width * height * 4 * sizeof(unsigned char);
    size_t convolved_size = (width - 2) * (height - 2) * 4 * sizeof(unsigned char);
    convolved = (unsigned char*)malloc(convolved_size);
    
    // allocate for GPU
    unsigned char* original_cuda;
    unsigned char* convolved_cuda;
    float* w_cuda;

    cudaMalloc((void**)&original_cuda, original_size);
    cudaMalloc((void**)&convolved_cuda, convolved_size);
    cudaMalloc((void**)&w_cuda, 3 * 3 * sizeof(float));

    cudaMemcpy(original_cuda, original, original_size, cudaMemcpyHostToDevice);
    cudaMemcpy(convolved_cuda, convolved, convolved_size, cudaMemcpyHostToDevice);
    cudaMemcpy(w_cuda, w[0], 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 6: convolve sequentially, record performance
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    clock_t startCPU = clock();
    convolveSequential(original, convolved, w[0], width, height);
    clock_t stopCPU = clock();

    printf("Sequential: %.6f ms\n", ((double)stopCPU - (double)startCPU) * 1000.0 / CLOCKS_PER_SEC);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 7: convolve in parallel, record performance
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    unsigned windows_per_thread = ((width - 2) * (height - 2)) / n_threads;

    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    convolveParallel<<<1, n_threads>>>(original_cuda, convolved_cuda, w_cuda, width, height, n_threads, windows_per_thread);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    float timeGPU;
    cudaEventElapsedTime(&timeGPU, startGPU, stopGPU);
    printf("Parallel Explicit: %.6f ms\n", timeGPU);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // step 8: write convolved image to file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cudaMemcpy(convolved, convolved_cuda, convolved_size, cudaMemcpyDeviceToHost);
    
    error = lodepng_encode32_file(argv[2], convolved, width - 2, height - 2);
    if (error) {
        printf("Error %d: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // ~~~~~~~~~~~~~~~~~~~~~
    // step 9: free at last!
    // ~~~~~~~~~~~~~~~~~~~~~

    free(original);
    free(convolved);
    cudaFree(original_cuda);
    cudaFree(convolved_cuda);
    cudaFree(w_cuda);

    return 0;
}