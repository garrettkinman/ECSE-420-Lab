#include <stdio.h>
#include​ ​<stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include​ ​"lodepng.h"


/* This method divides image into sections according to the number of threads */

void​ ​getDimensions​(​int​ ​numThreads​, ​int​* ​w​idth, ​int​* ​h​eight) { 

	*h​eight = ​1​;
	*w​idth = ​1​;

	bool​ side = ​true​;
​ 	while​ (numThreads > ​1​) {
​ 		if​ (side) { 
 			*w​idth *= ​2​;
 		} 
 		​else​ {
 			*h​eight *= ​2​; 
 		}
		side = !side;
		numThreads /= ​2​; 
	}
}


/* Function on GPU*/
__global__ ​void​ ​pool​(​unsigned​ ​char​* ​image​, ​unsigned​ ​char​* ​pool_image​, ​int​ ​pixelWidth​, int​ ​pixelHeight​, ​int​ ​w​idth, ​int​ ​h​eight) {
​ 	
 	int​ thread_index = ​threadIdx​.​x​;
​ 	
 	// index pixel sections assigned to thread ​
​ 	int​ row = thread_index / w​idth;
 	int​ col = thread_index % w​idth;
​ 	
 	// width and height of pixel sections
 	int​ sector_height = pixelHeight / h​eight;
 	int​ sector_width = pixelWidth * ​4​ / w​idth;
 	
 	sector_height -= sector_height % ​2​;
 	sector_width -= sector_width % ​2​; ​// even sized
​ 	
	// values and indices stored on 2x2 grid
​ 	int​ max_val; 
 	int t_left; 
 	int t_right; 
 	int b_left; 
 	int b_right;
​ 	int​ t_left_index; 
 	int t_right_index; 
 	int b_left_index; 
 	int b_right_index;

​ 	// global index of pixel
 	int​ glob_column, glob_row;
​ 	
  	// index of pool_image array
  	int​ index_pool, offset;

​ 	for​ (​int​ i = ​0​; i <= sector_height; i += 2) {
		index_pool = sector_width * col / 2 + (i/2 + sector_height/2*row) * pixelWidth * ​2​;
		offset = index_pool % 4;
		index_pool = index_pool - offset;

		for (int j = 0 - offset; j <= sector_width + 4 - offset; j += 8){
			// Calculating pixel location
			glob_row = i + sector_height * row;
			glob_column = j + sector_width * col;

			//Align indices
			glob_column = glob_column - glob_column % 4;

			//iterating through colours
			for (int colour = 0; colour < 4; colour++){
				t_left = 0;
				t_right = 0;
				b_left = 0;
				b_right = 0;

				// Get index of corner 2x2 region
				t_left_index = glob_column + colour + glob_row * pixelWidth * 4;
				t_right_index = glob_column + colour + 4 + glob_row * pixelWidth * 4;
				b_left_index = glob_column + colour + (glob_row + 1) * pixelWidth * 4;
				b_right_index = glob_column + colour + 4 + (glob_row + 1) * pixelWidth * 4;

				// Get value of corner
				if (t_left < pixelWidth * pixelHeight * 4){
					t_left = image[t_left_index];
				}

				if (glob_column + colour + 4 < pixelWidth * 4){
					t_right = image[t_right_index];
				}

				if (b_left_index < pixelWidth * pixelHeight * 4){
				 	b_left = image[b_left_index];
				}

				if (glob_column + colour + 4 < pixelWidth * 4 && b_right_index < pixelWidth * pixelHeight * 4){
				 	b_right = image[b_right_index];
				}

				// Calculating pool max
				max = t_left;

				if (t_right > max){
				 	max = t_right;
				}

				if (b_left > max){
				 	max = b_left;
				}

				if (b_right > max){
				 	max = b_right;
				}

				if (index_pool < pixelWidth * pixelHeight){
				 	pool_image[index_pool++] = max;
				}
				else{
					break;
				}

			}
		}
 	}
 	
}


__global__ void poolParallel(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // TODO
}

void poolSequential(unsigned char* og_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
    // TODO
}

int main(int argc, char *argv[]) {

	char input_filename[] = argv[1];
	char output_filename[] = argv[2];
	int numThreads = atoi(argv[3]); //number of threads

	int len_png;
	unsigned char* image, *new_image, *pool_image;
	unsigned height, width;
	unsigned error;

	int sectors_x, sectors_y;
	getDimensions(numThreads, &sectors_x, &sectors_y);

	//Load PNG image
	error = lodepng_decode32_file(&image, &width, &height, input_filename);

	// Error Check
	if(error){ exit(error); }

	//Calculate length of loaded PNG image 
	len_png = 4 * height * width * sizeof(unsigned char);

	//Allocated space for image in shareable memory
	cudaMallocManaged((void**) & new_image, len_png * sizeof(unsigned char));
	cudaMallocManaged((void**) & pool_image, len_png/4 * sizeof(unsigned char));

	//Initializing data array for image
	for (int i = 0; i < len_png; i++){
		new_imaged[i] = image[i];
	}

	// Launch pool() kernel on GPU with numThreads threads
	pool <<<1, numThreads>>> (new_image, pool_image, width, height, sectors_x, sectors_y);

	// Wait for threads to end on GPU
	cudaDeviceSynchronize();

	//Write resulting image to output file
	lodepng_ecode32_file(output_filename, pool_image, width/2, height/2);

	//Cleanup
	cudaFree(new_image);
	cudaFree(pool_image);
	free(image);



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
