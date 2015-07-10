#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <getopt.h>
#include "kernel.cu"
#define getIndex(i, j) ((i) * (width) + (j))

#define DEFAULT_BLOCK_SIZE 16
#define DEFAULT_WIDTH 16
#define DEFAULT_HEIGHT 16
#define DEFAULT_NUM_ITERATIONS 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void initGrid(float *grid, int width) {
	int i;
	for(i = 0; i < width; i++) {
		grid[getIndex(i, 0)] = 100;
		grid[getIndex(0, i)] = 100;
		grid[getIndex(width - 1, i)] = 100;
		grid[getIndex(i, width - 1)] = 100;
	}
}

int main(int argc, const char **argv) {
	unsigned int width = DEFAULT_WIDTH;
	unsigned int height = DEFAULT_HEIGHT;
	unsigned int blockSize = DEFAULT_BLOCK_SIZE;
	unsigned int iterations = DEFAULT_NUM_ITERATIONS;	
	bool printResult = false;
	int opt;

	while((opt = getopt(argc, (char * const *)argv, "s:w:i:p")) != -1) {
		switch(opt) {
			case 's':
				width = atoi(optarg);
				height = width;
				break;
			case 'w':
				blockSize = atoi(optarg);
				break;
			case 'i':
				iterations = atoi(optarg);
				break;
			case 'p':
				printResult = true;
				break;
			default:
				break;

		}
	}

	unsigned int gridLength = width * height;
	size_t gridSize = sizeof(float) * gridLength;

	float *grid = (float *) malloc(gridSize);
	initGrid(grid, width);
	
	float *deviceGrid;
	cudaMalloc((void **) &deviceGrid, gridSize);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(deviceGrid, grid, gridSize, cudaMemcpyHostToDevice);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	dim3 threads(blockSize, blockSize);
	dim3 gridDims(width / threads.x, width / threads.y);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	int i;
	for(i = 0; i < iterations; i++) {
		diffuse<<< gridDims, threads >>>(deviceGrid, width, blockSize);
	}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());	
	
	cudaMemcpy(grid, deviceGrid, gridSize, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if(printResult) {	
		int j;
		for(i = 0; i < width; i++) {
			for(j = 0; j < width; j++) {
				printf("%8.3f, ", grid[getIndex(i, j)]);
			}
			printf("\n");
		}
	}
	free(grid);
	cudaFree(grid);
}

