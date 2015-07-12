#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <getopt.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
//Convert a 2D array index to a 1d index
#define getIndex(i, j, width) (i) * (width) + (j)

#define DEFAULT_WORK_GROUP_SIZE 4
#define DEFAULT_WIDTH 16
#define DEFAULT_HEIGHT 16
#define DEFAULT_NUM_ITERATIONS 100

#define checkError(code) checkErr(code); assert(code == CL_SUCCESS);


const char *readFile(const char *filename){
	long int size = 0;
	FILE *file = fopen(filename, "rb");
	
	if(!file) {
		fputs("File error.\n", stderr);
		return NULL;
	}
	
	fseek(file, 0, SEEK_END);
	size = ftell(file);
	rewind(file);
	
	char *result = (char *) malloc(size);
	if(!result) {
		fputs("Memory error.\n", stderr);
		return NULL;
	}
	
	if(fread(result, 1, size, file) != size) {
		fputs("Read error.\n", stderr);
		return NULL;
	}
		
	fclose(file);
	result = realloc(result, size + sizeof(char));
	result[size] = '\0';
	return result;
}

static inline void checkErr(cl_int errorCode) {
	if(errorCode != CL_SUCCESS) {
		printf("Program failed with error code %i\n", errorCode);
	}

}


void gridInit(float *grid, unsigned int length, unsigned int width) {
	unsigned int i;
	//initialize left row to 100, since the rest is automatically set to 0
	for(i = 0; i < length; i++) {
		grid[getIndex(0, i, width)] = 100;
		grid[getIndex(i, 0, width)] = 100;
		grid[getIndex(i, width - 1, width)] = 100;
		grid[getIndex(width - 1 , i, width)] = 100;
	}
}

/**
 * Param 1: Grid length/width
 * Param 2: Work group size
 * Param 3: Iterations
 * Param 4: Print result (1/0)
 */
int main(int argc, const char * argv[]) {
	unsigned int width = DEFAULT_WIDTH;
	unsigned int height = DEFAULT_HEIGHT;
	unsigned int workGroupSize = DEFAULT_WORK_GROUP_SIZE;
	unsigned int iterations = DEFAULT_NUM_ITERATIONS;
	int opt;
	bool printResult = false;
	
	while((opt = getopt(argc, (char * const *)argv, "s:w:i:p")) != -1) {
		switch(opt) {
			case 's':
				width = atoi(optarg);
				height = width;
				break;
			case 'w':
				workGroupSize = atoi(optarg);
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
	
	int gridLength = width * height;
	size_t gridSize = gridLength * sizeof(float);

	//OpenCL only supports float by default
	float *grid = malloc(gridSize);
	
	gridInit(grid, width, height);

	//Initialize the things we will need to work with OpenCL
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	cl_kernel kernel;
	
	size_t dataBytes;
	size_t kernelLength;
	cl_int errorCode;
	
	cl_mem gridBuffer;
	
	cl_device_id* devices;
	cl_device_id gpu;
	
	cl_uint numPlatforms;

	errorCode = clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id platforms[numPlatforms];
	errorCode = clGetPlatformIDs(numPlatforms, platforms, NULL);
	
	checkError(errorCode);
	
	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (int) platforms[0], 0};

	context = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, 0, NULL, &errorCode);
	checkError(errorCode);
	
	errorCode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &dataBytes);
	devices = malloc(dataBytes);
	errorCode |= clGetContextInfo(context, CL_CONTEXT_DEVICES, dataBytes, devices, NULL);
	
	gpu = devices[0];
	
	commandQueue = clCreateCommandQueue(context, gpu, 0, &errorCode);
	checkError(errorCode);
	
	gridBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gridSize, grid, &errorCode);
	checkError(errorCode);
	
	const char* programBuffer = readFile("kernel.cl");
	kernelLength = strlen(programBuffer);
	program = clCreateProgramWithSource(context, 1, (const char **)&programBuffer, &kernelLength, &errorCode);
	checkError(errorCode);
	
	errorCode = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errorCode == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		
		// Allocate memory for the log
		char *log = (char *) malloc(log_size);
		
		// Get the log
		clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		
		// Print the log
		free(log);
		printf("%s\n", log);
	}
	checkError(errorCode);
	
	kernel = clCreateKernel(program, "diffuse", &errorCode);
	checkError(errorCode);

	size_t localWorkSize[2] = {workGroupSize, workGroupSize}, globalWorkSize[2] = {width, height};

	errorCode |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gridBuffer);
	errorCode |= clSetKernelArg(kernel, 1, sizeof(float) * workGroupSize * workGroupSize, NULL);
	errorCode |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&width);
	errorCode |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&workGroupSize);
	errorCode |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&iterations);
	checkError(errorCode);
	
	errorCode = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	checkError(errorCode);
	
	errorCode = clEnqueueReadBuffer(commandQueue, gridBuffer, CL_TRUE, 0, gridSize, grid, 0, NULL, NULL);
	checkError(errorCode);

	if(printResult) {
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				printf("%8.3f,", grid[getIndex(j, i, width)]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	free(grid);
	free(devices);
	free((void *)programBuffer);
	clReleaseContext(context);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	
    return 0;
}
