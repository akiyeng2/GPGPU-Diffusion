#ifndef ERROR_H
#define ERROR_H

#include <assert.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char* getErrorString(cl_int error);

#define checkError(code) \
	if(code != CL_SUCCESS) {\
		printf("Program failed: %s\n", getErrorString(code)); \
		assert(code == CL_SUCCESS); \
	}

#endif