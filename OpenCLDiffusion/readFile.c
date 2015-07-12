#include <stdio.h>
#include <stdlib.h>

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