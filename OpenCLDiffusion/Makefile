CC=gcc
FILES=readFile.c error.c main.c
OUT=main
OS=$(shell uname)

ifeq ($(OS),Darwin)
CCFLAGS=-framework OpenCL
else
CCFLAGS=-lOpenCL
endif
build: $(FILES)
	$(CC) -o $(OUT) $(FILES) $(CCFLAGS)

clean:
	rm -f *.o core

rebuild: clean build