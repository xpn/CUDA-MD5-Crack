INCLUDE_PATH = /usr/local/cuda/include
LIB_PATH = /usr/local/cuda/lib
CUDA_CC = /usr/local/cuda/bin/nvcc
CUDA_CFLAGS = -I$(INCLUDE_PATH) -keep #-DGPU_BENCHMARK -DDEBUG -DBENCHMARK
CC = gcc
CFLAGS = -I$(INCLUDE_PATH) -L$(LIB_PATH) -lcudart #-DDEBUG -DBENCHMARK

main: main.c main.h md5.o
	$(CC) main.c md5.o -o main $(CFLAGS)

md5.o: md5.cu main.h
	$(CUDA_CC) md5.cu -c -o md5.o $(CUDA_CFLAGS)


