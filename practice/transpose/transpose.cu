#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 32

// shared with bank conflict
__global__ void transpose_v0(const float* input, float* output, int M, int N) {
    __shared__ float shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    // base
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;

    // in matrix
    int xi = threadIdx.x + bx;
    int yi = threadIdx.y + by;

    // copy to shared
    if (xi < N && yi < M) {
        shared_mem[threadIdx.y][threadIdx.x] = input[yi * N + xi]; 
    }
    __syncthreads();
    
    // do transpose
    // origin point (bx, by) -> (by, bx)
    // shared men visited by y, write back by x
    int xt = by + threadIdx.x;
    int yt = bx + threadIdx.y;

    if (xt < M && yt < N) {
        // every warp, threadIdx.x++ will meet bank conflict
        output[yt * M + xt] = shared_mem[threadIdx.x][threadIdx.y];
    }
}

// padding to avoid conflict
__global__ void transpose_v1(const float* input, float* output, int M, int N) {
    __shared__ float shared_mem[BLOCK_SIZE][BLOCK_SIZE+1];

    // base
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;

    // in matrix
    int xi = threadIdx.x + bx;
    int yi = threadIdx.y + by;

    // copy to shared
    if (xi < N && yi < M) {
        shared_mem[threadIdx.y][threadIdx.x] = input[yi * N + xi]; 
    }
    __syncthreads();
    
    // do transpose
    // origin point (bx, by) -> (by, bx)
    // shared men visited by y, write back by x
    int xt = by + threadIdx.x;
    int yt = bx + threadIdx.y;

    if (xt < M && yt < N) {
        output[yt * M + xt] = shared_mem[threadIdx.x][threadIdx.y];
    }
}

// swizzling magic
__global__ void transpose_v2(const float* input, float* output, int M, int N) {
    __shared__ float shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    // base
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;

    // in matrix
    int xi = threadIdx.x + bx;
    int yi = threadIdx.y + by;

    // copy to shared
    if (xi < N && yi < M) {
        shared_mem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[yi * N + xi]; 
    }
    __syncthreads();
    
    // do transpose
    // origin point (bx, by) -> (by, bx)
    // shared men visited by y, write back by x
    int xt = by + threadIdx.x;
    int yt = bx + threadIdx.y;

    if (xt < M && yt < N) {
        output[yt * M + xt] = shared_mem[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}


int main(void) {
    return 0;
}