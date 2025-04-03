#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a+b-1)/(b))

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

__global__ void add(float* a, float* b, float *c, int N) {
    // single dim
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


int main (void) {
    constexpr int N = 356;
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        // a is 0 1 2 3 4 5 6
        // b is 6 5 4 3 2 1 0
        a_h[i] = i;
        b_h[i] = N-1-i;
    }



    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    CudaSafeCall(cudaMalloc(&a_d, N * sizeof(float)));
    CudaSafeCall(cudaMalloc(&b_d, N * sizeof(float)));
    CudaSafeCall(cudaMalloc(&c_d, N * sizeof(float)));
    CudaSafeCall(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));


    int block_size = 1024;
    int grid_size = CEIL(N, block_size);

    add<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i ++) {
        printf("%.2f %s", c_h[i], (i+1) % 10 == 0? "\n": "");
        
    }

    return 0;
}