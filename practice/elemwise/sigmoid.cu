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




__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}





int main (void) {
    constexpr int N = 333;
    float* x_h = (float*)malloc(N * sizeof(float));
    float* y_h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i ++) {
        x_h[i] = 1.0f / (1 + i);
    }

    
    float* x_d = nullptr;
    float* y_d = nullptr;
    CudaSafeCall(cudaMalloc(&x_d, N * sizeof(float)));
    CudaSafeCall(cudaMalloc(&y_d, N *sizeof(float)));
    cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice);
    

    int block_size = 256;
    int grid_size = CEIL(N, block_size);
    sigmoid<<<grid_size, block_size>>>(x_d, y_d, N);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i ++) {
        printf("%.2f %s", y_h[i], (i+1)%10 == 0? "\n":"");
    }


    



    return 0;
}