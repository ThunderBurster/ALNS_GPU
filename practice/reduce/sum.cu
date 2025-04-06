#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 256
#define WARP_SIZE 32

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






void host_reduce(float* x, const int N, float* sum) {
    *sum = 0.0;
    for (int i = 0; i < N; i ++) {
        *sum += x[i];
    }
}


// using global mem
// one block one result
__global__ void device_reduce_v0(float *x, float *y) {
    const int tid = threadIdx.x;
    float *x_s = x + blockDim.x * blockIdx.x;
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        // half thread in block do the job
        // block_size is odd
        if (tid < offset) {
            x_s[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[blockIdx.x] = x_s[0];
    } 
}

// using shared memory, one block one result
__global__ void device_reduce_v1(float *d_x, float *d_y, const int N) {
    const int tid = threadIdx.x;
    const int n = blockDim.x * blockIdx.x + tid;

    extern __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n]: 0.0f;

    __syncthreads();
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[blockIdx.x] = s_y[0];
    }
}

// using shared memory
// atomic add at last
__global__ void device_reduce_v2(float *d_x, float *d_y, const int N) {
    const int tid = threadIdx.x;
    const int n = blockDim.x * blockIdx.x + tid;

    extern __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n]: 0.0f;

    __syncthreads();
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

// warp reduce
// warp_size is 32, and block threads <= 1024(32 warps)
__global__ void device_reduce_v3(float *d_x, float *d_y, const int N) {
    __shared__ float s_y[WARP_SIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // one warp one value
    float val = (idx < N)? d_x[idx]: 0.0f;  // move from global to register
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // collect from multi warps
    if (laneId == 0) {
        s_y[warpId] = val;
    }
    __syncthreads();
    
    // first warp in block will do the reduce for block
    if (warpId == 0) {
        int warpCnt = blockDim.x / WARP_SIZE;
        val = (laneId < warpCnt)? s_y[laneId]: 0.0f;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // block to grid result
        if (laneId == 0) {
            atomicAdd(d_y, val);
        }
    }
}



int main(void) {


    return 0;
}