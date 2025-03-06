#ifndef OPERATOR
#define OPERATOR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "solution.cuh"

__device__ void greedyInsertion(Solution *ps, Data data, int k);
__device__ void regret2Insertion(Solution *ps, Data data, int k);
__device__ void randomRemoval(Solution *ps, Data data, int k);
__device__ void worstRemoval(Solution *ps, Data data, int k);
__device__ void relatedRemoval(Solution *ps, Data data, int k);
__device__ void nodePairRemoval(Solution *ps, Data data, int k);
//__device__ void clusterRemoval(Solution *ps, Data data, int k);


#endif // !OPERATOR
