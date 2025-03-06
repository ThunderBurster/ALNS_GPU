#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "solution.cuh"
#include "curand_kernel.h"
#include <time.h>
#include <limits.h>
#include "kernelPara.cuh"//define the BLOCK_SIZE and GRID_SIZE

#define COST 10000.0//used for init minCost





/*help fuctions*/
__device__ bool inSolution(Solution *ps, int i)
{
	return ps->next[i] >= 0 ? true : false;
}

__device__ bool feasible(Solution *ps, int insertPos,int toBeInsert, Data data)
{
	//CVRP, check the demand
	return ps->tourDemend[ps->tourid[insertPos]] + data.demand[toBeInsert] <= data.capacity;
}

__device__ int costFunction(Solution *ps, int x, int *removedList, int k, Data data)
{
	int i = x / k;//insertPos 
	int j = x - k * i;
	double c = COST * 2;
	if (inSolution(ps, i + 1) && feasible(ps, i + 1, removedList[j], data)) {
		c = dist(data, i + 1, removedList[j]) + dist(data, removedList[j], ps->next[i + 1]) - dist(data, i + 1, ps->next[i + 1]);
	}
	return c;
}

__device__ void minReduction(double *minCost, int *bestIdx)
{
	//the minCost and responding bestIdx will be move to idx 0
	//the array size if BLOCK_SIZE
	int size = BLOCK_SIZE;
	while (size > 1) {
		__syncthreads();
		if (threadIdx.x < size / 2 && minCost[threadIdx.x] > minCost[threadIdx.x + (size + 1) / 2]) {
			minCost[threadIdx.x] = minCost[threadIdx.x + (size + 1) / 2];
			bestIdx[threadIdx.x] = bestIdx[threadIdx.x + (size + 1) / 2];
		}
		size = (size + 1) / 2;
	}
	__syncthreads();
}

__device__ void getRemovedList(Solution *ps, int *list, int k) 
{
	k = 0;
	for (int i = 1;i <= ps->n;i++) {
		if (ps->next[i] < 0)
			list[k++] = i;
	}
}
/*help fuction ends*/








__device__ void greedyInsertion(Solution *ps, Data data, int k)
{
	__shared__ double minCost[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];
	__shared__ int *removedList;

	if (threadIdx.x == 0) {
		cudaMalloc(&removedList, k * sizeof(int));
	}

	while (k > 0) {
		__syncthreads();
		minCost[threadIdx.x] = COST;
		bestIdx[threadIdx.x] = -1;
		if (threadIdx.x == 0)
			getRemovedList(ps, removedList, k);

		__syncthreads();
		int x = threadIdx.x;
		while (x < k * (ps->m + ps->n)) {
			double c = costFunction(ps, x, removedList, k, data);
			if (c < minCost[threadIdx.x]) {
				minCost[threadIdx.x] = c;
				bestIdx[threadIdx.x] = x;
			}
			x += BLOCK_SIZE;
		}

		__syncthreads();
		minReduction(minCost, bestIdx);
		if (threadIdx.x == 0) {
			int i = bestIdx[0] / k;//insert pos
			int j = bestIdx[0] - k * i;
			insert(ps, i + 1, removedList[j], data);
		}
		k--;
	}

	if (threadIdx.x == 0)
		cudaFree(removedList);
	__syncthreads();
}

__device__ void regret2Insertion(Solution *ps, Data data, int k)
{
	__shared__ double r2Value[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];
	__shared__ int *removedList;

	if (threadIdx.x == 0) {
		cudaMalloc(&removedList, k * sizeof(int));
	}

	while (k > 0) {
		__syncthreads();
		r2Value[threadIdx.x] = -COST;
		bestIdx[threadIdx.x] = -1;
		if (threadIdx.x == 0) {
			getRemovedList(ps, removedList, k);
		}

		__syncthreads();
		int x = threadIdx.x;//x here is the idx in removed list
		while (x < k) {
			//compute the r2 value for removedList[x]
			double minCost = COST;
			double secondMinCost = COST;
			int thisIdx = -1;
			for (int i = 1;i <= ps->n + ps->m;i++) {
				double c = costFunction(ps, (i - 1) * k + x, removedList, k, data);
				if (c < minCost) {
					secondMinCost = minCost;
					minCost = c;
					thisIdx = (i - 1) * k + x;
				}
				else if (c < secondMinCost) {
					secondMinCost = c;
				}
			}
			double r2V = secondMinCost - minCost;//the r2V for removedList[x]

			//then update the shared r2Value&bestIdx
			if (r2V > r2Value[threadIdx.x]) {
				r2Value[threadIdx.x] = r2V;
				bestIdx[threadIdx.x] = thisIdx;
			}

			x += k;//thread 0 will consider r0,rk,r2k etc...
		}
		r2Value[threadIdx.x] = -r2Value[threadIdx.x];//to use minReduction as maxReduction

		__syncthreads();
		minReduction(r2Value, bestIdx);
		if (threadIdx.x == 0) {
			int i = bestIdx[0] / k;//insert pos
			int j = bestIdx[0] - k * i;
			insert(ps, i + 1, removedList[j], data);
		}
		k--;
	}

	if (threadIdx.x == 0)
		cudaFree(removedList);
	__syncthreads();
}

__device__ void randomRemoval(Solution *ps, Data data, int k)
{
	if (threadIdx.x == 0) {
		//this is a single thread operation
		int toBeRemoved = 0;
		curandState_t state;
		curand_init(clock(), blockIdx.x, 0, &state);
		
		while (k--) {
			while (ps->next[toBeRemoved] < 0) {
				toBeRemoved = (curand(&state) % ps->n) + 1;//random number between 1 -> n
			}
			remove(ps, toBeRemoved, data);
		}
	}
	__syncthreads();
}

__device__ void worstRemoval(Solution *ps, Data data, int k)
{
	__shared__ double minCost[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];

	while (k > 0) {
		__syncthreads();
		minCost[threadIdx.x] = COST;
		bestIdx[threadIdx.x] = -1;

		__syncthreads();
		int x = threadIdx.x + 1;//thread 0 will compute removing the first request
		while (x <= ps->n) {
			double c = COST * 2;
			if (inSolution(ps,x))
				c = dist(data, ps->next[x], ps->prev[x]) - dist(data, x, ps->next[x]) - dist(data, x, ps->prev[x]);
			if (c < minCost[threadIdx.x]) {
				minCost[threadIdx.x] = c;
				bestIdx[threadIdx.x] = x;
			}

			x += BLOCK_SIZE;
		}

		__syncthreads();
		minReduction(minCost, bestIdx);
		if (threadIdx.x == 0) {
			remove(ps, bestIdx[0], data);
		}
		k--;
	}
	__syncthreads();
}

__device__ void relatedRemoval(Solution *ps, Data data, int k)
{
	__shared__ double mostRelated[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];
	__shared__ int seed;
	if (threadIdx.x == 0) {
		curandState_t state;
		curand_init(clock(), blockIdx.x, 0, &state);
		seed = curand(&state) % ps->n + 1;
	}
	__syncthreads();
	int last = seed;

	while (k > 0) {
		__syncthreads();
		mostRelated[threadIdx.x] = COST;
		bestIdx[threadIdx.x] = -1;

		__syncthreads();
		int x = threadIdx.x + 1;
		while (x <= ps->n) {
			int d = 2 * COST;
			if (inSolution(ps, x))
				d = dist(data, x, last);
			if (d < mostRelated[threadIdx.x]) {
				mostRelated[threadIdx.x] = d;
				bestIdx[threadIdx.x] = x;
			}
			x += BLOCK_SIZE;
		}

		__syncthreads();
		minReduction(mostRelated, bestIdx);
		if (threadIdx.x == 0) {
			remove(ps, bestIdx[0], data);
		}
		last = bestIdx[0];//record the last removed;
		k--;
	}
	__syncthreads();
}

__device__ void nodePairRemoval(Solution *ps, Data data, int k)
{
	__shared__ double edgeValue[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];

	while (k > 0) {
		__syncthreads();
		edgeValue[threadIdx.x] = -COST;
		bestIdx[threadIdx.x] = -1;

		__syncthreads();
		int x = threadIdx.x + 1;
		while (x <= ps->n) {
			double value = -COST * 2;
			if (inSolution(ps, x))
				value = dist(data, x, ps->next[x]) + dist(data, x, ps->prev[x]);
			if (value > edgeValue[threadIdx.x]) {
				edgeValue[threadIdx.x] = value;
				bestIdx[threadIdx.x] = x;
			}
			x += BLOCK_SIZE;
		}
		edgeValue[threadIdx.x] = -edgeValue[threadIdx.x];

		__syncthreads();
		minReduction(edgeValue, bestIdx);
		if (threadIdx.x == 0) {
			remove(ps, bestIdx[0], data);
		}

		k--;
	}
	__syncthreads();
}

//__device__ void clusterRemoval(Solution s, Data data, int k)
//{
//
//}