#include "solution.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ void insert(Solution *ps, int insertPos, int toBeInserted, Data data)
{
	//insert a customer not in solution, and update the cost&demand
	if (toBeInserted >= 1 && toBeInserted <= ps->n && ps->next[toBeInserted] < 0 && ps->next[insertPos] >= 0) {
		//the insertPos has to be valid
		int prev = insertPos;
		int next = ps->next[prev];
		int tourid = ps->tourid[prev];
		//do the insertion
		ps->next[toBeInserted] = next;
		ps->prev[toBeInserted] = prev;
		ps->next[prev] = toBeInserted;
		ps->prev[next] = toBeInserted;
		//update cost&demand
		double costChange = dist(data, prev, toBeInserted) + dist(data, next, toBeInserted) - dist(data, next, prev);
		ps->totalCost += costChange;
		ps->tourDemend[tourid] += data.demand[toBeInserted];
	}
}

__device__ void remove(Solution *ps, int toBeRemoved, Data data)
{
	if (toBeRemoved >= 1 &&toBeRemoved <= ps->n && ps->next[toBeRemoved] >= 0) {
		//it has to be a correct index to be removed from the current solution
		int prev = ps->prev[toBeRemoved];
		int next = ps->next[toBeRemoved];
		int tourid = ps->tourid[toBeRemoved];
		//do the remove
		ps->next[prev] = next;
		ps->prev[next] = prev;
		ps->next[toBeRemoved] = -1;
		//update cost&demand
		double costChange = dist(data,prev,next) - dist(data,prev,toBeRemoved) - dist(data,next,toBeRemoved);
		ps->totalCost += costChange;
		ps->tourDemend[tourid] -= data.demand[toBeRemoved];
	}
}

__device__ double dist(Data data, int i, int j)
{
	double d = (data.x[i] - data.x[j]) * (data.x[i] - data.x[j]) + (data.y[i] - data.y[j]) * (data.y[i] - data.y[j]);
	return d;
}

__device__ Solution createEmptySolution(int n, int m)
{
	//a solution is created, with empty tour only
	Solution s;
	int size = m + n + 1;
	cudaMalloc(&s.next, size * sizeof(int));
	cudaMalloc(&s.prev, size * sizeof(int));
	cudaMalloc(&s.tourid, size * sizeof(int));
	cudaMalloc(&s.tourDemend, m * sizeof(int));

	s.m = m;
	s.n = n;
	s.totalCost = 0.0;
	for (int i = 0;i < size;i++) {
		if (i <= n) {
			//depot 0 and requests
			s.next[i] = -1;
		}
		else {
			//artificial depots
			s.next[i] = i + 1;
			s.prev[i] = i - 1;
			s.tourid[i] = i - n - 1;
		}
		if (i < m) {
			s.tourDemend[i] = 0;
		}
	}
	s.next[m + n] = 0;
	s.prev[0] = m + n;
	s.prev[n + 1] = -1;


	return s;
}
__device__ void deleteSolution(Solution solution)
{
	cudaFree(solution.next);
	cudaFree(solution.prev);
	cudaFree(solution.tourid);
	cudaFree(solution.tourDemend);
}

__device__ void copySolution(Solution* pTarget, Solution* pSource)
{
	//the solution is made of solution itself and the pointed memory
	int size = pSource->m + pSource->n + 1;
	int m = pSource->m;

	pTarget->m = pSource->m;
	pTarget->n = pSource->n;
	pTarget->totalCost = pSource->totalCost;
	cudaMemcpyAsync(pTarget->next, pSource->next, size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(pTarget->prev, pSource->prev, size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(pTarget->tourid, pSource->tourid, size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(pTarget->tourDemend, pSource->tourDemend, m * sizeof(int), cudaMemcpyDeviceToDevice);
}