#ifndef SOLUTION
#define SOLUTION

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct {
	int n,m;//n is the number of customers, m is the number of artificial depots, total 1+n+m nodes
	int *next;
	int *prev;
	int *tourid;

	double totalCost;
	int *tourDemend;//record the demand on this tour
} Solution;


typedef struct {
	int numCus;
	int capacity;

	int *demand;
	double *x;
	double *y;
} Data;

//这里最后也许要输入device

__device__ void insert (Solution *ps, int insertPos, int toBeInserted, Data data);//insert and updata the cost
__device__ void remove (Solution *ps, int toBeRemoved, Data data);
__device__ Solution createEmptySolution(int n, int m);
__device__ void deleteSolution(Solution solution);
__device__ double dist(Data data,int i, int j);
__device__ void copySolution(Solution* pTarget, Solution* pSource);

#endif 
