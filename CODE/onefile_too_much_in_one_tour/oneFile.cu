#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <time.h>
#include <limits.h>
#include <stdio.h>
#include <Windows.h>

typedef struct {
	int n, m;//n is the number of customers, m is the number of artificial depots, total 1+n+m nodes
	int* next;
	int* prev;
	int* tourid;

	double totalCost;
	int* tourDemend;//record the demand on this tour
} Solution;


typedef struct {
	int numCus;
	int capacity;

	int* demand;
	double* x;
	double* y;
} Data;



__device__ void insert(Solution* ps, int insertPos, int toBeInserted, Data data);//insert and updata the cost
__device__ void remove(Solution* ps, int toBeRemoved, Data data);
__device__ Solution createEmptySolution(int n, int m);
__device__ void deleteSolution(Solution solution);
__device__ double dist(Data data, int i, int j);
__device__ void copySolution(Solution* pTarget, Solution* pSource);

__device__ void greedyInsertion(Solution* ps, Data data, int k);
__device__ void regret2Insertion(Solution* ps, Data data, int k);
__device__ void randomRemoval(Solution* ps, Data data, int k);
__device__ void worstRemoval(Solution* ps, Data data, int k);
__device__ void relatedRemoval(Solution* ps, Data data, int k);
__device__ void nodePairRemoval(Solution* ps, Data data, int k);

__global__ void makeInitSolution(Solution* d_s, int* randomNumber, Data data);
__global__ void ALNSkernel(Solution* d_s, Data data, double** destoryWeight, double** repairWeight);
__device__ bool feasible(Solution* ps, int insertPos, int toBeInsert, Data data);

#define GRID_SIZE 8
#define BLOCK_SIZE 32
#define RANDOM_PARA 0.7
#define REMOVE_RATIO 0.5
#define NUM_DESTORY 4
#define NUM_REPAIR 2
#define INIT_WEIGHT 1.0
#define ALNS_REPEAT 100

#define ACCEPT_SCORE 1
#define NEW_BEST_SCORE 2;
#define RHO 0.5 

/*some error check code*/
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		//exit(-1);
	}
#endif

	return;
}
inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}
/*error check code ends*/


/*the code starts here*/
__device__ void insert(Solution* ps, int insertPos, int toBeInserted, Data data)
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

__device__ void remove(Solution* ps, int toBeRemoved, Data data)
{
	if (toBeRemoved >= 1 && toBeRemoved <= ps->n && ps->next[toBeRemoved] >= 0) {
		//it has to be a correct index to be removed from the current solution
		int prev = ps->prev[toBeRemoved];
		int next = ps->next[toBeRemoved];
		int tourid = ps->tourid[toBeRemoved];
		//do the remove
		ps->next[prev] = next;
		ps->prev[next] = prev;
		ps->next[toBeRemoved] = -1;
		//update cost&demand
		double costChange = dist(data, prev, next) - dist(data, prev, toBeRemoved) - dist(data, next, toBeRemoved);
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
	//CudaSafeCall(cudaMalloc(&s.next, size * sizeof(int)));
	//CudaSafeCall(cudaMalloc(&s.prev, size * sizeof(int)));
	//CudaSafeCall(cudaMalloc(&s.tourid, size * sizeof(int)));
	//CudaSafeCall(cudaMalloc(&s.tourDemend, m * sizeof(int)));
	s.next = (int*)malloc(size * sizeof(int));
	s.prev = (int*)malloc(size * sizeof(int));
	s.tourid = (int*)malloc(size * sizeof(int));
	s.tourDemend = (int*)malloc(m * sizeof(int));

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
	//cudaFree(solution.next);
	//cudaFree(solution.prev);
	//cudaFree(solution.tourid);
	//cudaFree(solution.tourDemend);
	free(solution.next);
	free(solution.prev);
	free(solution.tourid);
	free(solution.tourDemend);
}

__device__ void copySolution(Solution* pTarget, Solution* pSource)
{
	//the solution is made of solution itself and the pointed memory
	int size = pSource->m + pSource->n + 1;
	int m = pSource->m;

	pTarget->m = pSource->m;
	pTarget->n = pSource->n;
	pTarget->totalCost = pSource->totalCost;
	//cudaMemcpyAsync(pTarget->next, pSource->next, size * sizeof(int), cudaMemcpyDeviceToDevice);
	//cudaMemcpyAsync(pTarget->prev, pSource->prev, size * sizeof(int), cudaMemcpyDeviceToDevice);
	//cudaMemcpyAsync(pTarget->tourid, pSource->tourid, size * sizeof(int), cudaMemcpyDeviceToDevice);
	//cudaMemcpyAsync(pTarget->tourDemend, pSource->tourDemend, m * sizeof(int), cudaMemcpyDeviceToDevice);
	memcpy(pTarget->next, pSource->next, size * sizeof(int));
	memcpy(pTarget->prev, pSource->prev, size * sizeof(int));
	memcpy(pTarget->tourid, pSource->tourid, size * sizeof(int));
	memcpy(pTarget->tourDemend, pSource->tourDemend, m * sizeof(int));
}

#define COST 10000.0//used for init minCost





/*help fuctions*/
__device__ bool inSolution(Solution * ps, int i)
{
	return ps->next[i] >= 0 ? true : false;
}

__device__ bool feasible(Solution* ps, int insertPos, int toBeInsert, Data data)
{
	//CVRP, check the demand
	return ps->tourDemend[ps->tourid[insertPos]] + data.demand[toBeInsert] <= data.capacity;
}

__device__ int costFunction(Solution* ps, int x, int* removedList, int k, Data data)
{
	int i = x / k;//insertPos 
	int j = x - k * i;
	double c = COST * 2;
	if (inSolution(ps, i + 1) && feasible(ps, i + 1, removedList[j], data)) {
		c = dist(data, i + 1, removedList[j]) + dist(data, removedList[j], ps->next[i + 1]) - dist(data, i + 1, ps->next[i + 1]);
	}
	return c;
}

__device__ void minReduction(double* minCost, int* bestIdx)
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

__device__ void getRemovedList(Solution* ps, int* list, int k)
{
	k = 0;
	for (int i = 1;i <= ps->n;i++) {
		if (ps->next[i] < 0)
			list[k++] = i;
	}
}
/*help fuction ends*/




__device__ void greedyInsertion(Solution* ps, Data data, int k)
{
	__shared__ double minCost[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];
	__shared__ int* removedList;

	if (threadIdx.x == 0) {
		removedList = (int*)malloc(k * sizeof(int));
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
		free(removedList);
	__syncthreads();
}

__device__ void regret2Insertion(Solution* ps, Data data, int k)
{
	__shared__ double r2Value[BLOCK_SIZE];
	__shared__ int bestIdx[BLOCK_SIZE];
	__shared__ int* removedList;

	if (threadIdx.x == 0) {
		removedList = (int*)malloc(k * sizeof(int));
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
		free(removedList);
	__syncthreads();
}

__device__ void randomRemoval(Solution* ps, Data data, int k)
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

__device__ void worstRemoval(Solution* ps, Data data, int k)
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
			if (inSolution(ps, x))
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

__device__ void relatedRemoval(Solution* ps, Data data, int k)
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

__device__ void nodePairRemoval(Solution* ps, Data data, int k)
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


/*main file starts here*/
int getBestSolutionIdx(Solution s[])
{
	int idx = -1;
	double minCost = 10000.0;
	for (int i = 0;i < GRID_SIZE;i++) {
		if (minCost > s[i].totalCost) {
			minCost = s[i].totalCost;
			idx = i;
		}
	}
	return idx;
}
bool restart(Solution s[], Solution* d_s, int count, int noImprovement, double best)
{
	//first
	if (count > 125 && count <= 250 && noImprovement > 30)
		return false;
	else if (count > 250 && noImprovement > 10)
		return false;
	//second
	double deviation = 0.0;
	for (int i = 0;i < GRID_SIZE;i++) {
		deviation += (s[i].totalCost - best) * (s[i].totalCost - best);
	}
	deviation /= (double)(GRID_SIZE - 1);
	deviation = sqrt(deviation);
	double T = deviation / best;

	if (T < 0.005 && noImprovement>20)
		return false;
	else if (T < 0.00125 && noImprovement>5)
		return false;
	//third
	int countEqual = 0;
	for (int i = 0;i < GRID_SIZE;i++) {
		if (fabs(s[i].totalCost - best) < 1e-6)
			countEqual++;
	}
	if ((double)countEqual / GRID_SIZE > 0.15 && T < 0.002 && noImprovement>10)
		return false;

	//if no terminal condition is satisfied, return true to continue
	return true;
}

void input(Data* pdata, Data* pd_data)
{
	//to allocate the memory for pointers in Data, and to input data
	int nCus, capacity;
	scanf("%d %d", &nCus, &capacity);
	//to input data to the 'data' on the host first
	pdata->numCus = nCus;
	pdata->capacity = capacity;
	pdata->demand = (int*)malloc((nCus + 1) * sizeof(int));
	pdata->x = (double*)malloc((nCus + 1) * sizeof(double));
	pdata->y = (double*)malloc((nCus + 1) * sizeof(double));

	pdata->demand[0] = 0;
	pdata->x[0] = 0.0;
	pdata->y[0] = 0.0;
	for (int i = 1;i <= nCus;i++) {
		int demand;
		double x, y;
		scanf("%d %lf %lf", &demand, &x, &y);
		pdata->demand[i] = demand;
		pdata->x[i] = x;
		pdata->y[i] = y;
	}
	//then to copy the data to the device...ok let's do it
	CudaSafeCall(cudaMalloc(&pd_data->demand, (nCus + 1) * sizeof(int)));
	CudaSafeCall(cudaMalloc(&pd_data->x, (nCus + 1) * sizeof(double)));
	CudaSafeCall(cudaMalloc(&pd_data->y, (nCus + 1) * sizeof(double)));

	pd_data->capacity = capacity;
	pd_data->numCus = nCus;
	CudaSafeCall(cudaMemcpy(pd_data->demand, pdata->demand, (nCus + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(pd_data->x, pdata->x, (nCus + 1) * sizeof(double), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(pd_data->y, pdata->y, (nCus + 1) * sizeof(double), cudaMemcpyHostToDevice));
}

void initSolution(Solution s[], Solution* d_s, Data d_data)
{
	//to allocate the memory for pointers in Solution and make it an feasible solution(use kernel)
	//attention: s[] in the host memory is just a copy of *d_s in the device memory, it also points to the device memory
	//for each solution, random insert k request first and greedy insert the left

	//generate random numbers for further use
	int problemSize = d_data.numCus;
	int randomNumber[GRID_SIZE];
	int* d_randomNumber;
	CudaSafeCall(cudaMalloc(&d_randomNumber, GRID_SIZE * sizeof(int)));

	srand(time(NULL));
	for (int i = 0;i < GRID_SIZE;i++) {
		randomNumber[i] = rand() % (int)(problemSize * RANDOM_PARA);//up to 70% request can be insert randomly
	}
	CudaSafeCall(cudaMemcpy(d_randomNumber, randomNumber, GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice));



	makeInitSolution << <GRID_SIZE, BLOCK_SIZE >> > (d_s, d_randomNumber, d_data);
	CudaCheckError();//leave this to check
	cudaFree(d_randomNumber);
}

void ALNS(Solution s[], Solution* d_s, Data d_data)
{
	printf("in ALNS func\n");
	//this function help to lanuch ALNS kernel many times
	//allocate the memory for weight of operators, and set the initial weight value(1.0)
	double* destoryWeight[GRID_SIZE];
	double* repairWeight[GRID_SIZE];
	for (int i = 0;i < GRID_SIZE;i++) {
		CudaSafeCall(cudaMalloc(&destoryWeight[i], NUM_DESTORY * sizeof(double)));
		CudaSafeCall(cudaMalloc(&repairWeight[i], NUM_REPAIR * sizeof(double)));
	}
	double initDes[NUM_DESTORY];
	double initRe[NUM_REPAIR];
	for (int i = 0;i < NUM_DESTORY;i++)
		initDes[i] = INIT_WEIGHT;
	for (int i = 0;i < NUM_REPAIR;i++)
		initRe[i] = INIT_WEIGHT;
	for (int i = 0;i < GRID_SIZE;i++) {
		CudaSafeCall(cudaMemcpy(destoryWeight[i], initDes, NUM_DESTORY * sizeof(double), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(repairWeight[i], initRe, NUM_REPAIR * sizeof(double), cudaMemcpyHostToDevice));
	}
	double** d_destoryWeight, ** d_repairWeight;
	CudaSafeCall(cudaMalloc(&d_destoryWeight, GRID_SIZE * sizeof(double*)));
	CudaSafeCall(cudaMalloc(&d_repairWeight, GRID_SIZE * sizeof(double*)));
	CudaSafeCall(cudaMemcpy(d_destoryWeight, destoryWeight, GRID_SIZE * sizeof(double*), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_repairWeight, repairWeight, GRID_SIZE * sizeof(double*), cudaMemcpyHostToDevice));

	//go and lanuch the ALNS kernel!
	bool tag = true;
	int count = 0;
	int noImprovement = 0;
	while (tag) {
		//record the solution before start ALNS kernel
		count++;

		cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
		int bestIdx = getBestSolutionIdx(s);
		double lastBest = s[bestIdx].totalCost;

		ALNSkernel<<<GRID_SIZE, BLOCK_SIZE >>>(d_s, d_data, d_destoryWeight, d_repairWeight);
		printf("the %d time ALNS kernel finish\n", count);
		cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
		bestIdx = getBestSolutionIdx(s);
		double newBest = s[bestIdx].totalCost;

		if (newBest < lastBest) {
			noImprovement = 0;
		}
		else {
			noImprovement++;
		}

		tag = restart(s, d_s, count, noImprovement, newBest);

	}
	//do the work after the kernel need not to be restarted
	for (int i = 0;i < GRID_SIZE;i++) {
		cudaFree(destoryWeight[i]);
		cudaFree(repairWeight[i]);
	}
	cudaFree(d_destoryWeight);
	cudaFree(d_repairWeight);
}


__global__ void copyKernel(int* source, int* target,int size)
{
	for (int i = 0;i < size;i++) {
		target[i] = source[i];
	}
}
void printBest(Solution s[], Solution* d_s, Data data)
{
	cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
	int n = s[0].n;
	int m = s[0].m;

	int* next = (int*)malloc((m + n + 1) * sizeof(int));
	int* dnext;
	cudaMalloc(&dnext, (m + n + 1) * sizeof(int));
	int bestIdx = getBestSolutionIdx(s);
	//cudaMemcpy(next, s[bestIdx].next, (m + n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	copyKernel << <1, 1 >> > (s[bestIdx].next, dnext, m + n + 1);
	CudaSafeCall(cudaMemcpy(next, dnext, (m + n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	
	int cur = n + 1;//start from the first artificial depot
	while (cur >= 0) {
		//system("pause");//pause here
		//printf("cur value %d\n", cur);
		if (cur >= 1 && cur <= n) {
			//is a request
			printf("request %d, x:%.4f y:%.4f\n", cur, data.x[cur], data.y[cur]);
		}
		else
			printf("depot %d\n",cur);
		cur = next[cur];
	}
	free(next);
	cudaFree(dnext);
}

__global__ void freeKernel(Solution *s)
{
	for (int i = 0;i < GRID_SIZE;i++) {
		deleteSolution(s[i]);
	}
}
void afterWork(Solution s[], Solution* d_s, Data data, Data d_data)
{
	free(data.demand);
	free(data.x);
	free(data.y);
	cudaFree(d_data.demand);
	cudaFree(d_data.x);
	cudaFree(d_data.y);

	//cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
	//for (int i = 0;i < GRID_SIZE;i++) {
	//	cudaFree(s[i].next);
	//	cudaFree(s[i].prev);
	//	cudaFree(s[i].tourDemend);
	//	cudaFree(s[i].tourid);
	//}
	freeKernel << <1, 1 >> > (d_s);
	cudaFree(d_s);
}

int main(void)
{
	//the s[] array is stored in the host, but points to the device memory
	//the *d_s points to a Solution array in the device memory whose pointer points to the device momory
	Solution s[GRID_SIZE];
	Solution* d_s;
	CudaSafeCall(cudaMalloc(&d_s, GRID_SIZE * sizeof(Solution)));


	//data & d_data both exist in the host memory, but their pointer area is different
	//one points to the host memory and one points to the device memory
	//d_data in the host memory makes it possible to copy data from host to device
	//pass d_data to the kernel so the GPU can get access to the needed data
	Data data;
	Data d_data;


	//input data
	input(&data, &d_data);


	//to make empty solution, and to initiate them to feasible solutions
	initSolution(s, d_s, d_data);

	//to use ALNS
	ALNS(s, d_s, d_data);

	//after the ALNS kernel, print the result and do the after work
	printBest(s, d_s, data);
	afterWork(s, d_s, data, d_data);

	return 0;
}




//write the kernel(and some device help function) here


__device__ int choose(double* weight, int size)
{
	//this function help to choose the operator, return a idx chosen
	double totalValue = 0.0;
	for (int i = 0;i < size;i++) {
		totalValue += weight[i];
	}
	curandState_t state;
	curand_init(clock(), blockIdx.x, 0, &state);
	double thisValue = curand_uniform_double(&state) * totalValue;
	double curValue = 0.0;
	for (int i = 0;i < size;i++) {
		curValue += weight[i];
		if (curValue >= thisValue)
			return i;
	}
	return size - 1;//avoid bug if thisValue is equal to totalValue
}
__device__ bool accept(Solution* pNew, Solution* pCur, int step)
{
	double costChange = pNew->totalCost - pCur->totalCost;
	if (costChange < 0.0) {
		return true;
	}
	else {
		double p = exp(-costChange / (double)(ALNS_REPEAT - step));
		curandState_t state;
		curand_init(clock(), blockIdx.x, 0, &state);
		double thisP = curand_uniform_double(&state);
		return thisP < p;
	}
}


__global__ void makeInitSolution(Solution* d_s, int* randomNumber, Data data)
{

	//block i will take care of d_s[i] and insert randomNumber[i] requests randomly
	int problemSize = data.numCus;
	int nDepot = data.numCus;//this can be changed accroding to the problem
	Solution* pS = &d_s[blockIdx.x];

	//make empty solution and random insert first, only single thread operation
	if (threadIdx.x == 0) {
		*pS = createEmptySolution(problemSize, nDepot);
		int toInsert = problemSize + 1;//this is the first artificial depot
		int insertPos = 0;
		curandState_t state;
		curand_init(clock(), blockIdx.x, 0, &state);

		int k = randomNumber[blockIdx.x];
		for (int i = 0; i < k; i++) {
			while (pS->next[toInsert] >= 0) {
				toInsert = curand(&state) % problemSize + 1;//from 1 to n
			}
			insertPos = 0;
			while (pS->next[insertPos] < 0 || !feasible(pS, insertPos, toInsert, data)) {
				insertPos = curand(&state) % (problemSize + nDepot) + 1;//from 1 to m+n
			}
			insert(pS, insertPos, toInsert, data);
		}
	}
	//the do the greedy insertion for the left
	int k = problemSize - randomNumber[blockIdx.x];
	__syncthreads();
	greedyInsertion(pS, data, k);
}

__global__ void ALNSkernel(Solution* d_s, Data data, double** destoryWeight, double** repairWeight)
{
	Solution* pS = &d_s[blockIdx.x];//pointer to curent solution, every thread hold one
	double* desW = destoryWeight[blockIdx.x];
	double* reW = repairWeight[blockIdx.x];
	int n = pS->n;
	__shared__ int k;
	__shared__ int idx_des;
	__shared__ int idx_re;
	__shared__ int des_score[NUM_DESTORY];
	__shared__ int re_score[NUM_REPAIR];
	__shared__ int des_call[NUM_DESTORY];
	__shared__ int re_call[NUM_REPAIR];
	__shared__ Solution bestS;
	__shared__ Solution newS;


	if (threadIdx.x == 0) {
		//thread 0 in each block set the score to be 0 before start
		for (int i = 0;i < NUM_DESTORY;i++) {
			des_score[i] = 0;
			des_call[i] = 0;
		}
		for (int i = 0;i < NUM_REPAIR;i++) {
			re_score[i] = 0;
			re_call[i] = 0;
		}
		bestS = createEmptySolution(pS->n, pS->m);
		newS = createEmptySolution(pS->n, pS->m);
		copySolution(&bestS, pS);
	}
	__syncthreads();

	for (int i = 0;i < ALNS_REPEAT;i++) {
		//do it 100 times here
		if (threadIdx.x == 0) {
			//printf("this is block %d, %d times in ALNS kernel\n", blockIdx.x, i);
			idx_des = choose(desW, NUM_DESTORY);
			idx_re = choose(reW, NUM_REPAIR);
			des_call[idx_des] ++;
			re_call[idx_re] ++;

			curandState_t state;
			curand_init(clock(), blockIdx.x, 0, &state);
			k = curand(&state) % (int)(n * REMOVE_RATIO) + 1;//up to 70% will be removed

			copySolution(&newS, pS);
		}
		__syncthreads();
		//do the operator on the newS, all threads take part in
		switch (idx_des) {
		case 0:
			randomRemoval(&newS, data, k);
			break;
		case 1:
			worstRemoval(&newS, data, k);
			break;
		case 2:
			relatedRemoval(&newS, data, k);
			break;
		case 3:
			nodePairRemoval(&newS, data, k);
			break;
		}
		switch (idx_re) {
		case 0:
			greedyInsertion(&newS, data, k);
			break;
		case 1:
			regret2Insertion(&newS, data, k);
			break;
		}
		//oprator ends
		if (threadIdx.x == 0 && accept(&newS, pS, i)) {
			//accept?
			copySolution(pS, &newS);
			des_score[idx_des] += ACCEPT_SCORE;
			re_score[idx_re] += ACCEPT_SCORE;
		}
		if (threadIdx.x == 0 && newS.totalCost < pS->totalCost) {
			//new best?
			copySolution(&bestS, &newS);
			des_score[idx_des] += NEW_BEST_SCORE;
			re_score[idx_re] += NEW_BEST_SCORE;
		}
	}

	//update the weight by the scores
	if (threadIdx.x == 0) {
		for (int i = 0;i < NUM_DESTORY;i++) {
			desW[i] = (1.0 - RHO) * desW[i] + RHO * des_call[i] / ALNS_REPEAT * des_score[i];
		}
		for (int i = 0;i < NUM_REPAIR;i++) {
			desW[i] = (1.0 - RHO) * reW[i] + RHO * re_call[i] / ALNS_REPEAT * re_score[i];
		}
		//do the work before kernel ends
		copySolution(pS, &bestS);
		deleteSolution(bestS);
		deleteSolution(newS);
	}
	__syncthreads();
}