
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "solution.cuh"
#include "kernelPara.cuh"
#include "curand_kernel.h"
#include "operator.cuh"

__global__ void makeInitSolution(Solution* d_s, int *randomNumber, Data data);
__global__ void ALNSkernel(Solution* d_s, Data data, double** destoryWeight, double** repairWeight);

__device__ bool feasible(Solution* ps, int insertPos, int toBeInsert, Data data);

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

void input(Data *pdata, Data *pd_data)
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
	cudaMalloc(&pd_data->demand, (nCus + 1) * sizeof(int));
	cudaMalloc(&pd_data->x, (nCus + 1) * sizeof(double));
	cudaMalloc(&pd_data->y, (nCus + 1) * sizeof(double));

	/*cudaMemcpy(&pd_data->numCus, &pdata->numCus, sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(&pd_data->capacity, &pdata->capacity, sizeof(int), cudaMemcpyHostToDevice);*/
	pd_data->capacity = capacity;
	pd_data->numCus = nCus;
	cudaMemcpy(pd_data->demand, pdata->demand, (nCus + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pd_data->x, pdata->x, (nCus + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pd_data->y, pdata->y, (nCus + 1) * sizeof(double), cudaMemcpyHostToDevice);
}

void initSolution(Solution s[], Solution *d_s, Data d_data)
{
	//to allocate the memory for pointers in Solution and make it an feasible solution(use kernel)
	//attention: s[] in the host memory is just a copy of *d_s in the device memory, it also points to the device memory
	//for each solution, random insert k request first and greedy insert the left

	//generate random numbers for further use
	int problemSize = d_data.numCus;
	int randomNumber[GRID_SIZE];
	int *d_randomNumber;
	cudaMalloc(&d_randomNumber, GRID_SIZE * sizeof(int));

	srand(time(NULL));
	for (int i = 0;i < GRID_SIZE;i++) {
		randomNumber[i] = rand() % (int)(problemSize * RANDOM_PARA);//up to 70% request can be insert randomly
	}
	cudaMemcpy(d_randomNumber, randomNumber, GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	makeInitSolution<<<GRID_SIZE,BLOCK_SIZE>>>(d_s,d_randomNumber,d_data);
	cudaFree(d_randomNumber);
}

void ALNS(Solution s[], Solution *d_s, Data d_data)
{
	//this function help to lanuch ALNS kernel many times
	//allocate the memory for weight of operators, and set the initial weight value(1.0)
	double *destoryWeight[GRID_SIZE];
	double *repairWeight[GRID_SIZE];
	for (int i = 0;i < GRID_SIZE;i++) {
		cudaMalloc(&destoryWeight[i], NUM_DESTORY * sizeof(double));
		cudaMalloc(&repairWeight[i], NUM_REPAIR * sizeof(double));
	}
	double initDes[NUM_DESTORY];
	double initRe[NUM_REPAIR];
	for (int i = 0;i < NUM_DESTORY;i++)
		initDes[i] = INIT_WEIGHT;
	for (int i = 0;i < NUM_REPAIR;i++)
		initRe[i] = INIT_WEIGHT;
	for (int i = 0;i < GRID_SIZE;i++) {
		cudaMemcpy(destoryWeight[i], initDes, NUM_DESTORY * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(repairWeight[i], initRe, NUM_REPAIR * sizeof(double), cudaMemcpyHostToDevice);
	}
	double** d_destoryWeight, ** d_repairWeight;
	cudaMalloc(&d_destoryWeight, GRID_SIZE * sizeof(double*));
	cudaMalloc(&d_repairWeight, GRID_SIZE * sizeof(double*));
	cudaMemcpy(d_destoryWeight, destoryWeight, GRID_SIZE * sizeof(double*),cudaMemcpyHostToDevice);
	cudaMemcpy(d_repairWeight, repairWeight, GRID_SIZE * sizeof(double*), cudaMemcpyHostToDevice);
	
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

		ALNSkernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_s, d_data, d_destoryWeight, d_repairWeight);
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

void printBest(Solution s[], Solution *d_s, Data data)
{
	int n = s[0].n;
	int m = s[0].m;
	int* next = (int*)malloc((m + n + 1) * sizeof(int));

	cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
	int bestIdx = getBestSolutionIdx(s);
	cudaMemcpy(next, s[bestIdx].next, (m + n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	
	int cur = n + 1;//start from the first artificial depot
	while (cur >= 0) {
		if (cur >= 1 && cur <= n) {
			//is a request
			printf("request %d, x:%.4f y:%.4f\n", cur, data.x[cur], data.y[cur]);
		}
		else
			printf("depot 0\n");

		cur = next[cur];
	}
	free(next);
}


void afterWork(Solution s[], Solution *d_s, Data data, Data d_data)
{
	free(data.demand);
	free(data.x);
	free(data.y);
	cudaFree(d_data.demand);
	cudaFree(d_data.x);
	cudaFree(d_data.y);
	
	cudaMemcpy(s, d_s, GRID_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
	for (int i = 0;i < GRID_SIZE;i++) {
		cudaFree(s[i].next);
		cudaFree(s[i].prev);
		cudaFree(s[i].tourDemend);
		cudaFree(s[i].tourid);
	}
	cudaFree(d_s);
}

int main(void)
{
	//the s[] array is stored in the host, but points to the device memory
	//the *d_s points to a Solution array in the device memory whose pointer points to the device momory
	Solution s[GRID_SIZE];
	Solution *d_s;
	cudaMalloc(&d_s, GRID_SIZE * sizeof(Solution));

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


__global__ void makeInitSolution(Solution *d_s, int *randomNumber, Data data)
{
	//block i will take care of d_s[i] and insert randomNumber[i] requests randomly
	int problemSize = data.numCus;
	int nDepot = data.numCus;//this can be changed accroding to the problem
	Solution *pS = &d_s[blockIdx.x];

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

__global__ void ALNSkernel(Solution *d_s, Data data, double **destoryWeight, double **repairWeight)
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