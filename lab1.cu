#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

#define CSC(call)					\
do {								\
	cudaError_t status = call;		\
if (status != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
		exit(0);					\
	}								\
} while (0)

__global__ void kernel(double* arr1, double* arr2, int n) {
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
	int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for (i = idx; i < n; i += offset)
		arr1[i] = (arr1[i] <= arr2[i] ? arr1[i] : arr2[i]);
}

int main() {
	int i, n;
	double num;
	std::cin >> n;
	double* arr1 = (double*)malloc(sizeof(double) * n);
	double* arr2 = (double*)malloc(sizeof(double) * n);
	for (i = 0; i < n; i++) {
		std::cin >> num;
		arr1[i] = num;
	}
	for (i = 0; i < n; i++) {
		std::cin >> num;
		arr2[i] = num;
	}

	double*dev_arr1, *dev_arr2;
	CSC(cudaMalloc(&dev_arr1, sizeof(double) * n));
	CSC(cudaMalloc(&dev_arr2, sizeof(double) * n));
	CSC(cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice));

	kernel<<<256,256>>>(dev_arr1, dev_arr2, n);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(arr1, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr1));
	CSC(cudaFree(dev_arr2));
	for (i = 0; i < n; i++)
		printf("%f ", arr1[i]);
	printf("\n");
	free(arr1);
	return 0;
}
