#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
//#include <vector>


#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {		// Функция которая сравнивает объекты на "<"
		return fabs(a) < fabs(b); 									// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};


__global__ void forward(double* dev_matrix, int n, int i) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for (size_t y = idy + i + 1; y < n * 2; y += offsety) {
		for (size_t x = idx + i + 1; x < n; x += offsetx) {
			dev_matrix[y * n + x] -= dev_matrix[y * n + i] * dev_matrix[i * n + x];
		}
	}
}

__global__ void backward(double* dev_matrix, int n, int i) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for (size_t y = idy + i + 1; y < n * 2; y += offsety) {
		for (size_t x = idx; x <= i - 1; x += offsetx) {
			dev_matrix[y * n + x] -= dev_matrix[y * n + i] * dev_matrix[i * n + x];
		}
	}
}

__global__ void swap(double* dev_matrix, int n, int i, int i_max) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	double swap;
	for (size_t x = idx; x < n * 2; x += offset) {
		swap = dev_matrix[x * n + i_max];
		dev_matrix[x * n + i_max] = dev_matrix[x * n + i];
		dev_matrix[x * n + i] = swap;
	}
}

__global__ void divide_line(double *dev_matrix, int n, int i) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (size_t x = idx + i + 1; x < n * 2; x += offsetx) {
		dev_matrix[x * n + i] /= dev_matrix[i * n + i];
	}
}

int main() {
	int							n,
								i_max;
	double						*matrix, 
								*dev_matrix;
	comparator					cmp;
	thrust::device_ptr<double>	pi;

	std::cin >> n;
	matrix = new double[n * n * 2];
	for (size_t y = 0; y < n; y++) {
		for (size_t x = 0; x < n; x++)
			std::cin >> matrix[y + x*n];			
	}
	for (size_t y = 0; y < n; y++) {
		for (size_t x = n; x < n*2; x++)
			matrix[y + x * n] = (x == y+n? 1. : 0.);
	}
	
	CSC(cudaMalloc(&dev_matrix, sizeof(double) * n * n * 2));
	CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double)*n*n*2, cudaMemcpyHostToDevice));

	for (size_t i = 0; i < n-1; i++) {
		pi = thrust::device_pointer_cast(dev_matrix + i*n);
		i_max = thrust::max_element(pi + i, pi + n, cmp) - pi;
		if (i != i_max) swap <<<256, 256>>> (dev_matrix, n, i, i_max);
		divide_line << <256, 256 >> > (dev_matrix, n, i);
		CSC(cudaGetLastError());
		forward <<<dim3(32, 32), dim3(32, 32) >>> (dev_matrix, n, i);
		CSC(cudaGetLastError());	
	}

	divide_line <<<256, 256>>> (dev_matrix, n, n-1);
	CSC(cudaGetLastError());
	
	for (int i = n-1; i > 0; i--) {
		backward <<<dim3(32, 32), dim3(32, 32) >>> (dev_matrix, n, i);
		CSC(cudaGetLastError());
	}


	CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_matrix));

	std::cout.precision(10);
	std::cout.setf(std::ios::scientific);
	for (size_t y = 0; y < n; y++) {
		for (size_t x = n; x < n * 2; x++)
			std::cout << matrix[y + x * n] << ' ';
		std::cout << '\n';
	}
	free(matrix);
	return 0;
}
