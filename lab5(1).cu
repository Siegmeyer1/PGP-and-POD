#include <stdio.h>
#include <algorithm>
#include <float.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>

#define SPLIT_SIZE 32
#define BLOCK_SIZE 1024
#define GRID_SIZE  65535

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)




__global__ void SemiFilter(int N, int N_merge, int size, float* data) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	int step = N/2;
	for (int i = idx; i < size; i += offset) // blockId - IS NOT id of block of threads, IS id of block in bitonic sequence
	{										   //mergeBlockId - is kinda id of block in bitonic sequence of first semi-filter in current merge (damn this is hard to explain in words)
		int blockId = i/step;
		int first = i + step * blockId;
		int mergeBlockId = i / (N_merge/2);
		if ((mergeBlockId % 2 && data[first] < data[first + step]) || 
			(!(mergeBlockId % 2) && data[first] > data[first + step])) 
		{
			float tmp = data[first];
			data[first] = data[first + step];
			data[first + step] = tmp;
		}	
	}
}

void BitonicMerge(int size, int N_merge, float* data) {
	for (int N = N_merge; N >= 2; N /= 2)
	{
		SemiFilter<<<32, 32>>>(N, N_merge, size, data);
	}	
}

void BitonicSort(int size, float* data) {
	for (int N = 2; N <= size; N *= 2) {
		BitonicMerge(size, N, data);
	}
}



__global__ void _reduce(float* data, int size, bool max_out=false) {					//POSSIBLE TO MAKE WORK FASTER
	volatile __shared__ float shared_data[BLOCK_SIZE*2];
	int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	shared_data[threadIdx.x] = (max_out ? fmaxf(data[idx], data[idx + blockDim.x]) :
										fminf(data[idx], data[idx + blockDim.x]));
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			shared_data[threadIdx.x] = (max_out ? fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x+s]) :
												fminf(shared_data[threadIdx.x], shared_data[threadIdx.x+s]));
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) data[blockIdx.x] = shared_data[0];
}

float my_reduce_rec(float* data, int size, bool max_out=false) {
	float* result;
	result = (float*)malloc(sizeof(float));
	unsigned int block_size = BLOCK_SIZE; //since one thread compares 2 values, we can have block of n threads processing 2*n values
	unsigned int elems_in_block = block_size * 2;
	unsigned int grid_size = size / elems_in_block; //1 thread-block for 1 values-group-block
	if (size % elems_in_block) grid_size++;			// just in case

	_reduce<<<grid_size, block_size>>>(data, size, max_out);		//running reduction on whole array we have at the moment
	CSC(cudaGetLastError());

	if (grid_size <= elems_in_block) {																//if n of blocks is <= than n of values in one block than we can run
		_reduce<<<1, block_size>>>(data, size , max_out);			//our reduction again just once on just one block and get final answer
		CSC(cudaGetLastError());
		CSC(cudaMemcpy(result, data, sizeof(float), cudaMemcpyDeviceToHost));
	} else {
		result[0] = my_reduce_rec(data, grid_size, max_out);								//else it means we need more than one block, so we run this code again. We may run out
	}																						//of bounds of new size, but it`s ok since we won`t compare elements out of initial 
																							//array, so answer must be correct
	float r = result[0];
	free(result);
	return r;
}

float my_reduce(float* data, int n, bool max_out=false) {
	unsigned int values_n = BLOCK_SIZE * 2;
	int size = (n / values_n) * values_n;  //size of array completed to multiple of block size (element-wise)
	if (n % values_n) size += values_n;

	float *to_reduce = (float*)malloc(sizeof(float) * size); //extending array accordingly
	memcpy(to_reduce, data, sizeof(float)*n);
	std::fill(to_reduce+n, to_reduce+size, (max_out ? -FLT_MAX : FLT_MAX));

	float *gpu_to_reduce;									//moving to gpu
	CSC(cudaMalloc(&gpu_to_reduce, 	   sizeof(float) * size));
	CSC(cudaMemcpy(gpu_to_reduce, to_reduce, sizeof(float) * size, cudaMemcpyHostToDevice));

	float r = my_reduce_rec(gpu_to_reduce, size, max_out);  //magic!
	CSC(cudaFree(gpu_to_reduce));
	free(to_reduce);
	return r;
}

__global__ void hist(float* in_data, int* out_hist, float min, float max, int n_split, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	int split;
	for (size_t i = idx; i < size; i += offset)
	{
		split = (int)((in_data[i] - min) / (max - min) * n_split * 0.99999);
		atomicAdd(out_hist + split, 1);
	}
	
}

__global__ void count_sort(float* in_data, float* out_data, int* pref, float min, float max, int n_split, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	int split;
	for (size_t i = idx; i < size; i += offset)
	{
		split = (int)((in_data[i] - min) / (max - min) * n_split * 0.99999);
		out_data[atomicAdd(pref + split, 1)] = in_data[i];
	}
}

std::vector<int> bucketCalc(float* data, int n) {
	float 	min_el, max_el;
	float 	*dev_data;
	float	*dev_out;
	int		*host_pref;
	int		*dev_pref;
	int 	split_size 	= SPLIT_SIZE;
	int 	bucket_size	= BLOCK_SIZE;
	int 	n_split 	= (n - 1) / split_size + 1;

	min_el = my_reduce(data, n);
	max_el = my_reduce(data, n, true);
	//printf("%i: %f, %f\n", n, min_el, max_el);
	if (max_el == min_el) {
		std::vector<int> res;
		res.push_back(0);
		for (int i = 0; i < n; i += bucket_size) {
			res.push_back(i);
		} if (res.back() < n) res.push_back(n);
		return res;
	}

	CSC(cudaMalloc(&dev_data, 	   sizeof(float) * n));
	CSC(cudaMemcpy(dev_data, data, sizeof(float) * n, cudaMemcpyHostToDevice));
	CSC(cudaMalloc(&dev_out, 	   sizeof(float) * n));

	host_pref = (int*)malloc(sizeof(int) * (n_split));
	std::fill(host_pref, host_pref + n_split, 0);

	CSC(cudaMalloc(&dev_pref,  sizeof(int) * n_split));
	thrust::device_ptr<int> p_pref(dev_pref);
	CSC(cudaMemcpy(dev_pref, host_pref, sizeof(int) * n_split, cudaMemcpyHostToDevice));

	free(host_pref);

	hist<<<1024, 1024>>>(dev_data, dev_pref, min_el, max_el, n_split, n);
	CSC(cudaGetLastError());
	

	thrust::exclusive_scan(thrust::device, p_pref, p_pref + n_split, p_pref);

	count_sort<<<1024, 1024>>>(dev_data, dev_out, dev_pref, min_el, max_el, n_split, n);
	CSC(cudaGetLastError());
	CSC(cudaFree(dev_data));

	CSC(cudaMemcpy(data, dev_out, sizeof(float) * n, cudaMemcpyDeviceToHost));

	host_pref = (int*)malloc(sizeof(int) * (n_split + 1));
	host_pref[0] = 0;
	CSC(cudaMemcpy(host_pref + 1, dev_pref, sizeof(int) * n_split, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_pref));
	

	std::vector<int> res;
	res.push_back(0);
	int tmp;
	int count = 0;

	size_t i = 1;
	while (i <= n_split)
	{
		if (host_pref[i] - res.back() <= bucket_size) {
			tmp = host_pref[i];
			count++;
			i++;
		} 
		else if (count != 0) {
			res.push_back(tmp);
			count = 0;
		} 
		else {
			std::vector<int> tmp = bucketCalc(data + host_pref[i-1], host_pref[i] - host_pref[i-1]);
			int add = res.back();
			std::transform(std::begin(tmp), std::end(tmp), std::begin(tmp),[&add](int x){return x+add;});
			res.insert(res.end(), tmp.begin()+1, tmp.end());
			i++;
		}
	} if (res.back() < n) res.push_back(n);
	free(host_pref);
	return res;
}


__global__ void shared_BitonicSort(float* data, int* indicies, int ind_size) {
	for(int i = blockIdx.x; i < ind_size -1; i += gridDim.x) {
		__shared__ float shared_data[BLOCK_SIZE*2]; //???
		int l = indicies[i];
		int r = indicies[i + 1];
		if (threadIdx.x < r - l)
			shared_data[threadIdx.x] = data[l + threadIdx.x];
		else shared_data[threadIdx.x] = FLT_MAX;
		__syncthreads();

		for (int N_merge = 2; N_merge <= BLOCK_SIZE; N_merge *= 2) {
			for (int N = N_merge; N >= 2; N /= 2) {
				int step = N/2;
				int blockId = threadIdx.x / step;
				int first = threadIdx.x + step * blockId;
				int mergeBlockId = threadIdx.x / (N_merge/2);
				if ((mergeBlockId % 2 && shared_data[first] < shared_data[first + step]) || 
				(!(mergeBlockId % 2) && shared_data[first] > shared_data[first + step])) 
				{
					float tmp = shared_data[first];
					shared_data[first] = shared_data[first + step];
					shared_data[first + step] = tmp;
				}
				__syncthreads();
			}
			__syncthreads();
		}
		__syncthreads();
		if (threadIdx.x < r - l)
			data[l + threadIdx.x] = shared_data[threadIdx.x];
		__syncthreads();
	}
}


void bucketSort(float* data, int n) {
	std::vector<int> vtr = bucketCalc(data, n);
	int ind_size = vtr.size();
	int* indicies = vtr.data();
	//fprintf(stderr, "N of indicies: %i/n", ind_size);
	
	/*for (size_t i = 0; i < ind_size; i++) {
		printf("%i ", indicies[i]);
	} printf("\n");*/
	if (n == 0) return;
	
	float* dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(float) * n));
	CSC(cudaMemcpy(dev_data, data, sizeof(float) * n, cudaMemcpyHostToDevice));

	int* dev_indicies;
	CSC(cudaMalloc(&dev_indicies, sizeof(int) * ind_size));
	CSC(cudaMemcpy(dev_indicies, indicies, sizeof(int) * ind_size, cudaMemcpyHostToDevice));

	shared_BitonicSort<<<std::min(ind_size-1, GRID_SIZE), BLOCK_SIZE/*, BLOCK_SIZE*sizeof(float)*/>>>(dev_data, dev_indicies, ind_size);
	CSC(cudaGetLastError());
	CSC(cudaMemcpy(data, dev_data, sizeof(float) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_indicies));
}


int main() {
	int	  n;
	float *data;
	
	//FILE* file = fopen("test.data", "rb");
	//fread(&n, sizeof(int), 1, file);

	fread(&n, sizeof(int), 1, stdin);

	data = (float*)malloc(sizeof(float) * n);

	//fread(data, sizeof(float), n, file);
	//fclose(file);
	fread(data, sizeof(float), n, stdin);

	//fprintf(stderr, "size: %i\n", n);

	/*for (size_t i = 0; i < min(10, n); i++) {
		fprintf(stderr, "%f ", data[i]);
	} fprintf(stderr, "\n");*/

	bucketSort(data, n);

	/*for (size_t i = 0; i < min(10, n); i++) {
		fprintf(stderr, "%f ", data[i]);
	} fprintf(stderr, "\n");*/
	
	/*FILE* out_file = fopen("out.txt", "w");
	for (size_t i = 0; i < n; i++) {
		fprintf(out_file, "%f ", data[i]);
	} fprintf(out_file, "\n");
	fclose(out_file);*/

	/*int counter = 0;
	for (size_t i = 1; i < n; i++) {
		if (data[i-1] > data[i]) {
			counter++;
			printf("Error in: %zi\n", i);
		}
	}
	printf("ERRORS: %i\n", counter);*/


	fwrite(data, sizeof(float), n, stdout);
	free(data);
	return 0;
}
