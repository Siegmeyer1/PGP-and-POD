#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "mpi.h"



#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _i(i, j, k) ( ((k) + 1) * (block_dim[0] + 2) * (block_dim[1] + 2) + ((j) + 1) * (block_dim[0] + 2) + (i) + 1)
#define _i_real(i, j, k) ( ((k)) * (block_dim[0]) * (block_dim[1]) + ((j)) * (block_dim[0]) + (i))
#define _ib(i, j, k) ((k) * grid_dim[0] * grid_dim[1] + (j) * grid_dim[0] + (i))

enum bords {
	DOWN,
	UP,
	LEFT,
	RIGHT,
	FRONT,
	BACK,
	MIDDLE
};

__global__ void init_data(int ib, int jb, int kb, double *data, double *next, int *block_dim, int *grid_dim, double* borders, double init, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int offsetz = blockDim.z * gridDim.z;
	for (k = idz; k < block_dim[2]; k += offsetz)
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				data[_i(i, j, k)] = init;

	if (ib == 0)
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx) {
				data[_i(-1, j, k)] = borders[LEFT];
				next[_i(-1, j, k)] = borders[LEFT];
			}

	if (ib == grid_dim[0]-1)
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx) {
				data[_i(block_dim[0], j, k)] = borders[RIGHT];
				next[_i(block_dim[0], j, k)] = borders[RIGHT];
			}

	if (jb == 0)
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx) {
				data[_i(i, -1, k)] = borders[FRONT];
				next[_i(i, -1, k)] = borders[FRONT];
			}

	if (jb == grid_dim[1]-1)
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx) {
				data[_i(i, block_dim[1], k)] = borders[BACK];
				next[_i(i, block_dim[1], k)] = borders[BACK];
			}
	
	if (kb == 0)
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx) {
				data[_i(i, j, -1)] = borders[UP];
				next[_i(i, j, -1)] = borders[UP];
			}

	if (kb == grid_dim[2]-1)
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx) {
				data[_i(i, j, block_dim[2])] = borders[DOWN];
				next[_i(i, j, block_dim[2])] = borders[DOWN];
			}
	
}

__global__ void to_send_forward(int ib, int jb, int kb, double *data, int *block_dim, int *grid_dim, double *dibuff, double *djbuff, double *dkbuff, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	if (ib + 1 < grid_dim[0]) {					
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx)
				dibuff[k * block_dim[1] + j] = data[_i(block_dim[0] - 1, j, k)];
	}
	if (jb + 1 < grid_dim[1]) {					
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				djbuff[k * block_dim[0] + i] = data[_i(i, block_dim[1] - 1, k)];
	}
	if (kb + 1 < grid_dim[2]) {					
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				dkbuff[j * block_dim[0] + i] = data[_i(i, j, block_dim[2] - 1)];
	}
}

__global__ void to_send_backward(int ib, int jb, int kb, double *data, int *block_dim, int *grid_dim, double *dibuff, double *djbuff, double *dkbuff, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;	
	if (ib > 0) {					
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx)
				dibuff[k * block_dim[1] + j] = data[_i(0, j, k)];
	}
	if (jb > 0) {					
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				djbuff[k * block_dim[0] + i] = data[_i(i, 0, k)];
	}
	if (kb > 0) {					
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				dkbuff[j * block_dim[0] + i] = data[_i(i, j, 0)];
	}
	
}

__global__ void recieve_after_forward(int ib, int jb, int kb, double *data, int *block_dim, int *grid_dim, double *dibuff, double *djbuff, double *dkbuff, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;	
	if (ib > 0) {
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx)
				data[_i(-1, j, k)] = dibuff[k * block_dim[1] + j];
	}
	if (jb > 0) {
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				data[_i(i, -1, k)] = djbuff[k * block_dim[0] + i];
	}
	if (kb > 0) {
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				data[_i(i, j, -1)] = dkbuff[j * block_dim[0] + i];
	}
	
}

__global__ void recieve_after_backward(int ib, int jb, int kb, double *data, int *block_dim, int *grid_dim, double *dibuff, double *djbuff, double *dkbuff, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;	
	if (ib + 1 < grid_dim[0]) {
		for (k = idy; k < block_dim[2]; k += offsety)
			for (j = idx; j < block_dim[1]; j += offsetx)
				data[_i(block_dim[0], j, k)] = dibuff[k * block_dim[1] + j];
	}
	if (jb + 1 < grid_dim[1]) {
		for (k = idy; k < block_dim[2]; k += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				data[_i(i, block_dim[1], k)] = djbuff[k * block_dim[0] + i];
	}
	if (kb + 1 < grid_dim[2]) {
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				data[_i(i, j, block_dim[2])] = dkbuff[j * block_dim[0] + i];
	}
	
}

__global__ void calc(double *data, double *next, double *to_reduce, int *block_dim, double hx, double hy, double hz, int size) {
	int i, j, k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int offsetz = blockDim.z * gridDim.z;
	for (k = idz; k < block_dim[2]; k += offsetz)
		for (j = idy; j < block_dim[1]; j += offsety)
			for (i = idx; i < block_dim[0]; i += offsetx)
				{
				next[_i(i, j, k)] = 0.5 * ( (data[_i(i+1, j, k)] + data[_i(i-1, j, k)]) / (hx*hx)
										+ (data[_i(i, j+1, k)] + data[_i(i, j-1, k)]) / (hy*hy)
										+ (data[_i(i, j, k+1)] + data[_i(i, j, k-1)]) / (hz*hz) )
												/ ( 1.0 / (hx * hx) 
												+ 1.0 / (hy * hy) 
												+ 1.0 / (hz * hz) );
				to_reduce[_i_real(i, j, k)] = fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
				}
}

int main(int argc, char *argv[]) {
	int ib, jb, kb;
	int i, j, k;
	int deviceCount;
    //int grid_dim[3], block_dim[3];
	int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
    char file_name[100];

	double eps, hx, hy, hz, init;
	double *data, *temp, /*next,*/ *to_reduce, *ibuff, *jbuff, *kbuff, /*lineBuff,*/ *gather_stop, *borders, *brick_dim;// borders[6], brick_dim[3];
	double *ddata, *dnext, *dborders, *dibuff, *djbuff, *dkbuff;
	int *dblock_dim, *dgrid_dim, *block_dim, *grid_dim;
	block_dim = (int*)malloc(sizeof(int) * 3);
	grid_dim = (int*)malloc(sizeof(int) * 3);
	borders = (double*)malloc(sizeof(double) * 6);
	brick_dim = (double*)malloc(sizeof(double) * 3);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);
	cudaGetDeviceCount(&deviceCount);
	cudaSetDevice(id % deviceCount);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {					// Reading calculation parameters
        scanf("%d %d %d\n", &grid_dim[0], &grid_dim[1], &grid_dim[2]);
        scanf("%d %d %d\n", &block_dim[0], &block_dim[1], &block_dim[2]);
        scanf("%s", file_name);
        scanf("%lf", &eps);
        scanf("%lf %lf %lf\n", &brick_dim[0], &brick_dim[1], &brick_dim[2]);
        scanf("%lf %lf %lf %lf %lf %lf\n",
            &borders[DOWN], &borders[UP], &borders[LEFT], &borders[RIGHT], &borders[FRONT], &borders[BACK]);
        scanf("%lf", &init);

		fprintf(stderr, "%d %d %d\n%d %d %d\n%lf\n%lf %lf %lf\n%lf %lf %lf %lf %lf %lf\n%lf\n", grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2], eps,
						brick_dim[0], brick_dim[1], brick_dim[2], borders[0], borders[1], borders[2], borders[3], borders[4], borders[5], init);
	}
    MPI_Bcast(grid_dim, 3, MPI_INT, 0, MPI_COMM_WORLD);			// transmitting calc params to all procs
    MPI_Bcast(block_dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(brick_dim, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(borders, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(file_name, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	ib = id % grid_dim[0];
	jb = (id / grid_dim[0]) % grid_dim[1];
    kb = id / (grid_dim[0] * grid_dim[1]);

	hx = brick_dim[0] / (float)(grid_dim[0] * block_dim[0]);	
	hy = brick_dim[1] / (float)(grid_dim[1] * block_dim[1]);
    hz = brick_dim[2] / (float)(grid_dim[2] * block_dim[2]);
    
	CSC(cudaMalloc(&ddata, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2)));
	CSC(cudaMalloc(&dnext, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2)));
	CSC(cudaMalloc(&to_reduce, sizeof(double) * (block_dim[0]) * (block_dim[1]) * (block_dim[2])));

	CSC(cudaMalloc(&dborders, sizeof(double) * 6));
	CSC(cudaMalloc(&dblock_dim, sizeof(int) * 3));
	CSC(cudaMalloc(&dgrid_dim, sizeof(int) * 3));
	CSC(cudaMemcpy(dblock_dim, block_dim, sizeof(int)*3, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dgrid_dim, grid_dim, sizeof(int)*3, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dborders, borders, sizeof(double)*6, cudaMemcpyHostToDevice));

	CSC(cudaMalloc(&dibuff, sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2)));
	CSC(cudaMalloc(&djbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2)));
	CSC(cudaMalloc(&dkbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2)));

	data = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));	
	//next = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));
	ibuff = (double *)malloc(sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2));
	jbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2));
	kbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2));
	//lineBuff = (double *)malloc(sizeof(double) * (block_dim[0]+2));
	gather_stop = (double *)malloc(sizeof(double) * numproc);

	MPI_Request send_request[3], recv_request[3];
	MPI_Status status[3];

/*===================================Initializing data in each block(proc)====================================*/

	/*for(i = 0; i < block_dim[0]; i++)
		for(j = 0; j < block_dim[1]; j++)
            for(k = 0; k < block_dim[2]; k++)
			    data[_i(i, j, k)] = init;
	
	if (ib == 0)
	for(j = 0; j < block_dim[1]; j++)
        for(k = 0; k < block_dim[2]; k++) {
			data[_i(-1, j, k)] = borders[2];
			next[_i(-1, j, k)] = borders[2];
		}
	if (ib == grid_dim[0]-1)
	for(j = 0; j < block_dim[1]; j++)
        for(k = 0; k < block_dim[2]; k++) {
			data[_i(block_dim[0], j, k)] = borders[3];
			next[_i(block_dim[0], j, k)] = borders[3];
		}

	if (jb == 0)
	for(i = 0; i < block_dim[0]; i++)
		for(k = 0; k < block_dim[2]; k++) {
			data[_i(i, -1, k)] = borders[4];
			next[_i(i, -1, k)] = borders[4];
		}
	if (jb == grid_dim[1]-1)
	for(i = 0; i < block_dim[0]; i++)
		for(k = 0; k < block_dim[2]; k++) {
			data[_i(i, block_dim[1], k)] = borders[5];
			next[_i(i, block_dim[1], k)] = borders[5];
		}
	
	if (kb == 0)
	for(i = 0; i < block_dim[0]; i++)					
		for(j = 0; j < block_dim[1]; j++) {
			data[_i(i, j, -1)] = borders[1];
			next[_i(i, j, -1)] = borders[1];
		}
	if (kb == grid_dim[2]-1)
	for(i = 0; i < block_dim[0]; i++)					
		for(j = 0; j < block_dim[1]; j++) {
			data[_i(i, j, block_dim[2])] = borders[0];
			next[_i(i, j, block_dim[2])] = borders[0];
		}*/
	init_data<<<dim3(8,8,8),dim3(32,4,4)>>>(ib, jb, kb, ddata, dnext, dblock_dim, dgrid_dim, dborders, init, 1);
	CSC(cudaGetLastError());

	//CSC(cudaMemcpy(data, ddata, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));
	//CSC(cudaMemcpy(next, dnext, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));
	fprintf(stderr, "%d: init OK\n", id);


/*===================================End of initialization====================================*/
	//==========================start of calc loop==============================//
	bool stop = false;
	//bool local_stop = false;
	double max;
	//max = (double*)malloc(sizeof(double));
	//int tmp_counter = 0;
	while (!stop/*tmp_counter < 100*/) {
		//tmp_counter++;
		//fprintf(stderr, "%d: new calc loop\n", id);
		MPI_Barrier(MPI_COMM_WORLD);
		//===================================forward borders transfer====================================//
		/*if (ib + 1 < grid_dim[0]) {					
			for(j = 0; j < block_dim[1]; j++)
				for(k = 0; k < block_dim[2]; k++)
					ibuff[k * block_dim[1] + j] = data[_i(block_dim[0] - 1, j, k)];
			MPI_Isend(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		}
		if (jb + 1 < grid_dim[1]) {					
			for(i = 0; i < block_dim[0]; i++)
				for(k = 0; k < block_dim[2]; k++)
					jbuff[k * block_dim[0] + i] = data[_i(i, block_dim[1] - 1, k)];
			MPI_Isend(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		}
		if (kb + 1 < grid_dim[2]) {					
			for(i = 0; i < block_dim[0]; i++)
				for(j = 0; j < block_dim[1]; j++)
					kbuff[j * block_dim[0] + i] = data[_i(i, j, block_dim[2] - 1)];
			MPI_Isend(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD, &send_request[2]);
		}*/
		to_send_forward<<<dim3(32,32),dim3(32,32)>>>(ib, jb, kb, ddata, dblock_dim, dgrid_dim, dibuff, djbuff, dkbuff, 1);
		CSC(cudaGetLastError());

		if (ib + 1 < grid_dim[0]) {
			CSC(cudaMemcpy(ibuff, dibuff, sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));
			MPI_Isend(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		}
		if (jb + 1 < grid_dim[1]) {
			CSC(cudaMemcpy(jbuff, djbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));
			MPI_Isend(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		}
		if (kb + 1 < grid_dim[2]) {
			CSC(cudaMemcpy(kbuff, dkbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2), cudaMemcpyDeviceToHost));
			MPI_Isend(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD, &send_request[2]);
		}


		if (ib + 1 < grid_dim[0]) MPI_Wait(&send_request[0], status);
		if (jb + 1 < grid_dim[1]) MPI_Wait(&send_request[1], status);
		if (kb + 1 < grid_dim[2]) MPI_Wait(&send_request[2], status);
		
		//-------------------------------------------------------------------------------//
     	if (ib > 0) MPI_Irecv(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		if (jb > 0) MPI_Irecv(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		if (kb > 0) MPI_Irecv(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &recv_request[2]);
		if (ib > 0) MPI_Wait(&recv_request[0], status);
		if (jb > 0) MPI_Wait(&recv_request[1], status);
		if (kb > 0) MPI_Wait(&recv_request[2], status);
		if (ib > 0) CSC(cudaMemcpy(dibuff, ibuff, sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyHostToDevice));
		if (jb > 0) CSC(cudaMemcpy(djbuff, jbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2), cudaMemcpyHostToDevice));
		if (kb > 0) CSC(cudaMemcpy(dkbuff, kbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2), cudaMemcpyHostToDevice));
		
		/*if (ib > 0) {
			for(j = 0; j < block_dim[1]; j++)
				for(k = 0; k < block_dim[2]; k++)
					data[_i(-1, j, k)] = ibuff[k * block_dim[1] + j];
		}
		if (jb > 0) {
			for(i = 0; i < block_dim[0]; i++)
				for(k = 0; k < block_dim[2]; k++)
					data[_i(i, -1, k)] = jbuff[k * block_dim[0] + i];
		}
		if (kb > 0) {
			for(i = 0; i < block_dim[0]; i++)
				for(j = 0; j < block_dim[1]; j++)
					data[_i(i, j, -1)] = kbuff[j * block_dim[0] + i];
		}*/
		recieve_after_forward<<<dim3(32,32),dim3(32,32)>>>(ib, jb, kb, ddata, dblock_dim, dgrid_dim, dibuff, djbuff, dkbuff, 1);
		CSC(cudaGetLastError());
		//===================================end of forward borders transfer====================================//
		//===================================backward borders transfer====================================//

		/*if (ib > 0) {					
			for(j = 0; j < block_dim[1]; j++)
				for(k = 0; k < block_dim[2]; k++)
					ibuff[k * block_dim[1] + j] = data[_i(0, j, k)];
			MPI_Isend(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		}
		if (jb > 0) {					
			for(i = 0; i < block_dim[0]; i++)
				for(k = 0; k < block_dim[2]; k++)
					jbuff[k * block_dim[0] + i] = data[_i(i, 0, k)];
			MPI_Isend(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		}
		if (kb > 0) {					
			for(i = 0; i < block_dim[0]; i++)
				for(j = 0; j < block_dim[1]; j++)
					kbuff[j * block_dim[0] + i] = data[_i(i, j, 0)];
			MPI_Isend(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD, &send_request[2]);
		}*/
		to_send_backward<<<dim3(32,32),dim3(32,32)>>>(ib, jb, kb, ddata, dblock_dim, dgrid_dim, dibuff, djbuff, dkbuff, 1);
		CSC(cudaGetLastError());

		if (ib > 0) {	
			CSC(cudaMemcpy(ibuff, dibuff, sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));				
			MPI_Isend(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		}
		if (jb > 0) {	
			CSC(cudaMemcpy(jbuff, djbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));				
			MPI_Isend(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		}
		if (kb > 0) {	
			CSC(cudaMemcpy(kbuff, dkbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2), cudaMemcpyDeviceToHost));				
			MPI_Isend(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD, &send_request[2]);
		}

		if (ib > 0) MPI_Wait(&send_request[0], status);
		if (jb > 0) MPI_Wait(&send_request[1], status);
		if (kb > 0) MPI_Wait(&send_request[2], status);
		//----------------------------------------------------------------------------------//
     	if (ib + 1 < grid_dim[0]) MPI_Irecv(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		if (jb + 1 < grid_dim[1]) MPI_Irecv(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		if (kb + 1 < grid_dim[2]) MPI_Irecv(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &recv_request[2]);
		if (ib + 1 < grid_dim[0]) MPI_Wait(&recv_request[0], status);
		if (jb + 1 < grid_dim[1]) MPI_Wait(&recv_request[1], status);
		if (kb + 1 < grid_dim[2]) MPI_Wait(&recv_request[2], status);
		if (ib + 1 < grid_dim[0]) CSC(cudaMemcpy(dibuff, ibuff, sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyHostToDevice));
		if (jb + 1 < grid_dim[1]) CSC(cudaMemcpy(djbuff, jbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2), cudaMemcpyHostToDevice));
		if (kb + 1 < grid_dim[2]) CSC(cudaMemcpy(dkbuff, kbuff, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2), cudaMemcpyHostToDevice));

		recieve_after_backward<<<dim3(32,32),dim3(32,32)>>>(ib, jb, kb, ddata, dblock_dim, dgrid_dim, dibuff, djbuff, dkbuff, 1);
		CSC(cudaGetLastError());
		
		/*if (ib + 1 < grid_dim[0]) {
			for(j = 0; j < block_dim[1]; j++)
				for(k = 0; k < block_dim[2]; k++)
					data[_i(block_dim[0], j, k)] = ibuff[k * block_dim[1] + j];
		}
		if (jb + 1 < grid_dim[1]) {
			for(i = 0; i < block_dim[0]; i++)
				for(k = 0; k < block_dim[2]; k++)
					data[_i(i, block_dim[1], k)] = jbuff[k * block_dim[0] + i];
		}
		if (kb + 1 < grid_dim[2]) {
			for(i = 0; i < block_dim[0]; i++)
				for(j = 0; j < block_dim[1]; j++)
					data[_i(i, j, block_dim[2])] = kbuff[j * block_dim[0] + i];
		}*/

		/*===================================end of backward borders transfer====================================*/

		MPI_Barrier(MPI_COMM_WORLD);
		//if (!local_stop) {
		/*max = 0;
		for(i = 0; i < block_dim[0]; i++)
			for(j = 0; j < block_dim[1]; j++)
				for(k = 0; k < block_dim[2]; k++) {
					next[_i(i, j, k)] = 0.5 * ( (data[_i(i+1, j, k)] + data[_i(i-1, j, k)]) / (hx*hx)
											+ (data[_i(i, j+1, k)] + data[_i(i, j-1, k)]) / (hy*hy)
											+ (data[_i(i, j, k+1)] + data[_i(i, j, k-1)]) / (hz*hz) )
													/ ( 1.0 / (hx * hx) 
													+ 1.0 / (hy * hy) 
													+ 1.0 / (hz * hz) );
					max = std::max(max, std::abs(next[_i(i, j, k)] - data[_i(i, j, k)]));	
				}*/
		calc<<<dim3(8,8,8),dim3(32,4,4)>>>(ddata, dnext, to_reduce, dblock_dim, hx, hy, hz, 1);
		CSC(cudaGetLastError());
		thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(to_reduce);
		thrust::device_ptr<double> dmax = thrust::max_element(thrust::device, p_arr, p_arr + (block_dim[0]) * (block_dim[1]) * (block_dim[2]));
		//fprintf(stderr, "%e\n", dmax);
		CSC(cudaMemcpy(&max, to_reduce + (int)(dmax - p_arr), sizeof(double), cudaMemcpyDeviceToHost));
		//fprintf(stderr, "%d: iter %d, max = %e\n", id, tmp_counter, max);
		temp = dnext;
		dnext = ddata;
		ddata = temp;
		//local_stop = max < eps;
		//}
		MPI_Allgather(&max, 1, MPI_DOUBLE, gather_stop, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		stop = true;
		for (size_t eps_i = 0; eps_i < numproc; eps_i++)
		{
			if (gather_stop[eps_i] >= eps) {
				stop = false;
				break;
			}
		}
		
		

	}
	//==========================end of calc loop==============================//
	MPI_Barrier(MPI_COMM_WORLD);
	CSC(cudaMemcpy(data, ddata, sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2), cudaMemcpyDeviceToHost));

	int n_size = 14;
	char* text_buff = (char*)malloc(sizeof(char) * block_dim[0] * block_dim[1] * block_dim[2]  *n_size);
	memset(text_buff, ' ', sizeof(char) * block_dim[0] * block_dim[1] * block_dim[2]  *n_size);
	for(k = 0; k < block_dim[2]; k++)
		for (j = 0; j < block_dim[1]; j++)
			for (i = 0; i < block_dim[0]; i++)
				sprintf(text_buff + (i + block_dim[0] * j + block_dim[0] * block_dim[1] * (block_dim[2] -  k - 1))  * n_size, "%e", data[_i(i, j, k)]);
	for(i = 0; i < block_dim[0] * block_dim[1] * block_dim[2] * n_size; i++) {
		if (text_buff[i] == '\0') text_buff[i] = ' ';
		if (ib == grid_dim[0] - 1 && i != 0 && (i) % (n_size * block_dim[0]) == 0) text_buff[i-1] = '\n';
	}
	if (ib == grid_dim[0] - 1) text_buff[block_dim[0] * block_dim[1] * block_dim[2] * n_size - 1] = '\n';

	MPI_Datatype num;
	MPI_Type_contiguous(n_size, MPI_CHAR, &num);
    MPI_Type_commit(&num);

	int blocklens[block_dim[1]*block_dim[2]];
	int indicies[block_dim[1]*block_dim[2]];
	int a = 0;
	for (i = 0; i < block_dim[1]*block_dim[2]; i++)
	{
		blocklens[i] = block_dim[0];
		indicies[i] = a;
		a += block_dim[0]*grid_dim[0];
		if ((i + 1) % block_dim[1] == 0) a += block_dim[0]*grid_dim[0] * (grid_dim[1]-1) * block_dim[1];
	}

	MPI_Datatype dtype;
	MPI_Type_indexed(block_dim[1]*block_dim[2], blocklens, indicies, num, &dtype);
	MPI_Type_commit(&dtype);

	MPI_File fp;
	MPI_File_delete(file_name, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	int offset_i = sizeof(char) * n_size * block_dim[0] * ib;
	int offset_j = sizeof(char) * n_size * block_dim[0] *block_dim[1] *grid_dim[0] * jb;
	int offset_k = sizeof(char) * n_size * block_dim[0] *block_dim[1] * block_dim[2] * grid_dim[0] * grid_dim[1] * (grid_dim[2] - kb - 1);
	int offset_total = offset_i + offset_j + offset_k;
	MPI_File_set_view(fp, /*sizeof(char) * n_size * ib * block_dim[0]*/offset_total, MPI_CHAR, dtype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, text_buff, block_dim[0] * block_dim[1] * block_dim[2], num, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	/*if (id != 0) 
	{		
		for(k = block_dim[2]-1; k >= 0; k--)
		{
			for(j = 0; j < block_dim[1]; j++)
			{
				for(i = 0; i < block_dim[0]; i++)
				{
					lineBuff[i] = data[_i(i, j, k)];
				}
				MPI_Send(lineBuff, block_dim[0] + 2, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	} 
	else 
	{
		FILE* file = fopen(file_name, "w");
		for(kb = grid_dim[2]-1; kb >= 0; kb--) {
			for(k = block_dim[2]-1; k >= 0; k--) {
				for(jb = 0; jb < grid_dim[1]; jb++) {
					for(j = 0; j < block_dim[1]; j++) {
						for(ib = 0; ib < grid_dim[0]; ib++) {
							if (_ib(ib, jb, kb) == 0) {
								for(i = 0; i < block_dim[0]; i++) lineBuff[i] = data[_i(i, j, k)];
							}
							else {
								MPI_Recv(lineBuff, block_dim[0] + 2, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, status);
							}
							for(i = 0; i < block_dim[0]; i++) fprintf(file, "%e ", lineBuff[i]);
						}
						fprintf(file, "\n");
					}
					fprintf(file, "\n");
				}
			}
		}
		fclose(file);
	}*/

	MPI_Finalize();
	return 0;
}