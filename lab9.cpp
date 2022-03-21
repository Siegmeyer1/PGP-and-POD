#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include "mpi.h"
#include <omp.h>

#define _i(i, j, k) ( ((k) + 1) * (block_dim[0] + 2) * (block_dim[1] + 2) + ((j) + 1) * (block_dim[0] + 2) + (i) + 1)
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

int main(int argc, char *argv[]) {
	int ib, jb, kb;
	int i, j, k;
    int grid_dim[3], block_dim[3];
	int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
    char file_name[100];

	double eps, hx, hy, hz, init/*, start, end*/;
	double *data, *temp, *next,/* *ibuff, *jbuff, *kbuff, *lineBuff,*/ *gather_stop, borders[6], brick_dim[3];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);

	MPI_Barrier(MPI_COMM_WORLD);
	//start = MPI_Wtime();

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
    
	data = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));	
	next = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));
	/*ibuff = (double *)malloc(sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2));
	jbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2));
	kbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2));
	lineBuff = (double *)malloc(sizeof(double) * (block_dim[0]+2));*/
	gather_stop = (double *)malloc(sizeof(double) * numproc);

	MPI_Request send_request[3], recv_request[3];
	MPI_Status status[3];

/*===================================Initializing data in each block(proc)====================================*/

	for(i = 0; i < block_dim[0]; i++)
		for(j = 0; j < block_dim[1]; j++)
            for(k = 0; k < block_dim[2]; k++)
			    data[_i(i, j, k)] = init;
	
	if (ib == 0)
	for(j = 0; j < block_dim[1]; j++)
        for(k = 0; k < block_dim[2]; k++) {
			data[_i(-1, j, k)] = borders[LEFT];
			next[_i(-1, j, k)] = borders[LEFT];
		}
	if (ib == grid_dim[0]-1)
	for(j = 0; j < block_dim[1]; j++)
        for(k = 0; k < block_dim[2]; k++) {
			data[_i(block_dim[0], j, k)] = borders[RIGHT];
			next[_i(block_dim[0], j, k)] = borders[RIGHT];
		}

	if (jb == 0)
	for(i = 0; i < block_dim[0]; i++)
		for(k = 0; k < block_dim[2]; k++) {
			data[_i(i, -1, k)] = borders[FRONT];
			next[_i(i, -1, k)] = borders[FRONT];
		}
	if (jb == grid_dim[1]-1)
	for(i = 0; i < block_dim[0]; i++)
		for(k = 0; k < block_dim[2]; k++) {
			data[_i(i, block_dim[1], k)] = borders[BACK];
			next[_i(i, block_dim[1], k)] = borders[BACK];
		}
	
	if (kb == 0)
	for(i = 0; i < block_dim[0]; i++)					
		for(j = 0; j < block_dim[1]; j++) {
			data[_i(i, j, -1)] = borders[UP];
			next[_i(i, j, -1)] = borders[UP];
		}
	if (kb == grid_dim[2]-1)
	for(i = 0; i < block_dim[0]; i++)					
		for(j = 0; j < block_dim[1]; j++) {
			data[_i(i, j, block_dim[2])] = borders[DOWN];
			next[_i(i, j, block_dim[2])] = borders[DOWN];
		}

/*===================================End of initialization====================================*/

//===================================Defining mpi datatypes for borders=======================//

MPI_Datatype up_bord_t, front_bord_t, left_bord_t;
/*int block_lens_brd[block_dim[1]];
int indicies_brd[block_dim[1]];
int block_lens_brd_k[block_dim[2]];
int indicies_brd_k[block_dim[2]];
int block_lens_brd_i[block_dim[1]*block_dim[2]];
int indicies_brd_i[block_dim[1]*block_dim[2]];*/
int *block_lens_brd   = (int*)malloc(sizeof(int) * block_dim[1]);
int *indicies_brd     = (int*)malloc(sizeof(int) * block_dim[1]);
int *block_lens_brd_k = (int*)malloc(sizeof(int) * block_dim[2]);
int *indicies_brd_k   = (int*)malloc(sizeof(int) * block_dim[2]);
int *block_lens_brd_i = (int*)malloc(sizeof(int) * block_dim[1]*block_dim[2]);
int *indicies_brd_i   = (int*)malloc(sizeof(int) * block_dim[1]*block_dim[2]);
//====================================================================================//
int ind = block_dim[0] + 3;
for (i = 0; i < block_dim[1]; i++) {
	block_lens_brd[i] = block_dim[0];
	indicies_brd[i] = ind;
	ind += block_dim[0] + 2;
}
MPI_Type_indexed(block_dim[1], block_lens_brd, indicies_brd, MPI_DOUBLE, &up_bord_t);
MPI_Type_commit(&up_bord_t);
//====================================================================================//
ind = (block_dim[0] + 2) * (block_dim[1] + 2) + 1;
for (i = 0; i < block_dim[2]; i++) {
	block_lens_brd_k[i] = block_dim[0];
	indicies_brd_k[i] = ind;
	ind += (block_dim[0] + 2) * (block_dim[1] + 2);
}
MPI_Type_indexed(block_dim[2], block_lens_brd_k, indicies_brd_k, MPI_DOUBLE, &front_bord_t);
MPI_Type_commit(&front_bord_t);
//====================================================================================//
ind = (block_dim[0] + 2) * (block_dim[1] + 2) + block_dim[0] + 2;
for (k = 0; k < block_dim[2]; k++) {
	for (j = 0; j < block_dim[1]; j++) {
		block_lens_brd_i[k*block_dim[1]+j] = 1;
		indicies_brd_i[k*block_dim[1]+j] = ind;
		ind += (block_dim[0] + 2);
	}
	ind += 2*(block_dim[0] + 2);
}
MPI_Type_indexed(block_dim[1] * block_dim[2], block_lens_brd_i, indicies_brd_i, MPI_DOUBLE, &left_bord_t);
MPI_Type_commit(&left_bord_t);

free(block_lens_brd);
free(indicies_brd);
free(block_lens_brd_k);
free(indicies_brd_k);
free(block_lens_brd_i);
free(indicies_brd_i);
//===================================End of definition========================================//

	//==========================start of calc loop==============================//
	bool stop = false;
	int tmp_it = 0;
	//bool local_stop = false;
	double max;
	while (!stop/*tmp_it < 1*/) {
		tmp_it++;
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
		if (ib + 1 < grid_dim[0]) MPI_Isend(data + block_dim[0], 									 1,	left_bord_t,   _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		if (jb + 1 < grid_dim[1]) MPI_Isend(data + (block_dim[0]+2)*(block_dim[1]), 				 1,	front_bord_t,  _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		if (kb + 1 < grid_dim[2]) MPI_Isend(data + (block_dim[0]+2)*(block_dim[1]+2)*(block_dim[2]), 1, up_bord_t,     _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD, &send_request[2]);
		if (ib + 1 < grid_dim[0]) MPI_Wait(&send_request[0], status);
		if (jb + 1 < grid_dim[1]) MPI_Wait(&send_request[1], status);
		if (kb + 1 < grid_dim[2]) MPI_Wait(&send_request[2], status);
		//printf("%d: fwd send OK\n", id);
		//-------------------------------------------------------------------------------//
     	//if (ib > 0) MPI_Irecv(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		//if (jb > 0) MPI_Irecv(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		//if (kb > 0) MPI_Irecv(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &recv_request[2]);

		if (ib > 0) MPI_Irecv(data, 1, left_bord_t,  _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		if (jb > 0) MPI_Irecv(data, 1, front_bord_t, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		if (kb > 0) MPI_Irecv(data, 1, up_bord_t,    _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &recv_request[2]);

		if (ib > 0) MPI_Wait(&recv_request[0], status);
		if (jb > 0) MPI_Wait(&recv_request[1], status);
		if (kb > 0) MPI_Wait(&recv_request[2], status);
		//printf("%d: fwd recv OK\n", id);
		
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
		if (ib > 0) MPI_Isend(data + 1, 								1, left_bord_t,  _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD, &send_request[0]);
		if (jb > 0) MPI_Isend(data + block_dim[0]+2,				    1, front_bord_t, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD, &send_request[1]);
		if (kb > 0) MPI_Isend(data + (block_dim[0]+2)*(block_dim[1]+2), 1, up_bord_t,    _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD, &send_request[2]);
		if (ib > 0) MPI_Wait(&send_request[0], status);
		if (jb > 0) MPI_Wait(&send_request[1], status);
		if (kb > 0) MPI_Wait(&send_request[2], status);
		//printf("%d: bckd send OK\n", id);
		//----------------------------------------------------------------------------------//
     	//if (ib + 1 < grid_dim[0]) MPI_Irecv(ibuff, (block_dim[1]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		//if (jb + 1 < grid_dim[1]) MPI_Irecv(jbuff, (block_dim[0]+2) * (block_dim[2]+2), MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		//if (kb + 1 < grid_dim[2]) MPI_Irecv(kbuff, (block_dim[0]+2) * (block_dim[1]+2), MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &recv_request[2]);

		if (ib + 1 < grid_dim[0]) MPI_Irecv(data + block_dim[0]+1, 									   1, left_bord_t,  _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &recv_request[0]);
		if (jb + 1 < grid_dim[1]) MPI_Irecv(data + (block_dim[0]+2)*(block_dim[1]+1), 				   1, front_bord_t, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &recv_request[1]);
		if (kb + 1 < grid_dim[2]) MPI_Irecv(data + (block_dim[0]+2)*(block_dim[1]+2)*(block_dim[2]+1), 1, up_bord_t,    _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &recv_request[2]);

		if (ib + 1 < grid_dim[0]) MPI_Wait(&recv_request[0], status);
		if (jb + 1 < grid_dim[1]) MPI_Wait(&recv_request[1], status);
		if (kb + 1 < grid_dim[2]) MPI_Wait(&recv_request[2], status);
		//printf("%d: bckd recv OK\n", id);
		
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
		max = 0;
		#pragma omp parallel shared(data, next) private(i, j, k) reduction(max: max)
		{
			int thread_id = omp_get_thread_num();
			int offset = omp_get_num_threads();
			for (/*thread_id*/; thread_id < block_dim[0]*block_dim[1]*block_dim[2]; thread_id += offset) {
				i = thread_id % block_dim[0];
				j = (thread_id / block_dim[0]) % block_dim[1];
				k = thread_id / (block_dim[0] * block_dim[1]);
				/*for(i = idi; i < block_dim[0]; i += block_dim[0])
					for(j = idj; j < block_dim[1]; j += block_dim[1])
						for(k = idk; k < block_dim[2]; k += block_dim[2]) {*/
				next[_i(i, j, k)] = 0.5 * ( (data[_i(i+1, j, k)] + data[_i(i-1, j, k)]) / (hx*hx)
										+ (data[_i(i, j+1, k)] + data[_i(i, j-1, k)]) / (hy*hy)
										+ (data[_i(i, j, k+1)] + data[_i(i, j, k-1)]) / (hz*hz) )
												/ ( 1.0 / (hx * hx) 
												+ 1.0 / (hy * hy) 
												+ 1.0 / (hz * hz) );
				max = std::max(max, std::abs(next[_i(i, j, k)] - data[_i(i, j, k)]));	
						//}
			}
		} 
		/*for(i = 0; i < block_dim[0]; i++)
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
		temp = next;
		next = data;
		data = temp;
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

	/*if (id != 0) 
	{		
		for(k = block_dim[2]; k >= -1; k--)
		{
			for(j = -1; j <= block_dim[1]; j++)
			{
				for(i = -1; i <= block_dim[0]; i++)
				{
					lineBuff[i+1] = data[_i(i, j, k)];
				}
				MPI_Send(lineBuff, block_dim[0] + 2, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	} 
	else 
	{
		FILE* file = fopen(file_name, "w");
		for(kb = grid_dim[2]-1; kb >= 0; kb--) {
			for(k = block_dim[2]; k >= -1; k--) {
				for(jb = 0; jb < grid_dim[1]; jb++) {
					for(j = -1; j <= block_dim[1]; j++) {
						for(ib = 0; ib < grid_dim[0]; ib++) {
							if (_ib(ib, jb, kb) == 0) {
								for(i = -1; i <= block_dim[0]; i++) lineBuff[i+1] = data[_i(i, j, k)];
							}
							else {
								MPI_Recv(lineBuff, block_dim[0] + 2, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, status);
							}
							for(i = 0; i < block_dim[0]+2; i++) fprintf(file, "%e ", lineBuff[i]);
							fprintf(file, "\t");
						}
						fprintf(file, "\n");
					}
					fprintf(file, "\n");
				}
			}
		}
		fclose(file);
	}*/

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

	//int blocklens[block_dim[1]*block_dim[2]];
	//int indicies[block_dim[1]*block_dim[2]];
	int *blocklens = (int*)malloc(sizeof(int) * block_dim[1]*block_dim[2]);
	int *indicies  = (int*)malloc(sizeof(int) * block_dim[1]*block_dim[2]);
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

	free(blocklens);
	free(indicies);

	MPI_File fp;
	MPI_File_delete(file_name, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	int offset_i = sizeof(char) * n_size * block_dim[0] * ib;
	int offset_j = sizeof(char) * n_size * block_dim[0] *block_dim[1] *grid_dim[0] * jb;
	int offset_k = sizeof(char) * n_size * block_dim[0] *block_dim[1] * block_dim[2] * grid_dim[0] * grid_dim[1] * (grid_dim[2] - kb - 1);
	int offset_total = offset_i + offset_j + offset_k;
	MPI_File_set_view(fp, offset_total, MPI_CHAR, dtype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, text_buff, block_dim[0] * block_dim[1] * block_dim[2], num, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	MPI_Finalize();

	//if (id == 0) fprintf(stderr, "Time: %f\n", end-start);
	return 0;
}