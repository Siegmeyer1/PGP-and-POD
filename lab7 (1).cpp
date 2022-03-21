#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include "mpi.h"

#define _i(i, j, k) ( ((k) + 1) * (block_dim[0] + 2) * (block_dim[1] + 2) + ((j) + 1) * (block_dim[0] + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * grid_dim[0] * grid_dim[1] + (j) * grid_dim[0] + (i))

int main(int argc, char *argv[]) {
	int ib, jb, kb;
	int i, j, k;
    int grid_dim[3], block_dim[3];
	int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
    char file_name[100];

	double eps, hx, hy, hz, init;
	double *data, *temp, *next, *ibuff, *jbuff, *kbuff, *lineBuff, *gather_stop, borders[6], brick_dim[3];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {					// Reading calculation parameters
        scanf("%d %d %d\n", &grid_dim[0], &grid_dim[1], &grid_dim[2]);
        scanf("%d %d %d\n", &block_dim[0], &block_dim[1], &block_dim[2]);
        scanf("%s", file_name);
        scanf("%lf", &eps);
        scanf("%lf %lf %lf\n", &brick_dim[0], &brick_dim[1], &brick_dim[2]);
        scanf("%lf %lf %lf %lf %lf %lf\n",
            &borders[0], &borders[1], &borders[2], &borders[3], &borders[4], &borders[5]);
			//	down		up			left		right			front		back
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

	ib = id % grid_dim[0];
	jb = (id / grid_dim[0]) % grid_dim[1];
    kb = id / (grid_dim[0] * grid_dim[1]);

	hx = brick_dim[0] / (float)(grid_dim[0] * block_dim[0]);	
	hy = brick_dim[1] / (float)(grid_dim[1] * block_dim[1]);
    hz = brick_dim[2] / (float)(grid_dim[2] * block_dim[2]);
    
	data = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));	
	next = (double*)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2) * (block_dim[2]+2));
	ibuff = (double *)malloc(sizeof(double) * (block_dim[1]+2) * (block_dim[2]+2));
	jbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[2]+2));
	kbuff = (double *)malloc(sizeof(double) * (block_dim[0]+2) * (block_dim[1]+2));
	lineBuff = (double *)malloc(sizeof(double) * (block_dim[0]+2));
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
		}

/*===================================End of initialization====================================*/
	//==========================start of calc loop==============================//
	bool stop = false;
	//bool local_stop = false;
	double max;
	while (!stop) {
		MPI_Barrier(MPI_COMM_WORLD);
		//===================================forward borders transfer====================================//
		if (ib + 1 < grid_dim[0]) {					
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
		
		if (ib > 0) {
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
		}
		//===================================end of forward borders transfer====================================//
		
		//===================================backward borders transfer====================================//

		if (ib > 0) {					
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
		
		if (ib + 1 < grid_dim[0]) {
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
		}

		/*===================================end of backward borders transfer====================================*/

		MPI_Barrier(MPI_COMM_WORLD);
		//if (!local_stop) {
		max = 0;
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
				}
		temp = next;
		next = data;
		data = temp;
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

	if (id != 0) 
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
	}

	MPI_Finalize();
	return 0;
}