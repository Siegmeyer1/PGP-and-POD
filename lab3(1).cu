
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


__constant__ double average[32][3];
__constant__ double covMatrixInv[32][3][3];


__device__ int determineClass(uchar4 p, int nClasses) {
	double predicts[32];
	for (int i = 0; i < nClasses; i++)
	{
		predicts[i] = 0;
		double p_av[3], vec[3];
		for (int j = 0; j < 3; j++)
		{
			vec[j] = 0;
		}
		p_av[0] = p.x - average[i][0];
		p_av[1] = p.y - average[i][1];
		p_av[2] = p.z - average[i][2];
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				vec[j] += -p_av[k] * covMatrixInv[i][k][j];//i j k??
			}
			predicts[i] += vec[j] * p_av[j];
		}
	}
	double max = predicts[0];
	int index = 0;
	for (int i = 0; i < nClasses; i++)
	{
		if (predicts[i] > max)
		{
			max = predicts[i];
			index = i;
		}
	}
	return index;
}


__global__ void kernel(uchar4* data, int w, int h, int nClasses) {
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int x = idx; x < w*h; x += offset) {
			data[x].w = determineClass(data[x], nClasses);
		}
}

int main() {
	int w = 0, h = 0, nClasses = 0, p1 = 0, p2 = 0;
		unsigned long long np = 0;
	std::vector<std::vector<int2>> pixels;
	char file_name[256];

	std::cin >> file_name;
	FILE* fp = fopen(file_name, "rb");
	fread(&w, sizeof(unsigned int), 1, fp);
	fread(&h, sizeof(unsigned int), 1, fp);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	std::cin >> file_name;

	std::cin >> nClasses;
	for (int i = 0; i < nClasses; i++) {
		pixels.emplace_back(std::vector<int2>());
		std::cin >> np;
		for (unsigned long long j = 0; j < np; j++)
		{
			std::cin >> p1 >> p2;
			pixels[i].push_back(make_int2(p1, p2));
		}
	}
	//counting avg
	double averageHost[32][3];
	for (int i = 0; i < nClasses; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			averageHost[i][j] = 0.;
		}
	}
	for (int i = 0; i < nClasses; i++)
	{
		int np = pixels[i].size();
		for (int j = 0; j < np; j++)
		{
			uchar4 pixel = data[pixels[i][j].x + w * pixels[i][j].y];
			averageHost[i][0] += pixel.x;
			averageHost[i][1] += pixel.y;
			averageHost[i][2] += pixel.z;
		}
		for (int j = 0; j < 3; j++)
		{
			averageHost[i][j] /= np;
		}
	} //end counting avg

	//counting cov
	double covMatrixHost[32][3][3];
	for (int i = 0; i < nClasses; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				covMatrixHost[i][j][k] = 0;
			}
		}
	}
	for (int i = 0; i < nClasses; i++)
	{
		int np = pixels[i].size();
		for (int j = 0; j < np; j++)
		{
			uchar4 pixel = data[pixels[i][j].x + w * pixels[i][j].y];
			double pixelTmp[3];
			pixelTmp[0] = pixel.x - averageHost[i][0];
			pixelTmp[1] = pixel.y - averageHost[i][1];
			pixelTmp[2] = pixel.z - averageHost[i][2];
			for (int ax = 0; ax < 3; ax++)
			{
				for (int ay = 0; ay < 3; ay++)
				{
					covMatrixHost[i][ax][ay] += pixelTmp[ax] * pixelTmp[ay];
				}
			}
		}
		for (int ax = 0; ax < 3; ax++)
		{
			for (int ay = 0; ay < 3; ay++)
			{
				covMatrixHost[i][ax][ay] /= np - 1;
			}
		}
	} //end counting cov

	//counting inverse cov
	double covMatrixInvHost[32][3][3];
	double determinants[32];
	for (int i = 0; i < nClasses; i++)
	{
		determinants[i] = 0;
		for (int j = 0; j < 3; j++)
		{
			determinants[i] +=
				covMatrixHost[i][0][j] *
					(covMatrixHost[i][1][(j + 1) % 3] * covMatrixHost[i][2][(j + 2) % 3] -
					covMatrixHost[i][1][(j + 2) % 3] * covMatrixHost[i][2][(j + 1) % 3]);
		}
	}
	for (int i = 0; i < nClasses; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				covMatrixInvHost[i][j][k] =
					(covMatrixHost[i][(k + 1) % 3][(j + 1) % 3] * covMatrixHost[i][(k + 2) % 3][(j + 2) % 3] -
					covMatrixHost[i][(k + 1) % 3][(j + 2) % 3] * covMatrixHost[i][(k + 2) % 3][(j + 1) % 3]) / 
					determinants[i];

			}
		}
	} //finaly end counting inverse cov
	
	CSC(cudaMemcpyToSymbol(average, averageHost, sizeof(double) * 32 * 3));
	CSC(cudaMemcpyToSymbol(covMatrixInv, covMatrixInvHost, sizeof(double) * 32 * 3 * 3));
	/*
	// Ïîäãîòîâêà äàííûõ äëÿ òåêñòóðû
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Ïîäãîòîâêà òåêñòóðíîé ññûëêè, íàñòðîéêà èíòåðôåéñà ðàáîòû ñ äàííûìè
	tex.addressMode[0] = cudaAddressModeClamp;	// Ïîëèòèêà îáðàáîòêè âûõîäà çà ãðàíèöû ïî êàæäîìó èçìåðåíèþ
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Áåç èíòåðïîëÿöèè ïðè îáðàùåíèè ïî äðîáíûì êîîðäèíàòàì
	tex.normalized = false;						// Ðåæèì íîðìàëèçàöèè êîîðäèíàò: áåç íîðìàëèçàöèè

	// Ñâÿçûâàåì èíòåðôåéñ ñ äàííûìè
	CSC(cudaBindTextureToArray(tex, arr, ch));
	*/
	uchar4* dataDevice;
	CSC(cudaMalloc(&dataDevice, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(dataDevice, data, sizeof(uchar4)* w* h, cudaMemcpyHostToDevice));

	/*cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));*/
	kernel <<<32, 32>>> (dataDevice, w, h, nClasses);
	CSC(cudaGetLastError());
	/*CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
	printf("time = %f\n", t);*/
	CSC(cudaMemcpy(data, dataDevice, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	/*// Îòâÿçûâàåì äàííûå îò òåêñòóðíîé ññûëêè
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));*/
	CSC(cudaFree(dataDevice));

	fp = fopen(file_name, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
