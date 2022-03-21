
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4* out, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 w11, w12, w13, w31, w32, w33, w21, w23;
	float4 Gx, Gy;
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			w11 = tex2D(tex, x - 1, y - 1); w12 = tex2D(tex, x, y - 1); w13 = tex2D(tex, x + 1, y - 1);
			w31 = tex2D(tex, x - 1, y + 1); w32 = tex2D(tex, x, y + 1); w33 = tex2D(tex, x + 1, y + 1);
			w21 = tex2D(tex, x - 1, y); w23 = tex2D(tex, x + 1, y);
			Gx = make_float4((w13.x + w23.x + w33.x - w11.x - w21.x - w31.x),
							 (w13.y + w23.y + w33.y - w11.y - w21.y - w31.y), 
							 (w13.z + w23.z + w33.z - w11.z - w21.z - w31.z), 
							  w11.w);
			Gy = make_float4((w31.x + w32.x + w33.x - w11.x - w12.x - w13.x),
							 (w31.y + w32.y + w33.y - w11.y - w12.y - w13.y),
							 (w31.z + w32.z + w33.z - w11.z - w12.z - w13.z),
						 	  w11.w);
			float bx = Gx.x * 0.299 + Gx.y * 0.587 + Gx.z * 0.114;			
			float by = Gy.x * 0.299 + Gy.y * 0.587 + Gy.z * 0.114;
			//float bord_col = sqrt(Gx.x * Gx.x + Gx.y * Gx.y + Gx.z * Gx.z +
			//						Gy.x * Gy.x + Gy.y * Gy.y + Gy.z * Gy.z);
			float bord_col = sqrt(bx * bx + by * by);
			bord_col = min(255., bord_col);
			out[y * w + x] = make_uchar4(bord_col, bord_col, bord_col, w11.w);
		}
}

int main() {
	int w=0, h=0;
	char file_name[256];
	scanf("%s", file_name);
	FILE* fp = fopen(file_name, "rb");
	fread(&w, sizeof(unsigned int), 1, fp);
	fread(&h, sizeof(unsigned int), 1, fp);
	printf("%d, %d", w, h);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	scanf("%s", file_name);

	// Подготовка данных для текстуры
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4* dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

	kernel <<<dim3(32, 32), dim3(32, 32)>>> (dev_out, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(file_name, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
