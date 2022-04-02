#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <chrono>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#define N_TRIGS 66

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

enum OBJs {
	FLOOR,
	FIG1,
	FIG2,
	FIG3,
	ERROR
};

__device__ __host__ 
OBJs scene_element(int trig_num) {
	if 		(0 <= trig_num && trig_num <= 1) return FLOOR;
	else if (2 <= trig_num && trig_num <= 9) return FIG1;
	else if (10 <= trig_num && trig_num <= 45) return FIG2;
	else if (46 <= trig_num && trig_num <= 65) return FIG3;
	else return ERROR;
}

struct element_params {
	float3 center;
	float radius;
	float3 norm_color;
	float refract_coef;
	float transp_coef;
};

__device__ __host__
uchar4 to_color(float3 color) {
	return make_uchar4( color.x * 255, color.y * 255, color.z * 255, 0);
}

__device__ __host__ 
float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ 
float3 scale(float3 a, float b) {
	return {a.x * b, a.y * b, a.z * b};
}

__device__ __host__ 
float3 prod(float3 a, float3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ __host__ 
float3 norm(float3 v) {
	float l = sqrt(dot(v, v));
	return {v.x / l, v.y / l, v.z / l};
}

__device__ __host__ 
float3 diff(float3 a, float3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __host__ 
float3 add(float3 a, float3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __host__ 
float3 mult(float3 a, float3 b, float3 c, float3 v) {
	return {	a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z };
}

__device__ __host__ 
uchar4 color_sum(uchar4 a, uchar4 b) {
	return {(unsigned char)min(255, a.x + b.x),
			(unsigned char)min(255, a.y + b.y),
			(unsigned char)min(255, a.z + b.z),
			(unsigned char)min(255, a.w + b.w)};
}

__device__ __host__ 
uchar4 color_scale(uchar4 a, float coef) {
	return {(unsigned char)min(255, (int)(a.x * coef)),
			(unsigned char)min(255, (int)(a.y * coef)),
			(unsigned char)min(255, (int)(a.z * coef)),
			(unsigned char)min(255, (int)(a.w * coef))};
}

struct trig {
	float3 a;
	float3 b;
	float3 c;
	uchar4 color;
};

__device__ __host__
uchar4 get_from_tex(trig* _trigs, uchar4* tex, float floor_size, int tex_size, float _x, float _y) {
	int _i, _j;
	//printf("%f, %f\n", _x, _y);
	_i = min((int)(abs(_trigs[0].a.x - _x) / floor_size * tex_size), tex_size-1);
	_j = min((int)(abs(_trigs[0].a.y - _y) / floor_size * tex_size), tex_size-1);
	uchar4 color = tex[_i + _j*tex_size];
	color.w = 0;
	return color;
}

void octahedron(trig *trigs, float radius, float3 center, float3 norm_color) {
    uchar4 color = to_color(norm_color);

    float3 vertices[] = {
            float3{ center.x + radius, center.y,          center.z            },
            float3{ center.x - radius, center.y,          center.z            },
            float3{ center.x,          center.y + radius, center.z            },
            float3{ center.x,          center.y - radius, center.z            },
            float3{ center.x,          center.y,          center.z + radius   },
            float3{ center.x,          center.y,          center.z - radius   }
    };

    trigs[2] = trig{vertices[5], vertices[2], vertices[0], color};
    trigs[3] = trig{vertices[5], vertices[0], vertices[3], color};
    trigs[4] = trig{vertices[5], vertices[3], vertices[1], color};
    trigs[5] = trig{vertices[5], vertices[1], vertices[2], color};
    trigs[6] = trig{vertices[4], vertices[3], vertices[0], color};
    trigs[7] = trig{vertices[4], vertices[1], vertices[3], color};
    trigs[8] = trig{vertices[4], vertices[2], vertices[1], color};
    trigs[9] = trig{vertices[4], vertices[0], vertices[2], color};
}

void dodecahedron(trig *trigs, float radius, float3 center, float3 norm_color) {
    uchar4 color = to_color(norm_color);
    float phi = (1 + sqrt(5)) / 2;
	float sqr3 = (float)sqrt(3);

    float3 vertices[] = {
            float3{center.x + (-1/phi / sqr3 * radius),   center.y + ( 0            * radius),      center.z + ( phi     / sqr3 * radius)     },
            float3{center.x + ( 1/phi / sqr3 * radius),   center.y + ( 0            * radius),      center.z + ( phi     / sqr3 * radius)     },
            float3{center.x + (-1     / sqr3 * radius),   center.y + ( 1     / sqr3 * radius),      center.z + ( 1       / sqr3 * radius)     },
            float3{center.x + ( 1     / sqr3 * radius),   center.y + ( 1     / sqr3 * radius),      center.z + ( 1       / sqr3 * radius)     },
            float3{center.x + ( 1     / sqr3 * radius),   center.y + (-1     / sqr3 * radius),      center.z + ( 1       / sqr3 * radius)     },
            float3{center.x + (-1     / sqr3 * radius),   center.y + (-1     / sqr3 * radius),      center.z + ( 1       / sqr3 * radius)     },
            float3{center.x + ( 0            * radius),   center.y + (-phi   / sqr3 * radius),      center.z + ( 1/phi   / sqr3 * radius)     },
            float3{center.x + ( 0            * radius),   center.y + ( phi   / sqr3 * radius),      center.z + ( 1/phi   / sqr3 * radius)     },
            float3{center.x + (-phi   / sqr3 * radius),   center.y + (-1/phi / sqr3 * radius),      center.z + ( 0              * radius)     },
            float3{center.x + (-phi   / sqr3 * radius),   center.y + ( 1/phi / sqr3 * radius),      center.z + ( 0              * radius)     },
            float3{center.x + ( phi   / sqr3 * radius),   center.y + ( 1/phi / sqr3 * radius),      center.z + ( 0              * radius)     },
            float3{center.x + ( phi   / sqr3 * radius),   center.y + (-1/phi / sqr3 * radius),      center.z + ( 0              * radius)     },
            float3{center.x + ( 0            * radius),   center.y + (-phi   / sqr3 * radius),      center.z + (-1/phi   / sqr3 * radius)     },
            float3{center.x + ( 0            * radius),   center.y + ( phi   / sqr3 * radius),      center.z + (-1/phi   / sqr3 * radius)     },
            float3{center.x + ( 1     / sqr3 * radius),   center.y + ( 1     / sqr3 * radius),      center.z + (-1       / sqr3 * radius)     },
            float3{center.x + ( 1     / sqr3 * radius),   center.y + (-1     / sqr3 * radius),      center.z + (-1       / sqr3 * radius)     },
            float3{center.x + (-1     / sqr3 * radius),   center.y + (-1     / sqr3 * radius),      center.z + (-1       / sqr3 * radius)     },
            float3{center.x + (-1     / sqr3 * radius),   center.y + ( 1     / sqr3 * radius),      center.z + (-1       / sqr3 * radius)     },
            float3{center.x + ( 1/phi / sqr3 * radius),   center.y + ( 0            * radius),      center.z + (-phi     / sqr3 * radius)     },
            float3{center.x + (-1/phi / sqr3 * radius),   center.y + ( 0            * radius),      center.z + (-phi     / sqr3 * radius)     }
    };

    trigs[10] = trig{vertices[4],  vertices[0],  vertices[6],  color};
    trigs[11] = trig{vertices[0],  vertices[5],  vertices[6],  color};
    trigs[12] = trig{vertices[0],  vertices[4],  vertices[1],  color};
    trigs[13] = trig{vertices[0],  vertices[3],  vertices[7],  color};
    trigs[14] = trig{vertices[2],  vertices[0],  vertices[7],  color};
    trigs[15] = trig{vertices[0],  vertices[1],  vertices[3],  color};
    trigs[16] = trig{vertices[10], vertices[1],  vertices[11], color};
    trigs[17] = trig{vertices[3],  vertices[1],  vertices[10], color};
    trigs[18] = trig{vertices[1],  vertices[4],  vertices[11], color};
    trigs[19] = trig{vertices[5],  vertices[0],  vertices[8],  color};
    trigs[20] = trig{vertices[0],  vertices[2],  vertices[9],  color};
    trigs[21] = trig{vertices[8],  vertices[0],  vertices[9],  color};
    trigs[22] = trig{vertices[5],  vertices[8],  vertices[16], color};
    trigs[23] = trig{vertices[6],  vertices[5],  vertices[12], color};
    trigs[24] = trig{vertices[12], vertices[5],  vertices[16], color};
    trigs[25] = trig{vertices[4],  vertices[12], vertices[15], color};
    trigs[26] = trig{vertices[4],  vertices[6],  vertices[12], color};
    trigs[27] = trig{vertices[11], vertices[4],  vertices[15], color};
    trigs[28] = trig{vertices[2],  vertices[13], vertices[17], color};
    trigs[29] = trig{vertices[2],  vertices[7],  vertices[13], color};
    trigs[30] = trig{vertices[9],  vertices[2],  vertices[17], color};
    trigs[31] = trig{vertices[13], vertices[3],  vertices[14], color};
    trigs[32] = trig{vertices[7],  vertices[3],  vertices[13], color};
    trigs[33] = trig{vertices[3],  vertices[10], vertices[14], color};
    trigs[34] = trig{vertices[8],  vertices[17], vertices[19], color};
    trigs[35] = trig{vertices[16], vertices[8],  vertices[19], color};
    trigs[36] = trig{vertices[8],  vertices[9],  vertices[17], color};
    trigs[37] = trig{vertices[14], vertices[11], vertices[18], color};
    trigs[38] = trig{vertices[11], vertices[15], vertices[18], color};
    trigs[39] = trig{vertices[10], vertices[11], vertices[14], color};
    trigs[40] = trig{vertices[12], vertices[19], vertices[18], color};
    trigs[41] = trig{vertices[15], vertices[12], vertices[18], color};
    trigs[42] = trig{vertices[12], vertices[16], vertices[19], color};
    trigs[43] = trig{vertices[19], vertices[13], vertices[18], color};
    trigs[44] = trig{vertices[17], vertices[13], vertices[19], color};
    trigs[45] = trig{vertices[13], vertices[14], vertices[18], color};
}

void icosahedron(trig *trigs, float radius, float3 center, float3 norm_color) {
    uchar4 color = to_color(norm_color);
	float a = 4 * radius / (sqrt(2 * (5 + sqrt(5))));
	float offs_z = radius * (2 - (1 + sqrt(5)) / 2.0);
	float radius2 = sqrt(a*a + offs_z*offs_z);
	float angle18 = 18.0 * M_PI / 180.0;
	float angle54 = 54.0 * M_PI / 180.0;

	float3 vertices[] = {
		float3{center.x, center.y, center.z + radius},														//0
		float3{center.x, center.y, center.z - radius},														//1
		float3{center.x + radius2 * cos(angle18), center.y + radius2 * sin(angle18), center.z + offs_z},	//2
		float3{center.x + radius2 * cos(angle54), center.y - radius2 * sin(angle54), center.z + offs_z},	//3
		float3{center.x - radius2 * cos(angle54), center.y - radius2 * sin(angle54), center.z + offs_z},	//4
		float3{center.x - radius2 * cos(angle18), center.y + radius2 * sin(angle18), center.z + offs_z},	//5
		float3{center.x							, center.y + radius2			   , center.z + offs_z},	//6
		float3{center.x + radius2 * cos(angle18), center.y - radius2 * sin(angle18), center.z - offs_z},	//7
		float3{center.x + radius2 * cos(angle54), center.y + radius2 * sin(angle54), center.z - offs_z},	//8
		float3{center.x - radius2 * cos(angle54), center.y + radius2 * sin(angle54), center.z - offs_z},	//9
		float3{center.x - radius2 * cos(angle18), center.y - radius2 * sin(angle18), center.z - offs_z},	//10
		float3{center.x							, center.y - radius2			   , center.z - offs_z} 	//11
	};

	trigs[46] = trig{vertices[0], vertices[2], vertices[3], color}; //"top hat"
	trigs[47] = trig{vertices[0], vertices[3], vertices[4], color};
	trigs[48] = trig{vertices[0], vertices[4], vertices[5], color};
	trigs[49] = trig{vertices[0], vertices[5], vertices[6], color};
	trigs[50] = trig{vertices[0], vertices[6], vertices[2], color};

	trigs[51] = trig{vertices[2], vertices[3], vertices[7], color}; //"belt upper row"
	trigs[52] = trig{vertices[3], vertices[4], vertices[11], color};
	trigs[53] = trig{vertices[4], vertices[5], vertices[10], color};
	trigs[54] = trig{vertices[5], vertices[6], vertices[9], color};
	trigs[55] = trig{vertices[6], vertices[2], vertices[8], color};

	trigs[56] = trig{vertices[9], vertices[8], vertices[6], color}; //"belt lower row"
	trigs[57] = trig{vertices[8], vertices[7], vertices[2], color};
	trigs[58] = trig{vertices[7], vertices[11], vertices[3], color};
	trigs[59] = trig{vertices[11], vertices[10], vertices[4], color};
	trigs[60] = trig{vertices[10], vertices[9], vertices[5], color};

	trigs[61] = trig{vertices[1], vertices[9], vertices[8], color}; //"bottom hat"
	trigs[62] = trig{vertices[1], vertices[8], vertices[7], color};
	trigs[63] = trig{vertices[1], vertices[7], vertices[11], color};
	trigs[64] = trig{vertices[1], vertices[11], vertices[10], color};
	trigs[65] = trig{vertices[1], vertices[10], vertices[9], color};
}

void build_space(trig *trigs, float3 a, float3 b, float3 c, float3 d, element_params elem_params[]) {
	trigs[0] = {a, c, d, to_color(elem_params[FLOOR].norm_color)};
	trigs[1] = {a, c, b, to_color(elem_params[FLOOR].norm_color)};
	octahedron(trigs, elem_params[FIG1].radius, elem_params[FIG1].center, elem_params[FIG1].norm_color);
	dodecahedron(trigs, elem_params[FIG2].radius, elem_params[FIG2].center, elem_params[FIG2].norm_color);
	icosahedron(trigs, elem_params[FIG3].radius, elem_params[FIG3].center, elem_params[FIG3].norm_color);
}

__device__ __host__ 
uchar4 ray(float3 pos, float3 dir, trig* trigs, float refr_coefs[], float transp_coefs[], float3 ls, float3 ls_color, uchar4 *tex, int tex_size, bool to_ls=false, int recursion_counter=5) {
	pos = add(pos, scale(dir, 0.01));

	int k, k_min = -1;
	float ts_min;
	for(k = 0; k < N_TRIGS; k++) {
		float3 e1 = diff(trigs[k].b, trigs[k].a);
		float3 e2 = diff(trigs[k].c, trigs[k].a);
		float3 p = prod(dir, e2);	
		float div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		float3 t = diff(pos, trigs[k].a);
		float u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		float3 q = prod(t, e1);
		float v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		float ts = dot(q, e2) / div;
		if (ts <= 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if (k_min == -1) {
		if (!to_ls) return {0, 0, 0, 0};
		else return to_color(ls_color);
	}
	if (!to_ls) {
		OBJs object = scene_element(k_min);
		float3 norma = norm( prod(diff(trigs[k_min].b, trigs[k_min].a), diff(trigs[k_min].c, trigs[k_min].a)) );
		float3 cur_pos = add(scale(dir, ts_min), pos);
		float3 light_dir = norm(diff(ls, cur_pos));
		float lighting_coeff = abs(dot(norma, light_dir));
		uchar4 shading_color, surface_color;
		if (object == FLOOR) {
			float floor_size = max(abs(trigs[0].a.x - trigs[0].b.x), abs(trigs[0].a.y - trigs[0].b.y));
			surface_color = get_from_tex(trigs, tex, floor_size, tex_size, cur_pos.x, cur_pos.y);
		} else {
			surface_color = trigs[k_min].color;
		}
		shading_color = color_scale(to_color(ls_color), 0.5);
		if (recursion_counter) {
			shading_color = color_scale(ray(cur_pos, light_dir, trigs, refr_coefs, transp_coefs, ls, ls_color, tex, tex_size, true), lighting_coeff);
			float refract_coef = refr_coefs[object];
			float transp_coef = transp_coefs[object];
			if (refract_coef > 0.0) {
				float3 refract_dir = norm(add(scale(scale(norma, dot(dir, norma)), -2), dir));
				surface_color = color_sum(color_scale(surface_color, 1.0 - refract_coef), 
										  color_scale(ray(cur_pos, refract_dir, trigs, refr_coefs, transp_coefs, ls, ls_color, tex, tex_size, false, recursion_counter-1), refract_coef));
			}
			if (transp_coef > 0.0) {
				surface_color = color_sum(color_scale(surface_color, 1.0 - transp_coef), 
										  color_scale(ray(cur_pos, dir, trigs, refr_coefs, transp_coefs, ls, ls_color, tex, tex_size, false, recursion_counter-1), transp_coef));
			}
		}
		return color_sum(surface_color, shading_color);
	}
	else {
		return {0, 0, 0, 0};
	}
}

void render(float3 pc, float3 pv, trig* trigs, float refr_coefs[], float transp_coefs[], float3 ls, float3 ls_color, int w, int h, float angle, uchar4 *data, uchar4 *tex, int tex_size, int rec_depth) {
	int i, j;
	float dw = 2.0 / (w - 1);
	float dh = 2.0 / (h - 1);
	float z = 1.0 / tan(angle * M_PI / 360.0);
	float3 bz = norm(diff(pv, pc));
	float3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
	float3 by = prod(bx, bz);
	for(i = 0; i < w; i++)
		for(j = 0; j < h; j++) {
			float3 v = {-1.0f + dw * i, (-1.0f + dh * j) * h / w, z};
			float3 dir = norm(mult(bx, by, bz, v));
			data[(h - 1 - j) * w + i] = ray(pc, dir, trigs, refr_coefs, transp_coefs, ls, ls_color, tex, tex_size, false, rec_depth);
		}
}

__global__ void gpu_render(float3 pc, float3 pv, trig* trigs, float* refr_coefs, float* transp_coefs, float3 ls, float3 ls_color, int w, int h, float angle, uchar4 *data, uchar4 *tex, int tex_size, int rec_depth) {
	int i, j;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	float dw = 2.0 / (w - 1);
	float dh = 2.0 / (h - 1);
	float z = 1.0 / tan(angle * M_PI / 360.0);
	float3 bz = norm(diff(pv, pc));
	float3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
	float3 by = prod(bx, bz);
	for (j = idy; j < h; j += offsety)
		for (i = idx; i < w; i += offsetx) {
			float3 v = {-1.0f + dw * i, (-1.0f + dh * j) * h / w, z};
			float3 dir = norm(mult(bx, by, bz, v));
			data[(h - 1 - j) * w + i] = ray(pc, dir, trigs, refr_coefs, transp_coefs, ls, ls_color, tex, tex_size, false, rec_depth);
		}
}

void ssaa(uchar4* data, uchar4* res, int w, int h, int ssaa_scale) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			uint4 avg = {0, 0, 0, 0};
			for (int j = 0; j < ssaa_scale; j++) {
				for (int i = 0; i < ssaa_scale; i++) {
					avg.x += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].x;
					avg.y += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].y;
					avg.z += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].z;
				}		
			}
			avg.x /= ssaa_scale*ssaa_scale;
			avg.y /= ssaa_scale*ssaa_scale;
			avg.z /= ssaa_scale*ssaa_scale;
			res[x + w*y] = make_uchar4(avg.x, avg.y, avg.z, avg.w);			
		}		
	}	
}

__global__ void gpu_ssaa(uchar4* data, uchar4* res, int w, int h, int ssaa_scale) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for (int y = idy; y < h; y += offsety) {
		for (int x = idx; x < w; x += offsetx) {
			uint4 avg = {0, 0, 0, 0};
			for (int j = 0; j < ssaa_scale; j++) {
				for (int i = 0; i < ssaa_scale; i++) {
					avg.x += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].x;
					avg.y += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].y;
					avg.z += data[(i + x*ssaa_scale) + w*ssaa_scale*(j + y*ssaa_scale)].z;
				}		
			}
			avg.x /= ssaa_scale*ssaa_scale;
			avg.y /= ssaa_scale*ssaa_scale;
			avg.z /= ssaa_scale*ssaa_scale;
			res[x + w*y] = make_uchar4(avg.x, avg.y, avg.z, avg.w);			
		}		
	}	
}



int main(int argc, char *argv[]) {
	bool on_gpu = true;
	bool deflt = false;
	if (argc > 1) {
		if (strcmp(argv[1], "--cpu") == 0) on_gpu = false;
		if (strcmp(argv[1], "--default") == 0) deflt = true;
	}
	trig trigs[N_TRIGS];
	int k, w = 640, h = 480, tw, th, n_frames = 126, ssaa_scale = 2, recursion_depth = 5;
	char buff[256], out_dir[256] = "res/%d.data", path_to_tex[256] = "floor.data";
	float 	rc_0 = 7.0, zc_0 = 5.0, phic_0 = 0.0,
		  	Ac_r = 2.0, Ac_z = 0.0,
		  	wc_r = 2.0, wc_z = 6.0, wc_phi = 1.0, 
			pc_r = 0.0, pc_z = 0.0,

			rv_0 = 2.0, zv_0 = 2.0, phiv_0 = 0.0,
		  	Av_r = 0.0, Av_z = 0.0,
		  	wv_r = 1.0, wv_z = 4.0, wv_phi = 1.0, 
			pv_r = 0.0, pv_z = 0.0,
			
			fov = 120.0,

			radius1 = 2.0, radius2 = 2.0, radius3 = 2.0,

			refr_coef_floor = 0.25, refr_coef_1 = 0.25,
			refr_coef_2 = 0.25,		refr_coef_3 = 0.25,
			transp_coef_floor = 0.0, transp_coef_1 = 0.25,
			transp_coef_2 = 0.25,	  transp_coef_3 = 0.25;
	int		edge_lights1 = 0, edge_lights2 = 0, edge_lights3 = 0, n_lights = 1;

	float3 	center1 = float3{-3.0, -3.0, 2.0},
			center2 = float3{0.0, 0.0, 2.0},
			center3 = float3{3.0, 3.0, 2.0},
			color_floor = float3{0.0, 0.0, 1.0},
			color1 	= float3{0.5, 0.0, 0.0},
			color2 	= float3{0.0, 0.5, 0.0},
			color3 	= float3{0.5, 0.5, 0.0},
			floor_a = float3{-15, -15, 0},
			floor_b = float3{-15, 15, 0},
			floor_c = float3{15, 15, 0},
			floor_d = float3{15, -15, 0};
	float3 	light_source = float3{0.0, 2.5, 15.0},
			light_color  = float3{0.5, 0.5, 0.5},
			dummy_source, dummy_color;


	if (deflt) {
		printf("%d\n", n_frames);
		printf("%s\n", out_dir);
		printf("%d %d %f\n",w, h, fov);
		printf("%f %f %f\t%f %f\t%f %f %f\t%f %f\n",   	rc_0, zc_0, phic_0, Ac_r, Ac_z, 
													   	wc_r, wc_z, wc_phi, pc_r, pc_z);
		printf("%f %f %f\t%f %f\t%f %f %f\t%f %f\n",   	rv_0, zv_0, phiv_0, Av_r, Av_z, 
													   	wv_r, wv_z, wv_phi, pv_r, pv_z);
		printf("%f %f %f\t%f %f %f\t%f\t%f\t%f\t%d\n", 	center1.x, center1.y, center1.z,
													   	color1.x, color1.y, color1.z,
													   	radius1, refr_coef_1, transp_coef_1,
													   	edge_lights1);
		printf("%f %f %f\t%f %f %f\t%f\t%f\t%f\t%d\n", 	center2.x, center2.y, center2.z,
													   	color2.x, color2.y, color2.z,
													   	radius2, refr_coef_2, transp_coef_2,
													   	edge_lights2);
		printf("%f %f %f\t%f %f %f\t%f\t%f\t%f\t%d\n", 	center3.x, center3.y, center3.z,
													   	color3.x, color3.y, color3.z,
													   	radius3, refr_coef_3, transp_coef_3,
													   	edge_lights3);
		printf("%f %f %f\t%f %f %f\t%f %f %f\t%f %f %f\t%s\t%f %f %f\t%f\n",
														floor_a.x, floor_a.y, floor_a.z,
														floor_b.x, floor_b.y, floor_b.z,
														floor_c.x, floor_c.y, floor_c.z,
														floor_d.x, floor_d.y, floor_d.z,
														path_to_tex, color_floor.x, color_floor.y,
														color_floor.z, refr_coef_floor);
		printf("%d\n", n_lights);
		printf("%f %f %f\t%f %f %f\n", light_source.x, light_source.y, light_source.z,
										light_color.x, light_color.y, light_color.z);
		printf("%d %d\n", recursion_depth, ssaa_scale);
		return 0;
	} else {
		scanf("%d", &n_frames);
		scanf("%s", out_dir);
		scanf("%d %d %f",&w, &h, &fov);
		scanf("%f %f %f %f %f %f %f %f %f %f",   	&rc_0, &zc_0, &phic_0, &Ac_r, &Ac_z, 
													   	&wc_r, &wc_z, &wc_phi, &pc_r, &pc_z);
		scanf("%f %f %f %f %f %f %f %f %f %f",   	&rv_0, &zv_0, &phiv_0, &Av_r, &Av_z, 
													   	&wv_r, &wv_z, &wv_phi, &pv_r, &pv_z);
		scanf("%f %f %f %f %f %f %f %f %f %d", 	&center1.x, &center1.y, &center1.z,
													   	&color1.x, &color1.y, &color1.z,
													   	&radius1, &refr_coef_1, &transp_coef_1,
													   	&edge_lights1);
		scanf("%f %f %f %f %f %f %f %f %f %d", 	&center2.x, &center2.y, &center2.z,
													   	&color2.x, &color2.y, &color2.z,
													   	&radius2, &refr_coef_2, &transp_coef_2,
													   	&edge_lights2);
		scanf("%f %f %f %f %f %f %f %f %f %d", 	&center3.x, &center3.y, &center3.z,
													   	&color3.x, &color3.y, &color3.z,
													   	&radius3, &refr_coef_3, &transp_coef_3,
													   	&edge_lights3);
		scanf("%f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %f",
														&floor_a.x, &floor_a.y, &floor_a.z,
														&floor_b.x, &floor_b.y, &floor_b.z,
														&floor_c.x, &floor_c.y, &floor_c.z,
														&floor_d.x, &floor_d.y, &floor_d.z,
														path_to_tex, &color_floor.x, &color_floor.y,
														&color_floor.z, &refr_coef_floor);
		scanf("%d", &n_lights);
		scanf("%f %f %f %f %f %f", &light_source.x, &light_source.y, &light_source.z,
										&light_color.x, &light_color.y, &light_color.z);
		if (n_lights > 1) for (int l = 0; l < n_lights-1; l++) {
			scanf("%f %f %f %f %f %f", &dummy_source.x, &dummy_source.y, &dummy_source.z,
										&dummy_color.x, &dummy_color.y, &dummy_color.z);
		}		
		scanf("%d %d", &recursion_depth, &ssaa_scale);
	}

	FILE* fp = fopen(path_to_tex, "rb");
	fread(&tw, sizeof(unsigned int), 1, fp);
	fread(&th, sizeof(unsigned int), 1, fp);
	uchar4 *tex = (uchar4*)malloc(sizeof(uchar4) * tw * th);
	fread(tex, sizeof(uchar4), tw * th, fp);
	fclose(fp);

	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	uchar4 *dev_data, *big_data, *big_dev_data, *dev_tex;
	float3 pc, pv;

	element_params elem_params[4];
	float refr_coefs[4], *dev_refr_coefs;
	float transp_coefs[4], *dev_transp_coefs;
	elem_params[0].norm_color = color_floor;
	elem_params[0].refract_coef = refr_coef_floor;
	elem_params[1].center = center1;
	elem_params[1].norm_color = color1;
	elem_params[1].radius = radius1;
	elem_params[1].refract_coef = refr_coef_1;
	elem_params[2].center = center2;
	elem_params[2].norm_color = color2;
	elem_params[2].radius = radius2;
	elem_params[2].refract_coef = refr_coef_2;
	elem_params[3].center = center3;
	elem_params[3].norm_color = color3;
	elem_params[3].radius = radius3;
	elem_params[3].refract_coef = refr_coef_3;

	refr_coefs[FLOOR] = refr_coef_floor;
	refr_coefs[FIG1] = refr_coef_1;
	refr_coefs[FIG2] = refr_coef_2;
	refr_coefs[FIG3] = refr_coef_3;
	transp_coefs[FLOOR] = transp_coef_floor;
	transp_coefs[FIG1] = transp_coef_1;
	transp_coefs[FIG2] = transp_coef_2;
	transp_coefs[FIG3] = transp_coef_3;

	build_space(trigs, floor_a, floor_b, floor_c, floor_d, elem_params);

	int tex_size = min(tw, th);

	trig *dev_trigs;
	if (on_gpu) {
		CSC(cudaMalloc(&dev_tex, sizeof(uchar4) * tw * th));
		CSC(cudaMemcpy(dev_tex, tex, sizeof(uchar4) * tw * th, cudaMemcpyHostToDevice));
		free(tex);

		CSC(cudaMalloc(&dev_refr_coefs, sizeof(float) * 4));
		CSC(cudaMemcpy(dev_refr_coefs, refr_coefs, sizeof(float) * 4, cudaMemcpyHostToDevice));

		CSC(cudaMalloc(&dev_transp_coefs, sizeof(float) * 4));
		CSC(cudaMemcpy(dev_transp_coefs, transp_coefs, sizeof(float) * 4, cudaMemcpyHostToDevice));

		CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
		CSC(cudaMalloc(&big_dev_data, sizeof(uchar4) * w*ssaa_scale * h*ssaa_scale));

		CSC(cudaMalloc(&dev_trigs, sizeof(trig) * N_TRIGS));
		CSC(cudaMemcpy(dev_trigs, trigs, sizeof(trig) * N_TRIGS, cudaMemcpyHostToDevice));
	} else {
		big_data = (uchar4*)malloc(sizeof(uchar4) * w*ssaa_scale * h*ssaa_scale);
	}

	for(k = 0; k < n_frames; k++) {
		float t = 2*M_PI*k / n_frames;
		float rc = Ac_r * sin(wc_r*t + pc_r) + rc_0;
		float zc = Ac_z * sin(wc_z*t + pc_z) + zc_0;
		float phic = wc_phi*t + phic_0;
		float rv = Av_r * sin(wv_r*t + pv_r) + rv_0;
		float zv = Av_z * sin(wv_z*t + pv_z) + zv_0;
		float phiv = wv_phi*t + phiv_0;

		pc = float3{cos(phic)*rc, sin(phic)*rc, zc};
		pv = float3{cos(phiv)*rv, sin(phiv)*rv, zv};

		float time;

		if (!on_gpu) {
			auto begin = std::chrono::steady_clock::now();
			render(pc, pv, trigs, refr_coefs, transp_coefs, light_source, light_color, w*ssaa_scale, h*ssaa_scale, fov, big_data, tex, tex_size, recursion_depth);
			ssaa(big_data, data, w, h, ssaa_scale);
			auto end = std::chrono::steady_clock::now();
			time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
		}
		else {
			cudaEvent_t start, end;
			CSC(cudaEventCreate(&start));
			CSC(cudaEventCreate(&end));
			CSC(cudaEventRecord(start));	
			gpu_render<<<dim3(16, 16), dim3(16, 16)>>>(pc, pv, dev_trigs, dev_refr_coefs, dev_transp_coefs, light_source, light_color, w*ssaa_scale, h*ssaa_scale, fov, big_dev_data, dev_tex, tex_size, recursion_depth);
			CSC(cudaGetLastError());
			gpu_ssaa<<<dim3(16, 16), dim3(16, 16)>>>(big_dev_data, dev_data, w, h, ssaa_scale);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
			CSC(cudaEventRecord(end));
			CSC(cudaEventSynchronize(end));
			CSC(cudaEventElapsedTime(&time, start, end));
			CSC(cudaEventDestroy(start));
			CSC(cudaEventDestroy(end));
		}
		sprintf(buff, out_dir, k);
		printf("%d\t%fms\n", k, time);

		FILE *out = fopen(buff, "wb");
		fwrite(&w, sizeof(int), 1, out);
		fwrite(&h, sizeof(int), 1, out);
		fwrite(data, sizeof(uchar4), w * h, out);
		fclose(out);
	}
	free(data);
	if(on_gpu) {
		CSC(cudaFree(dev_tex));
		CSC(cudaFree(dev_refr_coefs));
		CSC(cudaFree(dev_transp_coefs));
		CSC(cudaFree(dev_data));
		CSC(cudaFree(big_dev_data));
		CSC(cudaFree(dev_trigs));
	} else {
		free(big_data);
	}
	return 0;
}