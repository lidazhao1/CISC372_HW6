//Simple optimized box blur
//by: Greg Silber
//Date: 5/1/2021
//This program reads an image and performs a simple averaging of pixels within a supplied radius.  For optimization,
//it does this by computing a running sum for each column within the radius, then averaging that sum.  Then the same for 
//each row.  This should allow it to be easily parallelized by column then by row, since each call is independent.

#define _CRT_SECURE_NO_WARNINGS
#define THREAD_BLOCK_SIZE 256

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define stride_bytes pWidth
#define gpu_compute_col computeColumn
#define gpu_compute_row computeRow
#define gpu_run(b,t) <<<b,t>>>
#define cuda_malloc cudaMalloc
#define cuda_memcpy cudaMemcpy
#define cuda_free cudaFree
#define cuda_error cudaError_t
#define cuda_device_synchronize cudaDeviceSynchronize

typedef struct _image {
	uint8_t* p_data;
	int width;
	int height;
	int bpp;
	long stride_bytes;
	long all_bytes;
} image;

typedef image* (*compute_action)(const image*, int);

//////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void single_thread_col(uint8_t* src, float* dest, int stride_bytes, int height, int radius, int bpp) {
	printf("start single_thread_col, thread index: %d.\n", threadIdx.x);
}

__global__ void single_thread_row(uint8_t* src, float* dest, int stride_bytes, int height, int radius, int bpp) {
	printf("start single_thread_row, thread index: %d.\n", threadIdx.x);
}

__global__ void gpu_compute_col(float* dst, uint8_t* src, int stride_bytes, int height, int radius, int bpp) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("start multi_thread_col, thread index: %d, block index: %d, block size: %d, col: %d.\n", threadIdx.x, blockIdx.x, blockDim.x, col);

	int i = 0;

	if (col >= stride_bytes) {
		return;
	}
	//initialize the first element of each column
	dst[col] = src[col];
	//start tue sum up to radius*2 by only adding
	for (i = 1;i <= radius * 2;i++)
		dst[i*stride_bytes + col] = src[i*stride_bytes + col] + dst[(i - 1)*stride_bytes + col];
	for (i = radius * 2 + 1;i < height;i++)
		dst[i*stride_bytes + col] = src[i*stride_bytes + col] + dst[(i - 1)*stride_bytes + col] - src[(i - 2 * radius - 1)*stride_bytes + col];
	//now shift everything up by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
	for (i = radius;i < height;i++) {
		dst[(i - radius)*stride_bytes + col] = dst[i*stride_bytes + col] / (radius * 2 + 1);
	}
	//now the first and last radius values make no sense, so blank them out
	for (i = 0;i < radius;i++) {
		dst[i*stride_bytes + col] = 0;
		dst[(height - 1)*stride_bytes - i*stride_bytes + col] = 0;
	}
}

__global__ void gpu_compute_row(float* dst, float* src, int stride_bytes, int height, int radius, int bpp) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("start multi_thread_row, thread index: %d, block index: %d, block size: %d, row: %d.\n", threadIdx.x, blockIdx.x, blockDim.x, row);

	int i = 0;
	int bradius = radius*bpp;

	if (row >= height) {
		return;
	}
	//initialize the first bpp elements so that nothing fails
	for (i = 0;i < bpp;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i];
	//start the sum up to radius*2 by only adding (nothing to subtract yet)
	for (i = bpp;i < bradius * 2 * bpp;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i] + dst[row*stride_bytes + i - bpp];
	for (i = bradius * 2 + bpp;i < stride_bytes;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i] + dst[row*stride_bytes + i - bpp] - src[row*stride_bytes + i - 2 * bradius - bpp];
	//now shift everything over by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
	for (i = bradius;i < stride_bytes;i++) {
		dst[row*stride_bytes + i - bradius] = dst[row*stride_bytes + i] / (radius * 2 + 1);
	}
	//now the first and last radius values make no sense, so blank them out
	for (i = 0;i < bradius;i++) {
		dst[row*stride_bytes + i] = 0;
		dst[(row + 1)*stride_bytes - 1 - i] = 0;
	}
}

//Computes a single column of the destination image by summing radius pixels
//Parameters: src: Teh src image as width*height*bpp 1d array
//            dest: pre-allocated array of size width*height*bpp to receive summed row
//            col: The current column number
//            pWidth: The width of the image * the bpp (i.e. number of bytes in a row)
//            height: The height of the source image
//            radius: the width of the blur
//            bpp: The bits per pixel in the src image
//Returns: None
void cpu_compute_col(float* dst, uint8_t* src, int col, int stride_bytes, int height, int radius, int bpp) {
	int i = 0;

	//initialize the first element of each column
	dst[col] = src[col];
	//start tue sum up to radius*2 by only adding
	for (i = 1;i <= radius * 2;i++)
		dst[i*stride_bytes + col] = src[i*stride_bytes + col] + dst[(i - 1)*stride_bytes + col];
	for (i = radius * 2 + 1;i < height;i++)
		dst[i*stride_bytes + col] = src[i*stride_bytes + col] + dst[(i - 1)*stride_bytes + col] - src[(i - 2 * radius - 1)*stride_bytes + col];
	//now shift everything up by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
	for (i = radius;i < height;i++) {
		dst[(i - radius)*stride_bytes + col] = dst[i*stride_bytes + col] / (radius * 2 + 1);
	}
	//now the first and last radius values make no sense, so blank them out
	for (i = 0;i < radius;i++) {
		dst[i*stride_bytes + col] = 0;
		dst[(height - 1)*stride_bytes - i*stride_bytes + col] = 0;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//Computes a single row of the destination image by summing radius pixels
//Parameters: src: Teh src image as width*height*bpp 1d array
//            dest: pre-allocated array of size width*height*bpp to receive summed row
//            row: The current row number
//            pWidth: The width of the image * the bpp (i.e. number of bytes in a row)
//            rad: the width of the blur
//            bpp: The bits per pixel in the src image
//Returns: None
void cpu_compute_row(float* dst, float* src, int row, int stride_bytes, int radius, int bpp) {
	int i = 0;
	int bradius = radius*bpp;

	//initialize the first bpp elements so that nothing fails
	for (i = 0;i < bpp;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i];
	//start the sum up to radius*2 by only adding (nothing to subtract yet)
	for (i = bpp;i < bradius * 2 * bpp;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i] + dst[row*stride_bytes + i - bpp];
	for (i = bradius * 2 + bpp;i < stride_bytes;i++)
		dst[row*stride_bytes + i] = src[row*stride_bytes + i] + dst[row*stride_bytes + i - bpp] - src[row*stride_bytes + i - 2 * bradius - bpp];
	//now shift everything over by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
	for (i = bradius;i < stride_bytes;i++) {
		dst[row*stride_bytes + i - bradius] = dst[row*stride_bytes + i] / (radius * 2 + 1);
	}
	//now the first and last radius values make no sense, so blank them out
	for (i = 0;i < bradius;i++) {
		dst[row*stride_bytes + i] = 0;
		dst[(row + 1)*stride_bytes - 1 - i] = 0;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
image* image_load(const char* filename) {
	image* p_image = (image*)malloc(sizeof(image));
	memset(p_image, 0, sizeof(image));
	p_image->p_data = stbi_load(filename, &p_image->width, &p_image->height, &p_image->bpp, 0);
	if (NULL == p_image->p_data) {
		free(p_image);
		return (NULL);
	}
	else {
		p_image->stride_bytes = p_image->width * p_image->bpp;
		p_image->all_bytes = p_image->stride_bytes * p_image->height;
	}
	return (p_image);
}

int    image_save(const image* p_image, const char* filename) {
	return (NULL == p_image ? -1 : stbi_write_png(filename, p_image->width, p_image->height, p_image->bpp, p_image->p_data, p_image->stride_bytes));
}

image* image_copy(const image* p_image_src, float* p_data) {
	int i = 0;
	image* p_image_dst = (image*)malloc(sizeof(image));
	memcpy(p_image_dst, p_image_src, sizeof(image));
	p_image_dst->p_data = (uint8_t*)malloc(sizeof(uint8_t)*p_image_src->all_bytes);
	for (i = 0; i < p_image_src->all_bytes; i++) {
		p_image_dst->p_data[i] = (uint8_t)p_data[i];
	}
	return (p_image_dst);
}

void   image_free(image* p_image) {
	if (NULL != p_image) {
		if (NULL != p_image->p_data) {
			free(p_image->p_data);
		}
		free(p_image);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
image* single_thread_cpu(const image* p_image_src, int radius) {
	long i = 0;
	float* p_matrix1 = (float*)malloc(sizeof(float)*p_image_src->all_bytes);
	float* p_matrix2 = (float*)malloc(sizeof(float)*p_image_src->all_bytes);
	for (i = 0; i < p_image_src->stride_bytes; i++) {
		cpu_compute_col( p_matrix1, p_image_src->p_data, i, p_image_src->stride_bytes, p_image_src->height, radius, p_image_src->bpp);
	}
	for (i = 0; i < p_image_src->height; i++) {
		cpu_compute_row(p_matrix2, p_matrix1, i, p_image_src->stride_bytes, radius, p_image_src->bpp);
	}
	free(p_matrix1);
	image* p_image_dst = image_copy(p_image_src, p_matrix2);
	free(p_matrix2);
	return (p_image_dst);
}

image* multi_thread_cpu(const image* p_image_src, int radius) {
	return (NULL);
}

image* single_thread_gpu(const image* p_image_src, int radius) {
	return (NULL);
}

image* multi_thread_gpu(const image* p_image_src, int radius) {
	cuda_error status = cudaSuccess;
	int i = 0, block_count = 1, thread_count = 256; // block size == thread_count == ? thread / 1 block
	uint8_t* p_matrix0 = NULL;
	float* p_matrix1 = NULL;
	float* p_matrix2 = NULL;
	float* p_matrix3 = (float*)malloc(sizeof(float)*p_image_src->all_bytes);

	status = cuda_malloc((void**)&p_matrix0, sizeof(uint8_t)*p_image_src->all_bytes);
	status = cuda_memcpy(p_matrix0, p_image_src->p_data, sizeof(uint8_t)*p_image_src->all_bytes, cudaMemcpyHostToDevice);

	status = cuda_malloc((void**)&p_matrix1, sizeof(float)*p_image_src->all_bytes);
	status = cuda_malloc((void**)&p_matrix2, sizeof(float)*p_image_src->all_bytes);

	block_count = (p_image_src->stride_bytes + thread_count - 1) / thread_count;
	printf("width: %d, thread count: %d, block count: %d\n", p_image_src->width, thread_count, block_count);
	gpu_compute_col gpu_run(block_count, thread_count) (p_matrix1, p_matrix0, p_image_src->stride_bytes, p_image_src->height, radius, p_image_src->bpp);
	cuda_device_synchronize();
	cuda_free(p_matrix0);

	block_count = (p_image_src->height + thread_count - 1) / thread_count;
	printf("height: %d, thread count: %d, block count: %d\n", p_image_src->height, thread_count, block_count);
	gpu_compute_row gpu_run(block_count, thread_count) (p_matrix2, p_matrix1, p_image_src->stride_bytes, p_image_src->height, radius, p_image_src->bpp);
	cuda_device_synchronize();
	cuda_free(p_matrix1);

	status = cuda_memcpy(p_matrix3, p_matrix2, sizeof(float)*p_image_src->all_bytes, cudaMemcpyDeviceToHost);
	cuda_free(p_matrix2);
	image* p_image_dst = image_copy(p_image_src, p_matrix3);
	free(p_matrix3);

	return (p_image_dst);
}

image* start(const char* message, compute_action action, const image* p_image_src, int radius) {
	long time_start = 0, time_end = 0;
	time_start = (long)time(NULL);
	image* p_image_dst = action(p_image_src, radius);
	time_end = (long)time(NULL);
	printf("Message: %s, Radius: %d, Time: %ld\n", message, radius, time_end - time_start);
	return (p_image_dst);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	image* p_image_dst = NULL;
	image* p_image_src = NULL;
	int status = 0;
	int radius = 11;
	char* filename = argc > 1 ? argv[1] : "gauss.jpg";

	if (argc < 2) {
		printf("%s: <filename> <blur radius>\n\tblur radius=pixels to average on any side of the current pixel\n", argv[0]);
	}
	else if (argc > 2) {
		sscanf(argv[2], "%d", &radius);
	}

	printf("Load: %s...", filename);
	p_image_src = image_load(filename);
	if (NULL == p_image_src) {
		printf("Error.\n");
		return (-1);
	}
	printf("OK.\n");

	p_image_dst = start("single_thread_cpu", single_thread_cpu, p_image_src, radius);
	status = image_save(p_image_dst, "/tmp/single_thread_cpu.png");
	image_free(p_image_dst);

	/*
	p_image_dst = start("multi_thread_cpu", multi_thread_cpu, p_image_src, radius);
	status = image_save(p_image_dst, "/tmp/multi_thread_cpu.png");
	image_free(p_image_dst);

	p_image_dst = start("single_thread_gpu", single_thread_gpu, p_image_src, radius);
	status = image_save(p_image_dst, "/tmp/single_thread_gpu.png");
	image_free(p_image_dst);
	*/

	p_image_dst = start("multi_thread_gpu", multi_thread_gpu, p_image_src, radius);
	status = image_save(p_image_dst, "/tmp/multi_thread_gpu.png");
	image_free(p_image_dst);

	image_free(p_image_src);
	return (status);
}
