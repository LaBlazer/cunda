// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Simple animation.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "Snowflake.hpp"


__global__ void kernel_crop(CudaPic input, CudaPic output, int2 position)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= output.m_size.y ) return;
	if (l_x >= output.m_size.x ) return;

	int offset_x = l_x + position.x, offset_y = l_y + position.y;

	if (offset_y >= input.m_size.y || offset_y < 0) return;
	if (offset_x >= input.m_size.x || offset_x < 0) return;

	// Store point into image
	output.at3(l_x, l_y) = input.at3(offset_x, offset_y);
}


__global__ void kernel_resize( CudaPic input, CudaPic output)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= (output.m_size.y - 1)) return;
	if (l_x >= (output.m_size.x - 1)) return;
	
	// new real position
	float input_x = l_x * ((float)input.m_size.x / (float)output.m_size.x);
	float input_y = l_y * ((float)input.m_size.y / (float)output.m_size.y);

	// diff x and y
	float diff_x = input_x - (int)input_x;
	float diff_y = input_y - (int)input_y;

	// points
	uchar3 bgr00 = input.at3((int)input_x,		(int)input_y);
	uchar3 bgr01 = input.at3((int)input_x,		(int)input_y + 1);
	uchar3 bgr10 = input.at3((int)input_x + 1,	(int)input_y);
	uchar3 bgr11 = input.at3((int)input_x + 1,	(int)input_y + 1);

	uchar3 out;

	// color calculation
	out.x = bgr00.x * (1 - diff_y) * (1 - diff_x) +
		bgr01.x * (1 - diff_y) * (diff_x)+
		bgr10.x * (diff_y) * (1 - diff_x) +
		bgr11.x * (diff_y) * (diff_x);

	out.y = bgr00.y * (1 - diff_y) * (1 - diff_x) +
		bgr01.y * (1 - diff_y) * (diff_x)+
		bgr10.y * (diff_y) * (1 - diff_x) +
		bgr11.y * (diff_y) * (diff_x);

	out.z = bgr00.z * (1 - diff_y) * (1 - diff_x) +
		bgr01.z * (1 - diff_y) * (diff_x)+
		bgr10.z * (diff_y) * (1 - diff_x) +
		bgr11.z * (diff_y) * (diff_x);

	output.at3(l_x, l_y) = out;
}

__global__ void kernel_animation(Snowflake snowflake, CudaPic background)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;

	int offset_x = snowflake.X + l_x, offset_y = snowflake.Y + l_y;

	if (offset_y >= background.m_size.y || l_y >= snowflake.picture.m_size.y) return;
	if (offset_x >= background.m_size.x || l_x >= snowflake.picture.m_size.x) return;
	if (l_y < 0) return;
	if (l_x < 0) return;

	if (snowflake.picture.at4(l_x, l_y).w == 0) return;
	
	// Store point into image
	background.at3(offset_x, offset_y).x = snowflake.picture.at4(l_x, l_y).x;
	background.at3(offset_x, offset_y).y = snowflake.picture.at4(l_x, l_y).y;
	background.at3(offset_x, offset_y).z = snowflake.picture.at4(l_x, l_y).z;
}

void cu_animation(CudaPic background, std::vector<Snowflake> &snowflakes) {
	cudaError_t l_cerr;
	
	for (const Snowflake &s : snowflakes) {
		// Grid creation, size of grid must be equal or greater than images
		int l_block_size = 32;
		dim3 l_blocks((s.picture.m_size.x + l_block_size - 1) / l_block_size,
			(s.picture.m_size.y + l_block_size - 1) / l_block_size);
		dim3 l_threads(l_block_size, l_block_size);
		kernel_animation<<<l_blocks, l_threads>>> (s, background);

		if ((l_cerr = cudaGetLastError()) != cudaSuccess)
			printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
	}

	cudaDeviceSynchronize();
}

void cu_crop( CudaPic input, CudaPic output, int2 position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( (output.m_size.x + l_block_size - 1 ) / l_block_size,
			       (output.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_crop<<< l_blocks, l_threads >>>(input, output, position);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

void cu_resize(CudaPic input, CudaPic output)
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks((output.m_size.x + l_block_size - 1) / l_block_size,
				(output.m_size.y + l_block_size - 1) / l_block_size);
	dim3 l_threads(l_block_size, l_block_size);
	kernel_resize << < l_blocks, l_threads >> > (input, output);

	if ((l_cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

	cudaDeviceSynchronize();
}

void cu_zoom(CudaPic input, CudaPic output, float2 center, float zoom)
{

	cv::Mat mat_temp(output.m_size.y * zoom, output.m_size.x * zoom, CV_8UC3);
	CudaPic temp(mat_temp.data, mat_temp.cols, mat_temp.rows);


	cu_resize(input, temp);

	cu_crop(temp, output, { (int)(temp.m_size.x * center.x) - (int)(output.m_size.x / 2),
							(int)(temp.m_size.y * center.y) - (int)(output.m_size.y / 2) });


}