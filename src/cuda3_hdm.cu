#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "math.h"
#include "pic_type.h"

// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( CudaPic t_cuda_pic )
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;

	int middle_y = t_cuda_pic.m_size.y / 2;

	uchar3 * pix = &t_cuda_pic.at3(l_x, l_y);
	
	int distance = abs(l_y - middle_y);
	if (distance < 45) {
		if (distance < 30) {
			pix->x = 0;
			pix->y = 0;
			pix->z = 0;
		}
		else {
			pix->x = 255;
			pix->y = 255;
			pix->z = 255;
		}
	}
}

void cu_run_animation( CudaPic t_pic, uint2 t_block_size )
{
	cudaError_t l_cerr;

	CudaPic l_cuda_pic;
	l_cuda_pic.m_size = t_pic.m_size;

	// Memory allocation in GPU device
	l_cerr = cudaMalloc( &l_cuda_pic.m_p_void, l_cuda_pic.m_size.x * l_cuda_pic.m_size.y * sizeof( uchar3 ) );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Copy data to GPU device
	l_cerr = cudaMemcpy( l_cuda_pic.m_p_void, t_pic.m_p_void, l_cuda_pic.m_size.x * l_cuda_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyHostToDevice );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Grid creation with computed organization
	dim3 l_grid( ( l_cuda_pic.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
				 ( l_cuda_pic.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( l_cuda_pic );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Copy data from GPU device to PC
	l_cerr = cudaMemcpy( t_pic.m_p_void, l_cuda_pic.m_p_void, t_pic.m_size.x * t_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Free memory
	cudaFree( l_cuda_pic.m_p_void );

	cudaDeviceSynchronize();
}
