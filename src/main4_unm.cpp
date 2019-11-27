// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.hpp"
#include "pic_type.hpp"

// Function prototype from .cu file
void cu_run_grayscale( CudaPic t_bgr_pic, CudaPic t_bw_pic );
void cu_run_bgr(CudaPic bgr, CudaPic b, CudaPic g, CudaPic r);

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	// Load image
	cv::Mat img_bgr = cv::imread("shrek.jpg", cv::IMREAD_COLOR );

	if ( !img_bgr.data )
	{
		printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
		return 1;
	}

	// create empty BW image
	cv::Mat img_b(img_bgr.size(), CV_8U);
	cv::Mat img_g(img_bgr.size(), CV_8U);
	cv::Mat img_r(img_bgr.size(), CV_8U);

	// data for CUDA
	unsigned x = img_bgr.size().width;
	unsigned y = img_bgr.size().height;
	CudaPic cuda_bgr(x, y), cuda_b(x, y), cuda_g(x, y), cuda_r(x, y);

	cuda_bgr.m_p_uchar3 = (uchar3 *) img_bgr.data;
	cuda_b.m_p_uchar1 = (uchar1 *)img_b.data;
	cuda_g.m_p_uchar1 = (uchar1 *)img_g.data;
	cuda_r.m_p_uchar1 = (uchar1 *)img_r.data;

	// Function calling from .cu file
	cv::imshow("Color before", img_bgr);

	cu_run_bgr(cuda_bgr, cuda_b, cuda_g, cuda_r);

	// Show the Color and BW image
	cv::imshow("Color after", img_bgr);
	cv::imshow("Blue", img_b);
	cv::imshow("Green", img_g);
	cv::imshow("Red", img_r);
	cv::waitKey( 0 );
}

