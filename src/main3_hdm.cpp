// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "pic_type.h"

// Prototype of function in .cu file
void cu_run_animation( CudaPic t_pic, uint2 t_block_size );

// Image size
#define SIZEX 500 // Width of image
#define	SIZEY 350 // Height of image
// Block size for threads
#define BLOCKX 10 // block width
#define BLOCKY 10 // block height

int main()
{
	// Creation of empty image.
	// Image is stored line by line.
	cv::Mat l_cv_img( SIZEY, SIZEX, CV_8UC3, cv::Scalar(0xdb, 0xaa, 0x75));

	CudaPic l_pic_img;
	l_pic_img.m_size.x = l_cv_img.size().width; // equivalent to cv_img.cols
	l_pic_img.m_size.y = l_cv_img.size().height; // equivalent to cv_img.rows
	l_pic_img.m_p_uchar3 = ( uchar3* ) l_cv_img.data;

	cv::imshow("Before", l_cv_img);

	// Function calling from .cu file
	uint2 l_block_size = { BLOCKX, BLOCKY };
	cu_run_animation( l_pic_img, l_block_size );

	// Show modified image
	cv::imshow( "After", l_cv_img );
	cv::waitKey( 0 );
}

