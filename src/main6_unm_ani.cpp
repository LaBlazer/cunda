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
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <time.h> 
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.hpp"
#include "pic_type.hpp"
#include "animation.hpp"
#include "Timer.hpp"

void cu_crop(CudaPic input, CudaPic output, int2 position);
void cu_resize(CudaPic input, CudaPic output);
void cu_zoom(CudaPic input, CudaPic output, float2 center, float zoom);

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	Animation l_animation;

	// Output images
	cv::Mat cropped(200, 200, CV_8UC3);
	cv::Mat upscaled(400, 400, CV_8UC3);
	cv::Mat zoomed(400, 400, CV_8UC3);

	// Ball image
	cv::Mat input = cv::imread("shrek.jpg", CV_LOAD_IMAGE_UNCHANGED);

	// Data for CUDA
	CudaPic cp_input(input.data, input.cols, input.rows), cp_cropped, cp_upscaled, cp_zoomed(zoomed.data, zoomed.cols, zoomed.rows);

	cp_cropped.m_size.x = cropped.cols;
	cp_cropped.m_size.y = cropped.rows;
	cp_cropped.m_p_uchar3 = ( uchar3 * )cropped.data;

	cp_upscaled.m_size.x = upscaled.cols;
	cp_upscaled.m_size.y = upscaled.rows;
	cp_upscaled.m_p_uchar3 = (uchar3 *)upscaled.data;

	cp_upscaled.m_size.x = upscaled.cols;
	cp_upscaled.m_size.y = upscaled.rows;
	cp_upscaled.m_p_uchar3 = (uchar3 *)upscaled.data;


	cv::imshow("Input", input);

	cu_crop(cp_input, cp_cropped, { 320, 20 });

	cv::imshow("Cropped", cropped);

	cu_resize(cp_cropped, cp_upscaled);

	cv::imshow("Upscaled", upscaled);


	float time = 0;
	while (cv::waitKey(10)) {
		time += 0.03;
		cu_zoom(cp_upscaled, cp_zoomed, { 0.5, 0.5 }, 1 + (sin(time) + 1));

		cv::imshow("Zoomed", zoomed);
	}

	cv::waitKey( 0 );
}

