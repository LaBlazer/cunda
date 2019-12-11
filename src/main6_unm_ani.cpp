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
#include "Snowflake.hpp"
#include "Timer.hpp"

void cu_animation(CudaPic background, std::vector<Snowflake> &snowflakes);

float randRangeFloat(float min, float max)
{
	assert(max > min);
	float random = ((float)rand()) / (float)RAND_MAX;

	float range = max - min;
	return (random*range) + min;
}

int randRange(int min, int max) //range : [min, max)
{
	return min + rand() % ((max + 1) - min);
}

int main( int t_numarg, char **t_arg )
{
	srand(time(NULL));
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	const unsigned width = 800, height = 500;
	
	// Load images
	cv::Mat input = cv::imread("snowflake.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat background = cv::imread("winter.jpg", CV_LOAD_IMAGE_UNCHANGED);

	// Data for CUDA
	CudaPic cuda_snowflake(input.data, input.cols, input.rows);
	

	std::vector<Snowflake> snowflakes;

	for (int i = 0; i < 60; i++) {
		snowflakes.emplace_back(randRange(-input.cols, background.cols), 
								randRange(-input.rows * 2, -input.rows),
								randRangeFloat(0.5, 2.5), 
								randRangeFloat(30, 40), 
								randRange(1, 7), 
								cuda_snowflake);
	}

	float time = 0;
	Timer t;
	while (cv::waitKey(10)) {
		time += t.elapsed();
		t.reset();
		for (Snowflake &s : snowflakes) {
			s.Y += s.speed;
			s.X = s.originX + (int)(sin(time * s.timeScale) * s.sinScale);

			if (s.Y > background.rows) {
				s.Y = randRange(-s.picture.m_size.y * 2, -s.picture.m_size.y);
				s.originX = randRange(0, background.cols);
			}
		}

		cv::Mat tempBackground = background.clone();
		CudaPic cuda_background(tempBackground.data, tempBackground.cols, tempBackground.rows);
		cu_animation(cuda_background, snowflakes);

		cv::imshow("Animation", tempBackground);
	}

	cv::waitKey( 0 );
}

