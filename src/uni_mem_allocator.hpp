
#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


class UniformAllocator : public cv::MatAllocator {
public:
	cv::UMatData* allocate(int dims, const int* sizes, int type,
                       void* data0, size_t* step,
		cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const override;

	bool allocate(cv::UMatData* u, cv::AccessFlag /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const override;

	void deallocate(cv::UMatData* u) const override;

};
