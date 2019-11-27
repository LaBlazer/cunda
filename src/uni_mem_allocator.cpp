#include "uni_mem_allocator.hpp"

cv::UMatData* UniformAllocator::allocate(int dims, const int* sizes, int type,
				   void* data0, size_t* step,
					cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const
{
	size_t total = CV_ELEM_SIZE(type);
	for (int i = dims-1; i >= 0; i--)
	{
		if (step)
		{
			if (data0 && step[i] != cv::Mat::AUTO_STEP)
			{
				CV_Assert(total <= step[i]);
				total = step[i];
			}
			else
			{
				step[i] = total;
			}
		}

		total *= sizes[i];
	}

	cv::UMatData* u = new cv::UMatData(this);
	u->size = total;

	if (data0)
	{
		u->data = u->origdata = static_cast<uchar*>(data0);
		u->flags |= cv::UMatData::USER_ALLOCATED;
	}
	else
	{
		void* ptr = 0;
		if(cudaMallocManaged(&ptr, total) != cudaSuccess)
		{
			abort();
		}


		u->data = u->origdata = static_cast<uchar*>(ptr);
	}

	return u;
}

bool UniformAllocator::allocate(cv::UMatData* u, cv::AccessFlag /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const
{
	return (u != NULL);
}

void UniformAllocator::deallocate(cv::UMatData* u) const
{
	if (!u)
		return;

	CV_Assert(u->urefcount >= 0);
	CV_Assert(u->refcount >= 0);

	if (u->refcount == 0)
	{
		if ( !(u->flags & cv::UMatData::USER_ALLOCATED) )
		{
			cudaFree(u->origdata);
			u->origdata = 0;
		}

		delete u;
	}
}

