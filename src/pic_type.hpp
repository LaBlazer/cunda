// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

// Structure definition for exchanging data between Host and Device
struct CudaPic
{
	CudaPic(unsigned x, unsigned y, unsigned z = 1) {
		m_size.x = x;
		m_size.y = y;
		m_size.z = z;
	}
  uint3 m_size;				// size of picture
  union {
	  void   *m_p_void;		// data of picture
	  uchar1 *m_p_uchar1;	// data of picture
	  uchar3 *m_p_uchar3;	// data of picture
	  uchar4 *m_p_uchar4;	// data of picture
  };

  __device__ __host__ uchar1 &at1(int x, int y) {
	  return m_p_uchar1[(y * m_size.x) + x];
  }

  __device__ __host__ uchar3 &at3(int x, int y) {
	  return m_p_uchar3[(y * m_size.x) + x];
  }

  __device__ __host__ uchar4 &at4(int x, int y) {
	  return m_p_uchar4[(y * m_size.x) + x];
  }
};
