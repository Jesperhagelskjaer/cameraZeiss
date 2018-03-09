///////////////////////////////////////////////////////////
//  SLMParentsCUDA.cu
//  Implementation of the Class AnalyseNeuronData
//  CUDA optimized
//  Created on:      05-june-2017 15:38:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cutil_inline_runtime.h"

#include <stdio.h>
#include "SLMParentsCUDA.h"

#define DEBUG_MSG  //printf

/**
 * CUDA Kernel Device code
 *
 */
__global__ void
genBinaryTemplate( unsigned char* dst, int seed, int strideDst)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0,  /* the sequence number is only important with multiple cores */
              0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  dst[row * strideDst + col] = curand(&state) % 2;
}

// THIS FUNCTION IS NOT WORKING YET KBE!!!!
void GenBinaryCUDA(unsigned char* matrixCUDA_, int Stride_)
{
	//setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( M / BLOCK_SIZE, M / BLOCK_SIZE );

   // DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    //DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    // Generate binary template
    genBinaryTemplate<<< grid, threads >>>(matrixCUDA_, (int)time(NULL), Stride_);
    //cutilSafeCall(cudaThreadSynchronize());
}

__global__ void
genBinaryInverseTemplate(unsigned char* dst, unsigned char* src, int strideDst, int strideSrc)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * strideDst + col] = 1 - src[row * strideSrc + col];
 
}

void GenBinaryInverseCUDA(unsigned char* dst, unsigned char* src, int strideDst, int strideSrc)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( M / BLOCK_SIZE, M / BLOCK_SIZE );

	// Generate binary template
    genBinaryInverseTemplate<<< grid, threads >>>(dst, src, strideDst, strideSrc);
    //cutilSafeCall(cudaThreadSynchronize());
}

__global__ void
multiplyTemplates(unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * strideDst + col] = src1[row * strideSrc + col] * src2[row * strideSrc + col];
 
}

void MultiplyCellCUDA(unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid( M / BLOCK_SIZE, M / BLOCK_SIZE );

	// Generate binary template
    multiplyTemplates<<< grid, threads >>>(dst, src1, src2, strideDst, strideSrc);
    //cutilSafeCall(cudaThreadSynchronize());
}

__global__ void
addTemplates( unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * strideDst + col] = src1[row * strideSrc + col] + src2[row * strideSrc + col];
 
}

void AddCellCUDA(unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc)
{
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid( M / BLOCK_SIZE, M / BLOCK_SIZE );

	// Generate binary template
    addTemplates<<< grid, threads >>>(dst, src1, src2, strideDst, strideSrc);
    //cutilSafeCall(cudaThreadSynchronize());
}
