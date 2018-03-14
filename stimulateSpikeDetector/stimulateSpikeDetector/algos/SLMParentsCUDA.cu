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

// Global random state variable on GPU
curandState_t* m_randStates = 0;

/**
* This GPU kernel function is used to initialize the random states
*
*/
__global__ void initRandom(unsigned int seed, curandState_t* states, int rowIdx) 
{
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int seq = rowIdx + col;

	if (seq < M*M) {
		/* we have to initialize the state */
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
			seq, /* the sequence number should be different for each core (unless you want all
						cores to get the same sequence of numbers for some reason - use thread id! */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[col]);
	}
}

void initCUDARandom(void)
{
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(M / BLOCK_SIZE, M / BLOCK_SIZE);

	/* allocate space on the GPU for the random states */
	if (m_randStates == 0)
		cudaMalloc((void**)&m_randStates, (M*M) * sizeof(curandState_t));

	/* invoke the GPU to initialize all of the random states */
	//initRandom<<< grid, threads >>> ((int)time(0), m_randStates);
	std::cout << "Initializing random numbers on GPU" << std::endl;
	for (int i = 0; i < M; i++) {
		initRandom << < M / BLOCK_SIZE, BLOCK_SIZE >> > ((int)time(0), &m_randStates[M*i], M*i);
		cutilSafeCall(cudaThreadSynchronize());
		std::cout << M-i << '\r';
	}
}

void freeCUDARandom(void)
{
	if (m_randStates != 0)
		cudaFree(m_randStates);
}

/**
 * CUDA Kernel Device code to generate binary random templates
 *
 */
__global__ void
genBinaryTemplate( unsigned char* dst, curandState_t* states, int strideDst)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * strideDst + col;

  /* curand works like rand - except that it takes a state as a parameter */
  dst[index] = curand(&states[index]) % 2;
}

void GenBinaryCUDA(unsigned char* matrixCUDA_, int Stride_)
{
	//setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( M / BLOCK_SIZE, M / BLOCK_SIZE );
   // DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    //DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    // Generate binary template
    genBinaryTemplate<<< grid, threads >>>(matrixCUDA_, m_randStates, Stride_);
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
    cutilSafeCall(cudaThreadSynchronize());
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
    cutilSafeCall(cudaThreadSynchronize());
}
