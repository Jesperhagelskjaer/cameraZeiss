///////////////////////////////////////////////////////////
//  SLMParentsCUDA.cpp
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge + Jesper Hagelskj�r
///////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "SLMParentsCUDA.h"

// Defined in SLMParentsCUDA.cu
void GenBinaryCUDA(unsigned char* matrixCUDA_, int Stride_);
void GenBinaryInverseCUDA(unsigned char* dst, unsigned char* src, int strideDst, int strideSrc);
void MultiplyCellCUDA(unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc);
void AddCellCUDA(unsigned char* dst, unsigned char* src1, unsigned char* src2, int strideDst, int strideSrc);


SLMTemplateCUDA::SLMTemplateCUDA(int binding) : SLMTemplate(binding)
{
	matrixCUDA_ = 0;
	Stride_ = 0;
	width_ = M;
	height_ = M;
}

//-------------------------------------------------------------------------------------------------------
// CUDA Implementation

SLMTemplateCUDA::~SLMTemplateCUDA()
{
	FreeMemoryOnCUDA();
}

bool SLMTemplateCUDA::MallocMatrixOnCUDA(void)
{
    cudaError_t cudaStatus;
	bool ok = true;

	if (matrixCUDA_ == 0) {
		// Allocation of memory for 2D source template in single precision format
		cudaStatus = cudaMallocPitch((void **)(&matrixCUDA_), &Stride_, width_ * sizeof(unsigned char), height_);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch matrixToCUDA_ failed!");
		    ok = false;
		}
		Stride_ /= sizeof(unsigned char);
	}
	return ok;
}

void SLMTemplateCUDA::FreeMemoryOnCUDA(void)
{
	if (matrixCUDA_ != 0) {
	  cudaFree(matrixCUDA_);
	  matrixCUDA_ = 0;
	}
}

bool SLMTemplateCUDA::CopyToCUDA(void)
{
    cudaError_t cudaStatus;
	bool ok = true;
	size_t Stride = M;
	
	// Copy input matrix from host memory to GPU buffers
    cudaStatus = cudaMemcpy2D(matrixCUDA_, Stride_ * sizeof(unsigned char),
                               matrix_, Stride * sizeof(unsigned char),
                               width_ * sizeof(unsigned char), height_,
                               cudaMemcpyHostToDevice);
    
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to CUDA failed!");
		ok = false;
    }

	return ok;
}

bool SLMTemplateCUDA::CopyFromCUDA(void)
{
    cudaError_t cudaStatus;
	bool ok = true;
	size_t Stride = M;

    // Copy input matrix from GPU buffers to host memory
    cudaStatus = cudaMemcpy2D(matrix_, Stride * sizeof(unsigned char),
							   matrixCUDA_, Stride_ * sizeof(unsigned char),
                               width_ * sizeof(unsigned char), height_,
                               cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2D from CUDA failed!");
		ok = false;
    }

	return ok;
}

//-------------------------------------------------------------------------------------------------------
SLMParentsCUDA::SLMParentsCUDA(int numParents, int numBindings) : 
	SLMParents(numParents, numBindings),
	BinaryTemplate1_(numBindings),
	BinaryTemplate2_(numBindings),
	Parent1_(numBindings),
	Parent2_(numBindings)
{
};

SLMParentsCUDA::~SLMParentsCUDA()
{
	for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
		SLMTemplate* pTemplate = *it;
		delete pTemplate;
	}
}

bool SLMParentsCUDA::InitCUDA(void)
{
   cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return false;
    }

	BinaryTemplate1_.MallocMatrixOnCUDA();
	BinaryTemplate2_.MallocMatrixOnCUDA();
	Parent1_.MallocMatrixOnCUDA();
	Parent2_.MallocMatrixOnCUDA();

	return true;
}

bool SLMParentsCUDA::ExitCUDA(void)
{
   cudaError_t cudaStatus;

   BinaryTemplate1_.FreeMemoryOnCUDA();
   BinaryTemplate2_.FreeMemoryOnCUDA();
   Parent1_.FreeMemoryOnCUDA();
   Parent2_.FreeMemoryOnCUDA();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return false;
    }
	return true;
}

void SLMParentsCUDA::GenerateNewParent(void)
{
	SLMTemplateCUDA *pNewTemplate =  new SLMTemplateCUDA(numBindings_);
	pNewTemplate->GenRandom();
	pNewTemplate->MallocMatrixOnCUDA();
	pNewTemplate->CopyToCUDA();
	pParentNew_ = pNewTemplate;
}

SLMTemplate *SLMParentsCUDA::GenerateOffspring(void)
{
	int number1, number2;
	SLMTemplateCUDA *pTemplate1, *pTemplate2;
	GetRandomTemplateIdx(number1, number2);
		
	//cout << endl;
	pTemplate1 = (SLMTemplateCUDA *)SLMTemplates_[number1];
	//cout << "Template1: " << endl;;
	//pTemplate1->Print();
	//cout << endl;

	pTemplate2 = (SLMTemplateCUDA *)SLMTemplates_[number2];

	//cout << "Template2: " << endl;;
	//pTemplate2->Print();
	//cout << endl << endl;

	GenBinaryCUDA(BinaryTemplate1_.matrixCUDA_, (int)BinaryTemplate1_.Stride_);

	GenBinaryInverseCUDA(BinaryTemplate2_.matrixCUDA_, BinaryTemplate1_.matrixCUDA_, (int)BinaryTemplate2_.Stride_, (int)BinaryTemplate1_.Stride_);

	//cout << "GenBinary1: " << endl;;
	//BinaryTemplate1_.Print();
	//cout << endl;
	//cout << "GenBinary2: " << endl;;
	//BinaryTemplate2_.Print();
	//cout << endl;

	MultiplyCellCUDA(Parent1_.matrixCUDA_, pTemplate1->matrixCUDA_, BinaryTemplate1_.matrixCUDA_, (int)Parent1_.Stride_, (int)BinaryTemplate1_.Stride_); 
	MultiplyCellCUDA(Parent2_.matrixCUDA_, pTemplate2->matrixCUDA_, BinaryTemplate2_.matrixCUDA_, (int)Parent2_.Stride_, (int)BinaryTemplate2_.Stride_); 
		
	SLMTemplateCUDA *pNewTemplate = new SLMTemplateCUDA(numBindings_);
	pNewTemplate->MallocMatrixOnCUDA();
	AddCellCUDA(pNewTemplate->matrixCUDA_, Parent1_.matrixCUDA_, Parent2_.matrixCUDA_, (int)pNewTemplate->Stride_, (int)Parent2_.Stride_);
	pNewTemplate->CopyFromCUDA();
		
	//cout << "Offspring parentNew: " << endl;;
	//pParentNew_->Print();
	//cout << endl;
		

	pNewTemplate->RandomMutation();
	pNewTemplate->CopyToCUDA();
	pParentNew_ = pNewTemplate;
	//cout << "Offspring mutation: " << endl;;
	//pParentNew_->Print();
	//cout << "test" << endl;;
	return pParentNew_;
}