///////////////////////////////////////////////////////////
//  SLMParentsCUDA.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;
#include "SLMParents.h"

#define BLOCK_SIZE		32   // CUDA block size

class SLMTemplateCUDA : public SLMTemplate
{
	friend class SLMParentsCUDA;

public:

	SLMTemplateCUDA(int binding);
	~SLMTemplateCUDA();

	// CUDA implementation
	bool MallocMatrixOnCUDA(void);
	void FreeMemoryOnCUDA(void);
	bool CopyToCUDA(void);
	bool CopyFromCUDA(void);

private:
	// CUDA member atributes
	unsigned char *matrixCUDA_;
	size_t Stride_;
	
	int height_;
	int width_;

};


class SLMParentsCUDA : public SLMParents
{
public:
	SLMParentsCUDA(int numParents, int numBinding);
	~SLMParentsCUDA();
	bool InitCUDA(void);
	bool ExitCUDA(void);

	// CUDA implementation
	virtual void GenerateNewParent(void);
	virtual SLMTemplate *GenerateOffspring(void);

private:
	SLMTemplateCUDA BinaryTemplate1C_, BinaryTemplate2C_;
	SLMTemplateCUDA Parent1C_, Parent2C_;

};
