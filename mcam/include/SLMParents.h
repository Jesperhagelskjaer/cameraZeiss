///////////////////////////////////////////////////////////
//  SLMParents.h
//  Implementation of the Class SLMTemplate, SLMParents
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include "defs.h"
using namespace std;

class SLMTemplate
{
public:
	SLMTemplate(int binding);
	void SetCost(double cost);
	double GetCost(void);
	void GenRandom(void);
	void GenBinary(void);
	void GenBinaryInverse(SLMTemplate &templateIn);
	void MultiplyCell(SLMTemplate &templateIn, SLMTemplate &templateOut);
	void AddCell(SLMTemplate &templateIn, SLMTemplate &templateOut);
	void RandomMutation(int n);
	void Print(void);
	unsigned char *GetMatrixPtr(void);

protected:
	unsigned char matrix_[M][M];
	double cost_;
	int binding_;
};

class SLMParents
{
public:
	SLMParents(int numParents, int numBinding);
	~SLMParents();
	bool IsTemplatesFull(void);
	void GenerateNewParent(void);
	void PrintTemplates(void);
	SLMTemplate *GenerateOffspring(void);
	void CompareCostAndInsertTemplate(double cost);
	//void GenParents(void);
	unsigned char* GetNewParentMatrixPtr(void);
	void DeleteTemplates(int num);

protected:
	void DeleteLastTemplate(void);
	int SigmoidRandomDistribution(void);
	void GetRandomTemplateIdx(int &number1, int &number2);

	SLMTemplate *pParentNew_;
	std::vector<SLMTemplate*> SLMTemplates_;
	int numParents_;
	int numBindings_;
	int numOffsprings_;

private:
	SLMTemplate BinaryTemplate1_, BinaryTemplate2_;
	SLMTemplate Parent1_, Parent2_;
};
