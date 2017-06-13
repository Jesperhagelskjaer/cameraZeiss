///////////////////////////////////////////////////////////
//  SLMParents.h
//  Implementation of the Class SLMTemplate
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
	SLMTemplate();
	void SetCost(double cost);
	double GetCost(void);
	void GenRandom(void);
	void GenBinary(void);
	void GenBinaryInverse(SLMTemplate &templateIn);
	void MultiplyCell(SLMTemplate &templateIn, SLMTemplate &templateOut);
	void AddCell(SLMTemplate &templateIn, SLMTemplate &templateOut);
	void RandomMutation(void);
	void Print(void);
	unsigned char *GetMatrixPtr(void);

private:
	unsigned char matrix_[M][M];
	double cost_;
};


class SLMParents
{
public:
	SLMParents(int num);
	~SLMParents();
	bool IsTemplatesFull(void);
	void GenerateNewParent(void);
	void PrintTemplates(void);
	SLMTemplate *GenerateOffspring(int NumBinding);
	void CompareCostAndInsertTemplate(double cost);
	void GenParents(int num);
	unsigned char* GetNewParentMatrixPtr(void);

private:
	void DeleteLastTemplate(void);
	void GetRandomTemplateIdx(int &number1, int &number2);

	SLMTemplate BinaryTemplate1_, BinaryTemplate2_;
	SLMTemplate Parent1_, Parent2_;
	SLMTemplate *pParentNew_;
	std::vector<SLMTemplate*> SLMTemplates_;
};
