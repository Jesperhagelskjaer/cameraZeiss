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
	void RandomMutation(int n = 1, int type = 1); // Perfoms mutation TYPE1 or TYPE2
	void RandomMutation1(void);
	void RandomMutation2(int n);
	void Print(void);
	unsigned char *GetMatrixPtr(void);
	double GetProbability(void) { return probabililty_; }

protected:
	unsigned char matrix_[M][M];
	double cost_;
	int binding_;
	double probabililty_;
};

class SLMParents
{
public:
	SLMParents(int numParents, int numBinding);
	~SLMParents();
	bool IsTemplatesFull(void);
	void GenerateNewParent(void);
	void PrintTemplatesCost(void);
	void PrintTemplates(void);
	SLMTemplate *GenerateOffspring(void);
	void CompareCostAndInsertTemplate(double cost);
	//void GenParents(void);
	unsigned char* GetNewParentMatrixPtr(void);
	unsigned char* GetMaxCostParentMatrixPtr(void);
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
