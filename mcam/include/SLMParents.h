#pragma once
#include <iostream>
#include <vector>

using namespace std;


#define M 512 // Matrix size of SLM timeplate
class SLMTemplate
{
public:
	SLMTemplate()
	{
		GenRandom();
		cost_ = 0;
		binding_ = 1;
	};

	void SetCost(unsigned int cost)
	{
		cost_ = cost;
	};

	void SetBinding(unsigned int binding)
	{
		binding_ = binding;
	};

	void GenRandom(void)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				matrix_[i][j] = rand() % 255;
	};

	void GenBinary(void)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				matrix_[i][j] = rand() % 2;
	};

	void GenBinaryInverse(SLMTemplate &templateIn)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				matrix_[i][j] = 1 - templateIn.matrix_[i][j];
	};

	void MultiplyCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				templateOut.matrix_[i][j] = matrix_[i][j] * templateIn.matrix_[i][j];
	};

	void AddCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				templateOut.matrix_[i][j] = matrix_[i][j] + templateIn.matrix_[i][j];
	};

	void RandomMutation(void)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				matrix_[i][j];
	};

	void Print(void)
	{
		printf("Matrix: %d %d %d %d %d %d\r\n", matrix_[0][0], matrix_[0][1], matrix_[0][2], matrix_[0][3], matrix_[0][4], matrix_[0][5]);
	}

private:
	unsigned char matrix_[M][M];
	unsigned int cost_;
	unsigned int binding_;
};


#define NUM_PARENTS 30 // Number of parents
class SLMParents
{
public:
	SLMParents(int num)
	{
		GenParents(num);
	};

	~SLMParents()
	{
		for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
			SLMTemplate* pTemplate = *it;
			delete pTemplate;
		}
	};

	void PrintTemplates(void) {
		int count = 1;
		for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
			SLMTemplate* pTemplate = *it;
			printf("%02d.", count++);
			pTemplate->Print();
		}
	}

	SLMTemplate *GenerateOffspring(int NumBinding)
	{
		int number1, number2;
		SLMTemplate *pTemplate1, *pTemplate2;
		GetRandomTemplateIdx(number1, number2);
		pTemplate1 = SLMTemplates_[number1];
		pTemplate2 = SLMTemplates_[number2];
		BinaryTemplate1_.GenBinary();
		BinaryTemplate2_.GenBinaryInverse(BinaryTemplate1_);
		pTemplate1->MultiplyCell(BinaryTemplate1_, Parent1_);
		pTemplate2->MultiplyCell(BinaryTemplate2_, Parent2_);
		Parent1_.AddCell(Parent2_, ParentNew_);
		Parent1_.RandomMutation();
		return &ParentNew_;
	}

	void GenParents(int num)
	{
		for (int i = 0; i < num; i++) {
			SLMTemplates_.push_back(new SLMTemplate());
		}
	};

private:

	void GetRandomTemplateIdx(int &number1, int &number2)
	{
		number1 = rand() % NUM_PARENTS; //NUM_PARENTS;
		number2 = rand() % NUM_PARENTS; //NUM_PARENTS;
	}

	SLMTemplate BinaryTemplate1_, BinaryTemplate2_;
	SLMTemplate Parent1_, Parent2_, ParentNew_;
	std::vector<SLMTemplate*> SLMTemplates_;
};
