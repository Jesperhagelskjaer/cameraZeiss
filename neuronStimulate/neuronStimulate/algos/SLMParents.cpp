///////////////////////////////////////////////////////////
//  SLMParents.cpp
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge + Jesper Hagelskjær
///////////////////////////////////////////////////////////

#include "SLMParents.h"

SLMTemplate::SLMTemplate()
{
	cost_ = 0;
};

void SLMTemplate::SetCost(double cost)
{
	cost_ = cost;
};

double SLMTemplate::GetCost(void)
{
	return cost_;
};

void SLMTemplate::GenRandom(void)
{
	unsigned char test;
	unsigned int bindingtest_ = BIND;

	for (int i = 0; i < M; i++) {
		while (i % bindingtest_ != 0 && i < M) {
			for (int k = 0; k < M; k++) {
				matrix_[i][k] = matrix_[i-1][k];
			}
			++i;
		} 
		if (i >= M) break;
		for (int j = 0; j < M; j++) {
			if (j % bindingtest_ == 0) {
				test = rand() % 255;
			}
			matrix_[i][j] = test;
		}				
	}	
};

/*
void SLMTemplate::GenRandom(void)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = rand() % 255;
};
*/

void SLMTemplate::GenBinary(void)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = rand() % 2;
};

void SLMTemplate::GenBinaryInverse(SLMTemplate &templateIn)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = 1 - templateIn.matrix_[i][j];
};

void SLMTemplate::MultiplyCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			templateOut.matrix_[i][j] = matrix_[i][j] * templateIn.matrix_[i][j];
};

void SLMTemplate::AddCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			templateOut.matrix_[i][j] = matrix_[i][j] + templateIn.matrix_[i][j];
};

void SLMTemplate::RandomMutation(void)
{
		//cout << "random mutation" << endl;
		float propabililty = (float)exp(-0.52); //exp(-0.52);
		//out << "probability:" << propabililty << endl;
		//cout << "signel rand() call: " << (float)rand() / (float)RAND_MAX << endl;
		int bindingtest_ = BIND;
		int j;
		int i;
		float threshold;

		for (i = 0; i < M; i += bindingtest_) {
			//cout << "i " << i << endl;
			for (j = 0; j < M; j += bindingtest_) {
				//cout << (int)matrix_[i][j] << " " ;
				threshold = (float)rand() / (float)RAND_MAX;
				if (propabililty > threshold) {  
					threshold = (float)1.1;
					int test = rand() % 255;
					for (int it2 = 0; it2 < bindingtest_; ++it2){
							 
						for (int jt2 = 0; jt2 < bindingtest_; ++jt2) {
							//cout << it2 << " " << jt2 << " " << i + it2 << " " << j + jt2  << " " << test << endl;
							matrix_[i + it2][j + jt2] = test;
							//cout << (int)matrix_[i + it2][j + jt2] << " ";
						}
					}				
				}
			}
		}
};

void SLMTemplate::Print(void)
{

	//#ifdef DEBUG_
	printf("cost: %f\r\n", cost_);
	printf("line1: %d %d %d %d %d %d %d\r\n", matrix_[0][0], matrix_[0][1], matrix_[0][2], matrix_[0][3], matrix_[0][4], matrix_[0][5], matrix_[0][6]);
	printf("line2: %d %d %d %d %d %d %d\r\n", matrix_[1][0], matrix_[1][1], matrix_[1][2], matrix_[1][3], matrix_[1][4], matrix_[1][5], matrix_[1][6]);
	printf("line3: %d %d %d %d %d %d %d\r\n", matrix_[2][0], matrix_[2][1], matrix_[2][2], matrix_[2][3], matrix_[2][4], matrix_[2][5], matrix_[2][6]);
	printf("line4: %d %d %d %d %d %d %d\r\n", matrix_[3][0], matrix_[3][1], matrix_[3][2], matrix_[3][3], matrix_[3][4], matrix_[3][5], matrix_[3][6]);
	printf("line5: %d %d %d %d %d %d %d\r\n", matrix_[4][0], matrix_[4][1], matrix_[4][2], matrix_[4][3], matrix_[4][4], matrix_[4][5], matrix_[4][6]);
	printf("line6: %d %d %d %d %d %d %d\r\n", matrix_[5][0], matrix_[5][1], matrix_[5][2], matrix_[5][3], matrix_[5][4], matrix_[5][5], matrix_[5][6]);
	printf("line7: %d %d %d %d %d %d %d\r\n", matrix_[6][0], matrix_[6][1], matrix_[6][2], matrix_[6][3], matrix_[6][4], matrix_[6][5], matrix_[6][6]);
	//#endif
};

unsigned char *SLMTemplate::GetMatrixPtr(void)
{
	return &matrix_[0][0];
}

SLMParents::SLMParents(int num)
{
	//GenParents(num);
	pParentNew_ = 0;
};

SLMParents::~SLMParents()
{
	for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
		SLMTemplate* pTemplate = *it;
		delete pTemplate;
	}
};

bool SLMParents::IsTemplatesFull(void)
{
	if (SLMTemplates_.size() >= NUM_PARENTS)
		return true;
	else
		return false;
}
	
void SLMParents::GenerateNewParent(void)
{
	pParentNew_ = new SLMTemplate();
	pParentNew_->GenRandom();
}

void SLMParents::PrintTemplates(void)
{
	int count = 1;
	for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
		SLMTemplate* pTemplate = *it;
#ifdef DEBUG_
		printf("%02d.", count++);
#endif
		pTemplate->Print();
	}
}

SLMTemplate *SLMParents::GenerateOffspring(int NumBinding)
{
	int number1, number2;
	SLMTemplate *pTemplate1, *pTemplate2;
	GetRandomTemplateIdx(number1, number2);
		
	//cout << endl;
	pTemplate1 = SLMTemplates_[number1];
	//cout << "Template1: " << endl;;
	//pTemplate1->Print();
	//cout << endl;

	pTemplate2 = SLMTemplates_[number2];

	//cout << "Template2: " << endl;;
	//pTemplate2->Print();
	//cout << endl << endl;

	BinaryTemplate1_.GenBinary();

	BinaryTemplate2_.GenBinaryInverse(BinaryTemplate1_);

	//cout << "GenBinary1: " << endl;;
	//BinaryTemplate1_.Print();
	//cout << endl;
	//cout << "GenBinary2: " << endl;;
	//BinaryTemplate2_.Print();
	//cout << endl;

	pTemplate1->MultiplyCell(BinaryTemplate1_, Parent1_);
	pTemplate2->MultiplyCell(BinaryTemplate2_, Parent2_);
		
	pParentNew_ = new SLMTemplate();
	Parent1_.AddCell(Parent2_, *pParentNew_);
		
	//cout << "Offspring parentNew: " << endl;;
	//pParentNew_->Print();
	//cout << endl;
		

	//Parent1_.RandomMutation(); This must be an error 
	pParentNew_->RandomMutation();
	//cout << "Offspring mutation: " << endl;;
	//pParentNew_->Print();
	//cout << "test" << endl;;
	return pParentNew_;
}

void SLMParents::CompareCostAndInsertTemplate(double cost)
{
	bool found = false;
	// Compare cost with all SLMTemplates
	// If cost better that smallest in poll
	// then insert ParentNew_ in SLMTemplates and
	// delete template with lowest cost
	pParentNew_->SetCost(cost);
	if (SLMTemplates_.size() == 0)
		SLMTemplates_.push_back(pParentNew_);
	else {
		for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
			SLMTemplate* pTemplate = *it;
			if (cost > pTemplate->GetCost()) {
				SLMTemplates_.insert(it, pParentNew_);
				found = true;
				break;
			}
		}
		if (found) {
			printf("New template inserted cost %.0f\r\n", cost);
			DeleteLastTemplate();
		} else {
			//printf("Template cost too low %.0f\r\n", cost);
			delete pParentNew_;
			pParentNew_ = 0;
		}
	}

}

void SLMParents::GenParents(int num)
{
	for (int i = 0; i < num; i++) {
		SLMTemplates_.push_back(new SLMTemplate());
	}
};

unsigned char* SLMParents::GetNewParentMatrixPtr(void)
{
	if (pParentNew_)
		return pParentNew_->GetMatrixPtr();
	else
		return 0;
};

void SLMParents::DeleteLastTemplate(void)
{
	//printf("%d\r\n", SLMTemplates_.size());
	if (SLMTemplates_.size() > NUM_PARENTS) {
		vector<SLMTemplate*>::iterator it = SLMTemplates_.end()-1;
		SLMTemplate* pParentLast = *it;
		//printf("Deleting template\r\n");
		//pParentLast->Print();
		SLMTemplates_.erase(it);
		delete pParentLast;
	}
}

void SLMParents::GetRandomTemplateIdx(int &number1, int &number2)
{
	number1 = rand() % NUM_PARENTS; //NUM_PARENTS;
	number2 = rand() % NUM_PARENTS; //NUM_PARENTS;
}
