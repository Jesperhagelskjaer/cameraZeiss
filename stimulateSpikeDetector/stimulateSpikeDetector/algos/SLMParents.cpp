///////////////////////////////////////////////////////////
//  SLMParents.cpp
//  Implementation of the Class SLMTemplate, SLMParents
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge + Jesper Hagelskjær
///////////////////////////////////////////////////////////

#include "SLMParents.h"

SLMTemplate::SLMTemplate(int binding)
{
	cost_ = 0;
	probabililty_ = 0;
	binding_ = binding;
}

SLMTemplate::~SLMTemplate()
{
	//std::cout << "Destroy template" << std::endl;
}

void SLMTemplate::SetCost(double cost)
{
	cost_ = cost;
}

double SLMTemplate::GetCost(void)
{
	return cost_;
}

void SLMTemplate::GenRandom(void)
{
	unsigned char test;

	for (int i = 0; i < M; i++) {
		while (i % binding_ != 0 && i < M) {
			for (int k = 0; k < M; k++) {
				matrix_[i][k] = matrix_[i-1][k];
			}
			++i;
		} 
		if (i >= M) break;
		for (int j = 0; j < M; j++) {
			if (j % binding_ == 0) {
				test = rand() % 255;
			}
			matrix_[i][j] = test;
		}				
	}	
}

/*
void SLMTemplate::GenRandom(void)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = rand() % 255;
}
*/

void SLMTemplate::GenBinary(void)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = rand() % 2;
}

void SLMTemplate::GenBinaryInverse(SLMTemplate &templateIn)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			matrix_[i][j] = 1 - templateIn.matrix_[i][j];
}

void SLMTemplate::MultiplyCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			templateOut.matrix_[i][j] = matrix_[i][j] * templateIn.matrix_[i][j];
}

void SLMTemplate::AddCell(SLMTemplate &templateIn, SLMTemplate &templateOut)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			templateOut.matrix_[i][j] = matrix_[i][j] + templateIn.matrix_[i][j];
}

void SLMTemplate::RandomMutation(int n, int type)
{
	if (type == 2)
		RandomMutation2(n); // Random mutation according to paper
	else
		RandomMutation1(); // First version of random mutation
}

void SLMTemplate::RandomMutation1(void)
{
		//cout << "random mutation" << endl;
		probabililty_ = MUT_PROBABILITY_TYPE1; //exp(-0.52);
		//out << "probability:" << propabililty << endl;
		//cout << "signel rand() call: " << (float)rand() / (float)RAND_MAX << endl;
		int j;
		int i;
		float threshold;

		for (i = 0; i < M; i += binding_) {
			//cout << "i " << i << endl;
			for (j = 0; j < M; j += binding_) {
				//cout << (int)matrix_[i][j] << " " ;
				threshold = (float)rand() / (float)RAND_MAX;
				if (probabililty_ > threshold) {
					threshold = (float)1.1;
					int test = rand() % 255;
					for (int it2 = 0; it2 < binding_; ++it2){
							 
						for (int jt2 = 0; jt2 < binding_; ++jt2) {
							//cout << it2 << " " << jt2 << " " << i + it2 << " " << j + jt2  << " " << test << endl;
							matrix_[i + it2][j + jt2] = test;
							//cout << (int)matrix_[i + it2][j + jt2] << " ";
						}
					}				
				}
			}
		}
}

void SLMTemplate::RandomMutation2(int n)
{
	// Creates and clear matrix to hold modes modified
	static unsigned char modifiedMatrix[M][M];
	memset(&modifiedMatrix[0][0], 0, sizeof(modifiedMatrix));

	// Computes the number modes to modify
	probabililty_ = MUT_PROBABILITY_TYPE2(n);
	int numRandModes = (int)round(probabililty_);
	//printf("RandomMutation2 enter %d\r\n", numRandModes);

	// Modifies random position in matrix_
	while (numRandModes > 0) {
		int randIdx = rand() % (M*M); // Random index to matrix
		//printf("%d\r\n", randIdx);
		int i = randIdx / M; // Computes matrix row position
		int j = randIdx % M; // Computes matrix coloum position
		//printf("%d,%d\r\n", i, j);
		int oi = i % binding_; // Computes binding row offset
		int oj = j % binding_; // Computes binding coloum offset
		//printf("%d,%d\r\n", oi, oj);
		i -= oi; // Binding row position
		j -= oj; // Binding coloum position
		//printf("%d,%d\r\n", i, j);

		// Check if random binding position already modified
		if (modifiedMatrix[i][j] == 0) {

			// Mark matrix binding position changed
			modifiedMatrix[i][j] = 1;
			numRandModes--;
			//printf("Modes left %d\r\n", numRandModes);

			// Generate new random mode
			unsigned char mode = rand() % 255;
			
			// Change mode in matrix binding position
			for (int ib = i; ib < i+binding_; ++ib) {
				for (int jb = j; jb < j+binding_; ++jb) {
					matrix_[ib][jb] = mode;
					//cout << (int)matrix_[ib][jb] << " ";
				}
			}
		}
	}

	//printf("RandomMutation2 exit %d\r\n", numRandModes);

}

void SLMTemplate::Print(void)
{
	//#ifdef DEBUG_
	printf("cost: %f\r\n", cost_);

	for (int j = 0; j < 10; j++) {
		printf("line%02d:", j + 1);
		for (int i = 0; i < 16; i++)
			printf(" %3d", matrix_[j][i]);
		printf("\r\n");
	}
	//#endif
}

unsigned char *SLMTemplate::GetMatrixPtr(void)
{
	return &matrix_[0][0];
}

SLMParents::SLMParents(int numParents, int numBindings) :
	BinaryTemplate1_(numBindings),
	BinaryTemplate2_(numBindings),
	Parent1_(numBindings),
	Parent2_(numBindings)
{
	pParentNew_ = 0;
	numOffsprings_ = 0;
	numParents_ = numParents;
	numBindings_ = numBindings;
	//GenParents();
	// Seed random number generator
	srand((unsigned int)time(NULL));
}

SLMParents::~SLMParents()
{
	for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
		SLMTemplate* pTemplate = *it;
		delete pTemplate;
	}
}

bool SLMParents::IsTemplatesFull(void)
{
	if (SLMTemplates_.size() >= numParents_)
		return true;
	else
		return false;
}
	
void SLMParents::GenerateNewParent(void)
{
	pParentNew_ = new SLMTemplate(numBindings_);
	pParentNew_->GenRandom();
}

void SLMParents::PrintTemplatesCost(void) 
{
	int count = 1;
	printf("Template costs: ");
	for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
		SLMTemplate* pTemplate = *it;
		printf("%.2f, ", pTemplate->GetCost());
	}
	printf("\r\n");
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

SLMTemplate *SLMParents::GenerateOffspring(void)
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
		
	pParentNew_ = new SLMTemplate(numBindings_);
	Parent1_.AddCell(Parent2_, *pParentNew_);
		
	//cout << "Offspring parentNew: " << endl;;
	//pParentNew_->Print();
	//cout << endl;
		

	//Parent1_.RandomMutation(); This must be an error 
	//pParentNew_->RandomMutation1();
	pParentNew_->RandomMutation(++numOffsprings_, MUT_TYPE);
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
	if (SLMTemplates_.size() == 0) {
		SLMTemplates_.push_back(pParentNew_); // Empty list
		printf("New first cost %.0f, %.1f  \r\n", cost, pParentNew_->GetProbability());
	} else {

		for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
			SLMTemplate* pTemplate = *it;
			if (cost > pTemplate->GetCost()) {
				SLMTemplates_.insert(it, pParentNew_); // Template higher cost insert
				printf("New cost %.0f, %.1f  \r\n", cost, pParentNew_->GetProbability());
				found = true;
				break;
			}
		}

		if (SLMTemplates_.size() < numParents_) {
			// First number of random templates
			if (!found) {
				printf("New back cost %.0f, %.1f  \r\n", cost, pParentNew_->GetProbability());
				SLMTemplates_.push_back(pParentNew_); // Template lower cost and list not full
			}
		} else if (found) {
			// Offspring templates
			DeleteLastTemplate();
		} else {
			//printf("Template cost too low %.0f\r\n", cost);
			delete pParentNew_;
			pParentNew_ = 0;
		}
	}

}

/*
void SLMParents::GenParents(void)
{
	for (int i = 0; i < numParents_; i++) {
		SLMTemplates_.push_back(new SLMTemplate(numBindings_));
	}
}
*/

unsigned char* SLMParents::GetNewParentMatrixPtr(void)
{
	if (pParentNew_)
		return pParentNew_->GetMatrixPtr();
	else
		return 0;
};

unsigned char* SLMParents::GetMaxCostParentMatrixPtr(void)
{
	SLMTemplate *pTemplateMaxCost;
	if (SLMTemplates_.size() > 0) {
		vector<SLMTemplate*>::iterator it = SLMTemplates_.begin();
		pTemplateMaxCost = *it;
		//printf("Using template with max cost %.0f\r\n", pTemplateMaxCost->GetCost());
		return pTemplateMaxCost->GetMatrixPtr();
	}
	else
		return 0;
};

void SLMParents::DeleteLastTemplate(void)
{
	//printf("%d\r\n", SLMTemplates_.size());
	if (SLMTemplates_.size() > numParents_) {
		vector<SLMTemplate*>::iterator it = SLMTemplates_.end()-1;
		SLMTemplate* pParentLast = *it;
		//printf("Deleting template\r\n");
		//pParentLast->Print();
		SLMTemplates_.erase(it);
		delete pParentLast;
	}
}

void SLMParents::DeleteTemplates(int num)
{
	if (SLMTemplates_.size() > num) {
		printf("Deleted templates with cost: ");
		for (int n = 0; n < num; n++) {
			vector<SLMTemplate*>::iterator it = SLMTemplates_.end() - 1;
			SLMTemplate* pParentLast = *it;
			printf("%.0f, ", pParentLast->GetCost());
			SLMTemplates_.erase(it);
			delete pParentLast;
		}
		printf("\r\n");
	}
}


/**
% Try MATLAB script below to see logistic probability distribution:
	MAX = 20;
	x = rand([1 1000])*MAX;
	hist(x)
	y = ones(size(x)). / (1 + exp(x)); % Sigmoid function, logistic sigmoid
	figure, plot(y, '.')
	y = floor(MAX * 2*y)
	figure, hist(y)
*/
int SLMParents::SigmoidRandomDistribution(void)
{
	double x = ((double)rand() / RAND_MAX) * numParents_;
	double y = 1 / (1 + exp(0.19*x)); // Sigmoid function
	return (int)floor(numParents_*2*y);
}

void SLMParents::GetRandomTemplateIdx(int &number1, int &number2)
{
	if (RAND_PROPABILITY == 0) {
		number1 = rand() % numParents_;
		number2 = rand() % numParents_;
		while (number1 == number2)
			number2 = rand() % numParents_;
	}
	else {
		// Probability higer picking af template with high cost (logistic probability distribution)
		number1 = SigmoidRandomDistribution();
		number2 = SigmoidRandomDistribution();
		while (number1 == number2)
			number2 = SigmoidRandomDistribution();
		//printf("Logistic Random Distribution %d, %d\r\n", number1, number2);
	}

}
