#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

//#define M 6  // Matrix size of SLM timeplate
#define M 512  // Matrix size of SLM timeplate
class SLMTemplate
{
public:
	SLMTemplate()
	{
		cost_ = 0;
		binding_ = 3;
	};

	void SetCost(double cost)
	{
		cost_ = cost;
	};

	double GetCost(void)
	{
		return cost_;
	};

	void SetBinding(unsigned int binding)
	{
		binding_ = binding;
	};

	/* Problem KBE!!!
	void GenRandom(void)
	{
		unsigned char test;
		unsigned int bindingtest_ = 3;

		for (int i = 0; i < M; i++) {
				while (i % bindingtest_ != 0){
					for (int k = 0; k < M; k++){
						matrix_[i][k] = matrix_[i-1][k];
					}
			++i;} //KBE????
			
			for (int j = 0; j < M; j++) {
				if (j % bindingtest_ == 0) {
					test = rand() % 255;
				}
				matrix_[i][j] = test;
			}				
		}
		
		
	};
	*/

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

			cout << "random mutation" << endl;
			float propabililty = (float)exp(-0.52); //exp(-0.52);
			//out << "probability:" << propabililty << endl;
			//cout << "signel rand() call: " << (float)rand() / (float)RAND_MAX << endl;
			int bindingtest_ = 3;
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

	void Print(void)
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

	unsigned char *GetMatrixPtr(void) 
	{
		return &matrix_[0][0];
	}

private:
	unsigned char matrix_[M][M];
	double cost_;
	unsigned int binding_;
};


#define NUM_PARENTS 2//30 // Number of parents
class SLMParents
{
public:
	SLMParents(int num)
	{
		//GenParents(num);
		pParentNew_ = 0;
	};

	~SLMParents()
	{
		for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
			SLMTemplate* pTemplate = *it;
			delete pTemplate;
		}
	};

	bool IsTemplatesFull(void)
	{
		if (SLMTemplates_.size() >= NUM_PARENTS)
			return true;
		else
			return false;
	}
	
	void GenerateNewParent(void)
	{
		pParentNew_ = new SLMTemplate();
		pParentNew_->GenRandom();
	}

	void PrintTemplates(void) 
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

	SLMTemplate *GenerateOffspring(int NumBinding)
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
		
		cout << "Offspring parentNew: " << endl;;
		pParentNew_->Print();
		cout << endl;
		

		//Parent1_.RandomMutation(); This must be an error 
		pParentNew_->RandomMutation();
		cout << "Offspring mutation: " << endl;;
		pParentNew_->Print();
		cout << "test" << endl;;
		return pParentNew_;
	}

	void CompareCostAndInsertTemplate(double cost)
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
			int currentPos = 0;
			for (vector<SLMTemplate*>::iterator it = SLMTemplates_.begin(); it != SLMTemplates_.end(); ++it) {
				SLMTemplate* pTemplate = *it;
				if (cost >= pTemplate->GetCost()) {
					SLMTemplates_.insert(it, pParentNew_);
					found = true;
					break;
				}
				currentPos++;
			}
			if (found)
				DeleteLastTemplate();
			else {
				delete pParentNew_;
				pParentNew_ = 0;
			}
		}

	}

	void GenParents(int num)
	{
		for (int i = 0; i < num; i++) {
			SLMTemplates_.push_back(new SLMTemplate());
		}
	};

	unsigned char* GetNewParentMatrixPtr(void)
	{
		if (pParentNew_)
			return pParentNew_->GetMatrixPtr();
		else
			return 0;
	};

private:

	void DeleteLastTemplate(void)
	{
		//printf("%d\n\r", SLMTemplates_.size());
		if (SLMTemplates_.size() > NUM_PARENTS) {
			vector<SLMTemplate*>::iterator it = SLMTemplates_.end()-1;
			SLMTemplate* pParentLast = *it;
			//pParentLast->Print();
			SLMTemplates_.erase(it);
			delete pParentLast;
		}
	}

	void GetRandomTemplateIdx(int &number1, int &number2)
	{
		number1 = rand() % NUM_PARENTS; //NUM_PARENTS;
		number2 = rand() % NUM_PARENTS; //NUM_PARENTS;
	}

	SLMTemplate BinaryTemplate1_, BinaryTemplate2_;
	SLMTemplate Parent1_, Parent2_;
	SLMTemplate *pParentNew_;
	std::vector<SLMTemplate*> SLMTemplates_;
};
