#pragma once

#include <windows.h>
#include "SLMParents.h"
#include "TemplateImages.h"
#include "SLMInterface.h"

#define NUM_PARENTS 2 // Number of parents

class GenericAlgo {
public:
	
	GenericAlgo()
	{
		pSLMParents_ = new SLMParents(NUM_PARENTS);
		pImg_ = new CamImage();
		pSLMInterface_ = new SLMInterface();
	};

	GenericAlgo(Blink_SDK *pSLMsdk)
	{
		pSLMParents_ = new SLMParents(NUM_PARENTS);
		pImg_ = new CamImage();
		pSLMInterface_ = new SLMInterface(pSLMsdk);
	};

	void StartSLM()
	{
		if (pSLMParents_->IsTemplatesFull()) {
			//pSLMParents_->PrintTemplates();
			pSLMParents_->GenerateOffspring(1);
		} else {
			pSLMParents_->GenerateNewParent();
		}
		//pSLMInterface_->SendTestPhase(pSLMParents_->GetNewParentMatrixPtr(), M);
		pSLMInterface_->SendPhase(pSLMParents_->GetNewParentMatrixPtr());
	};

	void TestComputeIntencity(double cost)
	{
		pSLMParents_->CompareCostAndInsertTemplate(cost);
		pSLMParents_->PrintTemplates();
	}

	void ComputeIntencity(unsigned short *pImage, RECT rec)
	{
		double cost;
		int width, height;
		IMAGE_HEADER* header = (IMAGE_HEADER*)pImage;

		bool isColorImage = 0;
		// painting raw data to image
		unsigned short* pixel = NULL;

		width = header->roiWidth / header->binX;
		height = header->roiHeight / header->binY;

		isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;

		// painting raw camera data to image
		pixel = (unsigned short*)pImage + header->headerSize / 2;

		printf("Image taken L%d, R%d, T%d, B%d\r\n", rec.left, rec.right, rec.top, rec.bottom);
		pImg_->CopyImage(pixel, height, width, rec);
		cost = pImg_->ComputeIntencity();
		pSLMParents_->CompareCostAndInsertTemplate(cost);
	};

	~GenericAlgo()
	{
		delete pSLMParents_;
		delete pImg_;
		delete pSLMInterface_;
	};

private:
	SLMParents *pSLMParents_;
	CamImage *pImg_;
	SLMInterface *pSLMInterface_;
};
