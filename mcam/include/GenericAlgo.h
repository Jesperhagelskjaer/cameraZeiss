#pragma once

#include <windows.h>
#include "SLMParents.h"
#include "TemplateImages.h"

#define NUM_PARENTS 30 // Number of parents

class GenericAlgo {
public:
	
	GenericAlgo() 
	{
		pSLMParents_ = new SLMParents(NUM_PARENTS);
		pTemplateImg_ = new TemplateImages();
	};

	void StartSLM()
	{
		pSLMParents_->PrintTemplates();
	};

	void ComputeIntencity(unsigned short *pImage, RECT rec)
	{
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
	};

	~GenericAlgo()
	{
		delete pSLMParents_;
		delete pTemplateImg_;
	};

private:
	SLMParents *pSLMParents_;
	TemplateImages *pTemplateImg_;
};
