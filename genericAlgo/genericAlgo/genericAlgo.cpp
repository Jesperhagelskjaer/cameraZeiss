// genericAlgo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "mcam_zei_ex.h"
#include "GenericAlgo.h"
#include "time.h"

SLMParents slmParents(NUM_PARENTS);


#define CAM_WIDTH 1936
#define CAM_HEIGHT 1460

typedef struct _CAM_IMAGE{
	IMAGE_HEADER header;
	unsigned short image[CAM_WIDTH * CAM_HEIGHT];
} CAM_IMAGE;

CAM_IMAGE camImage;

int main()
{
	srand(time(NULL));
	SLMTemplate *pParentNew;
	GenericAlgo *pGenericAlgo = new GenericAlgo();
	RECT rect;

	memset(&camImage, 0, sizeof(CAM_IMAGE));
	camImage.header.headerSize = sizeof(IMAGE_HEADER);
	camImage.header.binX = 2;
	camImage.header.binY = 2;
	camImage.header.roiWidth = 2 * CAM_WIDTH;
	camImage.header.roiHeight = 2 * CAM_HEIGHT;
	camImage.header.bitsPerPixel = MCAM_BPP_MONO;
	rect.left = 0;
	rect.right = COLS;
	rect.top = 0;
	rect.bottom = ROWS;
	pGenericAlgo->ComputeIntencity((unsigned short *)&camImage, rect);

	//slmParents.PrintTemplates();
	pParentNew = slmParents.GenerateOffspring(1);
	//pParentNew->Print();
    return 0;
}

