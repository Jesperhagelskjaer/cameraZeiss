// genericAlgo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "mcam_zei_ex.h"
#include "GenericAlgo.h"
#include "time.h"

//SLMParents slmParents(NUM_PARENTS);

#define CAM_WIDTH 1936
#define CAM_HEIGHT 1460

typedef struct _CAM_IMAGE{
	IMAGE_HEADER header;
	unsigned short image[CAM_WIDTH * CAM_HEIGHT];
	} CAM_IMAGE;

CAM_IMAGE camImage;

//#define TEST_SIZE 4
//double costTest[TEST_SIZE] = { 2, 4, 1, 3 };

#define TEST_SIZE 6
double costTest[TEST_SIZE] = { 0, 0, 2, 4, 1, 3 };

int main()
{
	srand(time(NULL));
	//camImage.image[0] = 100;
	//cout << camImage.image[1] << "test" << endl;
	//printf("Matrix: %d %d %d %d %d %d %d\r\n", camImage.image[0],camImage.image[1], camImage.image[2], camImage.image[3], camImage.image[4], camImage.image[5], camImage.image[6]);
	
	//SLMTemplate *pParentNew;
	GenericAlgo *pGenericAlgo = new GenericAlgo();

	for (int i = 0; i < TEST_SIZE; i++) {

		pGenericAlgo->StartSLM(); // Start og tag billede
		pGenericAlgo->TestComputeIntencity(costTest[i]);
	}

	// Billede er taget kaldes fra Qt
	memset(&camImage, 0, sizeof(CAM_IMAGE));

	camImage.image[1+ CAM_WIDTH] = 10000;
	camImage.image[1 + CAM_HEIGHT] = 1;
	camImage.header.headerSize = sizeof(IMAGE_HEADER);
	camImage.header.binX = 2;
	camImage.header.binY = 2;
	camImage.header.roiWidth = 2 * CAM_WIDTH;
	camImage.header.roiHeight = 2 * CAM_HEIGHT;
	camImage.header.bitsPerPixel = MCAM_BPP_MONO;
	//cout << "Image værdi = " << camImage.image[0] << endl;

	RECT rect;
	rect.left = 0;
	rect.right = COLS;
	rect.top = 0;
	rect.bottom = ROWS;
	pGenericAlgo->ComputeIntencity((unsigned short *)&camImage, rect);

	//slmParents.PrintTemplates();
	//pParentNew = slmParents.GenerateOffspring(1);
	//pParentNew->Print();
    return 0;
}

