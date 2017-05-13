#include <math.h>
#include <Application.hpp>
#include <QImagereader.h>
#include "MCamRemote.hpp"
#include "BmpUtil.h"
#include "GenericAlgo.h"

void *remote_proc_main(void *parm);
static MCamRemote* mcamRemotePtr;

void *remote_proc_main(void *parm)
{
	return mcamRemotePtr->remoteProcMain(parm);
}

MCamRemote::MCamRemote(Application *applicationPtr)
{
	this->applicationPtr = applicationPtr;
	mcamRemotePtr = this;
	pGenericAlgo = new GenericAlgo();

	// Remote single shot handling
	threadStarted = false;
	stopProcessing = false;
	strcpy(fileName, START_FILE);

	connect(this, SIGNAL(doSingleImage()), this->applicationPtr, SLOT(doSingleShot()));
}

MCamRemote::~MCamRemote()
{
	delete pGenericAlgo;
}

int MCamRemote::startRemoteThread()
{
	MCAM_LOG_INIT("MCamRemote::startRemoteThread")
	
	if (threadStarted == false)
	{
		stopProcessing = false;

		MCAM_LOGF_STATUS("Starting remote thread");
		sem_init(&psem, 0, 0);

		if (pthread_create(&remoteTid, NULL, &remote_proc_main, NULL) != 0)
			MCAM_LOGF_ERROR("error creating image_proc_main thread");

	}
	return 0;
}

int MCamRemote::stopRemoteThread()
{
	MCAM_LOG_INIT("MCamRemote::stopRemoteThread")
	int done = 0;
	int i;
	int ret = 0;
	stopProcessing = true;
	sem_post(&psem);
	for (i = 0; i < 100; i++) {
		if (!threadStarted) {
			done = 1;
			break;
		}
		MCamUtil::sleep(50);
	}
	if (done)
		ret = 0;
	else {
		ret = 1;
		MCAM_LOGF_ERROR("stopRemoteThread failed!");
	}

	return ret;
}

int MCamRemote::waitForStart(void)
{
	char systemCmd[50];
	int result = 0;
	hStartFile = fopen(fileName, "r"); // Check for file exist
	while (hStartFile == NULL) {
		MCamUtil::sleep(50);
		hStartFile = fopen(fileName, "r"); // Check for file exist
	}
	fclose(hStartFile);
	sprintf(systemCmd, "del %s", fileName);
	system(systemCmd);
	return result;
}

// incoming requests from "MCamRemote::imagCallback"
void *MCamRemote::remoteProcMain(void *parm)
{
	MCAM_LOG_INIT("MCamRemote::remoteProcMain")

	threadStarted = true;
	MCAM_LOGF_STATUS("Generic Algo thread started");

	// Wait for file to start processing
	while (!stopProcessing) {

		// Wait for start.txt file to start processing
		waitForStart();
		printf("Generic Algo StartSLM\r\n");

		int maxLoops = pGenericAlgo->GetNumIterations();
		for (int loop = 0; loop < maxLoops && !stopProcessing; loop++) {

			timeMeas.setStartTime();
			pGenericAlgo->StartSLM();
			timeMeas.printDuration("Generic and SLM");

			timeMeas.setStartTime();
			//createTestImage();
			doSingleImage();
			
			sem_wait(&psem);

			//MCamUtil::sleep(10);
			printf("%d\r", loop+1);
		}

		// Create stop.txt file to indicate completed
		createStopFile();
	}
	
	threadStarted = false;
	MCAM_LOGF_STATUS("Stopped remote thread");

	return NULL;
}


void MCamRemote::createStopFile(void)
{
	FILE *hStopFile;
	hStopFile = fopen(STOP_FILE, "w"); // Create stop file
	fclose(hStopFile);
}

// ski.bmp
#define TEST_FILE   "ski.bmp"
#define IMG_WIDTH   2988
#define IMG_HEIGHT  5312
#define BYTES_PIXEL 2

void MCamRemote::createTestImage(void)
{
	ROI imgROI;
	unsigned short* pixel = NULL;
	int imageSize = IMG_WIDTH*IMG_HEIGHT*3 + sizeof(IMAGE_HEADER)*2;
	unsigned short* imageData = (unsigned short*)malloc(imageSize);

	if (imageData == 0) {
		printf("Error allocating memory to image\r\n");
		return;
	}
	IMAGE_HEADER* header = (IMAGE_HEADER*)imageData;

	header->headerSize = sizeof(IMAGE_HEADER);
	header->bitsPerPixel = MCAM_BPP_COLOR;
	header->binX = BYTES_PIXEL;
	header->binY = BYTES_PIXEL;
	pixel = (unsigned short*)imageData + header->headerSize / 2;

	LoadBmpAsGray(TEST_FILE, &imgROI, (byte *)pixel);
	printf("Image loaded %s\r\n", TEST_FILE);
	header->roiWidth = imgROI.width*BYTES_PIXEL;
	header->roiHeight = imgROI.height*BYTES_PIXEL;
	printf("Width %d, Height %d, Header %d\r\n", header->roiWidth/BYTES_PIXEL, header->roiHeight/BYTES_PIXEL, header->headerSize);

	saveImage(imageData, true);
}

void MCamRemote::saveImage(unsigned short *imageData, bool test)
{
	RECT rec = applicationPtr->getCurrentFrameSize();
	
	// KBE For test only
#if 0
	rec.left = 0;
	rec.top = 0;
	rec.right = COLS-1;
	rec.bottom = ROWS-1;
#endif

	timeMeas.printDuration("Do Single Image");

	timeMeas.setStartTime();
	pGenericAlgo->ComputeIntencity(imageData, rec);
	timeMeas.printDuration("Compute Intencity");

	//printf("Generic iter completed\r\n");
	sem_post(&psem);

/*
	ROI imgROI;
	IMAGE_HEADER* header = (IMAGE_HEADER*)imageData;

	bool isColorImage = 0;
	// painting raw data to image
	unsigned short* pixel = NULL;

	imgROI.width = header->roiWidth / header->binX;
	imgROI.height = header->roiHeight / header->binY;

	isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;

	// painting raw camera data to image
	pixel = (unsigned short*)imageData + header->headerSize / 2;

	printf("Saving image file\r\n");

	if (test)
		DumpBmpAsGray(IMAGE_FILE, (byte *)pixel, imgROI);
	else
	    DumpBmpShortAsGray(IMAGE_FILE, pixel, imgROI);

	createStopFile();
*/

}
