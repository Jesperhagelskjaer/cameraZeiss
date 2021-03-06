#include <math.h>
#include <Application.hpp>
#include <QImagereader.h>
#include "defs.h"
#include "MCamRemote.hpp"
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
	pGenericAlgo = 0;

	// Remote single shot handling
	threadStarted = false;
	stopProcessing = false;
	strcpy(fileName, START_FILE);

	// Default rectangle inside image
	recAlgo_.left = 245;
	recAlgo_.top = 245;
	recAlgo_.right = 255;
	recAlgo_.bottom = 255;

	imgROI_.width = 0;
	imgROI_.height = 0;

	// User default parameters
	numBindings_ = NUM_BINDINGS;
	numParents_ = NUM_PARENTS;
	numIterations_ = GEN_ITERATIONS;
	delayMS_ = DELAY_MS;
	pauseMS_ = PAUSE_MS;
	laserIntensity_ = LASER_INTENSITY;
	numBetweenSave_ = NUM_BETWEEN_SAVE_IMG;
	numEndIterations_ = NUM_END_ITERATIONS;

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

		MCAM_LOGF_STATUS("Starting remote thread and open laser port");

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

		pGenericAlgo = new GenericAlgo(numParents_, numBindings_, numIterations_);

		if (pGenericAlgo == 0) {
			printf("FAILED TO ALLOCATE GENERIC ALGORITHM\r\n");
			stopProcessing = true;
			break;
		}

		pGenericAlgo->OpenLaserPort(LASER_PORT, laserIntensity_);

		printf("Generic Algo StartSLM: %d, %d, %d, %d, %.2f, %d, %d\r\n", 
			    numIterations_, numParents_, numBindings_, numBetweenSave_, 
			    laserIntensity_, delayMS_, pauseMS_);
		iteration_ = 0;

		int numLoopsTrain = pGenericAlgo->GetNumIterations();
		int maxLoops = numLoopsTrain + numEndIterations_;
		for (int loop = 0; loop < maxLoops && !stopProcessing; loop++) {

			//timeMeas.setStartTime();
			//pGenericAlgo->TurnLaserOn(); // KBE???
			trainMode_ = (loop < numLoopsTrain);
			pGenericAlgo->StartSLM(trainMode_); // Training generic algorihm, use parent with max cost after training
			pGenericAlgo->TurnLaserOn();
			if (delayMS_ > 0)
				MCamUtil::sleep(delayMS_);

			//timeMeas.printDuration("Generic and SLM");

			//timeMeas.setStartTime();
#ifdef TEST_WITHOUT_CAMERA_
			createTestImage();
#else
			doSingleImage();
#endif
			
			sem_wait(&psem);

			printf("%d\r\n", loop+1);
			iteration_++;

			if (pauseMS_ > 0)
				MCamUtil::sleep(pauseMS_);
		}

		// Create stop.txt file to indicate completed
		saveDataFile(maxLoops); // Save data results file
		createStopFile();
		delete pGenericAlgo;
		pGenericAlgo = 0;
	}

	threadStarted = false;
	MCAM_LOGF_STATUS("Stopped remote thread");

	return NULL;
}

void MCamRemote::createStartFile(void)
{
	FILE *hStartFile;
	hStartFile = fopen(START_FILE, "w"); // Create start file
	fclose(hStartFile);
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
	printf("Width %d, Height %d, Header %d\r\n", header->roiWidth/BYTES_PIXEL, 
												 header->roiHeight/BYTES_PIXEL, header->headerSize);

	saveImage(imageData, true);
	free(imageData);
}

void MCamRemote::setRecAlgo(RECT rec)
{
	recAlgo_.left = rec.left;
	recAlgo_.top = rec.top;
	recAlgo_.right = rec.right;
	recAlgo_.bottom = rec.bottom;
}

void MCamRemote::saveDataFile(int maxLoops)
{
	FILE *hDataFile;
	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);

	hDataFile = fopen(DATA_FILE, "w"); // Create data file

	if (hDataFile != 0) {
		fprintf(hDataFile, "%d,%02d,%02d,%02d,%02d,%02d,",
							now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
							now->tm_hour, now->tm_min, now->tm_sec);
		fprintf(hDataFile, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%0.2f\r\n", 
			                imgROI_.width, imgROI_.height, 
			                recAlgo_.left, recAlgo_.top, recAlgo_.right, recAlgo_.bottom,
							numParents_, numBindings_, maxLoops, numBetweenSave_, 
							pGenericAlgo->GetLaserIntensity());
		fclose(hDataFile);
	}
}

void MCamRemote::saveImageData(unsigned short *imageData, long cost)
{
	char fileName[50];
	IMAGE_HEADER* header = (IMAGE_HEADER*)imageData;
	bool isColorImage = 0;
	// painting raw data to image
	unsigned short* pixel = NULL;

	imgROI_.width = header->roiWidth / header->binX;
	imgROI_.height = header->roiHeight / header->binY;

	isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;

	// painting raw camera data to image
	pixel = (unsigned short*)imageData + header->headerSize / 2;

	sprintf(fileName, IMG_FILES, iteration_, cost);
	printf("Saving image file %s\r\n", fileName);

	//DumpBmpAsGray(fileName, (byte *)pixel, imgROI_);
	DumpImgShortAsBinary(fileName, pixel, imgROI_);
	//DumpBmpShortAsGray(fileName, pixel, imgROI_);
}

long MCamRemote::saveImage(unsigned short *imageData, bool test)
{
	long newCost = 0;
	//RECT rec = applicationPtr->getCurrentFrameSize();

	//timeMeas.printDuration("Do Single Image");

	//timeMeas.setStartTime();
	pGenericAlgo->TurnLaserOff();

	newCost = (long)pGenericAlgo->ComputeIntencity(imageData, recAlgo_, trainMode_);

	if (trainMode_) {
		//timeMeas.printDuration("Compute Intencity");
		if (NUM_RAND_ITERATIONS > 0 && (iteration_%NUM_RAND_ITERATIONS == 0)) {
			// Delete num parents each NUM_RAND_ITERATIONS
			pGenericAlgo->DeleteTemplates(NUM_RAND_TEMPLATES);
		}
	}

	if (iteration_ % numBetweenSave_ == 0) {
		// Save image and data to files
		saveImageData(imageData, newCost);
	}

	//printf("Generic iter completed\r\n");
	sem_post(&psem);

	return newCost;
}

