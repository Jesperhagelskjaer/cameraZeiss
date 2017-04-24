#include <math.h>
#include <Application.hpp>
#include <QImagereader.h>
#include "MCamRemote.hpp"
#include "BmpUtil.h"

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

	// Remote single shot handling
	threadStarted = false;
	stopProcessing = false;
	strcpy(fileName, START_FILE);

	connect(this, SIGNAL(doSingleImage()), this->applicationPtr, SLOT(doSingleShot()));
}

MCamRemote::~MCamRemote()
{
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
	MCAM_LOGF_STATUS("Remote communication thread started");

	while (!stopProcessing) {
		waitForStart();
		printf("Remote single shot\r\n");
		//createTestImage(); // FOR TESTING ONLY KBE???
		doSingleImage();
		//sem_wait(&psem);
		//MCamUtil::sleep(10);
		MCAM_LOGF_INFO("Remote trigger");
	}
	
	threadStarted = false;
	MCAM_LOGF_STATUS("Stopped remote thread");

	return NULL;
}

/* Experiments
void MCamRemote::createTestImage(void)
{
	QString imageTestFile("ski2.jpg");
	QImage image;

	if (image.load(imageTestFile, "jpg"))
		saveImage(&image);
	else
		printf("Could not load test image\r\n");
}

void MCamRemote::createTestImage(void)
{
	QImageReader reader("C:\\AxiocamSDK\\mcam\\ski2.jpg");
	reader.setAutoDetectImageFormat(true);
	QImage image = reader.read();
	if (image.isNull()) 
	{
		printf("C:\\AxiocamSDK\\mcam\\ski2.jpg ");
		printf("Could not load test image\r\n");
	}
	else
		saveImage(&image);

}

void MCamRemote::createTestImage(void)
{
	QImage image(3, 3, QImage::Format_RGB32);
	QRgb value;

	value = qRgb(189, 149, 39); // 0xffbd9527
	image.setPixel(1, 1, value);

	value = qRgb(122, 163, 39); // 0xff7aa327
	image.setPixel(0, 1, value);
	image.setPixel(1, 0, value);

	value = qRgb(237, 187, 51); // 0xffedba31
	image.setPixel(2, 1, value);

	saveImage(&image);
}

void MCamRemote::createTestImage(void)
{
QImage image("ski2", "jpg");
if (image.isNull())
printf("Could not load test image\r\n");
else
saveImage(&image);
}

void MCamRemote::createTestImage(void)
{
	QString imageTestFile("ski2.jpg");
	QImage image;
	FILE *hFile;

	hFile = fopen("ski2.jpg", "r"); // Check for file exist
	if (hFile == NULL) {
		printf("Could not read test image\r\n");
	} else
		fclose(hFile);

	if (image.load(imageTestFile, "jpg"))
		saveImage(&image);
	else
		printf("Could not load test image\r\n");
}
*/

// NOT WORKING!!!
void MCamRemote::saveImage(QImage *pImage)
{
	QString imageFile(IMAGE_FILE);
	printf("Saving image file\r\n");
	if (!pImage->save(imageFile, "jpg"))
		printf("Could not save image\r\n");
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
	unsigned short* imageData = new unsigned short[imageSize];

	IMAGE_HEADER* header = (IMAGE_HEADER*)imageData;

	header->headerSize = sizeof(IMAGE_HEADER);
	header->bitsPerPixel = MCAM_BPP_COLOR;
	header->binX = BYTES_PIXEL;
	header->binY = BYTES_PIXEL;
	pixel = (unsigned short*)imageData + header->headerSize / 2;

	LoadBmpAsGray(TEST_FILE, &imgROI, (byte *)pixel);
	header->roiWidth = imgROI.width*BYTES_PIXEL;
	header->roiHeight = imgROI.height*BYTES_PIXEL;
	printf("Width %d, Height %d, Header %d\r\n", header->roiWidth/BYTES_PIXEL, header->roiHeight/BYTES_PIXEL, header->headerSize);

	saveImage(imageData, true);
}

void MCamRemote::saveImage(unsigned short *imageData, bool test)
{
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
}
