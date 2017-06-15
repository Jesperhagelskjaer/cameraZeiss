/**
* @file MCamRemote.h
* @author kbe
* @date 20.04.2017
*
* @brief header file for mcam image handling
*
* Copyright Aarhus University 2017, All rights reserved.
*/
#ifndef MCAMREMOTE_H_
#define MCAMREMOTE_H_

#include "mcam.h"
#include <QObject>
#include "BmpUtil.h"
#include "TimeMeasure.h"
#include "GenericAlgo.h"

#define START_FILE "start.txt"
#define STOP_FILE  "stop.txt"
#define IMAGE_FILE "image.bmp"
#define DATA_FILE  "data\\data.txt"
#define IMG_FILES  "data\\IM%06d_%d.bin"

class Application;

class MCamRemote: QObject
{
	Q_OBJECT
	Application *applicationPtr;
    
	// Remote thread communication handling
	bool threadStarted;
	bool stopProcessing;
	pthread_t remoteTid;
	sem_t psem;

	// Start FILE
	char fileName[100];
	FILE* hStartFile;

	// Generic Algorithm
	GenericAlgo *pGenericAlgo;

public:
	MCamRemote(Application *applicationPtr);
	virtual ~MCamRemote();

	int startRemoteThread();
	int stopRemoteThread();
	void *remoteProcMain(void *parm);
	long saveImage(unsigned short *imageData, bool test = false);
	void setRecAlgo(RECT rec);
	void getRecAlgo(RECT *pRec) {
		*pRec = recAlgo_;
	}

signals:
	void doSingleImage(void);

private:
	void createTestImage(void);
	int waitForStart(void);
	void createStopFile(void);
	void saveImageData(unsigned short *imageData, long cost);
	void saveDataFile(int maxLoops);
	TimeMeasure timeMeas;
	RECT recAlgo_;
	ROI imgROI_;
	int iteration_;
	int numBetweenSave_;
};

#endif

