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
#include "TimeMeasure.h"
#include "GenericAlgo.h"

#define START_FILE "start.txt"
#define STOP_FILE  "stop.txt"
#define IMAGE_FILE "image.bmp"

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
	void saveImage(unsigned short *imageData, bool test = false);

signals:
	void doSingleImage(void);

private:
	void createTestImage(void);
	int waitForStart(void);
	void createStopFile(void);
	TimeMeasure timeMeas;
};

#endif

