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

	void createStartFile(void); // Used to start generic algo from GUI
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

	// User parameters
	int numBindings_;
	int numParents_;
	int numIterations_;
	int numBetweenSave_;
	float laserIntensity_;
	int delayMS_;
	int pauseMS_;

public:
	void setNumBindings(int bindings) { numBindings_ = bindings; };
	void setNumParents(int parents) { numParents_ = parents; };
	void setNumIterations(int iterations) { numIterations_ = iterations; };
	void setNumBetweenSave(int betweenSave) { numBetweenSave_ = betweenSave; };
	void setLaserIntensity(float intensity) { laserIntensity_ = intensity; };
	void setDelayMS(int delay) { delayMS_ = delay; };
	void setPauseMS(int pause) { pauseMS_ = pause; };

	int getNumBindings(void) { return numBindings_; };
	int getNumParents(void) { return numParents_; };
	int getNumIterations(void) { return numIterations_; };
	int getNumBetweenSave(void) { return numBetweenSave_; };
	float getLaserIntensity(void) { return laserIntensity_; };
	int getDelayMS(void) { return delayMS_; };
	int getPauseMS(void) { return pauseMS_; };
};

#endif

