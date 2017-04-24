/**
 * @file MCamTest.cpp
 * @author ggraf
 * @date 15.02.2015
 *
 * @brief Test modude of mcam
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#include "mcam_zei.h"
#include "mcam_zei_ex.h"

#include <Application.hpp>
#include "MCamTest.hpp"

static MCamTest* mcamTestPtr;

void *stress_proc_main(void *parm) {
	return (mcamTestPtr->stressProcMain(parm));
}

MCamTest::MCamTest(Application *applicationPtr) {
	mcamTestPtr = this;
	this->applicationPtr = applicationPtr;
	stopStressProc = 0;
	stressMsgBox = NULL;
	stressTestResult = 0;
	numberOfPixelClocks=1;
	maxROI.bottom = 0;
	maxROI.top = 0;
	maxROI.left = 0;
	maxROI.right = 0;
	minROI.bottom = 0;
	minROI.top = 0;
	minROI.left = 0;
	minROI.right = 0;
	//iStressTid = 0;
}

MCamTest::~MCamTest() {
}

// generate random number between min and max
int MCamTest::getRand(int min, int max) {
	if (min >= max) {
		return min;
	}
	return rand() % (max - min + 1) + min;
}

// change camera parameter for stress test
long MCamTest::executeModifier(int contShotRunning) {
	MCAM_LOG_INIT("MCamTest::executeModifier")
	long result = 0;
	long result2 = 0;
	int modifier = getRand(0, 4);
	if (modifier == 0) {
		; // no modification
	} else if (modifier == 1 && !contShotRunning) {
		// pixelfreq
		int pixelFreq = getRand(0, 1);

		if ((!applicationPtr->isTriggerEnabled()) && (numberOfPixelClocks > 1)) {
			result = McammSetPixelClock(applicationPtr->getCameraIndex(), pixelFreq);
			if (result == NOERR)
				MCAM_LOGF_INFO("### MOD PIXEL ### Set Pixel Clock to %d, result=%d", pixelFreq, result);
			else {
				MCAM_LOGF_ERROR("### ERROR MOD PIXEL ### Set Pixel Clock to %d failed, result=%d (error ignored, assume USB 2.0)", pixelFreq,
						result);
				result = 0;
			}
		}
	} else if (modifier == 2) {
		// ROI
		RECT newROI;
		RECT currROI;
		int x = getRand(0, maxROI.right-maxROI.left);
		int y = getRand(0, maxROI.bottom-maxROI.top);
		int width = getRand(minROI.right, maxROI.right-maxROI.left);
		int height = getRand(minROI.bottom, maxROI.bottom-maxROI.top);
		newROI.left = x;
		newROI.right = x + width;
		newROI.top = y;
		newROI.bottom = y + height;
		if (newROI.right >  maxROI.right-maxROI.left)
		  newROI.right = maxROI.right-maxROI.left;
		if (newROI.bottom > maxROI.bottom-maxROI.top) {
		  newROI.bottom = maxROI.bottom-maxROI.top;
		}

		result = McammSetFrameSize(applicationPtr->getCameraIndex(), &newROI);
		result2 = McammGetCurrentFrameSize(applicationPtr->getCameraIndex(), &currROI);
		if (result == NOERR && result2 == NOERR) {
			MCAM_LOGF_INFO("### MOD FRAME ### write result=%ld read result=%ld new frame: %d %d %d %d", result, result2, newROI.bottom,
					           newROI.left, newROI.right, newROI.top);
		} else {
			MCAM_LOGF_ERROR("### ERROR MOD FRAME ### write set-result=%ld  get-result2=%ld new frame: left=%d right=%d bottom=%d top=%d readback: left=%d right=%d bottom=%d top=%d",
			      result, result2, newROI.left,	newROI.right, newROI.bottom, newROI.top, currROI.left, currROI.right, currROI.bottom, currROI.top);
			if (result2 != NOERR)
				result = result2;
		}
	} else if (modifier == 3) {
		int rand = getRand(1, 5);
		int binning = 1;
		BOOL hasBinning = FALSE;

		if (cameraInfo[applicationPtr->getCameraIndex()].Type == mcamRGB) {
			rand = getRand(1, 3);
			if (rand == 1)
				binning = 1;
			else if (rand == 2)
				binning = 3;
			else
				binning = 5;
		} else {
			rand = getRand(1, 5);
			binning = rand;
		}
		result = McammHasBinning(applicationPtr->getCameraIndex(), binning, &hasBinning);
		if ((result == NOERR) && hasBinning) {
            result = McammSetBinning(applicationPtr->getCameraIndex(), binning);
            if (result == NOERR)
                MCAM_LOGF_INFO("### MOD BINNING ###  to %d, result=%d", binning, result);
            else
                MCAM_LOGF_ERROR("### ERROR MOD BINNING ###  to %d, result=%d", binning, result);
		}
	} else if (modifier == 4) {
		int exposure = getRand(10000, 300000);
		result = McammSetExposure(applicationPtr->getCameraIndex(), exposure);
		if (result == NOERR)
			MCAM_LOGF_INFO("### MOD EXPOSURE ### to %d ms, result=%d", exposure / 1000, result);
		else
			MCAM_LOGF_ERROR("### ERROR MOD EXPOSURE ### to %d ms, result=%d", exposure / 1000, result);
	}
	return result;
}

// main() of stress test thread
void *MCamTest::stressProcMain(void *parm) {
	MCAM_LOG_INIT("MCamTest::stressProcMain")
	int mode = 0;
	int modeSW = 0;
	long counter = 0;
	bool nowait = false;
	int stressContShotRunning = 0;
	MCAM_LOGF_INFO("Started stress thread");
	long result = McammGetNumberOfPixelClocks(applicationPtr->getCameraIndex(), &numberOfPixelClocks);
	if (result != NOERR) {
	    stressTestResult++;
	}
	MCAM_LOGF_INFO("McammGetNumberOfPixelClocks = %d", numberOfPixelClocks);
  // set min. frame size
	minROI.bottom = 0;
  minROI.top = 0;
  minROI.left = 0;
  minROI.right = 0;
  result = McammSetFrameSize(applicationPtr->getCameraIndex(), &minROI);
  if (result != NOERR) {
     stressTestResult++;
  }
  result = McammGetCurrentFrameSize(applicationPtr->getCameraIndex(), &minROI);
  if (result != NOERR) {
     stressTestResult++;
  }
  MCAM_LOGF_INFO("McammGetCurrentFrameSize minROI: left=%d right=%d bottom=%d top=%d", minROI.left, minROI.right, minROI.bottom, minROI.top);
	// set max. frame size
	result = McammSetFrameSize(applicationPtr->getCameraIndex(), NULL);
	if (result != NOERR) {
	      stressTestResult++;
	}
	result = McammGetCurrentFrameSize(applicationPtr->getCameraIndex(), &maxROI);
	if (result != NOERR) {
	   stressTestResult++;
	}

  MCAM_LOGF_INFO("McammGetCurrentFrameSize maxROI: left=%d right=%d bottom=%d top=%d", maxROI.left, maxROI.right, maxROI.bottom, maxROI.top);


  while (!stopStressProc) {
		mode = getRand(0, 6);
		modeSW = getRand(0, 100);
		nowait = false;
		if (mode < 3 && stressContShotRunning == 0 && (counter % 50) == 0) {
			stressTestResult += applicationPtr->continuousShotStartStop(true);
			stressContShotRunning = 1;
		} else if (mode < 5 && stressContShotRunning == 1 && (counter % 50) == 0) {
			stressTestResult += applicationPtr->continuousShotStartStop(false);
			stressContShotRunning = 0;
		} else if (mode == 6 && stressContShotRunning == 0) {
			stressTestResult += applicationPtr->singleShot();
			nowait = true;
			counter++;
		}

		if (modeSW == 1) {
			// test SW trigger
			if (stressContShotRunning)
				stressTestResult += applicationPtr->continuousShotStartStop(false);
			MCAM_LOGF_STATUS("sw trigger stress starting");
			for ( int i = 0; i < 10; i++) {
				long randSleep = getRand(10, 400);
				stressTestResult += applicationPtr->thisMCamImagePtr->setSoftwareTrigger(applicationPtr->getCameraIndex(), true);
				if (stressTestResult != NOERR)
					break;
				stressTestResult += applicationPtr->continuousShotStartStop(true);
				if (stressTestResult != NOERR)
					break;

				MCamUtil::sleep(randSleep);

				stressTestResult += applicationPtr->continuousShotStartStop(false);
				if (stressTestResult != NOERR)
					break;
				stressTestResult += applicationPtr->thisMCamImagePtr->setSoftwareTrigger(applicationPtr->getCameraIndex(), false);
				if (stressTestResult != NOERR)
					break;

				stressTestResult += applicationPtr->continuousShotStartStop(true);
				if (stressTestResult != NOERR)
					break;

				MCamUtil::sleep(randSleep);

				stressTestResult += applicationPtr->continuousShotStartStop(false);
				if (stressTestResult != NOERR)
					break;
				if (stopStressProc)
					break;
			}
			if (stressTestResult > 0 )
				MCAM_LOGF_ERROR("sw trigger stress test ended in error: %d", stressTestResult);
			else
				MCAM_LOGF_STATUS("sw trigger stress test done");

			if (stressContShotRunning)
				stressTestResult += applicationPtr->continuousShotStartStop(true);
		}

		if (!nowait) {
			MCamUtil::sleep(100);
			counter++;
		}
		if ((counter % 20) == 0) {
			stressTestResult += executeModifier(stressContShotRunning);
		}

		if (stressTestResult != NOERR) {
			MCAM_LOGF_ERROR("### Stress Test failed!");
			break;
		}
	}
	if (stressContShotRunning)
		stressTestResult += applicationPtr->continuousShotStartStop(false);
	MCAM_LOGF_INFO("Stopped stress thread");
	return NULL;
}

// start stress test
void MCamTest::startStressTest() {
	MCAM_LOG_INIT("MCamTest::startStressTest")
	long result = 0;
	stopStressProc = false;
	QAbstractButton *clicked = NULL;

	stressTestResult = 0;

	if (applicationPtr->getCameraIndex() < 0) {
		MCAM_LOGF_ERROR("no camera available -> abort!");
		return;
	}
	if (applicationPtr->getMcamImagePtr()->isContShotRunning()) {
		MCAM_LOGF_ERROR("cont shot running -> abort!");
		return;
	}
	result = McammInfo(applicationPtr->getCameraIndex(), &(cameraInfo[applicationPtr->getCameraIndex()]));

	if (result != 0) {
		MCAM_LOGF_ERROR("camera cannot be accessed -> abort!");
		return;
	}

	if (pthread_create(&iStressTid, NULL, &::stress_proc_main, NULL) != 0) {
		MCAM_LOGF_ERROR("error creating stress_proc_main thread");
		return;
	}
	stressMsgBox = new QMessageBox(this);
	stressMsgBox->setButtonText(QMessageBox::Ok, "Stop");
	stressMsgBox->setWindowTitle(tr("MCam Stress Test"));
	stressMsgBox->setText(tr("Stress test running - press 'Stop' to end test!   "));
	stressMsgBox->show();
	do {
		QApplication::processEvents();
		MCamUtil::sleep(20);
		clicked = stressMsgBox->clickedButton();
	} while (clicked == NULL && stressTestResult == 0);
	if (clicked == NULL)
		stressMsgBox->close();
	delete stressMsgBox;

	stopStressProc = true;
	pthread_join(iStressTid, NULL);

	if (stressTestResult) {
		QMessageBox::warning(this, tr("Stress Test Error"), tr("<p>MCam Stress Test ended in Error!</p>"));
	}

	applicationPtr->updateCameraGUIParamter(applicationPtr->getCameraIndex());

	return;
}
