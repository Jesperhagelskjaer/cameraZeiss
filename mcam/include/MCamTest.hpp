/**
 * @file MCamTest.hpp
 * @author ggraf
 * @date 15.02.2015
 *
 * @brief mcam test header file
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#ifndef MCAMTEST_HPP_
#define MCAMTEST_HPP_

#include "mcam.h"

#include <QFileDialog>
#include <QMessageBox>

class Application;

class MCamTest: QDialog
{
    Application *applicationPtr;
    SMCAMINFO cameraInfo[MCAM_MAX_NO_OF_CAMERAS];
    bool stopStressProc;
    long stressTestResult;
    QMessageBox *stressMsgBox;
    pthread_t iStressTid;
    long numberOfPixelClocks;
    RECT maxROI;
    RECT minROI;

public:
    MCamTest(Application *applicationPtr);
    virtual ~MCamTest();
    void *stressProcMain(void *parm);
    int getRand(int min, int max);
    long executeModifier(int contShotRunning);
    void startStressTest();
};

#endif /* MCAMTEST_HPP_ */
