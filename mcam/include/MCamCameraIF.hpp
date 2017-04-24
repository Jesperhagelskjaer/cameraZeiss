/**
 * @file MCamCameraIF.h
 * @author ggraf
 * @date 14.02.2015
 *
 * @brief include file for camera interface
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#ifndef MCAMCAMERAIF_H_
#define MCAMCAMERAIF_H_

#include "mcam.h"

#include <QObject>

class Application;

#define MCAM_MAX_NO_OF_CAMERAS 20

typedef struct
{
    unsigned long exposureTime;
    int pixelFreqIndex;
    bool useSensorTapsRight;
    bool useSensorTapsBottom;
    bool cameraBuffering;
    int binning;
    bool doMonoImage;
    bool doColorProcessing;
    unsigned long userFrameTime;
    int whitePointRed;
    int whitePointGreen;
    int whitePointBlue;

} McamProperties;

enum ECameraType {
	MCAM_CAMERA_CCD = 0,
	MCAM_CAMERA_CMOS,
	MCAM_CAMERA_UNKNOWN
};

class MCamCameraIF: public QObject
{
    Q_OBJECT
    Application *applicationPtr;
    pthread_t cameraSupervisionTid;
    bool mcammInitializedState[MCAM_MAX_NO_OF_CAMERAS];
    bool cameraValidState[MCAM_MAX_NO_OF_CAMERAS];
    SMCAMINFO cameraInfo[MCAM_MAX_NO_OF_CAMERAS];
    long availableCameras;
    char propertyPath[MAX_PATH * 2];
    char logPath[MAX_PATH * 2];

    McamProperties mcamProperties;
    MCammColorMatrixOptimizationMode colorMatrixMode;

    // original value from property file
    double whiteRedOriginal, whiteGreenOriginal, whiteBlueOriginal;

public:
    MCamCameraIF(Application *applicationPtr, char *propertyPath, char *logPath);
    virtual ~MCamCameraIF();
    long setSessionProperties();
    long init();
    long deInit();
    void *cameraSuperVisionProcMain(void *parm);
    long loadMcamFileProperties();
    static const char *mcamGetErrorString(int result);
    int cameraCallback(long cameraIndex, McammCameraState state);
    long getNumberOfCameras();
    bool isCameraInitialized(long cameraIndex);
    long initializeCamera(long cameraIndex);
    int getDeviceString(long cameraIndex, char*deviceString);
    MCammColorMatrixOptimizationMode getColorMatrixMode();
    void setCameraDefaults(long cameraIndex);
    long setWhiteBalance(long cameraIndex, double red, double green, double blue);
    void setWhitePoint(double red, double green, double blue);
    long getWhitePoint(double *redPtr, double *greenPtr, double *bluePtr);

    double getWhiteRedOriginalValue();
    double getWhiteGreenOriginalValue();
    double getWhiteBlueOriginalValue();

    void resetWhitePoint();
    void saveMcamProperties();
    McamProperties* getMcamPropertyPtr();
    int getCameraType(long cameraIndex, ECameraType *cameraType);

    signals:
    void setBusyLock(bool enabled);
    int selectCamera(long cameraIndex);
    void updateDevices();
    int showMessageBox(QString);
    int dismissMessageBox();

};

#endif /* MCAMCAMERAIF_H_ */
