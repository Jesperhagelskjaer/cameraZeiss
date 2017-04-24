/**
 * @file MCamCameraIF.cpp
 * @author ggraf
 * @date 14.02.2015
 *
 * @brief camera DLL interface
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#include <errno.h>
#include <QFileDialog>

extern "C" {
#include <pthread.h>
#include <semaphore.h>
}

#include "MCamCameraIF.hpp"
#include <Application.hpp>
#include <ConfigReader.hpp>

static MCamCameraIF* cameraIFPtr;
static void *supervision_proc_main(void *parm);

// default values for mcam
#define MCAM_PROPERTY_FILE  "mcam.properties"

static void cameraCallback(long cameraIndex, McammCameraState state)
{
    cameraIFPtr->cameraCallback(cameraIndex, state);
}

void *supervision_proc_main(void *parm)
{
    return cameraIFPtr->cameraSuperVisionProcMain(parm);
}

MCamCameraIF::MCamCameraIF(Application *applicationPtr, char *propertyPath, char *logPath)
{
    MCAM_LOG_INIT("MCamCameraIF::MCamCameraIF")
    cameraIFPtr = this;
    strcpy(this->logPath, logPath);
    this->applicationPtr = applicationPtr;
    availableCameras = 0;

    connect(this, SIGNAL(selectCamera(long)), applicationPtr, SLOT(selectCamera(long)));
    connect(this, SIGNAL(setBusyLock(bool)), applicationPtr, SLOT(setBusyLock(bool)));
    connect(this, SIGNAL(showMessageBox(QString)), applicationPtr, SLOT(showMessageBox(QString)));
    connect(this, SIGNAL(dismissMessageBox()), applicationPtr, SLOT(dismissMessageBox()));

    connect(this, SIGNAL(updateDevices()), applicationPtr, SLOT(doUpdateDevices()));

    for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; i++) {
        mcammInitializedState[i] = false;
        cameraValidState[i] = false;
    }
    whiteRedOriginal = whiteGreenOriginal = whiteBlueOriginal = 255;
    memset(&mcamProperties, '\0', sizeof(mcamProperties));
    colorMatrixMode = mcammNoOptimization; // overwritten in loadMcamFileProperties()
    if (pthread_create(&cameraSupervisionTid, NULL, &supervision_proc_main, NULL) != 0)
        MCAM_LOGF_ERROR("error creating image_proc_main thread");
}

MCamCameraIF::~MCamCameraIF()
{
}

void *MCamCameraIF::cameraSuperVisionProcMain(void *parm)
{
    MCAM_LOG_INIT("MCamCameraIF::cameraSuperVisionProcMain")
    MCAM_LOGF_STATUS("cameraSuperVisionProcMain thread started");

    emit showMessageBox(tr("Searching and initializing Axiocam USB 3.0 cameras, please wait!"));

    long result = init();

    if (result != NOERR)
        MCAM_LOGF_ERROR("init failed");
    else {
        emit updateDevices();
    }
    emit dismissMessageBox();


    if (availableCameras == 0)
        emit showMessageBox(tr("No Zeiss Axiocam USB 3.0 cameras found!"));
    else {
        setBusyLock(false);
        emit selectCamera(0);
        applicationPtr->doLoadSettings();
    }
    return NULL;
}

long MCamCameraIF::setSessionProperties()
{
    MCAM_LOG_INIT("MCamCameraIF::setSessionProperties")
    long result = NOERR;
    int err = 0;
#ifdef _WIN32
    if (strlen(logPath) >0) {
        // set session properties for axcam.log
        McammSessionProperties sessionProps;

        memset(&sessionProps, '\0', sizeof(McammSessionProperties));
        result = McammGetSessionProperties(&sessionProps, sizeof(sessionProps));
        if (result == NOERR) {
            sessionProps.propertyVersion = MCAM_SESSION_PROPERTY_VERSION;
            if (_access(logPath, 06) == 0) { // read / write
                strcpy(sessionProps.logfilePath, logPath);
                //printf("Logfile path set to '%s'\n", logPath);
            } else {
                printf("Error logfile path does not exist '%s' - ignored", logPath);
                err++;
            }
            if (strlen(propertyPath) > 0)
            if (_access(propertyPath, 04) == 0)  // read only access
            strcpy(sessionProps.propertyFilePath, propertyPath);
            else {
                printf("Error property path does not exist '%s' - ignored\n", propertyPath);
                err++;
            }
        }
        if (err < 2) {
            result = McammSetSessionProperties(&sessionProps, sizeof(sessionProps));
            if (result != NOERR)
            printf("Error setting SessionProperties, result=%d\n", result);
        }
    }
#else
/*  Test only
    McammSessionProperties sessionProps;
	memset(&sessionProps, '\0', sizeof(McammSessionProperties));
	result = McammGetSessionProperties(&sessionProps, sizeof(sessionProps));
	if (result == NOERR) {
		sessionProps.propertyVersion = MCAM_SESSION_PROPERTY_VERSION;
		strcpy(sessionProps.logfilePath, "/tmp");
		result = McammSetSessionProperties(&sessionProps, sizeof(sessionProps));
		printf("set SessionProperties, result=%d\n", result);
	}
*/
#endif
    return 0;
}

// initialize camera interface and initialize all available cameras
long MCamCameraIF::init()
{
    MCAM_LOG_INIT("MCamCameraIF::setSessionProperties")
    long result = NOERR;
    MCAM_LOGF_STATUS("searching and initializing Axiocam USB 3.0 cameras, please wait!");

    result = setSessionProperties();
    if (result != NOERR) {
        MCAM_LOGF_ERROR("setSessionProperties failed result=%d", result);
        return result;
    }
    result = McammLibInit(false); // includes logger initialization
    if (result != NOERR) {
        MCAM_LOGF_ERROR("McammLibInit failed result=%d", result);
        return result;
    }

    availableCameras = McamGetNumberofCameras();

    MCAM_LOGF_STATUS("number of cameras=%d", availableCameras);

    // startup -> deliver sorted number of cameras
    // no callback for initial cameras
    for (int i = 0; i < availableCameras; i++) {
        cameraValidState[i] = true;
        initializeCamera(i);
    }

    result = McammAddCameraListener(&::cameraCallback);
    if (result != NOERR) {
        MCAM_LOGF_ERROR("McammAddCameraListener failed result=%d", result);
        return result;
    }
    return result;
}

long MCamCameraIF::deInit()
{
    MCAM_LOG_INIT("MCamCameraIF::deInit")
    for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; i++) {
        if (isCameraInitialized(i)) {
            MCAM_LOGF_INFO("mcam calling McammClose for cameraindex=%d", i);
            McammClose(i);
        }
    }
    MCAM_LOGF_INFO("mcam calling McammLibTerm");
    McammLibTerm();
    MCAM_LOGF_STATUS("mcam calling exit");
    return NOERR;
}

int MCamCameraIF::cameraCallback(long cameraIndex, McammCameraState state)
{
    MCAM_LOG_INIT("MCamCameraIF::cameraCallback")

    MCAM_LOGF_DEBUG("cameraCallback cameraindex=%ld state=%d", cameraIndex, state);

    if (cameraIndex < 0)
        return NODEVICEFOUND;

    if (state == mcammCameraAdded) {
        MCAM_LOGF_STATUS("Camera CAM#%d added", cameraIndex);
        cameraValidState[cameraIndex] = true;

        long result = initializeCamera(cameraIndex);
        if (result != NOERR)
            cameraValidState[cameraIndex] = false;

        availableCameras++;

        emit updateDevices();

    } else if (state == mcammCameraRemoved) {
        MCAM_LOGF_STATUS("Camera CAM#%d removed", cameraIndex);
        McammClose(cameraIndex);
        cameraValidState[cameraIndex] = false;
        mcammInitializedState[cameraIndex] = false;
        if (availableCameras > 0)
          availableCameras--;
        emit updateDevices();
    }
    return NOERR;
}

long MCamCameraIF::getNumberOfCameras()
{
    return availableCameras;
}

bool MCamCameraIF::isCameraInitialized(long cameraIndex)
{
    if (cameraIndex < 0)
        return false;
    if (cameraIndex < MCAM_MAX_NO_OF_CAMERAS)
        return cameraValidState[cameraIndex];
    else
        return false;
}

long MCamCameraIF::initializeCamera(long cameraIndex)
{
    MCAM_LOG_INIT("MCamCameraIF::initializeCamera")
    long result = NOERR;
    long pMax=0;
    long pMin=0;

    if (cameraIndex < 0)
        return NODEVICEFOUND;

    if (cameraIndex >= MCAM_MAX_NO_OF_CAMERAS)
        return PARAMERR;

    if (cameraValidState[cameraIndex] && (!mcammInitializedState[cameraIndex])) {
        int result = McammInit(cameraIndex);
        MCAM_LOGF_INFO("McammInit result=%d", result);
        if (result == NOERR)
            mcammInitializedState[cameraIndex] = true;
    }
    setCameraDefaults(cameraIndex);

    // get and log parameter
    for (int i = 0; i < (int) mcammParmEnumSize; i++) {
        BOOL supported;
        result = MCammHasParameter(cameraIndex, (MCammSupportedParameterType) i, &supported);
        if (result == NOERR)
            MCAM_LOGF_INFO("MCammHasParameter: MCammSupportedParameterType [%d] %s", i, (supported ==0)?"FALSE":"TRUE");
        else
            MCAM_LOGF_ERROR("MCammHasParameter: MCammSupportedParameterType [%d] error=%d ", i, result);
    }

    result = MCammGetParameterRange(cameraIndex, mcammParmExposure, &pMin, &pMax);
    if (result == NOERR)
        MCAM_LOGF_INFO("MCammGetParameterRange: MCammSupportedParameterType [%d] %d",mcammParmExposure, pMax);
    else
        MCAM_LOGF_ERROR("MCammGetParameterRange: MCammSupportedParameterType [%d] error=%d ", mcammParmExposure, result);

    return result;
}

int MCamCameraIF::getDeviceString(long cameraIndex, char*deviceString)
{
    MCAM_LOG_INIT("MCamCameraIF::getDeviceString")
    char buffer[128];
    char *colorMono = (char*) "";
    long result;

    if (cameraIndex < 0)
        return NODEVICEFOUND;

    result = McammInfo(cameraIndex, &(cameraInfo[cameraIndex]));
    if (result == 0) {
        if (cameraInfo[cameraIndex].Features == 0) {
            strcpy(deviceString, "Axiocam ");
        } else {
            sprintf(deviceString, "Axiocam %03ld ", cameraInfo[cameraIndex].Features);
        }
        if (cameraInfo[cameraIndex].Type == mcamRGB) {
            colorMono = (char*) "c";
        } else
            colorMono = (char*) "m";
        MCAM_LOGF_STATUS("CAM#%d S/N=%d%s", cameraIndex, cameraInfo[cameraIndex].SerienNummer, colorMono);
        if (cameraInfo[cameraIndex].Type == mcamRGB) {
            strcat(deviceString, "color");
            colorMono = (char*) "c";
        } else {
            strcat(deviceString, "mono");
            colorMono = (char*) "m";
        }
        sprintf(buffer, " #%ld%s", cameraInfo[cameraIndex].SerienNummer, colorMono);
        strcat(deviceString, buffer);
    } else
        MCAM_LOGF_ERROR("McammInfo ERROR result=%d", result);
    return 0;
}

int MCamCameraIF::getCameraType(long cameraIndex, ECameraType *cameraType)
{
    MCAM_LOG_INIT("MCamCameraIF::getCameraType")
	long result = NOERR;

    if (cameraIndex < 0)
        return NODEVICEFOUND;

    *cameraType = MCAM_CAMERA_UNKNOWN;

    result = McammInfo(cameraIndex, &(cameraInfo[cameraIndex]));
    if (result == 0) {
    	if (cameraInfo[cameraIndex].Features == 503 ||
    		cameraInfo[cameraIndex].Features == 506 ||
    		cameraInfo[cameraIndex].Features == 512) {
    		*cameraType = MCAM_CAMERA_CCD;
    	} else
    		*cameraType = MCAM_CAMERA_CMOS;
    } else {
    	 MCAM_LOGF_ERROR("McammInfo ERROR result=%d", result);
    }
    return result;
}

McamProperties * MCamCameraIF::getMcamPropertyPtr()
{
    return &mcamProperties;
}

long MCamCameraIF::loadMcamFileProperties()
{
    MCAM_LOG_INIT("MCamCameraIF::loadMcamFileProperties")
    ConfigReader config(MCAM_PROPERTY_FILE);

    memset(&mcamProperties, '\0', sizeof(mcamProperties));

    // set exposure time
    mcamProperties.exposureTime = config.read("exposureTime", 40000);

    mcamProperties.pixelFreqIndex = config.read("pixelFreqIndex", 1);

    mcamProperties.useSensorTapsRight = config.read("useSensorTapsRight", true);
    mcamProperties.useSensorTapsBottom = config.read("useSensorTapsBottom", true);

    mcamProperties.cameraBuffering = config.read("cameraBuffering", false);

    mcamProperties.userFrameTime = config.read("userFrameTime", 0);

    colorMatrixMode = mcammPipelineStageExceptMatrix;

    mcamProperties.binning = config.read("binning", 1);

    mcamProperties.doMonoImage = config.read("doMonoImage", false);

    mcamProperties.doColorProcessing = config.read("doColorProcessing", true);

    mcamProperties.whitePointRed = config.read("whitePoint.red", WHITE_POINT_DEFAULT_RED);
    mcamProperties.whitePointGreen = config.read("whitePoint.green", WHITE_POINT_DEFAULT_GREEN);
    mcamProperties.whitePointBlue = config.read("whitePoint.blue", WHITE_POINT_DEFAULT_BLUE);

    whiteRedOriginal = mcamProperties.whitePointRed;
    whiteGreenOriginal = mcamProperties.whitePointGreen;
    whiteBlueOriginal = mcamProperties.whitePointBlue;
    MCAM_LOGF_STATUS("Loaded Red=%lf Green=%lf Blue=%lf", whiteRedOriginal, whiteGreenOriginal, whiteBlueOriginal);
    return NOERR;
}

void MCamCameraIF::setCameraDefaults(long cameraIndex)
{
    MCAM_LOG_INIT("MCamCameraIF::setCameraDefaults")
    long result = NOERR;
    BOOL hasPixelClocks = FALSE;
    BOOL hasMultiplePortModes = FALSE;
    BOOL bufferingIsFaster = FALSE;
    BOOL isCompressionFaster = FALSE;

    if (cameraIndex < 0)
        return;

    MCAM_LOGF_STATUS("setCameraDefaults start for cameraIndex=%ld", cameraIndex);
    result = McammSetColorMatrixOptimizationMode(cameraIndex, colorMatrixMode);
    if (result != NOERR)
        MCAM_LOGF_ERROR("set colorMatrixMode failed result=%d", result);
	else
		 MCAM_LOGF_INFO("set colorMatrixMode = %d", (int) colorMatrixMode);		
    result = McammSetFrameTime(cameraIndex, mcamProperties.userFrameTime);
    if (result != NOERR)
        MCAM_LOGF_ERROR("set McammSetFrameTime failed, result =%d", result);

    result = MCammHasParameter(cameraIndex, mcammParmPixelClocks, &hasPixelClocks);
    if (hasPixelClocks) {
        result = McammSetPixelClock(cameraIndex, mcamProperties.pixelFreqIndex);
        if (result != NOERR)
            MCAM_LOGF_WARN("McammSetPixelClock call failed, result=%d (no error, if camera connected via USB 2.0 only or ContShot running)",
                                result);
    }
    result = MCammHasParameter(cameraIndex, mcammParmMultiplePortModes, &hasMultiplePortModes);
    if (hasMultiplePortModes) {
        result = McammUseSensorTaps(cameraIndex, mcamProperties.useSensorTapsRight, mcamProperties.useSensorTapsBottom);
        if (result != NOERR)
            MCAM_LOGF_ERROR("McammUseSensorTaps call failed, result=%d", result);
    }

    result = MCammIs8BitCompressionFaster(cameraIndex, &isCompressionFaster);
    if (result == NOERR) {
    	McammEnable8bitCompression(cameraIndex, isCompressionFaster);
    }

    result = MCammIsImageBufferingFaster(cameraIndex, &bufferingIsFaster);
        if (result == NOERR) {
        result = McammSetCameraBuffering(cameraIndex, bufferingIsFaster);
        if (result != NOERR)
            MCAM_LOGF_ERROR("McammSetCameraBuffering call failed, result=%d", result);
    }

    result = McammSetHardwareTriggerMode(cameraIndex, true, false);
    if (result != NOERR)
        MCAM_LOGF_ERROR("McammSetHardwareTriggerMode failed, result=%d", result);

    for (int gpoIndex=0; gpoIndex <3; gpoIndex++) {
      result = McammSetGPOSource(cameraIndex, gpoIndex, mcammGPOOff);
      if (result != NOERR)
             MCAM_LOGF_ERROR("McammSetGPOSource failed, result=%d", result);
      result = McammSetGPOSettings(cameraIndex, gpoIndex, 0, 0, false);
        if (result != NOERR)
            MCAM_LOGF_ERROR("McammSetGPOSettings gpoIndex=%d failed to set, result=%ld", gpoIndex, result);
    }
    // set binning
    result = McammSetBinning(cameraIndex, mcamProperties.binning);
    if (result != NOERR)
        MCAM_LOGF_ERROR("set McammSetBinning failed", result);

    // enable/disable monochrome image
    result = McammGetCurrentBitsPerPixel(cameraIndex);
    if (result == MCAM_BPP_COLOR && mcamProperties.doMonoImage) {
        result = McammSetBitsPerPixel(cameraIndex, MCAM_BPP_MONO);
        if (result != NOERR)
            MCAM_LOGF_ERROR("set McammSetBitsPerPixel failed, result=%ld", result);
    }
    result = McammSetFrameSize(cameraIndex, NULL);
    if (result != NOERR)
        MCAM_LOGF_ERROR("set McammSetFrameSize failed, result=%ld", result);
    // enable/disable color processing

    // disable discard mode - slow processing on PC will slow down camera
    result = McammSetImageDiscardMode(cameraIndex, false);
    if (result != NOERR)
        MCAM_LOGF_ERROR("set McammSetImageDiscardMode failed, result=%ld", result);

    result = McammEnableColorProcessing(cameraIndex, mcamProperties.doColorProcessing);
    if (result == NOERR) {
        // set image white point
        result = setWhiteBalance(cameraIndex, mcamProperties.whitePointRed, mcamProperties.whitePointGreen, mcamProperties.whitePointBlue);
        if (result == NOERR) {
            MCAM_LOGF_INFO("Setting image white balance OK cameraIndex=%ld result=%ld red=%ld green=%ld blue=%ld", cameraIndex, result,
                            mcamProperties.whitePointRed, mcamProperties.whitePointGreen, mcamProperties.whitePointBlue);
        } else {
            MCAM_LOGF_ERROR("Setting image white balance failed cameraIndex=%ld result=%ld red=%ld green=%ld blue=%ld", cameraIndex, result,
                            mcamProperties.whitePointRed, mcamProperties.whitePointGreen, mcamProperties.whitePointBlue);
        }
    } else {
        MCAM_LOGF_ERROR("enable color processing failed");
    }
    MCAM_LOGF_INFO("setCameraDefaults done");
}

MCammColorMatrixOptimizationMode MCamCameraIF::getColorMatrixMode()
{
    return colorMatrixMode;
}

long MCamCameraIF::setWhiteBalance(long cameraIndex, double red, double green, double blue)
{
    MCAM_LOG_INIT("MCamCameraIF::setWhiteBalance")

    if (cameraIndex < 0)
        return NODEVICEFOUND;

    MCAM_LOGF_INFO("MCamCameraIF::setWhiteBalance cameraIndex=%ld red=%lf green=%lf blue=%lf", cameraIndex, red, green, blue);
    if (cameraIndex >= 0) {
        long result = McammSetWhiteBalance(cameraIndex, red, green, blue);
        if (result != NOERR) {
            MCAM_LOGF_ERROR("cameraIndex=%ld Error calling McammSetImageWhiteBalance,red=%lf green=%lf blue=%lf cameraIndex=%d result=%ld",
                            cameraIndex, red, green, blue, cameraIndex, result);
        }
        return result;
    }
    return PARAMERR;
}

void MCamCameraIF::resetWhitePoint()
{
    MCAM_LOG_INIT("MCamCameraIF::resetWhitePoint")

    mcamProperties.whitePointRed = whiteRedOriginal;
    mcamProperties.whitePointGreen = whiteGreenOriginal;
    mcamProperties.whitePointBlue = whiteBlueOriginal;
    MCAM_LOGF_INFO("MCamCameraIF::resetWhitePoint red=%ld green=%ld blue=%ld", mcamProperties.whitePointRed, mcamProperties.whitePointGreen,
                    mcamProperties.whitePointBlue);
}

void MCamCameraIF::setWhitePoint(double red, double green, double blue)
{
    MCAM_LOG_INIT("MCamCameraIF::setWhitePoint")
    MCAM_LOGF_INFO("MCamCameraIF::setWhitePoint red=%lf green=%lf blue=%lf", red, green, blue);
    mcamProperties.whitePointRed = red;
    mcamProperties.whitePointGreen = green;
    mcamProperties.whitePointBlue = blue;
}

long MCamCameraIF::getWhitePoint(double *redPtr, double *greenPtr, double *bluePtr)
{
    *redPtr = mcamProperties.whitePointRed;
    *greenPtr = mcamProperties.whitePointGreen;
    *bluePtr = mcamProperties.whitePointBlue;
    return NOERR;
}

void MCamCameraIF::saveMcamProperties()
{
    MCAM_LOG_INIT("MCamCameraIF::saveMcamProperties")
    MCAM_LOGF_INFO("saveMcamProperties Red=%d Green=%d Blue=%d", mcamProperties.whitePointRed, mcamProperties.whitePointGreen,
                    mcamProperties.whitePointBlue);
    FILE *fp = fopen(MCAM_PROPERTY_FILE, "wb");
    if (fp != NULL) {
        fprintf(fp, "# mcam.properties\r\n");
        fprintf(fp, "exposureTime = %ld\r\n", mcamProperties.exposureTime);
        fprintf(fp, "# pixel frequency 0: LOW 1: HIGH\r\n");
        fprintf(fp, "pixelFreqIndex = %d\r\n", mcamProperties.pixelFreqIndex);
        fprintf(fp, "useSensorTapsRight= %d\r\n", mcamProperties.useSensorTapsRight);
        fprintf(fp, "useSensorTapsBottom = %d\r\n", mcamProperties.useSensorTapsBottom);
        fprintf(fp, "cameraBuffering = %d\r\n", mcamProperties.cameraBuffering);
        fprintf(fp, "binning = %d\r\n", mcamProperties.binning);
        fprintf(fp, "# global settings\r\n");
        fprintf(fp, "doMonoImage = %d\r\n", mcamProperties.doMonoImage);
        fprintf(fp, "doColorProcessing = %d\r\n", mcamProperties.doColorProcessing);
        fprintf(fp, "userFrameTime = %ld\r\n", mcamProperties.userFrameTime);
        fprintf(fp, "whitePoint.red = %d\r\n", mcamProperties.whitePointRed);
        fprintf(fp, "whitePoint.green = %d\r\n", mcamProperties.whitePointGreen);
        fprintf(fp, "whitePoint.blue = %d\r\n", mcamProperties.whitePointBlue);
        fclose(fp);
        printf(MCAM_PROPERTY_FILE " saved.");
    } else {
        MCAM_LOGF_ERROR("cannot open '%s', error=%d", MCAM_PROPERTY_FILE, errno);
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Critical);
        if (errno == EACCES)
            msgBox.setText("Error saving properties - Do you have proper access rights?");
        msgBox.setText("Error saving properties!");
        msgBox.exec();
    }
}

const char *MCamCameraIF::mcamGetErrorString(int result)
{
    if (result == 0)
        return "NOERRR";
    else if (result == DOWNLOADERR)
        return "download error";                // 1
    else if (result == INITERR)
        return "Initialization error";
    else if (result == NOCAMERA)
        return "no camera found";               // 3
    else if (result == ABORTERR)
        return "abort error";
    else if (result == WHITEERR)
        return "white error";
    else if (result == IMAGESIZEERR)
        return "image size error";
    else if (result == NOMEMERR)
        return "out of memory";
    else if (result == PARAMERR)
        return "parameter error";               // 8
    else if (result == PIEZOCALBADIMAGE)
        return "piezo calibration error";
    else if (result == CAMERABUSY)
        return "camera busy";                  // 10
    else if (result == CAMERANOTSTARTED)
        return "camera not started";
    else if (result == BLACKREFTOOBRIGHT)
        return "too bright for black reference";    // 12
    else if (result == WHITEREFTOOBRIGHT)
        return "too bright for white reference";    // 13
    else if (result == WHITEREFTOODARK)
        return "too dark for white reference";      // 14
    else if (result == NOTIMPLEMENTED)
        return "not implemented";                   // 15
    else if (result == NODEVICEFOUND)
        return "no camera device found";            // 16
    else if (result == HARDWAREVERSIONCONFLICT)
        return "hardware version conflict";
    else if (result == FIRMWAREVERSIONCONFLICT)
        return "firmware version conflict";
    else if (result == CONTEXTVERSIONCONFLICT)
        return "context version conflict";
    else if (result == READERROR)
        return "read error";                        // 20
    else if (result == TRIGGERERROR)
        return "trigger error";
    else if (result == BANDWIDTHERROR)
        return "bandwidth error";
    else if (result == RESOURCEERROR)
        return "resource error";
    else if (result == ATTACHERROR)
        return "attach error";
    else if (result == CHANNELERROR)
        return "channel error";
    else if (result == STOPERROR)
        return "stop error";
    else if (result == WRITEERROR)
        return "write error";                       // 27
    else if (result == EPROMERR)
        return "EEPROM error";
    else if (result == BUSRESETERR)
        return "bus reset error";
    else if (result == MULTISHOTTIMEOUT)
        return "multishot timeout";
    else
        return "unknown result error code";         // 31
}

