/*=============================================================

 MCAM_ZEI.H

 Header file for Microscopy Camera
 Resolution up to dim=3
 Author Udo Lenz
 Revision 0.0 / 30-Jul-99

 Extensions by Horst Schwarz
 Revision 0.29.0

 ==================================================================*/

#ifndef MCAM_ZEI_H_
#define MCAM_ZEI_H_

#if defined (_WIN32)

#include <windows.h>

#define DLLEXPORT __declspec(dllexport)

#else

#define DLLEXPORT
#define WINAPI
#define BOOL bool
#define LONG long
#define LONGLONG long long

// Priority Mappings for Linux
#define THREAD_PRIORITY_IDLE -15
#define THREAD_PRIORITY_LOWEST -2
#define THREAD_PRIORITY_BELOW_NORMAL -1
#define THREAD_PRIORITY_NORMAL 0
#define THREAD_PRIORITY_ABOVE_NORMAL 1
#define THREAD_PRIORITY_HIGHEST 2
#define THREAD_PRIORITY_TIME_CRITICAL 15

typedef struct _RECT
{
    LONG left;LONG top;LONG right;LONG bottom;
} RECT, *PRECT;

#endif /* defined (_WIN32) */

#if defined (__cplusplus)

extern "C" {

#endif /* defined (__cplusplus) */

typedef struct tagSMCAMCALIBRATIONERROR
{
    float x_err_max, y_err_max, /* Maximal positioning errors in x and y */
    x_err_rms, y_err_rms, /* Root mean square positioning errors */
    x_pos_min, x_pos_max, /* Smallest and largest piezo values in x */
    y_pos_min, y_pos_max; /* Smallest and largest piezo values in y */
} SMCAMCALIBRATIONERROR;

typedef enum
{
    mcamBW = 0, mcamRGB = 1
} eMcamType;


typedef struct tagSMCAMINFO
{
    long Revision;
    long SerienNummer;
    eMcamType Type;
    //  long Type;
    long Features;  // Mapped to camera model: "0": unknown, "503": Axiocam 503, "506": Axiocam 506
} SMCAMINFO;

#ifndef HRFW
typedef enum
{
    mcamScanModeLowQuality = 0,
    mcamScanModeHighQuality,
    mcamScanModeRaw,
    mcamScanModeHDRLowQuality,
    mcamScanModeHDRHighQuality,
} eMcamScanMode;
#else
typedef enum
{   mcamScanModeLowQuality=0,
    mcamScanModeHighQuality,
    mcamScanModeRaw,
    mcamScanModePiezodim1,
    mcamScanModePiezodim2,
    mcamScanModePiezodim3,
    mcamScanMode9Shot
}eMcamScanMode;
#endif

/*
 typedef enum
 { mcamScanModeDoubleSnap=0,
 mcamScanModeSnap,
 mcamScanModeDoubleField,
 mcamScanModeRaw
 } eMcamScanMode;
 */
typedef enum
{
    mcamStatusStarted = 1, mcamStatusReadout = 2
} eMcamStatus;

typedef enum
{
    mcamMultiShotFilterOff = 0, mcamMultiShotFilterOn = 1, mcamMultiShotFilterAuto = 2
} eMcamMultiShotFilterMode;

typedef BOOL (*McamImageProc)(long done, long total, eMcamStatus Status);

typedef BOOL (*McamImageProcEx)(long done, long total, eMcamStatus Status, void *UserParam);

typedef void (*McamBusresetProc)(void *UserParam);

typedef BOOL (*ContinuousCallbackProc)(unsigned short *img, long bytesize, long currbufnr,
        LONGLONG FrameTime, void *UserParam);

typedef void (*ImageDoneCallbackProc)(void *UserParam);

DLLEXPORT long WINAPI McamInit();

DLLEXPORT void WINAPI McamClose();

DLLEXPORT long WINAPI McamInfo(SMCAMINFO *pMcamInfo);

DLLEXPORT long WINAPI McamGetNumberOfResolutions();

DLLEXPORT long WINAPI McamGetResolutionValues(long Index, long *pWidth, long *pHeight,
        eMcamScanMode *pMode);

DLLEXPORT long WINAPI McamSetResolution(long Index);

DLLEXPORT long WINAPI McamGetCurrentResolution();

DLLEXPORT BOOL WINAPI McamHasBitsPerPixel(long wishedbpp);

DLLEXPORT long WINAPI McamSetBitsPerPixel(long bpp);

DLLEXPORT long WINAPI McamGetCurrentBitsPerPixel();

DLLEXPORT BOOL WINAPI McamHasBinning(long wishedBinning);

DLLEXPORT long WINAPI McamSetBinning(long Binning);

DLLEXPORT long WINAPI McamGetCurrentBinning();

DLLEXPORT BOOL WINAPI McamHasImageSize();

DLLEXPORT BOOL WINAPI McamHasFrame();

DLLEXPORT long WINAPI McamSetImageSize(RECT *prcSize);

DLLEXPORT long WINAPI McamGetCurrentImageSize(RECT *prcSize);

DLLEXPORT long WINAPI McamSetFrameSize(RECT *prcSize);

DLLEXPORT long WINAPI McamGetCurrentFrameSize(RECT *prcSize);

DLLEXPORT long WINAPI McamGetCurrentDataSize(long *width, long *height);

DLLEXPORT long WINAPI McamAcquisition(unsigned short *pImageData, long allocatedSize,
        McamImageProc pCallBack);

DLLEXPORT long WINAPI McamAcquisitionEx(unsigned short *pImageData, long allocatedSize,
        McamImageProcEx pCallBack, void *UserParam);

DLLEXPORT long WINAPI McamCalculateBlackRef(McamImageProc pCallBack);

DLLEXPORT long WINAPI McamCalculateBlackRefEx(McamImageProcEx pCallBack, void *UserParam);

DLLEXPORT long WINAPI McamSaveBlackRef(unsigned short *ref, long bytesize);

DLLEXPORT long WINAPI McamRestoreBlackRef(unsigned short *ref, long bytesize);

DLLEXPORT long WINAPI McamSetBlackRef(BOOL benable);

DLLEXPORT BOOL WINAPI McamGetCurrentBlackRef();

DLLEXPORT BOOL WINAPI McamHasBlackRef();

DLLEXPORT long WINAPI McamCalculateWhiteRef(McamImageProc pCallBack);

DLLEXPORT long WINAPI McamCalculateWhiteRefEx(McamImageProcEx pCallBack, void *UserParam);

DLLEXPORT long WINAPI McamSetWhiteRef(BOOL benable);

DLLEXPORT BOOL WINAPI McamGetCurrentWhiteRef();

DLLEXPORT BOOL WINAPI McamHasWhiteRef();

DLLEXPORT long WINAPI McamSetWhiteTable(short *table);

DLLEXPORT long WINAPI McamGetWhiteTable(short *table);

DLLEXPORT long WINAPI McamGetExposureRange(long *pMin, long *pMax, long *pInc);

DLLEXPORT long WINAPI McamSetExposure(long Microseconds);

DLLEXPORT long WINAPI McamGetCurrentExposure();

DLLEXPORT long WINAPI McamSetImageWhiteBalance(double Red, double Green, double Blue);

DLLEXPORT long WINAPI McamSetWhiteBalance(double Red, double Green, double Blue);

DLLEXPORT long WINAPI McamGetCurrentWhiteBalance(double *pRed, double *pGreen, double *pBlue);

DLLEXPORT BOOL WINAPI McamHasFastAcquisition();

DLLEXPORT long WINAPI McamStartFastAcquisition();

DLLEXPORT long WINAPI McamAbortFastAcquisition();

DLLEXPORT long WINAPI McamIsFastAcquisitionReady(unsigned short *pImageData, long allocatedSize,
        BOOL bStartNext);

DLLEXPORT BOOL WINAPI McamIsColorProcessingEnabled();

DLLEXPORT void WINAPI McamEnableColorProcessing(BOOL benable);

DLLEXPORT long WINAPI McamShutterControl(BOOL bautouse, BOOL binvert);

DLLEXPORT long WINAPI McamSetErrorMessages(BOOL benable);

DLLEXPORT BOOL WINAPI McamGetCurrentErrorMessages();

DLLEXPORT void WINAPI McamSetColorSaturation(double redsaturation, double greensaturation,
        double bluesaturation);

DLLEXPORT long WINAPI McamEnableHardwareTrigger(BOOL benable);

DLLEXPORT long WINAPI McamSetHardwareTriggerPolarity(BOOL binvert);

DLLEXPORT long WINAPI McamIsHardwareTriggerEnabled(BOOL *benable);

DLLEXPORT long WINAPI McamSetShutterDelay(long OpenDelay, long CloseDelay);

DLLEXPORT long WINAPI McamGetCurrentShutterDelay(long* OpenDelay, long* CloseDelay);

DLLEXPORT long WINAPI McamWriteFirmware(unsigned char *buf, long size);

DLLEXPORT long WINAPI McamGetNumberofCameras();

DLLEXPORT long WINAPI McamSetBusresetCallback(McamBusresetProc pCallBack, void *UserParam);

DLLEXPORT long WINAPI McamGetContextVersionfromContext(void *pContext, unsigned long * pVersion);

DLLEXPORT long WINAPI McamGetCurrentContextVersion(unsigned long * pVersion);

DLLEXPORT long WINAPI McamGetDLLNamefromContext(void *pContext, char * pName, long namelength);

// mutiple camera functions

DLLEXPORT long WINAPI McammInit(long camnum);

DLLEXPORT void WINAPI McammClose(long camnum);

DLLEXPORT long WINAPI McammInfo(long cameraindex, SMCAMINFO *pMcamInfo);

DLLEXPORT long WINAPI McammGetNumberOfResolutions(long cameraindex);

DLLEXPORT long WINAPI McammGetResolutionValues(long cameraindex, long Index, long *pWidth,
        long *pHeight, eMcamScanMode *pMode);

DLLEXPORT long WINAPI McammSetResolution(long cameraindex, long Index);

DLLEXPORT long WINAPI McammGetCurrentResolution(long cameraindex);

DLLEXPORT long WINAPI McammHasBitsPerPixel(long cameraindex, long wishedbpp, BOOL *bhasbits);

DLLEXPORT long WINAPI McammSetBitsPerPixel(long cameraindex, long bpp);

DLLEXPORT long WINAPI McammGetCurrentBitsPerPixel(long cameraindex);

DLLEXPORT long WINAPI McammHasBinning(long cameraindex, long wishedBinning, BOOL *bhasbinning);

DLLEXPORT long WINAPI McammSetBinning(long cameraindex, long Binning);

DLLEXPORT long WINAPI McammGetCurrentBinning(long cameraindex, long *binning);

DLLEXPORT long WINAPI McammHasImageSize(long cameraindex, BOOL *bhassize);

DLLEXPORT long WINAPI McammHasFrame(long cameraindex, BOOL *bhasframe);

DLLEXPORT long WINAPI McammSetImageSize(long cameraindex, RECT *prcSize);

DLLEXPORT long WINAPI McammGetCurrentImageSize(long cameraindex, RECT *prcSize);

DLLEXPORT long WINAPI McammSetFrameSize(long cameraindex, RECT *prcSize);

DLLEXPORT long WINAPI McammGetCurrentFrameSize(long cameraindex, RECT *prcSize);

DLLEXPORT long WINAPI McammGetCurrentDataSize(long cameraindex, long *width, long *height);

DLLEXPORT long WINAPI McammAcquisition(long cameraindex, unsigned short *pImageData,
        long allocatedSize, McamImageProc pCallBack);

DLLEXPORT long WINAPI McammAcquisitionEx(long cameraindex, unsigned short *pImageData,
        long allocatedSize, McamImageProcEx pCallBack, void *UserParam);

DLLEXPORT long WINAPI McammCalculateBlackRef(long cameraindex, McamImageProc pCallBack);

DLLEXPORT long WINAPI McammCalculateBlackRefEx(long cameraindex, McamImageProcEx pCallBack,
        void *UserParam);

DLLEXPORT long WINAPI McammSaveBlackRef(long cameraindex, unsigned short *ref, long bytesize);

DLLEXPORT long WINAPI McammRestoreBlackRef(long cameraindex, unsigned short *ref, long bytesize);

DLLEXPORT long WINAPI McammSetBlackRef(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammGetCurrentBlackRef(long cameraindex, BOOL *bgetbref);

DLLEXPORT long WINAPI McammHasBlackRef(long cameraindex, BOOL *bhasbref);

DLLEXPORT long WINAPI McammCalculateWhiteRef(long cameraindex, McamImageProc pCallBack);

DLLEXPORT long WINAPI McammCalculateWhiteRefEx(long cameraindex, McamImageProcEx pCallBack,
        void *UserParam);

DLLEXPORT long WINAPI McammSetWhiteRef(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammGetCurrentWhiteRef(long cameraindex, BOOL *bgetwref);

DLLEXPORT long WINAPI McammHasWhiteRef(long cameraindex, BOOL *bhaswref);

DLLEXPORT long WINAPI McammSetWhiteTable(long cameraindex, short *table);

DLLEXPORT long WINAPI McammGetWhiteTable(long cameraindex, short *table);

DLLEXPORT long WINAPI McammGetExposureRange(long cameraindex, long *pMin, long *pMax, long *pInc);

DLLEXPORT long WINAPI McammSetExposure(long cameraindex, long Microseconds);

DLLEXPORT long WINAPI McammGetCurrentExposure(long cameraindex, long *Microseconds);

DLLEXPORT long WINAPI McammSetImageWhiteBalance(long cameraindex, double Red, double Green,
        double Blue);

DLLEXPORT long WINAPI McammSetWhiteBalance(long cameraindex, double Red, double Green, double Blue);

DLLEXPORT long WINAPI McammGetCurrentWhiteBalance(long cameraindex, double *pRed, double *pGreen,
        double *pBlue);

DLLEXPORT long WINAPI McammHasFastAcquisition(long cameraindex, BOOL *bhasfast);

DLLEXPORT long WINAPI McammSetImagedoneCallback(long cameraindex,
        ImageDoneCallbackProc pImagedoneCallback, void *UserParam);

DLLEXPORT long WINAPI McammStartFastAcquisition(long cameraindex);

DLLEXPORT long WINAPI McammAbortFastAcquisition(long cameraindex);

DLLEXPORT long WINAPI McammIsFastAcquisitionReady(long cameraindex, unsigned short *pImageData,
        long allocatedSize, BOOL bStartNext);

DLLEXPORT long WINAPI McammGetIPInfo(long cameraindex, void **pContext,
        unsigned long * pContextByteSize, ContinuousCallbackProc cbproc,
        unsigned long *pImgByteSize);

DLLEXPORT long WINAPI McammStartContinuousAcquisition(long cameraindex, int Priority,
        void *UserParam);

DLLEXPORT long WINAPI McammStopContinuousAcquisition(long cameraindex);

DLLEXPORT long WINAPI McammIsColorProcessingEnabled(long cameraindex, BOOL *bisenabled);

DLLEXPORT long WINAPI McammEnableColorProcessing(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammShutterControl(long cameraindex, BOOL bautouse, BOOL binvert);

DLLEXPORT long WINAPI McammShutterControlTTLOut2(long cameraindex, BOOL bautouse, BOOL binvert,
        BOOL bPreSkip, BOOL bReadOut, BOOL bPostSkip);

DLLEXPORT long WINAPI McammSetColorSaturation(long cameraindex, double redsaturation,
        double greensaturation, double bluesaturation);

DLLEXPORT long WINAPI McammEnableHardwareTrigger(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammSetHardwareTriggerPolarity(long cameraindex, BOOL binvert);

DLLEXPORT long WINAPI McammIsHardwareTriggerEnabled(long cameraindex, BOOL *benable);

DLLEXPORT long WINAPI McammSetShutterDelay(long cameraindex, long OpenDelay, long CloseDelay);

DLLEXPORT long WINAPI McammGetCurrentShutterDelay(long cameraindex, long* OpenDelay,
        long* CloseDelay);

DLLEXPORT long WINAPI McammWriteFirmware(long cameraindex, unsigned char *buf, long size);

DLLEXPORT long WINAPI McammEnablePacking(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammEnable8bitCompression(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammExecuteIPFunction(void *pContext, unsigned short *pImageData,
        unsigned short *pImageDataToProcess);

DLLEXPORT long WINAPI McammInitializeAcquisition(long cameraindex);

DLLEXPORT long WINAPI McammNextAcquisition(long cameraindex, unsigned short *pImageData,
        long allocatedSize, long microseconds);

DLLEXPORT long WINAPI McammFinalizeAcquisition(long cameraindex);

DLLEXPORT long WINAPI McammReadAdvancedFeatureRegister(long cameraindex, long offset,
        unsigned long *data);

DLLEXPORT long WINAPI McammReadAnalogGain(long cameraindex, long *loggain);

DLLEXPORT long WINAPI McammWriteAnalogGain(long cameraindex, long loggain);

DLLEXPORT long WINAPI McammSetExposureSequence(long cameraindex, long LengthOfSequence,
        unsigned long *Microseconds);

DLLEXPORT long WINAPI McammSetBusresetCallback(long cameraindex, McamBusresetProc pCallBack,
        void *UserParam);

DLLEXPORT long WINAPI McammSetFrameDelay(long cameraindex, unsigned long Microseconds);

DLLEXPORT long WINAPI McammSetFrameTime(long cameraindex, unsigned long Microseconds);

DLLEXPORT long WINAPI McammGetCurrentReadouttime(long cameraindex, unsigned long *Microseconds);

DLLEXPORT long WINAPI McammEnableOverlappedTriggeredAcquisition(long cameraindex, BOOL benable);

#define NOERR 0
#define DOWNLOADERR 1
#define INITERR 2
#define NOCAMERA 3
#define ABORTERR 4
#define WHITEERR 5
#define IMAGESIZEERR 6
#define NOMEMERR 7
#define PARAMERR 8
#define PIEZOCALBADIMAGE 9
#define CAMERABUSY 10
#define CAMERANOTSTARTED 11
#define BLACKREFTOOBRIGHT 12
#define WHITEREFTOOBRIGHT 13
#define WHITEREFTOODARK 14
#define NOTIMPLEMENTED 15
#define NODEVICEFOUND 16
#define HARDWAREVERSIONCONFLICT 17
#define FIRMWAREVERSIONCONFLICT 18
#define CONTEXTVERSIONCONFLICT 19

#define READERROR 20
#define TRIGGERERROR 21
#define BANDWIDTHERROR 22
#define RESOURCEERROR 23
#define ATTACHERROR 24
#define CHANNELERROR 25
#define STOPERROR 26
#define WRITEERROR 27
#define EPROMERR 28
#define BUSRESETERR 29
#define MULTISHOTTIMEOUT 30

// extra HR Firewire Definitions

DLLEXPORT long WINAPI McammSetFadingCorrection(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammGetCurrentFadingCorrection(long cameraindex, BOOL *benabled);

DLLEXPORT long WINAPI McammHasCalibration(long cameraindex, BOOL *bhascal);

typedef BOOL (*McamCalibrationProc)(long DoneSteps, long TotalSteps, SMCAMCALIBRATIONERROR *pError);

DLLEXPORT long
WINAPI McammCalibrate(long cameraindex, long MaxSteps, McamCalibrationProc pCallBack);

typedef BOOL (*McamCalibrationProcEx)(long DoneSteps, long TotalSteps,
        SMCAMCALIBRATIONERROR *pError, void *UserParam);

DLLEXPORT long WINAPI McammCalibrateEx(long cameraindex, long MaxSteps,
        McamCalibrationProcEx pCallBack, void * UserParam);

DLLEXPORT long WINAPI McammLoadCalibration(long cameraindex, char *pFileName);

DLLEXPORT long WINAPI McammSaveCalibration(long cameraindex, char *pFileName);

DLLEXPORT long WINAPI McammResetCalibration(long cameraindex);

DLLEXPORT long WINAPI McammEnableSharpener(long cameraindex, BOOL benable);

DLLEXPORT long WINAPI McammIsSharpenerEnabled(long cameraindex, BOOL *benabled);

DLLEXPORT long WINAPI McammSetMultiShotFilter(long cameraindex, eMcamMultiShotFilterMode mode);

DLLEXPORT long WINAPI McammGetCurrentMultiShotFilter(long cameraindex,
        eMcamMultiShotFilterMode *mode);

// veraltetes bzw. mrc5-irrelevantes Zeugs

DLLEXPORT long WINAPI McamButtonWasPressed(BOOL *buttonpressed);

DLLEXPORT BOOL WINAPI McamHasCalibration();

typedef BOOL (*McamCalibrationProc)(long DoneSteps, long TotalSteps, SMCAMCALIBRATIONERROR *pError);

DLLEXPORT long WINAPI McamCalibrate(long MaxSteps, McamCalibrationProc pCallBack);

typedef BOOL (*McamCalibrationProcEx)(long DoneSteps, long TotalSteps,
        SMCAMCALIBRATIONERROR *pError, void *UserParam);

DLLEXPORT long WINAPI McamCalibrateEx(long MaxSteps, McamCalibrationProcEx pCallBack,
        void *UserParam);

DLLEXPORT long WINAPI McamLoadCalibration(char *pFileName);

DLLEXPORT long WINAPI McamSaveCalibration(char *pFileName);

DLLEXPORT long WINAPI McamResetCalibration();

DLLEXPORT void WINAPI McamEnableSharpener(BOOL benable);

DLLEXPORT BOOL WINAPI McamIsSharpenerEnabled();

DLLEXPORT long WINAPI McamSetFadingCorrection(BOOL benable);

DLLEXPORT BOOL WINAPI McamGetCurrentFadingCorrection();

DLLEXPORT long WINAPI McamSetFadingCorrectionGainLimit(double gainlimit);

DLLEXPORT long WINAPI McamSetMultiShotFilter(eMcamMultiShotFilterMode mode);

DLLEXPORT eMcamMultiShotFilterMode WINAPI McamGetCurrentMultiShotFilter();

DLLEXPORT long WINAPI McamSetBloomingVoltage(long volt);

DLLEXPORT long WINAPI McamGetBloomingVoltage(long *volt);

DLLEXPORT long WINAPI McamSetPeltierVoltage(long volt);

DLLEXPORT long WINAPI McamGetPeltierVoltage(long *volt);

DLLEXPORT long WINAPI MCamIsAvailable();

DLLEXPORT void WINAPI McamStripeAdjust(double factor, BOOL bauto);

DLLEXPORT BOOL WINAPI Mcam_corrblems(BOOL bcorr);

DLLEXPORT unsigned long WINAPI ReadEprom(unsigned short address, short bytenr);

DLLEXPORT void WINAPI WriteEprom(unsigned short address, unsigned long data, short bytenr);

DLLEXPORT void WINAPI outcomm(unsigned long value);

DLLEXPORT long WINAPI McammSetBloomingVoltage(long cameraindex, long volt);

DLLEXPORT long WINAPI McammGetBloomingVoltage(long cameraindex, long *volt);

#if defined (__cplusplus)

}

#endif /* defined (__cplusplus) */

#endif /* MCAM_ZEI_H_ */

