/*
 * MCam.hpp
 *
 *  Created on: Oct 21, 2011
 *      Author: horst
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#ifndef MCAM_HPP_
#define MCAM_HPP_

#include "mcam.h"

class MCam
{
public:
    MCam();
    virtual ~MCam();

#if defined (_WIN32)
    HINSTANCE handle;
#else
    void* handle;
#endif /* defined (_WIN32) */

    // mcam_zei_ex.h
    typedef long (*McammGetSessionProperties_t)(McammSessionProperties *, long);
    McammGetSessionProperties_t McammGetSessionProperties;

    typedef long (*McammSetSessionProperties_t)(McammSessionProperties *, long);
    McammSetSessionProperties_t McammSetSessionProperties;

    typedef long (*McammLibInit_t)();
    McammLibInit_t McammLibInit;

    typedef long (*McammLibTerm_t)();
    McammLibTerm_t McammLibTerm;

    typedef long (*McammGetCurrentImageDataSize_t)(long, long*);
    McammGetCurrentImageDataSize_t McammGetCurrentImageDataSize;

    // mcam_zei.h

    typedef long (*McamGetNumberofCameras_t)();
    McamGetNumberofCameras_t McamGetNumberofCameras;

    typedef long (*McammInit_t)(long);
    McammInit_t McammInit;

    typedef long (*McammClose_t)(long);
    McammClose_t McammClose;

    typedef long (*McammSetResolution_t)(long, long);
    McammSetResolution_t McammSetResolution;

    typedef long (*McammSetBinning_t)(long, long);
    McammSetBinning_t McammSetBinning;

    typedef long (*McammGetCurrentBinning_t)(long, long*);
    McammGetCurrentBinning_t McammGetCurrentBinning;

    typedef long (*McammGetResolutionValues_t)(long, long, long*, long*, eMcamScanMode*);
    McammGetResolutionValues_t McammGetResolutionValues;

    typedef long (*McammGetCurrentBitsPerPixel_t)(long);
    McammGetCurrentBitsPerPixel_t McammGetCurrentBitsPerPixel;

    typedef long (*McammGetCurrentFrameSize_t)(long, RECT*);
    McammGetCurrentFrameSize_t McammGetCurrentFrameSize;

    typedef long (*McammSetFrameSize_t)(long, RECT*);
    McammSetFrameSize_t McammSetFrameSize;

    typedef long (*McammGetCurrentExposure_t)(long, long*);
    McammGetCurrentExposure_t McammGetCurrentExposure;

    typedef long (*McammSetExposure_t)(long, long);
    McammSetExposure_t McammSetExposure;

    typedef long (*McammAcquisitionEx_t)(long, unsigned short*, long, McamImageProcEx, void*);
    McammAcquisitionEx_t McammAcquisitionEx;

    typedef long (*McammGetIPInfo_t)(long, void**, unsigned long*, ContinuousCallbackProc, unsigned long*);
    McammGetIPInfo_t McammGetIPInfo;

    typedef long (*McammExecuteIPFunction_t)(void*, unsigned short*, unsigned short*);
    McammExecuteIPFunction_t McammExecuteIPFunction;

    typedef long (*McammStartContinuousAcquisition_t)(long, int, void*);
    McammStartContinuousAcquisition_t McammStartContinuousAcquisition;

    typedef long (*McammStopContinuousAcquisition_t)(long);
    McammStopContinuousAcquisition_t McammStopContinuousAcquisition;

    typedef long (*McammSetFrameDelay_t)(long, unsigned long);
    McammSetFrameDelay_t McammSetFrameDelay;

    typedef long (*McammGetCurrentFrameTime_t)(long cameraindex, unsigned long *Microseconds);
    McammGetCurrentFrameTime_t McammGetCurrentFrameTime;

    typedef long (*McammCalculateBlackRefEx_t)(long, McamImageProcEx, void*);
    McammCalculateBlackRefEx_t McammCalculateBlackRefEx;

    typedef long (*McammSetBlackRef_t)(long, bool);
    McammSetBlackRef_t McammSetBlackRef;

    typedef long (*McammSaveBlackRef_t)(long, unsigned short*, long);
    McammSaveBlackRef_t McammSaveBlackRef;

    typedef long (*McammRestoreBlackRef_t)(long, unsigned short*, long);
    McammRestoreBlackRef_t McammRestoreBlackRef;

    typedef long (*McammCalculateWhiteRefEx_t)(long, McamImageProcEx, void*);
    McammCalculateWhiteRefEx_t McammCalculateWhiteRefEx;

    typedef long (*McammSetWhiteRef_t)(long, bool);
    McammSetWhiteRef_t McammSetWhiteRef;

    typedef long (*McammGetWhiteTable_t)(long, short*);
    McammGetWhiteTable_t McammGetWhiteTable;

    typedef long (*McammSetWhiteTable_t)(long, short*);
    McammSetWhiteTable_t McammSetWhiteTable;

    typedef long (*McammEnableColorProcessing_t)(long, bool);
    McammEnableColorProcessing_t McammEnableColorProcessing;

    typedef long (*McammSetImageWhiteBalance_t)(long, double, double, double);
    McammSetImageWhiteBalance_t McammSetImageWhiteBalance;

    typedef long (*McammSetBitsPerPixel_t)(long, long);
    McammSetBitsPerPixel_t McammSetBitsPerPixel;

    typedef long (*McammGetBlackReferenceDataSize_t)(long, long*);
    McammGetBlackReferenceDataSize_t McammGetBlackReferenceDataSize;

    typedef long (*McammGetWhiteReferenceDataSize_t)(long, long*);
    McammGetWhiteReferenceDataSize_t McammGetWhiteReferenceDataSize;

    typedef long (*McammSetCameraBuffering_t)(long, bool);
    McammSetCameraBuffering_t McammSetCameraBuffering;

    typedef long (*McammGetCurrentCameraBuffering_t)(long, bool*);
    McammGetCurrentCameraBuffering_t McammGetCurrentCameraBuffering;

    typedef long (*McammSetColorMatrixOptimizationMode_t)(long, MCammColorMatrixOptimizationMode);
    McammSetColorMatrixOptimizationMode_t McammSetColorMatrixOptimizationMode;

    typedef long (*McammGetColorMatrixOptimizationMode_t)(long, MCammColorMatrixOptimizationMode*);
    McammGetColorMatrixOptimizationMode_t McammGetColorMatrixOptimizationMode;

    typedef long (*McammEnableHardwareTrigger_t)(long, BOOL);
    McammEnableHardwareTrigger_t McammEnableHardwareTrigger;

    typedef long (*McammSetHardwareTriggerMode_t)(long, BOOL, BOOL);
    McammSetHardwareTriggerMode_t McammSetHardwareTriggerMode;

    typedef long (*McammSetGPOSource_t)(long, long, MCammGPOSource);
    McammSetGPOSource_t McammSetGPOSource;

    typedef long (*McammSetGPOSettings_t)(long, long, long, long, bool);
    McammSetGPOSettings_t McammSetGPOSettings;

    typedef long (*McammInfo_t)(long, SMCAMINFO*);
    McammInfo_t McammInfo;

    typedef long (*McammSetWhiteBalance_t)(long, double, double, double);
    McammSetWhiteBalance_t McammSetWhiteBalance;

    typedef long (*McammSetSoftwareTrigger_t)(long, BOOL);
    McammSetSoftwareTrigger_t McammSetSoftwareTrigger;

    typedef long (*McammGetSoftwareTrigger_t)(long, BOOL*);
    McammGetSoftwareTrigger_t McammGetSoftwareTrigger;

    typedef long (*McammTriggerReady_t)(long, BOOL*);
    McammTriggerReady_t McammTriggerReady;

    typedef long (*McammExecuteSoftwareTrigger_t)(long);
    McammExecuteSoftwareTrigger_t McammExecuteSoftwareTrigger;

    typedef long (*McammSetTriggerWaitFrameDelay_t)(long, BOOL);
    McammSetTriggerWaitFrameDelay_t McammSetTriggerWaitFrameDelay;

    typedef long (*McammGetTriggerWaitFrameDelay_t)(long, BOOL*);
    McammGetTriggerWaitFrameDelay_t McammGetTriggerWaitFrameDelay;

    typedef long (*McammGetMaxRawImageDataSize_t)(long, long *);
    McammGetMaxRawImageDataSize_t McammGetMaxRawImageDataSize;

    typedef long (*McammGetWhiteRefImage_t)(long, unsigned short *, long, McamImageProcEx, void *);
    McammGetWhiteRefImage_t McammGetWhiteRefImage;

    typedef long (*McammCalculateWhiteRefFromImage_t)(long, unsigned short *, long);
    McammCalculateWhiteRefFromImage_t McammCalculateWhiteRefFromImage;

    void loadLibrary();
    void unloadLibrary();

};

#endif /* MCAM_HPP_ */
