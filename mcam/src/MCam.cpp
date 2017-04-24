/*
 * MCam.cpp
 *
 *  Created on: Oct 21, 2011
 *      Author: horst
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#include "mcam.h"
#include <MCam.hpp>

#if defined (_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif /* defined (_WIN32) */

MCam::MCam()
{
    loadLibrary();
}

MCam::~MCam()
{
    unloadLibrary();
}

void MCam::loadLibrary()
{
  MCAM_LOG_INIT("MCam::loadLibrary")

    // loading 'AxIP' library
#if defined (_WIN32)
    HINSTANCE handle = LoadLibrary("axcam64.dll");
#else
    void* handle = dlopen("libaxcam64.so", RTLD_LAZY);
#endif /* defined (_WIN32) */

    if (handle != NULL) {
      // loading 'McammSetSessionProperties' function
#if defined (_WIN32)
        McammGetSessionProperties = (McammGetSessionProperties_t) GetProcAddress((HMODULE)handle, "McammGetSessionProperties");
#else
        McammGetSessionProperties = (McammGetSessionProperties_t) dlsym(handle, "McammGetSessionProperties");
#endif
        if (!McammGetSessionProperties) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetSessionProperties' function." );
        }

        // loading 'McammSetSessionProperties' function
  #if defined (_WIN32)
          McammSetSessionProperties = (McammSetSessionProperties_t) GetProcAddress((HMODULE)handle, "McammSetSessionProperties");
  #else
          McammSetSessionProperties = (McammSetSessionProperties_t) dlsym(handle, "McammSetSessionProperties");
  #endif
          if (!McammSetSessionProperties) {
              MCAM_LOGF_ERROR("Unable to load 'McammSetSessionProperties' function." );
          }


        // loading 'McammLibInit' function
#if defined (_WIN32)
        McammLibInit = (McammLibInit_t) GetProcAddress((HMODULE)handle, "McammLibInit");
#else
        McammLibInit = (McammLibInit_t) dlsym(handle, "McammLibInit");
#endif
        if (!McammLibInit) {
            MCAM_LOGF_ERROR("Unable to load 'McammLibInit' function." );
        }

        // loading 'McammLibTerm' function
#if defined (_WIN32)
        McammLibTerm = (McammLibTerm_t) GetProcAddress((HMODULE)handle, "McammLibTerm");
#else
        McammLibTerm = (McammLibTerm_t) dlsym(handle, "McammLibTerm");
#endif
        if (!McammLibTerm) {
            MCAM_LOGF_ERROR("Unable to load 'McammLibTerm' function." );
        }

        // loading 'McammGetCurrentImageDataSize' function
#if defined (_WIN32)
        McammGetCurrentImageDataSize = (McammGetCurrentImageDataSize_t) GetProcAddress((HMODULE)handle, "McammGetCurrentImageDataSize");
#else
        McammGetCurrentImageDataSize = (McammGetCurrentImageDataSize_t) dlsym(handle,
                "McammGetCurrentImageDataSize");
#endif
        if (!McammGetCurrentImageDataSize) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentImageDataSize' function." );
        }

        // loading 'McamGetNumberofCameras' function
#if defined (_WIN32)
        McamGetNumberofCameras = (McamGetNumberofCameras_t) GetProcAddress((HMODULE)handle, "McamGetNumberofCameras");
#else
        McamGetNumberofCameras = (McamGetNumberofCameras_t) dlsym(handle, "McamGetNumberofCameras");
#endif
        if (!McamGetNumberofCameras) {
            MCAM_LOGF_ERROR("Unable to load 'McamGetNumberofCameras' function." );
        }

        // loading 'McammInit' function
#if defined (_WIN32)
        McammInit = (McammInit_t) GetProcAddress((HMODULE)handle, "McammInit");
#else
        McammInit = (McammInit_t) dlsym(handle, "McammInit");
#endif
        if (!McammInit) {
            MCAM_LOGF_ERROR("Unable to load 'McammInit' function." );
        }

        // loading 'McammClose' function
#if defined (_WIN32)
        McammClose = (McammClose_t) GetProcAddress((HMODULE)handle, "McammClose");
#else
        McammClose = (McammClose_t) dlsym(handle, "McammClose");
#endif
        if (!McammClose) {
            MCAM_LOGF_ERROR("Unable to load 'McammClose' function." );
        }

        // loading 'McammSetBinning' function
#if defined (_WIN32)
        McammSetBinning = (McammSetBinning_t) GetProcAddress((HMODULE)handle, "McammSetBinning");
#else
        McammSetBinning = (McammSetBinning_t) dlsym(handle, "McammSetBinning");
#endif
        if (!McammSetBinning) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetBinning' function." );
        }

        // loading 'McammGetCurrentBinning' function
#if defined (_WIN32)
        McammGetCurrentBinning = (McammGetCurrentBinning_t) GetProcAddress((HMODULE)handle, "McammGetCurrentBinning");
#else
        McammGetCurrentBinning = (McammGetCurrentBinning_t) dlsym(handle, "McammGetCurrentBinning");
#endif
        if (!McammGetCurrentBinning) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentBinning' function." );
        }

        // loading 'McammSetResolution' function
#if defined (_WIN32)
        McammSetResolution = (McammSetResolution_t) GetProcAddress((HMODULE)handle, "McammSetResolution");
#else
        McammSetResolution = (McammSetResolution_t) dlsym(handle, "McammSetResolution");
#endif
        if (!McammSetResolution) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetResolution' function." );
        }

        // loading 'McammGetResolutionValues' function
#if defined (_WIN32)
        McammGetResolutionValues = (McammGetResolutionValues_t) GetProcAddress((HMODULE)handle, "McammGetResolutionValues");
#else
        McammGetResolutionValues = (McammGetResolutionValues_t) dlsym(handle, "McammGetResolutionValues");
#endif
        if (!McammGetResolutionValues) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetResolutionValues' function." );
        }

        // loading 'McammGetCurrentBitsPerPixel' function
#if defined (_WIN32)
        McammGetCurrentBitsPerPixel = (McammGetCurrentBitsPerPixel_t) GetProcAddress((HMODULE)handle, "McammGetCurrentBitsPerPixel");
#else
        McammGetCurrentBitsPerPixel = (McammGetCurrentBitsPerPixel_t) dlsym(handle, "McammGetCurrentBitsPerPixel");
#endif
        if (!McammGetCurrentBitsPerPixel) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentBitsPerPixel' function." );
        }

        // loading 'McammGetCurrentFrameSize' function
#if defined (_WIN32)
        McammGetCurrentFrameSize = (McammGetCurrentFrameSize_t) GetProcAddress((HMODULE)handle, "McammGetCurrentFrameSize");
#else
        McammGetCurrentFrameSize = (McammGetCurrentFrameSize_t) dlsym(handle,
                "McammGetCurrentFrameSize");
#endif
        if (!McammGetCurrentFrameSize) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentFrameSize' function." );
        }

        // loading 'McammSetFrameSize' function
#if defined (_WIN32)
        McammSetFrameSize = (McammSetFrameSize_t) GetProcAddress((HMODULE)handle, "McammSetFrameSize");
#else
        McammSetFrameSize = (McammSetFrameSize_t) dlsym(handle, "McammSetFrameSize");
#endif
        if (!McammSetFrameSize) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetFrameSize' function." );
        }

        // loading 'McammGetCurrentExposure' function
#if defined (_WIN32)
        McammGetCurrentExposure = (McammGetCurrentExposure_t) GetProcAddress((HMODULE)handle, "McammGetCurrentExposure");
#else
        McammGetCurrentExposure = (McammGetCurrentExposure_t) dlsym(handle,
                "McammGetCurrentExposure");
#endif
        if (!McammGetCurrentExposure) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentExposure' function." );
        }

        // loading 'McammSetExposure' function
#if defined (_WIN32)
        McammSetExposure = (McammSetExposure_t) GetProcAddress((HMODULE)handle, "McammSetExposure");
#else
        McammSetExposure = (McammSetExposure_t) dlsym(handle, "McammSetExposure");
#endif
        if (!McammSetExposure) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetExposure' function." );
        }

        // loading 'McammAcquisitionEx' function
#if defined (_WIN32)
        McammAcquisitionEx = (McammAcquisitionEx_t) GetProcAddress((HMODULE)handle, "McammAcquisitionEx");
#else
        McammAcquisitionEx = (McammAcquisitionEx_t) dlsym(handle, "McammAcquisitionEx");
#endif
        if (!McammAcquisitionEx) {
            MCAM_LOGF_ERROR("Unable to load 'McammAcquisitionEx' function." );
        }

        // loading 'McammGetIPInfo' function
#if defined (_WIN32)
        McammGetIPInfo = (McammGetIPInfo_t) GetProcAddress((HMODULE)handle, "McammGetIPInfo");
#else
        McammGetIPInfo = (McammGetIPInfo_t) dlsym(handle, "McammGetIPInfo");
#endif
        if (!McammGetIPInfo) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetIPInfo' function." );
        }

        // loading 'McammExecuteIPFunction' function
#if defined (_WIN32)
        McammExecuteIPFunction = (McammExecuteIPFunction_t) GetProcAddress((HMODULE)handle, "McammExecuteIPFunction");
#else
        McammExecuteIPFunction = (McammExecuteIPFunction_t) dlsym(handle, "McammExecuteIPFunction");
#endif
        if (!McammExecuteIPFunction) {
            MCAM_LOGF_ERROR("Unable to load 'McammExecuteIPFunction' function." );
        }

        // loading 'McammStartContinuousAcquisition' function
#if defined (_WIN32)
        McammStartContinuousAcquisition = (McammStartContinuousAcquisition_t) GetProcAddress((HMODULE)handle, "McammStartContinuousAcquisition");
#else
        McammStartContinuousAcquisition = (McammStartContinuousAcquisition_t) dlsym(handle, "McammStartContinuousAcquisition");
#endif
        if (!McammGetIPInfo) {
            MCAM_LOGF_ERROR("Unable to load 'McammStartContinuousAcquisition' function." );
        }

        // loading 'McammStopContinuousAcquisition' function
#if defined (_WIN32)
        McammStopContinuousAcquisition = (McammStopContinuousAcquisition_t) GetProcAddress((HMODULE)handle, "McammStopContinuousAcquisition");
#else
        McammStopContinuousAcquisition = (McammStopContinuousAcquisition_t) dlsym(handle, "McammStopContinuousAcquisition");
#endif
        if (!McammGetIPInfo) {
            MCAM_LOGF_ERROR("Unable to load 'McammStopContinuousAcquisition' function." );
        }

        // loading 'McammSetFrameDelay' function
#if defined (_WIN32)
        McammSetFrameDelay = (McammSetFrameDelay_t) GetProcAddress((HMODULE)handle, "McammSetFrameDelay");
#else
        McammSetFrameDelay = (McammSetFrameDelay_t) dlsym(handle, "McammSetFrameDelay");
#endif
        if (!McammSetFrameDelay) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetFrameDelay' function." );
        }


        // loading 'McammGetCurrentFrameTime' function
#if defined (_WIN32)
        McammGetCurrentFrameTime = (McammGetCurrentFrameTime_t) GetProcAddress((HMODULE)handle, "McammGetCurrentFrameTime");
#else
        McammGetCurrentFrameTime = (McammGetCurrentFrameTime_t) dlsym(handle, "McammGetCurrentFrameTime");
#endif
        if (!McammGetCurrentFrameTime) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentFrameTime' function." );
        }


        // loading 'McammCalculateBlackRefEx' function
#if defined (_WIN32)
        McammCalculateBlackRefEx = (McammCalculateBlackRefEx_t) GetProcAddress((HMODULE)handle, "McammCalculateBlackRefEx");
#else
        McammCalculateBlackRefEx = (McammCalculateBlackRefEx_t) dlsym(handle, "McammCalculateBlackRefEx");
#endif
        if (!McammCalculateBlackRefEx) {
            MCAM_LOGF_ERROR("Unable to load 'McammCalculateBlackRefEx' function." );
        }

        // loading 'McammSetBlackRef' function
#if defined (_WIN32)
        McammSetBlackRef = (McammSetBlackRef_t) GetProcAddress((HMODULE)handle, "McammSetBlackRef");
#else
        McammSetBlackRef = (McammSetBlackRef_t) dlsym(handle, "McammSetBlackRef");
#endif
        if (!McammSetBlackRef) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetBlackRef' function." );
        }

        // loading 'McammSaveBlackRefEx' function
#if defined (_WIN32)
        McammSaveBlackRef = (McammSaveBlackRef_t) GetProcAddress((HMODULE)handle, "McammSaveBlackRef");
#else
        McammSaveBlackRef = (McammSaveBlackRef_t) dlsym(handle, "McammSaveBlackRef");
#endif
        if (!McammSaveBlackRef) {
            MCAM_LOGF_ERROR("Unable to load 'McammSaveBlackRef' function." );
        }

        // loading 'McammRestoreBlackRefEx' function
#if defined (_WIN32)
        McammRestoreBlackRef = (McammRestoreBlackRef_t) GetProcAddress((HMODULE)handle, "McammRestoreBlackRef");
#else
        McammRestoreBlackRef = (McammRestoreBlackRef_t) dlsym(handle, "McammRestoreBlackRef");
#endif
        if (!McammRestoreBlackRef) {
            MCAM_LOGF_ERROR("Unable to load 'McammRestoreBlackRef' function." );
        }

        // loading 'McammCalculateWhiteRefEx' function
#if defined (_WIN32)
        McammCalculateWhiteRefEx = (McammCalculateWhiteRefEx_t) GetProcAddress((HMODULE)handle, "McammCalculateWhiteRefEx");
#else
        McammCalculateWhiteRefEx = (McammCalculateWhiteRefEx_t) dlsym(handle, "McammCalculateWhiteRefEx");
#endif
        if (!McammCalculateWhiteRefEx) {
            MCAM_LOGF_ERROR("Unable to load 'McammCalculateWhiteRefEx' function." );
        }

        // loading 'McammSetWhiteRef' function
#if defined (_WIN32)
        McammSetWhiteRef = (McammSetWhiteRef_t) GetProcAddress((HMODULE)handle, "McammSetWhiteRef");
#else
        McammSetWhiteRef = (McammSetWhiteRef_t) dlsym(handle, "McammSetWhiteRef");
#endif
        if (!McammSetWhiteRef) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetWhiteRef' function." );
        }

        // loading 'McammGetWhiteTable' function
#if defined (_WIN32)
        McammGetWhiteTable = (McammGetWhiteTable_t) GetProcAddress((HMODULE)handle, "McammGetWhiteTable");
#else
        McammGetWhiteTable = (McammGetWhiteTable_t) dlsym(handle, "McammGetWhiteTable");
#endif
        if (!McammGetWhiteTable) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetWhiteTable' function." );
        }

        // loading 'McammSetWhiteTable' function
#if defined (_WIN32)
        McammSetWhiteTable = (McammSetWhiteTable_t) GetProcAddress((HMODULE)handle, "McammSetWhiteTable");
#else
        McammSetWhiteTable = (McammSetWhiteTable_t) dlsym(handle, "McammSetWhiteTable");
#endif
        if (!McammSetWhiteTable) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetWhiteTable' function." );
        }

        // loading 'McammEnableColorProcessing' function
#if defined (_WIN32)
        McammEnableColorProcessing = (McammEnableColorProcessing_t) GetProcAddress((HMODULE)handle, "McammEnableColorProcessing");
#else
        McammEnableColorProcessing = (McammEnableColorProcessing_t) dlsym(handle, "McammEnableColorProcessing");
#endif
        if (!McammEnableColorProcessing) {
            MCAM_LOGF_ERROR("Unable to load 'McammEnableColorProcessing' function." );
        }

        // loading 'McammSetImageWhiteBalance' function
#if defined (_WIN32)
        McammSetImageWhiteBalance = (McammSetImageWhiteBalance_t) GetProcAddress((HMODULE)handle, "McammSetImageWhiteBalance");
#else
        McammSetImageWhiteBalance = (McammSetImageWhiteBalance_t) dlsym(handle, "McammSetImageWhiteBalance");
#endif
        if (!McammSetImageWhiteBalance) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetImageWhiteBalance' function." );
        }

        // loading 'McammSetBitsPerPixel' function
#if defined (_WIN32)
        McammSetBitsPerPixel = (McammSetBitsPerPixel_t) GetProcAddress((HMODULE)handle, "McammSetBitsPerPixel");
#else
        McammSetBitsPerPixel = (McammSetBitsPerPixel_t) dlsym(handle, "McammSetBitsPerPixel");
#endif
        if (!McammSetBitsPerPixel) {
            MCAM_LOGF_ERROR("Unable to load 'McammSetBitsPerPixel' function." );
        }

        // loading 'McammGetBlackReferenceDataSize' function
#if defined (_WIN32)
        McammGetBlackReferenceDataSize = (McammGetBlackReferenceDataSize_t) GetProcAddress((HMODULE)handle, "McammGetBlackReferenceDataSize");
#else
        McammGetBlackReferenceDataSize = (McammGetBlackReferenceDataSize_t) dlsym(handle, "McammGetBlackReferenceDataSize");
#endif
        if (!McammGetBlackReferenceDataSize) {
            MCAM_LOGF_ERROR("Unable to load 'McammGetBlackReferenceDataSize' function." );
        }

        // loading 'McammGetWhiteReferenceDataSize' function
#if defined (_WIN32)
        McammGetWhiteReferenceDataSize = (McammGetWhiteReferenceDataSize_t) GetProcAddress((HMODULE)handle, "McammGetWhiteReferenceDataSize");
#else
        McammGetWhiteReferenceDataSize = (McammGetWhiteReferenceDataSize_t) dlsym(handle, "McammGetWhiteReferenceDataSize");
#endif
	if (!McammGetWhiteReferenceDataSize) {
		MCAM_LOGF_ERROR("Unable to load 'McammGetWhiteReferenceDataSize' function." );
	}

// loading 'McammSetCameraBuffering' function
#if defined (_WIN32)
	McammSetCameraBuffering = (McammSetCameraBuffering_t) GetProcAddress((HMODULE)handle, "McammSetCameraBuffering");
#else
	McammSetCameraBuffering = (McammSetCameraBuffering_t) dlsym(handle, "McammSetCameraBuffering");
#endif
if (!McammSetCameraBuffering) {
	MCAM_LOGF_ERROR("Unable to load 'McammSetCameraBuffering' function." );
}

// loading 'McammGetCurrentCameraBuffering' function
#if defined (_WIN32)
McammGetCurrentCameraBuffering = (McammGetCurrentCameraBuffering_t) GetProcAddress((HMODULE)handle, "McammGetCurrentCameraBuffering");
#else
McammGetCurrentCameraBuffering = (McammGetCurrentCameraBuffering_t) dlsym(handle, "McammGetCurrentCameraBuffering");
#endif
if (!McammGetCurrentCameraBuffering) {
	MCAM_LOGF_ERROR("Unable to load 'McammGetCurrentCameraBuffering' function." );
}


// loading 'McammSetColorMatrixOptimizationMode' function
#if defined (_WIN32)
	McammSetColorMatrixOptimizationMode = (McammSetColorMatrixOptimizationMode_t) GetProcAddress((HMODULE)handle, "McammSetColorMatrixOptimizationMode");
#else
	McammSetColorMatrixOptimizationMode = (McammSetColorMatrixOptimizationMode_t) dlsym(handle, "McammSetColorMatrixOptimizationMode");
#endif
if (!McammSetColorMatrixOptimizationMode) {
	MCAM_LOGF_ERROR("Unable to load 'McammSetColorMatrixOptimizationMode' function." );
}

// loading 'McammSetColorMatrixOptimizationMode' function
#if defined (_WIN32)
	McammGetColorMatrixOptimizationMode = (McammGetColorMatrixOptimizationMode_t) GetProcAddress((HMODULE)handle, "McammGetColorMatrixOptimizationMode");
#else
	McammGetColorMatrixOptimizationMode = (McammGetColorMatrixOptimizationMode_t) dlsym(handle, "McammGetColorMatrixOptimizationMode");
#endif
if (!McammGetColorMatrixOptimizationMode) {
	MCAM_LOGF_ERROR("Unable to load 'McammGetColorMatrixOptimizationMode' function." );
}

// loading 'McammEnableHardwareTrigger' function
#if defined (_WIN32)
McammEnableHardwareTrigger = (McammEnableHardwareTrigger_t) GetProcAddress((HMODULE)handle, "McammEnableHardwareTrigger");
#else
McammEnableHardwareTrigger = (McammEnableHardwareTrigger_t) dlsym(handle, "McammEnableHardwareTrigger");
#endif
if (!McammEnableHardwareTrigger) {
	MCAM_LOGF_ERROR("Unable to load 'McammEnableHardwareTrigger' function." );
}

// loading 'McammSetHardwareTriggerMode' function
#if defined (_WIN32)
McammSetHardwareTriggerMode = (McammSetHardwareTriggerMode_t) GetProcAddress((HMODULE)handle, "McammSetHardwareTriggerMode");
#else
McammSetHardwareTriggerMode = (McammSetHardwareTriggerMode_t) dlsym(handle, "McammSetHardwareTriggerMode");
#endif
if (!McammSetHardwareTriggerMode) {
	MCAM_LOGF_ERROR("Unable to load 'McammSetHardwareTriggerMode' function." );
}


// loading 'McammSetGPOSettings' function
#if defined (_WIN32)
McammSetGPOSource = (McammSetGPOSource_t) GetProcAddress((HMODULE)handle, "McammSetGPOSource");
#else
McammSetGPOSource = (McammSetGPOSource_t) dlsym(handle, "McammSetGPOSource");
#endif
if (!McammSetGPOSource) {
	MCAM_LOGF_ERROR("Unable to load 'McammSetGPOSource' function." );
}


// loading 'McammSetGPOSettings' function
#if defined (_WIN32)
McammSetGPOSettings = (McammSetGPOSettings_t) GetProcAddress((HMODULE)handle, "McammSetGPOSettings");
#else
McammSetGPOSettings = (McammSetGPOSettings_t) dlsym(handle, "McammSetGPOSettings");
#endif
if (!McammSetGPOSettings) {
	MCAM_LOGF_ERROR("Unable to load 'McammSetGPOSettings' function." );
}

// loading 'McammInfo' function
#if defined (_WIN32)
McammInfo = (McammInfo_t) GetProcAddress((HMODULE)handle, "McammInfo");
#else
McammInfo = (McammInfo_t) dlsym(handle, "McammInfo");
#endif
if (!McammInfo) {
	MCAM_LOGF_ERROR("Unable to load 'McammInfo' function." );
}

// loading 'McammSetWhiteBalance' function
#if defined (_WIN32)
McammSetWhiteBalance = (McammSetWhiteBalance_t) GetProcAddress((HMODULE)handle, "McammSetWhiteBalance");
#else
McammSetWhiteBalance = (McammSetWhiteBalance_t) dlsym(handle, "McammSetWhiteBalance");
#endif
if (!McammSetWhiteBalance) {
  MCAM_LOGF_ERROR("Unable to load 'McammSetWhiteBalance' function." );
}

// loading 'McammSetSoftwareTrigger' function
#if defined (_WIN32)
McammSetSoftwareTrigger = (McammSetSoftwareTrigger_t) GetProcAddress((HMODULE)handle, "McammSetSoftwareTrigger");
#else
McammSetSoftwareTrigger = (McammSetSoftwareTrigger_t) dlsym(handle, "McammSetSoftwareTrigger");
#endif
if (!McammSetSoftwareTrigger) {
  MCAM_LOGF_ERROR("Unable to load 'McammSetSoftwareTrigger' function." );
}

// loading 'McammGetSoftwareTrigger' function
#if defined (_WIN32)
McammGetSoftwareTrigger = (McammGetSoftwareTrigger_t) GetProcAddress((HMODULE)handle, "McammGetSoftwareTrigger");
#else
McammGetSoftwareTrigger = (McammGetSoftwareTrigger_t) dlsym(handle, "McammGetSoftwareTrigger");
#endif
if (!McammGetSoftwareTrigger) {
  MCAM_LOGF_ERROR("Unable to load 'McammGetSoftwareTrigger' function." );
}

// loading 'McammTriggerReady' function
#if defined (_WIN32)
McammTriggerReady = (McammTriggerReady_t) GetProcAddress((HMODULE)handle, "McammTriggerReady");
#else
McammTriggerReady = (McammTriggerReady_t) dlsym(handle, "McammTriggerReady");
#endif
if (!McammTriggerReady) {
  MCAM_LOGF_ERROR("Unable to load 'McammTriggerReady' function." );
}

// loading 'McammExecuteSoftwareTrigger' function
#if defined (_WIN32)
McammExecuteSoftwareTrigger = (McammExecuteSoftwareTrigger_t) GetProcAddress((HMODULE)handle, "McammExecuteSoftwareTrigger");
#else
McammExecuteSoftwareTrigger = (McammExecuteSoftwareTrigger_t) dlsym(handle, "McammExecuteSoftwareTrigger");
#endif
if (!McammExecuteSoftwareTrigger) {
  MCAM_LOGF_ERROR("Unable to load 'McammExecuteSoftwareTrigger' function." );
}

// loading 'McammSetTriggerWaitFrameDelay' function
#if defined (_WIN32)
McammSetTriggerWaitFrameDelay = (McammSetTriggerWaitFrameDelay_t) GetProcAddress((HMODULE)handle, "McammSetTriggerWaitFrameDelay");
#else
McammSetTriggerWaitFrameDelay = (McammSetTriggerWaitFrameDelay_t) dlsym(handle, "McammSetTriggerWaitFrameDelay");
#endif
if (!McammSetTriggerWaitFrameDelay) {
  MCAM_LOGF_ERROR("Unable to load 'McammSetTriggerWaitFrameDelay' function." );
}

// loading 'McammGetTriggerWaitFrameDelay' function
#if defined (_WIN32)
McammGetTriggerWaitFrameDelay = (McammGetTriggerWaitFrameDelay_t) GetProcAddress((HMODULE)handle, "McammGetTriggerWaitFrameDelay");
#else
McammGetTriggerWaitFrameDelay = (McammGetTriggerWaitFrameDelay_t) dlsym(handle, "McammGetTriggerWaitFrameDelay");
#endif
if (!McammGetTriggerWaitFrameDelay) {
  MCAM_LOGF_ERROR("Unable to load 'McammGetTriggerWaitFrameDelay' function." );
}

// loading 'McammGetMaxRawImageDataSize' function
#if defined (_WIN32)
McammGetMaxRawImageDataSize = (McammGetMaxRawImageDataSize_t) GetProcAddress((HMODULE)handle, "McammGetMaxRawImageDataSize");
#else
McammGetMaxRawImageDataSize = (McammGetMaxRawImageDataSize_t) dlsym(handle, "McammGetMaxRawImageDataSize");
#endif
if (!McammGetMaxRawImageDataSize) {
  MCAM_LOGF_ERROR("Unable to load 'McammGetMaxRawImageDataSize' function." );
}

// loading 'McammGetWhiteRefImage' function
#if defined (_WIN32)
McammGetWhiteRefImage = (McammGetWhiteRefImage_t) GetProcAddress((HMODULE)handle, "McammGetWhiteRefImage");
#else
McammGetWhiteRefImage = (McammGetWhiteRefImage_t) dlsym(handle, "McammGetWhiteRefImage");
#endif
if (!McammGetWhiteRefImage) {
  MCAM_LOGF_ERROR("Unable to load 'McammGetWhiteRefImage' function." );
}

// loading 'McammCalculateWhiteRefFromImage' function
#if defined (_WIN32)
McammCalculateWhiteRefFromImage = (McammCalculateWhiteRefFromImage_t) GetProcAddress((HMODULE)handle, "McammCalculateWhiteRefFromImage");
#else
McammCalculateWhiteRefFromImage = (McammCalculateWhiteRefFromImage_t) dlsym(handle, "McammCalculateWhiteRefFromImage");
#endif
if (!McammCalculateWhiteRefFromImage) {
  MCAM_LOGF_ERROR("Unable to load 'McammCalculateWhiteRefFromImage' function." );
}

////////////////////////////////////////////////////////////////////////////////////
    } else {
#if defined (_WIN32)
        MCAM_LOGF_ERROR("Unable to load library: %d" ,GetLastError() );
#else
        MCAM_LOGF_ERROR("Unable to load library: %d ", dlerror() );
#endif
    }

}

void MCam::unloadLibrary()
{

#if defined (_WIN32)
    FreeLibrary((HMODULE)handle);
#else
    // dlclose(handle);
#endif /* defined (_WIN32) */
}
