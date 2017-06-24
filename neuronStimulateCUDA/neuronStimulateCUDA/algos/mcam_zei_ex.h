/*
 * mcam_zei_ex.h
 *
 *  Created on: Mar 22, 2012
 *      Author: horst
 */

#ifndef MCAM_ZEI_EX_H_
#define MCAM_ZEI_EX_H_

#ifndef NOERR
#include <mcam_zei.h>
#endif

#if defined (__cplusplus)

extern "C" {

#endif /* defined (__cplusplus) */

#define MCAM_DB_PER_LSB 0.0359 // constant from AD9970 data sheet
#define MCAM_GAIN1 0
#define MCAM_GAIN2 168 // derived from above (6.0206dB / 0.0359 = 167,7)
#define MCAM_GAIN3 266 // derived from above (9.5424dB / 0.0359 = 265,8)

#define CMOS_DB_PER_LSB 0.1 // constant from IMX174 data sheet
#define CMOS_GAIN1 0
#define CMOS_GAIN2 120
#define CMOS_GAIN3 240

#define PACKED
#pragma pack(push,1)

// header values

#define MCAM_HEADER_VERSION 2

// header signature
#define MCAM_HEADER_SIGNATURE 0x41474852
#define MCAM_HEADER_SIGNATURE_TAIL 0xFFFFFFFF

// backend streaming mode
#define MCAM_STREAMING_MODE_SPLIT_Y_MASK 0x00000001
#define MCAM_STREAMING_MODE_SQRT_MASK 0x00000002

// pixel format, as it comes from the camera
#define MCAM_PIXEL_FORMAT_MONO_8 0
#define MCAM_PIXEL_FORMAT_BAYER_RG_8 1   // ICX674
#define MCAM_PIXEL_FORMAT_BAYER_GR_8 2   // ICX694
#define MCAM_PIXEL_FORMAT_MONO_14 4
#define MCAM_PIXEL_FORMAT_BAYER_RG_14 5  // ICX674
#define MCAM_PIXEL_FORMAT_BAYER_GR_14 6  // ICX694
#define MCAM_PIXEL_FORMAT_BAYER_GR_16 3
// CMOS Start
#define MCAM_PIXEL_FORMAT_MONO_12 7
#define MCAM_PIXEL_FORMAT_BAYER_RG_12 8
#define MCAM_PIXEL_FORMAT_BAYER_GR_12 9
// CMOS END
#define MCAM_PIXEL_FORMAT_BAYER_BG_14 10 // ICX834
#define MCAM_PIXEL_FORMAT_BAYER_BG_8 11  // ICX834

// bits per pixel, as it changes during processing
#define MCAM_BPP_SQRT 8              // before expansion to 14bit linear
#define MCAM_BPP_IMX12_PACKED 12     // CMOS before expansion to 14bit linear
#define MCAM_BPP_MONO 14             // final size for b/w, intermediate size for color
#define MCAM_BPP_COLOR 42            // after demosaicing
// image processing header flags

// decoding
#define MCAM_IP_MIRRORED 0x00000001
#define MCAM_IP_DEINTERLACED 0x00000002

// preprocessing (inplace)
#define MCAM_IP_SWBINNED 0x00000004
#define MCAM_IP_TILESADJUSTED 0x00000008

// processing
#define MCAM_IP_BLACKREFAPPLIED 0x00000010
#define MCAM_IP_WHITEREFAPPLIED 0x00000020
#define MCAM_IP_BLEMISHESCORRECTED 0x00000040
#define MCAM_IP_DEMOSAICED_LOWQ 0x00000080
#define MCAM_IP_DEMOSAICED_MEDIUMQ 0x00000100
#define MCAM_IP_DEMOSAICED_HIGHQ 0x00000200
#define MCAM_IP_COLORMATRIXAPPLIED 0x00000400
#define MCAM_IP_TOMONOCONVERTED 0x00000800

// error handling
#define MCAM_IP_VALIDDATAMASK 0x3FFF
#define MCAM_IP_INVALIDDATAMASK 0xC000

typedef struct _IMAGE_HEADER
{
    unsigned int signature;
    unsigned int headerVersion;
    unsigned int headerSize;       // in bytes
    unsigned int reserved;

    unsigned int numWordsPerLineLeft;   // Effective number of pixel left half with binning
    unsigned int numWordsPerLineRight;  // Effective number of pixel right half with binning
                                    // (zero if not used)

    unsigned int streamingMode;    // info about image streaming (sqrt, splitY)
    unsigned int numLines;         // no of lines transferred per tile

    unsigned int numPackets;       // no of packets transferred for whole frame
    unsigned int fillerSize;       // size of padding data in bytes

    unsigned int blackLevelTopLeft;     // black level * (binY-1) per tile
    unsigned int blackLevelTopRight;    // black level * (binY-1) per tile

    unsigned int blackLevelBottomLeft;  // black level * (binY-1) per tile
    unsigned int blackLevelBottomRight; // black level * (binY-1) per tile
    unsigned int binX;             // horizontal binning factor
    unsigned int blackLevel;
    unsigned int binY;             // vertical binning factor

    // ccdWidth/ccdHeight in sensor coordinates
    unsigned int ccdWidth;         // total number of active columns on CCD
    unsigned int ccdHeight;        // total number of active lines on CCD

    // ROI in sensor coordinates
    unsigned int roiX;             // CCD's active area upper left corner
    unsigned int roiY;             // ROI's upper left corner relative to
    unsigned int roiLeftWidth;     // ROI's width on CCD's left half
    unsigned int roiRightWidth;    // ROI's width of CCD's right half (zero if not used)
    unsigned int roiTopHeight;     // ROI's height on CCD's upper half
    unsigned int roiBottomHeight;  // ROI's height on CCD's lower half (zero if not used)

    // pixel representation
    unsigned int pixelFormat;
    unsigned int bitsPerPixel;
    unsigned int bytesPerPixel;    // given as fraction to support packed formats:
                               // numerator in upper, denominator in lower 16 bits

    // camera information
    // index=28
    unsigned int cameraId;         // color type in leftmost nibble (0: mono, 1: color)
                               // ccd selector in next 12 bits,
                               // serial number in rightmost 16 bits
    unsigned int cameraHwVersion;  // hardware version (from EEPROM)
    unsigned int cameraFwVersion;  // firmware version (from firmware header file "cam.h")
    unsigned int cameraVersion;    // AXIOCAM_DELIVERY_TAG (written by axif)

    // image information

    // ------------ index position of the following values in header must correspondent to HW --START
    // written in HW see header.vhd
    unsigned int timestamp;        // exposure start, one microsecond resolution
    unsigned int exposureDelay;    // lower 24 bits: time in microseconds exposure of this image
                               //                was delayed
                               // upper 8 bits:  reason for the delay
                               //                0 - MemAlmostFull
                               //                1 - HdrDone ISR took too long
    // ------------ index position in header must correspondent to HW --ENDS
    // index=34
    unsigned int imageNumber;
    signed int  sequenceLength;   // number of images for multiShots
    unsigned int acqMode;
    unsigned int triggerMode;
    unsigned int pixelClock;       // pixel clock for readout in Hertz
    unsigned int exposureTime;     // exposure time of this image in microseconds
    unsigned int gain;
    unsigned int roiWidth;
    unsigned int roiHeight;
    unsigned int testPatternTopLeft;        // here are the image test patterns (0x0000 ... 0xFFFF)
    unsigned int testPatternTopRight;       // contain 0xFFFFFFFF if not in test pattern mode
    unsigned int testPatternBottomLeft;     // note that test patterns may be modified
    unsigned int testPatternBottomRight;    // by horizontal hardware binning
    unsigned int ccdTemperature;  // current CCD temperature in 1/16th. °C
    unsigned int antiGlowStatus;  // 1 when antiGlow was active during exposure, 0 otherwise
    unsigned int readoutTime;     // readout time of this image in microseconds
    unsigned int frameTime;       // frame time of this image in microseconds
                              // user frame time limitation and bandwidth limitation
                              // is taken into account for this entry

    // image processing flags
    unsigned int imageProcessing;

    // original header values before sw changes
    unsigned int origPixelFormat;
    unsigned int origBitsPerPixel;
    unsigned int origBinX;

    // data error markers
    unsigned int syncErrorCount;

    ////// CAUTION ////////
    unsigned int signatureTail;  // signatureTail must stay at the same position
                             // otherwise shading and black reference cannot be restored in ZEN

    // moved two attributes to be compatible with older header versions since readoutTime and frameTime had been added
    unsigned int firstSyncErrorLine; // roi relative line number
    unsigned int lastSyncErrorLine;  // roi relative line number

    unsigned int finalBinXY;     // contains the binning factor to be applied to the image so that the image aspect ratio is kept
                             // note that the camera can support a different binning factor in HW. These values are stored
                             // in the binX, binY fields
    unsigned int sequencePosition; // image number in sequence starting with e.g. HDR Mode: 0 or 1

    unsigned int acquistionOptions; // acq. options like: HDR, sub-sampling
     // rest of header memory block is preset with zero
} IMAGE_HEADER PACKED;

typedef enum {
  mcammSessionDefault = 0, /**<Default behavior */
  mcammSessionAxioVisionCompatibility /**< Compatibility deviations for AxioVision */

} MCammSessionType;

#define MCAM_SESSION_PROPERTY_VERSION   0
#define MCAM_MAX_PATH_LEN            1024

typedef struct _McammSessionProperties {
  long propertyVersion;                         // Currently Latest = 0
  MCammSessionType sessionType;                 // default: mcammSessionDefault
  BOOL stripImageHeader;                        // default : false
  char logfilePath[MCAM_MAX_PATH_LEN];         // default='\0' ('\0' terminated)
  char propertyFilePath[MCAM_MAX_PATH_LEN];    // default='\0' ('\0' terminated)
  char reserved[1024]; // must be set to 0 -> memset(reserved, '\0', sizeof(reserved))
} McammSessionProperties;

#pragma pack(pop)
#undef PACKED

/**
 * Tile Adjustment Modes
 */
typedef enum {

  mcammTileAdjustmentOff = 0, /**< Disables tile adjustment */
  mcammTileAdjustmentLinear,  /**< Linear tile adjustment */
  mcammTileAdjustmentBiLinear /**< BiLinear tile adjustment */
} MCammTileAdjustmentMode;


typedef enum {

  mcammLineFlickerSuppressionOff = 0,	/**< Disables tile adjustment */
  mcammLineFlickerSuppressionLinear, 	/**< Linear tile adjustment */
  mcammLineFlickerSuppressionBiLinear   /**< BiLinear tile adjustment */
} MCammLineFlickerSuppressionMode;

/**
 * General Purpose Output Sources
 */
typedef enum {
  mcammGPOOff, /**< Disables the GPO (low if not inverted, high if inverted) */
  mcammGPOTriggered, /**< A trigger signal has been received, but the exposure not started yet */
  mcammGPOExposure, /**< An image is exposed */
  mcammGPOReadout, /**< The image sensor is read out */
  mcammGPOSyncTriggerReady, /**< An input trigger signal is accepted */
  mcammGPOAsyncTriggerReady /**< An input trigger signal is accepted immediately (without  line jitter) */
} MCammGPOSource;

typedef enum {
  mcammCameraAdded, mcammCameraRemoved
} McammCameraState;

/**
 * Predefined Color Temperatures
 */
typedef enum {
  mcamm3200k = 0, /**< tungsten */
  mcamm5500k, /**< daylight */
  mcammCustomWhitePoint, /**< custom */
} MCammColorTemperature;

/**
 * Readout Modes
 */
typedef enum {
  mcammSinglePortReadout = 0, /**< one quadrant */
  mcammDualPortReadout, /**< two quadrants, horizontal splitted image */
  mcammQuadPortReadout /**< four quadrants */
} MCammReadoutMode;

typedef void (*McammCameraListenerProc)(long cameraIndex,
    McammCameraState state);

typedef struct {
  bool locked;
  unsigned long startTimeUs;
  unsigned long stopTimeUs;
  unsigned long count;
  unsigned long long timeUs;
  unsigned long minTimeUs;
  unsigned long maxTimeUs;
} McammMetricsData;

typedef struct {
  McammMetricsData bufferExchangeTime;
  McammMetricsData busReadTime;
  McammMetricsData busWaitTime;
  McammMetricsData bufferCallbackTime;
  McammMetricsData decoding;
  McammMetricsData preprocessing;
  McammMetricsData tileAdjustment;
  McammMetricsData processing;
  McammMetricsData applyReferences;
  McammMetricsData removeBlemishes;
  McammMetricsData binAndDemosaic;
  McammMetricsData demosaic;
  McammMetricsData applyColorMatrix;
  McammMetricsData convertToMono;
  McammMetricsData bayerToMono;
} McammMetrics;

typedef enum {
  mcammNoOptimization = 0, /**< Off */
  mcammPipelineStageExceptMatrix, /**< Move all image processing but ColorMatrix to Dispatcher pipeline stage (*/
  mcammAllPipelineStage /**< Move all image processing to Dispatcher pipeline stage */
} MCammColorMatrixOptimizationMode;


// enum for camera features
typedef enum {
  mcammParmColor = 0,
  mcammParmROI,
  mcammParmExposure,
  mcammParmAnalogGain,
  mcammParmHDR,
  mcammParmBinning,              // 5
  mcammParmPixelClocks,
  mcammParmBlackReference,
  mcammParmWhiteReference,
  mcammParmCoolingSupport,
  mcammParmTemperatureSensor,     // 10
  mcammParmSyncHardwareTrigger,
  mcammParmAsyncHardwareTrigger,
  mcammParmIPQuality,
  mcammParmGPOs,
  mcammParmAntiGlow,             //15
  mcammParm8BitCompression,
  mcammParm10BitMode,
  mcammParmHighImageRateMode,
  mcammParmMultiplePortModes,
  mcammParmTileAdjustment,       // 20
  mcammParmLineFlickerSuppression,
  mcammParmPiezoCalibration,
  mcammParmPiezoScan,
  mcammParmSubSampling,
  mcammParmEnumSize              // 25
 } MCammSupportedParameterType;


/**
 * Get MCamm session properties
 *
 * @param pSessionProperties Pointer to user allocated session property
 *
 * @param byteSizeOfSessionProperties = sizeof(McammSessionProperties)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetSessionProperties(
    McammSessionProperties *pSessionProperties,
    long byteSizeOfSessionProperties);

/**
 * Set MCamm session properties
 *
 * To be called only before McammLibInit()
 *
 * Important: Always overwrite propertyVersion with MCAM_SESSION_PROPERTY_VERSION
 * even if read with McammGetSessionProperties
 *
 * @param pSessionProperties Pointer to user allocated session property
 *        propertyVersion         set to MCAM_SESSION_PROPERTY_VERSION
 *        MCammSessionType        default: mcammSessionDefault
 *        stripImageHeader        default : false
 *        reserved                set to '0'
 *
 * @param byteSizeOfSessionProperties = sizeof(McammSessionProperties)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetSessionProperties(
    McammSessionProperties *pSessionProperties,
    long byteSizeOfSessionProperties);

/**
 * Initialize the library with all it camera independent features.
 *
 * @param ipOnly True if library is only used for image post-processing.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammLibInit(bool ipOnly);

/**
 * Terminate the library with all it camera independent features.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammLibTerm();

/**
 * Sets the white balance according to the predefined EEPROM values.
 *
 * @param cameraIndex The selected camera.
 *
 * @param colorTemperature The predefined color temperature (EEPROM values).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetPredefinedWhiteBalance(long cameraIndex,
    MCammColorTemperature colorTemperature);

/**
 * Gets the current white balance and the according EEPROM values.
 *
 * @param cameraIndex The selected camera.
 *
 * @param colorTemperature Current color temperature.
 *
 * @param red Red color value of current white balance.
 *
 * @param green Green color value of current white balance.
 *
 * @param blue Blue color value of current white balance.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentPredefinedWhiteBalance(long cameraIndex,
    MCammColorTemperature* colorTemperature, double* red, double* green,
    double* blue);

/**
 * Gets the white balance color values for the given color temperature.
 *
 * @param cameraIndex The selected camera.
 *
 * @param colorTemperature Color temperature.
 *
 * @param red Red color value.
 *
 * @param green Green color value.
 *
 * @param blue Blue color value.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetPredefinedWhiteBalanceValues(long cameraIndex,
    MCammColorTemperature colorTemperature, double* red, double* green,
    double* blue);

/**
 * Stops the currently active single shot acquisition.
 *
 * @param cameraIndex The selected camera.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammStopCurrentAcquisition(long cameraIndex);

/**
 * Enables the multi shot acquisition mode. If activated, the continuous shot stops automatically after
 * the given count of acquired images.
 *
 * @param cameraIndex The selected camera.
 *
 * @param enabled 'True' to enable multi shot.
 *
 * @param imageCount The image count of multi shot acquisition.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammEnableMultiShot(long cameraIndex, bool enabled,
    long imageCount);

/**
 * Gets the current multi shot acquisition state.
 *
 * @param cameraIndex The selected camera.
 *
 * @param enabled 'True' if multi shot is enabled.
 *
 * @param imageCount The image count of multi shot acquisition.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammIsMultiShotEnabled(long cameraIndex, bool* enabled,
    long* imageCount);

/**
 * Gets the current frame time in microseconds.
 *
 * @param cameraIndex The selected camera.
 *
 * @param frameTimeMs Current frame time.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentFrameTime(long cameraindex,
    unsigned long* frameTimeMs);

/**
 * Calculate the size of black reference in bytes.
 *
 * @param cameraIndex The selected camera.
 *
 * @param size Black reference size pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetBlackReferenceDataSize(long cameraIndex,
    long* size);

/**
 * Calculate the size of calculated white reference in bytes.
 *
 * @param cameraIndex The selected camera.
 *
 * @param size White reference size pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetWhiteReferenceDataSize(long cameraIndex,
    long* size);

/**
 * Validate and adjust the frame size.
 *
 * @param cameraIndex The selected camera.
 *
 * @param frame The frame to validate.
 *
 * @param binning The binning factor.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammValidateFrameSize(long cameraIndex, RECT* frame,
    long binning);

/**
 * Calculate the size of current image in bytes.
 *
 * @param cameraIndex The selected camera.
 *
 * @param size Image size pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentImageDataSize(long cameraIndex,
    long* size);

/**
 * Calculate the size of current raw image in bytes.
 *
 * @param cameraindex The selected camera.
 *
 * @param size Image size pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentRawImageDataSize(long cameraindex,
    long* size);

/**
 * Calculate the maximum size of an image in bytes.
 *
 * @param cameraIndex The selected camera.
 *
 * @param size Image size pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetMaxImageDataSize(long cameraIndex, long* size);

/**
 * Gets current hardware trigger polarity.
 *
 * @param cameraIndex The selected camera.
 *
 * @param binvert True if inverted.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentHardwareTriggerPolarity(long cameraindex,
    BOOL* binvert);

/**
 * Changes the hardware trigger mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param toEdge True if edge mode, false if level mode.
 *
 * @param debounce True if trigger signal has to be debounced.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetHardwareTriggerMode(long cameraindex, BOOL toEdge,
    BOOL debounce);

/**
 * Gets current hardware trigger mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param toEdge True if edge mode, false if level mode.
 *
 * @param debounce True if trigger signal has to be debounced.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentHardwareTriggerMode(long cameraindex,
    BOOL* toEdge, BOOL* debounce);

/**
 * Sets the delay in microseconds between the incoming hardware trigger signal and the image exposure.
 * During the active delay no other trigger is accepted. It can be interrupted by changing the trigger
 * mode or by stopping the current acquisition.
 *
 * @param cameraIndex The selected camera.
 *
 * @param delay Delay time in microseconds.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetHardwareTriggerDelay(long cameraindex,
    long delay);

/**
 * Gets current hardware trigger delay.
 *
 * @param cameraIndex The selected camera.
 *
 * @param delay Delay time in microseconds.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentHardwareTriggerDelay(long cameraindex,
    long* delay);

/**
 * Gets the number of available parameter sets. The valid index for parameter sets ranges from 0 to (number - 1).
 *
 * @param cameraIndex The selected camera.
 *
 * @param number The number of parameter sets.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetNumberOfParameterSets(long cameraIndex,
    long* number);

/**
 * Select the parameter set used for next changes on relevant parameters.
 *
 * @param cameraIndex The selected camera.
 *
 * @param index The parameter set index.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetSelectedParameterSet(long cameraIndex,
    long index);

/**
 * Get the currently selected parameter set.
 *
 * @param cameraIndex The selected camera.
 *
 * @param selector Index pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetSelectedParameterSet(long cameraIndex,
    long *index);

/**
 * Activate the parameter set used for next image acquisitions.
 *
 * @param cameraIndex The selected camera.
 *
 * @param index The parameter set index.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetActiveParameterSet(long cameraIndex, long index);

/**
 * Get the currently active parameter set.
 *
 * @param cameraIndex The selected camera.
 *
 * @param selector Index pointer.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetActiveParameterSet(long cameraIndex, long *index);

/**
 * Copy all relevant values from one parameter set to another.
 *
 * @param cameraIndex The selected camera.
 *
 * @param sourceIndex Source parameter set index.
 *
 * @param destinationIndex Destination parameter set index.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammCopyParameterSet(long cameraIndex, long sourceIndex,
    long destinationIndex);

/**
 * Acquisition of a raw image (without pre- and post-processing).
 *
 * @param cameraindex The selected camera.
 *
 * @param data The array where the image is stored.
 *
 * @param size The size of the image data array.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammRawImageAcquisition(long cameraindex,
    unsigned short* data, long size);

/**
 * Sets the parameter set cycle count.
 *
 *     count == 0: Cycle: 0-0-0-0-...
 *     count == 1: Cycle: 1-1-1-1-...
 *     count == 2: Cycle: 1-2-1-2-...
 *     count == 3: Cycle: 1-2-3-1-...
 *
 * @param cameraIndex The selected camera.
 *
 * @param count The count of parameter set cycle.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetParameterSetCount(long cameraIndex, long count);

/**
 * Gets the number of available pixel clocks. The valid index for pixel clock ranges from 0 to (number - 1).
 *
 * @param cameraIndex The selected camera.
 *
 * @param number The number of pixel clock values.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetNumberOfPixelClocks(long cameraIndex,
    long* number);

/**
 * Gets the pixel clock value for given pixel clock.
 *
 * @param cameraIndex The selected camera.
 *
 * @param index The pixel clock index.
 *
 * @param value The pixel clock value in hertz.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetPixelClockValue(long cameraIndex, long index,
    long* value);

/**
 * Sets the pixel clock frequency by a selector.
 *
 * @param cameraIndex The selected camera.
 *
 * @param index The pixel clock index.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetPixelClock(long cameraIndex, long index);

/**
 * Gets currently activated pixel clock in hertz.
 *
 * @param cameraIndex The selected camera.
 *
 * @param index Current pixel clock index.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentPixelClock(long cameraIndex, long* index);

/**
 * Gets the desired sensor temperature range.
 *
 * @param cameraIndex The selected camera.
 *
 * @param minDegrees Minimum desired temperature in degrees.
 *
 * @param maxDegrees Maximum desired temperature in degrees.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetSensorTemperatureRange(long cameraIndex,
    long* minDegrees, long* maxDegrees);

/**
 * Sets the desired sensor temperature.
 *
 * @param cameraIndex The selected camera.
 *
 * @param degrees Desired temperature in degrees (see McammGetSensorTemperatureRange).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetSensorTemperature(long cameraIndex, long degrees);

/**
 * Gets the desired sensor temperature.
 *
 * @param cameraIndex The selected camera.
 *
 * @param degrees Desired temperature in degrees.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetSensorTemperature(long cameraIndex,
    long* degrees);

/**
 * Resets the desired sensor temperature to the default value.
 *
 * @param cameraIndex The selected camera.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammResetSensorTemperature(long cameraIndex);

/**
 * Gets the current sensor temperature.
 *
 * @param cameraIndex The selected camera.
 *
 * @param degrees Current temperature in degrees.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentSensorTemperature(long cameraIndex,
    float* degrees);

/**
 * Checks if the given readout mode is available.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mode Readout mode to be checked.
 *
 * @param available True if the mode is available.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammHasReadoutMode(long cameraIndex,
    MCammReadoutMode mode, bool* available);

/**
 * Gets info about currently used readout mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mode Currently active readout mode.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentReadoutMode(long cameraIndex,
    MCammReadoutMode* mode);

/**
 * Sets the readout mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mode Readout mode.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetReadoutMode(long cameraIndex,
    MCammReadoutMode mode);

/// FOR BACKWARD COMPATIBILITY ONLY ///

/**
 * Gets info about available sensor taps.
 *
 * @param cameraIndex The selected camera.
 *
 * @param right Right hand side sensor taps.
 *
 * @param bottom Bottom sensor taps.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammHasSensorTaps(long cameraIndex, bool* right,
    bool* bottom);

/**
 * Gets info about currently used sensor taps.
 *
 * @param cameraIndex The selected camera.
 *
 * @param right Right hand side sensor taps.
 *
 * @param bottom Bottom sensor taps.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammCurrentUsedSensorTaps(long cameraIndex, bool* right,
    bool* bottom);

/**
 * Enables or disables available sensor taps.
 *
 * @param cameraIndex The selected camera.
 *
 * @param right Right hand side sensor taps.
 *
 * @param bottom Bottom sensor taps.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammUseSensorTaps(long cameraIndex, bool right,
    bool bottom);

/// FOR BACKWARD COMPATIBILITY ONLY ///

/**
 * Enables the image adjustment of visible sensor tile borders.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mode Enable/disable tile adjustment.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetTileAdjustmentMode(long cameraIndex,
    MCammTileAdjustmentMode mode);

/**
 * Gets current tile adjustment mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mode Current tile adjustment mode.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetTileAdjustmentMode(long cameraIndex,
    MCammTileAdjustmentMode* mode);

/**
 * Sets the power state of device frontend (sensor and analog components)
 * and/or forced cooling (available only with cable on second USB port).
 * In the auto power mode the frontend is automatically switched on or off
 * on camera initialization or termination.
 *
 * @param cameraIndex The selected camera.
 *
 * @param powerOn Enable/disable frontend power.
 *
 * @param autoPower Enable/disable automatic frontend power.
 *
 * @param forceCooling Force cooling.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetPowerState(long cameraIndex, bool powerOn,
    bool autoPower, bool forceCooling);

/**
 * Gets device power settings.
 *
 * @param cameraIndex The selected camera.
 *
 * @param powerOn Is frontend powered.
 *
 * @param autoPower Is frontend power in automatic mode.
 *
 * @param forcedCooling Is cooling mode forced.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetPowerState(long cameraIndex, bool* powerOn,
    bool* autoPower, bool* forcedCooling);

/**
 * Gets current device power state.
 *
 * @param cameraIndex The selected camera.
 *
 * @param powerOn Frontend components active.
 *
 * @param forcedCooling Cooling active.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCurrentPowerState(long cameraIndex, bool* powerOn,
    bool* autoPower, bool* cooling);

/**
 * Sets the anti glow mode
 *
 * @param cameraIndex The selected camera.
 *
 * @param use glow mode on edge trigger (default on)
 *
 * @param use glow mode on level trigger (default off)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetAntiGlowMode(long cameraindex,
    bool enableOnEdgeTrigger, bool enableOnLevelTrigger);

/**
 * Gets anti glow mode
 *
 * @param cameraIndex The selected camera.
 *
 * @param using glow mode on edge trigger (default on)
 *
 * @param using glow mode on level trigger (default off)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetAntiGlowMode(long cameraindex,
    bool *enabledOnEdgeTriggerPtr, bool *enabledOnLevelTriggerPtr);

/**
 * Gets the number of available Gereral Purpose Outputs.
 *
 * @param cameraIndex The selected camera.
 *
 * @param number Number of available GPOs.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetNumberOfGPOs(long cameraIndex, long* number);

/**
 * Sets the source for the given Gereral Purpose Output.
 *
 * @param cameraIndex The selected camera.
 *
 * @param gpoIndex The index of the GPO to be used (from 0 to McammGetNumberOfGPOs() - 1).
 *
 * @param source The selected GPO source (see MCammGPOSource enumeration for details).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetGPOSource(long cameraIndex, long gpoIndex,
    MCammGPOSource source);

/**
 * Gets the source for the given General Purpose Output.
 *
 * @param cameraIndex The selected camera.
 *
 * @param gpoIndex The index of the GPO to be used (from 0 to McammGetNumberOfGPOs() - 1).
 *
 * @param source The selected GPO source (see MCammGPOSource enumeration for details).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetGPOSource(long cameraIndex, long gpoIndex,
    MCammGPOSource* source);

/**
 * Gets settings ranges for the given General Purpose Output.
 *
 * @param cameraIndex The selected camera.
 *
 * @param delay Minimum value for signal to source delay.
 *
 * @param delay Maximum value for signal to source delay.
 *
 * @param delay Minimum value for pulse width of signal.
 *
 * @param delay Maximum value for pulse width of signal.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetGPOSettingRanges(long cameraIndex, long* minDelay,
    long* maxDelay, long* minPulseWidth, long* maxPulseWidth);

/**
 * Changes settings for the given General Purpose Output.
 *
 * @param cameraIndex The selected camera.
 *
 * @param gpoIndex The index of the GPO to be used (from 0 to McammGetNumberOfGPOs() - 1).
 *
 * @param delay Specifies how many microseconds the signal from the source is delayed.
 *
 * @param pulseWidth Determines if either the source signal is sent directly (value 0), or the rising edge of
 *        the (possibly delayed) signal triggers a pulse. If greater than 0, then the pulse width is specified
 *        in microseconds.
 *
 * @param invert Defines the polarity of the output signal ("False" if same polarity as the source signal, else inverted).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetGPOSettings(long cameraIndex, long gpoIndex,
    long delay, long pulseWidth, bool invert);

/**
 * Gets the settings for the given General Purpose Output.
 *
 * @param cameraIndex The selected camera.
 *
 * @param gpoIndex The index of the GPO to be used (from 0 to McammGetNumberOfGPOs() - 1).
 *
 * @param delay Specifies how many microseconds the signal from the source is delayed.
 *
 * @param pulseWidth Determines if either the source signal is sent directly (value 0), or the rising edge of
 *        the signal (possibly delayed)  triggers a pulse. If greater than 0, then the pulse width is specified
 *        in microseconds.
 *
 * @param inverted Defines the polarity of the output signal ("False" if same polarity as the source signal, else inverted).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetGPOSettings(long cameraIndex, long gpoIndex,
    long* delay, long* pulseWidth, bool* inverted);

/**
 * Sets the camera buffering mode.
 *
 * Buffering ON may increase frame rate but also may increase latency
 *
 * @param cameraIndex The selected camera.
 *
 * @param enabled True if camera should buffering images.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetCameraBuffering(long cameraIndex, bool enabled);

/**
 * Gets current camera buffering mode.
 *
 * @param cameraIndex The selected camera.
 *
 * @param enabled True if camera is buffering images.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *
 * @note see also MCammLifeModeIsImageBufferingFaster
 *
 */
DLLEXPORT long WINAPI McammGetCurrentCameraBuffering(long cameraIndex,
    bool* enabled);

/**
 * Sets the camera color matrix optimization mode.
 *
 * The option has no effect if camera buffering is ON i.e. will only be used in "life ContinuousAcquisition mode"
 *
 * If switched on instead of a raw image a (partly) processed image will be delivered by the callback
 * If switched on for optimization reasons McammExecuteIPFunction shall be called with the
 * parameter pImageData == NULL because no copy of the image is required in these modes
 *
 * @param cameraindex of the selected camera.
 *
 * @param Mode to be selected
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetColorMatrixOptimizationMode(long cameraindex,
    MCammColorMatrixOptimizationMode colorMatrixMode);

/**
 * Gets current color matrix optimzation mode.
 *
 * @param cameraindex of the selected camera.
 *
 * @param current selected color matrix optimization mode
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetColorMatrixOptimizationMode(long cameraindex,
    MCammColorMatrixOptimizationMode *colorMatrixModePtr);

/**
 * Sets the intensity of main- and trigger-LED. The maximum intensity values are available
 * through McammGetMaxLEDIntensity. The values are corrected automatically by the camera
 * and should therefore requested after this call.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mainLED Intensity of main LED (from 0 to max value).
 *
 * @param triggerLED Intensity of trigger LED (from 0 max value).
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetLEDIntensity(long cameraIndex, long mainLED,
    long triggerLED);

/**
 * Gets the current intensity of main- and trigger-LED.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mainLED Intensity of main LED.
 *
 * @param triggerLED Intensity of trigger LED.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetLEDIntensity(long cameraIndex, long* mainLED,
    long* triggerLED);

/**
 * Gets the maximum intensity of main- and trigger-LED.
 *
 * @param cameraIndex The selected camera.
 *
 * @param mainLED Maximum intensity of main LED .
 *
 * @param triggerLED Maximum intensity of trigger LED.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetMaxLEDIntensity(long cameraIndex,
    long* maxMainLED, long* maxTriggerLED);

/**
 * Adds a camera listener, which is informed about added and removed cameras. A defective
 * and self repaired device is removed and added again with default configuration. Then
 * McammInit should be called again.
 *
 * @param listener The callback function.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammAddCameraListener(McammCameraListenerProc listener);

/**
 * Removes a camera listener.
 *
 * @param listener The callback function.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammRemoveCameraListener(
    McammCameraListenerProc listener);

/**
 * Gets current bus data handling and image processing metrics.
 *
 * @param cameraIndex The selected camera.
 *
 * @param metrics Pointer to the metrics collection structure.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetMetrics(long cameraIndex, McammMetrics** metrics);

/**
 * Resets bus data handling and image processing metrics to zero.
 *
 * @param cameraIndex The selected camera.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammResetMetrics(long cameraIndex);

/**
 * Starts the measurement of the given metrics subject. The passed time until "McammStopMeasurement" is measured.
 *
 * @param metricsData The metrics data structure.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammStartMeasurement(McammMetricsData* metricsData);

/**
 * Stops the measurement of the given metrics subject.
 *
 * @param metricsData The metrics data structure.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammStopMeasurement(McammMetricsData* metricsData);

/**
 * Create an empty image data buffer.
 *
 * @param imageDataBuffer The buffer to be created.
 *
 * @param length The buffer length.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammCreateImageDataBuffer(
    unsigned short** imageDataBuffer, long length);

/**
 * Delete image analytics structure.
 *
 * @param imageDataBuffer The buffer to be deleted.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammDeleteImageDataBuffer(
    unsigned short** imageDataBuffer);

/**
 * Enable or disable software trigger for continuous shot
 *
 * @param cameraIndex The selected camera.
 *
 * @param enable or disable trigger (default = disable)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *
 * @note:   After the software trigger is active an acquisition is implicitly active
 *          i.e. changes to acquisition parameters (e.g. ROI or exposure time), from now on have only effects on the
 *          images _after_ the next image.
 *          Be sure to setup all acquisition parameters for the 1st image BEFORE activating the SoftwareTrigger
 */
DLLEXPORT long WINAPI McammSetSoftwareTrigger(long cameraindex, BOOL benable);

/**
 * Get current software trigger setting for continuous shot
 *
 * @param cameraIndex The selected camera.
 *
 * @param trigger enabled or disabled
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetSoftwareTrigger(long cameraindex, BOOL *benabled);

/**
 * Get whether a trigger is possible
 *
 * @param cameraIndex The selected camera.
 *
 * @param if true is returned:  a subsequent call to McammExecuteSoftwareTrigger is safe
 *        if false is returned: a subsequent call to McammExecuteSoftwareTrigger may result in flickr or a busy error
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammTriggerReady(long cameraindex, BOOL *ready);

/**
 * Trigger an image acquisition
 *
 * @param cameraIndex The selected camera.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *          "CAMERABUSY" if camera is not ready yet. This may be used to execute the
 *          software triggers w/o using McammTriggerReady() to achieve the max. image rate
 * @note
 */
DLLEXPORT long WINAPI McammExecuteSoftwareTrigger(long cameraindex);

/**
 * Enable or disable wait time=framedelay for software trigger for continuous shot
 *
 * Works for both hardware and software trigger
 *
 * If active the camera returns "busy" until it is save to trigger
 * If false camera only returns busy during exposure time.
 *
 * @param cameraIndex The selected camera.
 *
 * @param enable or disable trigger (default = disable)
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetTriggerWaitFrameDelay(long cameraindex,
    BOOL benable);

/**
 * Get current "wait frame delay" setting for software trigger setting for continuous shot
 *
 * @param cameraIndex The selected camera.
 *
 * @param wait frame delay enabled or disabled
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetTriggerWaitFrameDelay(long cameraindex,
    BOOL *benabled);

/**
 * Start Fast Acquisition
 * To be used for AxioVision
 *
 * @param cameraIndex The selected camera.
 *
 * @param true:  fast mode for live image
 *        false: image overflow protected mode for experiment mode based on software trigger
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *
 * @note: use  McammAbortFastAcquisition to stop FastAcquisitionEx
 */
DLLEXPORT long WINAPI McammStartFastAcquisitionEx(long cameraindex,
    BOOL bLifeMode);

/**
 * Description see McammIsFastAcquisitionReadyEx
 * To be used for AxioVision
 *
 * additional parameter:
 * bLifeMode @param true:  fast mode for live image
 *                  false: image overflow protected mode for experiment mode based on software trigger
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammIsFastAcquisitionReadyEx(long cameraindex,
    unsigned short* pImageData, long allocatedSize, BOOL bStartNext,
    BOOL bLifeMode);

/**
 * Get maximum possible raw frame size for the camera
 * No binning is assumed e.g. binningX = binningY=1
 *
 * @param cameraIndex The selected camera.
 *
 * @param pointer to variable to receive the frame size in bytes
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetMaxRawImageDataSize(long cameraindex, long* size);

/**
 * Get an image from camera to be used for calculating the white reference later
 *
 * @param cameraIndex The selected camera.
 *
 * @param pointer to user user allocated memory to hold the image to be retrieved
 *        see McammGetMaxRawImageDataSize() to get the image size
 *
 * @param image size
 *
 * @param progress callback
 *
 * @param user parameter for progress callback
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetWhiteRefImage(long cameraindex,
    unsigned short *pImageData, long allocatedSize, McamImageProcEx pCallBack,
    void *UserParam);

/**
 * Calculate white reference out of user supplied image
 *
 * @param cameraIndex The selected camera.
 *
 * @param pointer to user user allocated memory which holds an image to be used for calculation
 *
 * @param image size
 *
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammCalculateWhiteRefFromImage(long cameraindex,
    unsigned short *pImageData, long allocatedSize);

/**
 * Determine whether 8 bit compression is enabled
 *
 * @param cameraIndex The selected camera.
 *
 * @param pointer to user user allocated memory which receives the result value
 *
 * @note 8 bit compression is used to reduce the bandwidth requirement on the USB Bus
 *       e.g. if only USB 2.0 is available
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI Mcammis8bitCompressionEnabled(long cameraindex,
    BOOL *benabled);

/**
 * Get Unit cell size / Pixel distance of CCD sensor
 *
 * @param cameraIndex The selected camera.
 *
 * @param pointer to user user allocated memory which receives the result value
 *
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetCCDPixelDistance(long cameraindex,
    double *pixelDistanceMicroMeter);


/**
 * Set Image Discard Mode
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param true:  Images are discard in overload situations (default)
 *        false: Images are never discarded. Image acquisition is delayed
 *        		 in overload situations
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *
 * @note: if set to 'false' image latency might increase
 * 		  'false' may be a good option for "experiments"
 */
DLLEXPORT long WINAPI McammSetImageDiscardMode(long cameraindex, BOOL benable);

/**
 * Get Image Discard Mode
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param true:  Images are discard in overload situations (default)
 *        false: Images are never discarded. Image acquisition is delayed
 *        		 in overload situations
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 *
 * @note: if set to 'false' image latency might increase
 * 		  'false' may be a good option for "experiments"
 */
DLLEXPORT long WINAPI McammGetImageDiscardMode(long cameraindex, BOOL *benable);

/**
 * Configures the flicker suppression mode, Axiocam 702,727 only!
 *
 * @param cameraIndex The selected camera.
 *
 * @param flicker suppression mode.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammSetLineFlickerSuppressionMode(long cameraIndex, MCammLineFlickerSuppressionMode mode);

/**
 * Gets current the flicker suppression mode, Axiocam 702,727 only!
 *
 * @param cameraIndex The selected camera.
 *
 * @param current flicker suppression mode.
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetLineFlickerSuppressionMode(long cameraIndex, MCammLineFlickerSuppressionMode* mode);


/**
 * Gets currently supported range of gain values
 *
 * For CMOS cameras the upper half of the range maps to digital gain
 *
 * @param cameraIndex The selected camera.
 *
 * @param minimum value returned
 * *
 * @param maximum value returned
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long WINAPI McammGetAnalogGainRange(long cameraIndex, long* pMin, long* pMax);

/**
 * Enables or disables HDR Mode
 *
 * Only supported for For CMOS cameras
 * You should disable discard mode if HDR is used @see McammSetImageDiscardMode
 *
 * Exposure range might change and current exposure time might be corrected
 * to be inside the range
 * @see also McammGetExposureRange() which will report the current valid range after
 * switch to HDR
 *
 * @param cameraIndex The selected camera.
 *
 * @param true: HDR on  false: HDR off
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long McammSetHDRMode(long camerIndex, BOOL doEnable);


/**
 * Get current HDR Mode
 *
 * Only supported for For CMOS cameras
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param returns true: HDR on false: HDR off
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long McammGetHDRMode(long cameraindex, BOOL *pEnabled);

/**
 * Get info whether it is recommended to switch to 8 bit compressed mode
 *
 * *** Currently a DUMMY implementation: Binning==2 + Biining==4 -> false else true
 * ****************************************************************
 * For Axiocam 7xx only!
 *
 * Only supported for For CMOS cameras
 *
 * @param cameraIndex The selected camera.
 *
 * @param returns true: switch on compression  false: do not switch on compression
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long MCammIs8BitCompressionFaster(long cameraindex, BOOL *isFaster);

/**
 * Enables or disables High Image Rate Mode
 *
 * Only supported for For CMOS cameras
 *
 * Changes are effective after the next acquisition started
 *
 * Default is: "Enabled"
 *
 * If enabled the frame delay between images is minimized
 * With Axiocam 702m 1000 fps e.g. with a ROI of 512 x 32 are achievable
 *
 * If enabled random image acquisition delays (up to 1 ms) are likely to occur.
 *
 * @param cameraIndex The selected camera.
 *
 * @param true: Image Rate Mode on  false: Image Rate Mode off
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long McammEnableHighImageRateMode(long camerIndex, BOOL doEnable);


/**
 * Get current state of High Image Rate Mode
 *
 * Only supported for For CMOS cameras
 *
 * @param cameraIndex The selected camera.
 *
 * @param true: Image Rate Mode on  false: Image Rate Mode off
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long McammIsHighImageRateModeEnabled(long cameraindex, BOOL *pEnabled);


/**
 * Get current image number increment
 *
 * Reports the current image number increment.
 * The image number is provided in the image header
 * In standard mode the increment is always "1", e.g. in HDR mode it might be "2"
 *
 * @param cameraIndex The selected camera.
 *
 * @param increment
 *
 * @return  NOERR on success.\n
 *          Error code on failure.
 */
DLLEXPORT long McammGetCurrentImageNumberIncrement(long cameraindex, long *increment);


/**
 * Query supported camera parameters (features)
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param Pointer to boolean, returns "true" if feature is valid, "false" otherwise
 *
 * @return  NOERR on success.\n
 *          PARAMERR on unsupported parameter.
 */
DLLEXPORT long MCammHasParameter(long cameraindex, MCammSupportedParameterType parameter, BOOL *supported);

/**
 * Query min/max value for a camera parameter
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param Pointer to long, returns the min. supported value
 *
 * @param Pointer to long, returns the man. supported value
 *
 *
 * @return  NOERR on success.\n
 *          PARAMERR on unsupported parameter.
 *
 * @note supported for: mcammParmExposure, mcammParmAnalogGain, mcammParmBinning, mcammParmPixelClocks
 *
 */
DLLEXPORT long MCammGetParameterRange(long cameraindex, MCammSupportedParameterType parameter, long *pMin, long *pMax);


/**
 * Determine whether camera buffering could lead to higher frame rates
 * Query whether the frame rate can will raise, if camera buffering is enabled.
 *
 * Result depends on camera model an whether compression is on or or off.
 * A gain in frame rate is typically seen with e.g. short aquisition times, full ROI etc
 *
 *
 * @param cameraIndex The selected camera.
 *
 * @param Pointer to BOOL, "true" is returned, if camera buffering should be switched on (using McammSetCameraBuffering)
 *                                for maximizing the frame rate
 *                         "false" is returned, if camera buffer will usually not increase the frame rate
 *
 * @return  NOERR on success.\n
 *
 */
DLLEXPORT long MCammIsImageBufferingFaster(long cameraindex, BOOL *isFaster);


#if defined (__cplusplus)

}

#endif /* defined (__cplusplus) */

#endif /* MCAM_ZEI_EX_H_ */
