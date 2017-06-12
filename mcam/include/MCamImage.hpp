/**
 * @file MCamImage.h
 * @author ggraf
 * @date 14.02.2015
 *
 * @brief header file for mcam image handling
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#ifndef MCAMIMAGE_H_
#define MCAMIMAGE_H_

#include "mcam.h"

#include <QObject>

class Application;

#define REFRESH_SINGLE_SHOT_RATE	5
#define NUMBER_OF_IMG_BUFFERS		2

typedef enum _imgbuf_stat
{
    ACQ_IMGBUF_UNUSED,    // free buffer - not used / already processed
    ACQ_IMGBUF_WRITING,   // buffer is currently used by USB
    ACQ_IMGBUF_READING,   // buffer will be currently read/ processed
    ACQ_IMGBUF_USED       // buffer retrieval completed but buffer not yet started to process
} Imgbuf_stat;

// image buffer maintainance
typedef struct _imgbuf
{
    unsigned short cameraindex;
    unsigned char *buffer;
    unsigned int bufferSize;
    Imgbuf_stat state;
    long seqno;
    long overwrites;
    long size;
    long long timeStamp;
    void* userParam;
} Buffer_info_t;

class MCamImage: QObject
{
    Q_OBJECT
    Application *applicationPtr;
    bool disableQtPainting;
    long activeCameraIndex;
    bool isLinGainImage;

    // ContinuousShot handling
    bool threadStarted;
    bool stopProcessing;
    sem_t psem;
    pthread_t imageTid;
    pthread_t iTriggerTid;
    Buffer_info_t bufinfo[NUMBER_OF_IMG_BUFFERS];
    pthread_mutex_t bufMutex;
    int bufferId;

    unsigned short *tmpProcessedImageData;
    void* cameraContext;

    bool swTriggerEnabled;
    long incomingImageCounter;
    long imageCounter;

    bool contShotRunning;
    long contShotRunningCameraIndex;

    bool triggerProcMainRunning;

    long incomingLastImageCounter;
    long lastImageCounter;
    long maxColorImageDataSize;
    long paintCounter;
    unsigned long currentFrameTime;
    bool stopTriggerProc;
    bool cameraChangePending;
    long pendingCameraIndex;
    long pendingSize;
	long singleShotCount;

    // performance measurement
    uint64_t callback_start_us;
    uint64_t setimage_us;
    uint64_t qimage_us;
    uint64_t ipTime_us;
    uint64_t ipTimeSum_us;
    QString rateStr;
    uint64_t lastQis;

    bool histogramEnabled;

public:
    MCamImage(Application *applicationPtr);
    virtual ~MCamImage();

    int init();
    int deInit();
    int selectCamera(long cameraIndex);
    void setHighGain(bool isLinGainImage);
    void enableHistogram(bool doEnable);
    void setSqrt(bool isSqrtImage);
    long setSoftwareTrigger(long cameraIndex, bool enableSWTrigger);
    void *imageProcMain(void *parm);
    void *triggerProcMain(void *parm);
    bool processImage(unsigned short* data, long byteSize, long long timeStamp, void* userParam);
    bool setImage(const QImage* image);
    long doSingleShot(long cameraIndex);
    long doContinuousShot(long cameraIndex, bool start);
    bool isContShotRunning();
    long getActiveCameraIndex();
    bool imageCallback(unsigned short* data, long byteSize, long bufferNumber, long long timeStamp, void* userParam);
    void createQImage(unsigned short* sourceImage, long byteSize, QImage* image);
    void colorTemperatureChanged(long cameraIndex, int value);
    int startImageThread();
    int stopImageThread();
    int lockBufferMutex();

    int unlockBufferMutex();

    signals:
    void updateTransferRate(QString rateStr);
    void contShotStart(bool start);
    int cameraSelected(long cameraIndex);
	void updateCost(long cost);

};

#endif /* MCAMIMAGE_H_ */
