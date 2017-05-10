/**
 * @file MCamImage.cpp
 * @author ggraf
 * @date 4.02.2015
 *
 * @brief Image handling of mcam
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#include <math.h>
#include <Application.hpp>
#include "MCamImage.hpp"
#include "ui_mcam.h"

// table for color temperature calculation
double RgbTab[][3] =
                {
                                { 419.87, 202.46, 72.27 }, // 2000K
                                { 396.73, 214.34, 100.46 }, { 376.68, 223.14, 124.19 }, { 359.03, 229.83, 144.99 },
                                { 343.32, 235.03, 163.53 }, { 329.21, 239.14, 180.19 }, { 316.38, 242.44, 195.22 },
                                { 304.68, 245.11, 208.91 }, { 292.49, 248.87, 214.08 }, { 282.10, 250.86, 225.48 },
                                { 272.43, 252.50, 236.06 }, { 263.45, 253.86, 245.87 },
                                { 255.00, 255.00, 255.00 }, // 5000K
                                { 247.06, 255.95, 263.52 }, { 239.57, 256.76, 271.43 }, { 232.51, 257.43, 278.77 },
                                { 225.86, 258.01, 285.58 }, { 219.56, 258.50, 291.91 }, { 213.54, 258.92, 297.84 },
                                { 207.84, 259.27, 303.35 }, { 202.39, 259.59, 308.50 }, { 197.15, 259.86, 313.35 },
                                { 192.10, 260.09, 317.92 }, { 187.36, 260.29, 322.14 }, { 182.73, 260.47, 326.15 }, // 8000K
                };

void *image_proc_main(void *parm);
static MCamImage* mcamImagePtr;

void *image_proc_main(void *parm)
{
    return mcamImagePtr->imageProcMain(parm);
}

void *trigger_proc_main(void *parm)
{
    return mcamImagePtr->triggerProcMain(parm);
}

static BOOL imageCallback(unsigned short* data, long byteSize, long bufferNumber, LONGLONG timeStamp, void* userParam)
{
    return mcamImagePtr->imageCallback(data, byteSize, bufferNumber, timeStamp, userParam);
}

MCamImage::MCamImage(Application *applicationPtr)
{
    this->applicationPtr = applicationPtr;
    mcamImagePtr = this;

    disableQtPainting = false; // for test purposes

    callback_start_us = 0;
    setimage_us = 0;
    qimage_us = 0;
    ipTime_us = 0;
    ipTimeSum_us = 0;

    // continousShot handling
    threadStarted = false;
    stopProcessing = false;
    bufferId = -1;
    //imageTid = 0;

    //iTriggerTid = 0;
    swTriggerEnabled = false;

    incomingImageCounter = 0;
    imageCounter = 0;

    contShotRunning = false;
    contShotRunningCameraIndex = 0;

    triggerProcMainRunning = false;

    incomingLastImageCounter = 0;
    lastImageCounter = 0;
    maxColorImageDataSize = 0;
    paintCounter = 0;
    currentFrameTime = 0;

    stopTriggerProc = false;

    isLinGainImage = false;

    activeCameraIndex = -1;

    cameraChangePending = false;
    pendingCameraIndex = -1;
    pendingSize = 0;

    lastQis = 0;

    histogramEnabled = false;
    tmpProcessedImageData = NULL;
    cameraContext = NULL;
	singleShotCount = 0;

    connect(this, SIGNAL(updateTransferRate(QString)), applicationPtr, SLOT(updateTransferRate(QString)));
    connect(this, SIGNAL(contShotStart(bool)), applicationPtr, SLOT(contShotStart(bool)));

    connect(this, SIGNAL(cameraSelected(long)), applicationPtr, SLOT(cameraSelected(long)));
}

MCamImage::~MCamImage()
{
  if (tmpProcessedImageData != NULL)
    free(tmpProcessedImageData);
}

int MCamImage::init()
{
    return startImageThread();
}

int MCamImage::deInit()
{
    return stopImageThread(); //synchronous
}

int MCamImage::lockBufferMutex()
{
    return pthread_mutex_lock(&bufMutex);
}

int MCamImage::unlockBufferMutex()
{
    return pthread_mutex_unlock(&bufMutex);
}

void MCamImage::setHighGain(bool isLinGainImage)
{
    this->isLinGainImage = isLinGainImage;
}

void MCamImage::enableHistogram(bool doEnable) {
	histogramEnabled = doEnable;
}

long MCamImage::setSoftwareTrigger(long cameraIndex, bool enableSWTrigger)
{
    MCAM_LOG_INIT("MCamImage::setSoftwareTrigger")
    if (cameraIndex < 0)
        return NODEVICEFOUND;
    swTriggerEnabled = enableSWTrigger;
    long result = McammSetSoftwareTrigger(cameraIndex, enableSWTrigger);
    MCAM_LOGFI_INFO("set to %d, result=%ld", swTriggerEnabled, result);
    result = McammSetTriggerWaitFrameDelay(cameraIndex, swTriggerEnabled); // camera return busy until it is possible to trigger
    MCAM_LOGFI_INFO("wait frame delay set to %d, result=%ld", swTriggerEnabled, result);
    return result;
}

// caution: no buffers allocated yet
int MCamImage::startImageThread()
{
    MCAM_LOG_INIT("MCamImage::startImageThread")
    int i;
    stopProcessing = false;
    bufferId = -1;
    for (i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
        bufinfo[i].state = ACQ_IMGBUF_UNUSED;
        bufinfo[i].buffer = NULL;
        bufinfo[i].bufferSize = 0;
        bufinfo[i].seqno = 0;
        bufinfo[i].overwrites = 0;
    }

    pthread_mutex_init(&bufMutex, NULL);

    sem_init(&psem, 0, 0);

    if (pthread_create(&imageTid, NULL, &image_proc_main, NULL) != 0)
        MCAM_LOGF_ERROR("error creating image_proc_main thread");
    return 0;
}


int MCamImage::selectCamera(long cameraIndex) {
  MCAM_LOG_INIT("MCamImage::cameraSelected")
  long size = 0;
  long result = NOERR;

  MCAM_LOGFI_INFO("cameraSelected, try to change to cameraIndex=%d", cameraIndex);

  if (activeCameraIndex == cameraIndex)
    return 0;
  result = McammGetMaxRawImageDataSize(cameraIndex, &size);
  if (result != NOERR) {
    MCAM_LOGFI_ERROR("McammGetMaxRawImageDataSize failed, result = %ld", result);
    return 1;
  }
  cameraChangePending = true;
  pendingCameraIndex = cameraIndex;
  pendingSize = size*3; // *3: could be a color image!
  MCAM_LOGFI_INFO("trigger change to cameraIndex=%d size=%ld bytes", pendingCameraIndex, pendingSize);
  if (tmpProcessedImageData != NULL)
     free(tmpProcessedImageData);
  tmpProcessedImageData = (unsigned short *) malloc(pendingSize + 16384);
  sem_post(&psem);
  return 0;
}

int MCamImage::stopImageThread()
{
    MCAM_LOG_INIT("MCamImage::stopImageThread")
    int done = 0;
    int i;
    int ret = 0;
    stopProcessing = true;
    sem_post(&psem);
    for (i = 0; i < 100; i++) {
        if (!threadStarted) {
            done = 1;
            break;
        }
        MCamUtil::sleep(50);
    }
    if (done)
        ret = 0;
    else {
        ret = 1;
        MCAM_LOGF_ERROR("stopImageProcessing failed!");
    }

    return ret;
}

// trigger  new images with low latency in a loop
// use case: request a well defined number of images in a user controlled speed
void *MCamImage::triggerProcMain(void *parm)
{
    MCAM_LOG_INIT("MCamImage::triggerProcMain")
    long result;
    BOOL ready = false;
    triggerProcMainRunning = true;
    MCAM_LOGF_INFO("started sw trigger thread");
    while (!stopTriggerProc) {
        result = McammExecuteSoftwareTrigger(activeCameraIndex);
        if (result != NOERR) {
            if (result != CAMERABUSY) {
                MCAM_LOGF_ERROR("#####ERROR McammExecuteSoftwareTrigger result=%ld", result);
                break;
            } else {
                MCamUtil::sleep(5);
            }
        }
    }
    MCAM_LOGF_INFO("SW trigger thread ends");
    triggerProcMainRunning = false;
    return NULL;
}

// used by singleShot and ContShot processing
// input:  raw camera image RGB
// output: QImage
void MCamImage::createQImage(unsigned short* imageData, long byteSize, QImage* image)
{
    IMAGE_HEADER* header = (IMAGE_HEADER*) imageData;

    long width = 0;
    long height = 0;

    long hist[65536];
    for (int i = 0; i < 65536; i++) hist[i] = 0;

    bool isColorImage = 0;

    // painting raw data to image
    unsigned short* pixel = NULL;

    width = header->roiWidth / header->binX;
    height = header->roiHeight / header->binY;

    isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;

    // painting raw camera data to image
    pixel = (unsigned short*) imageData + header->headerSize / 2;
    QRgb* line;
    int blue, green, red, corr, basket, basketvalue;

    basket = (16383 / width) + 1;

    // create histogram
    for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++){
                if (isColorImage) {
                    hist[*pixel++]++;
                    hist[*pixel++]++;
                    hist[*pixel++]++;
                } else {
                    hist[*pixel++]++;
                }
            }

    if (histogramEnabled) {
		pixel = (unsigned short*) imageData + header->headerSize / 2;

		for(int x=0;x<16384/basket;x++) {
			basketvalue = hist[basket*x];
			for(int y=1;y<basket;y++){
				basketvalue += hist[basket*x+y];
			}
			corr = (int32_t)(sqrt((double)basketvalue) + 0.5);
			if(corr >= height-1)
				corr = height-2;
			if (isColorImage) {
				for(int y=1;y<=corr;y++){
					pixel[3*(y * width + x)] = 16383;
					pixel[3*(y * width + x)+1] = 16383;
					pixel[3*(y * width + x)+2] = 16383;
				}
			}
			else {
				for(int y=1;y<=corr;y++)
					pixel[y * width + x] = 16383;
			}
		}
    }

    long entries = width*height;
    if (isColorImage) entries *= 3;
    long lowerlimit = 0;

    int i = 0;
    while(lowerlimit < entries/100) lowerlimit += hist[i++];
    lowerlimit = i;

    long upperlimit = 0;

    i = 65535;
    while(upperlimit < entries/100) upperlimit += hist[i--];
    upperlimit = i;
    if(upperlimit <= lowerlimit) lowerlimit = upperlimit - 1;

    for(i=0;i<lowerlimit;i++) hist[i] = 0;
    for(i=lowerlimit;i<upperlimit;i++) hist[i] = 255*(i-lowerlimit)/(upperlimit-lowerlimit);
    for(i=upperlimit;i<65536;i++) hist[i] = 255;

    pixel = (unsigned short*) imageData + header->headerSize / 2;
    for (int y = 0; y < height; ++y) {
        line = (QRgb*) image->scanLine(y);
        if (!isLinGainImage)
            for (int x = 0; x < width; ++x) {
                if (isColorImage) {
                    blue = *pixel++ >> 6;
                    green = *pixel++ >> 6;
                    red = *pixel++ >> 6;
                } else {
                    red = green = blue = *pixel++ >> 6;
                }
                line[x] = qRgb(red, green, blue);
            }
        else
            for (int x = 0; x < width; ++x) {
                if (isColorImage) {
                	blue = hist[*pixel++];
                    if (blue < 0)
                        blue = 0;
                    if (blue > 255)
                        blue = 255;
                    green = hist[*pixel++];
                    if (green < 0)
                        green = 0;
                    if (green > 255)
                        green = 255;
                    red = hist[*pixel++];
                    if (red < 0)
                        red = 0;
                    if (red > 255)
                        red = 255;
                } else {
                    red = green = blue = hist[*pixel++];
                    if (blue < 0)
                        red = green = blue = 0;
                    if (blue > 255)
                        red = green = blue = 255;
                }
                line[x] = qRgb(red, green, blue);
            }
    }
}

// used by MCamImage::imageProcMain (ContShot)
// measure and display image
// allocates QImage, free in setImage()
bool MCamImage::processImage(unsigned short* data, long byteSize, long long timeStamp, void* userParam)
{
    MCAM_LOG_INIT("MCamImage::processImage")
    IMAGE_HEADER* header = (IMAGE_HEADER*) data;
    QImage* image = NULL;
    uint64_t qis = MCamUtil::getTimeMiroseconds();

    image = new QImage(header->roiWidth / header->binY, header->roiHeight / header->binY, QImage::Format_RGB32);

    qimage_us += MCamUtil::getTimeMiroseconds() - qis;

    qis = MCamUtil::getTimeMiroseconds();

    createQImage(data, byteSize, image);

    // delegate drawing of image (pixmap) to GUI thread!
    applicationPtr->executePaintImage(image);

    qimage_us += MCamUtil::getTimeMiroseconds() - qis;
    qimage_us -= ipTime_us;
    ipTimeSum_us += ipTime_us;

    imageCounter++;
    if (qis - lastQis > 1000000) {
    	lastQis = qis;
        if (lastImageCounter < 0) {
        	// initialize once
            callback_start_us = MCamUtil::getTimeMiroseconds();
            lastImageCounter = imageCounter;
            incomingLastImageCounter = incomingImageCounter;
            ipTimeSum_us = 0;
            setimage_us = 0;
            qimage_us = 0;
            paintCounter = 0;
        } else {
            // measure back to last call
            uint64_t callback_diff_us = MCamUtil::getTimeMiroseconds() - callback_start_us;

            float f = (imageCounter - lastImageCounter) * 1000000;
            if (callback_diff_us > 0)
                f /= callback_diff_us;
            else
                f = 0;

            float fi = (incomingImageCounter - incomingLastImageCounter) * 1000000;
            if (callback_diff_us > 0)
                fi /= callback_diff_us;
            else
                fi = 0;

            float fp = paintCounter * 1000000;
            if (setimage_us + qimage_us > 0)
                fp /= setimage_us + qimage_us;
            else
                fp = 0;

            MCAM_LOGF_INFO("CAM#%d total images= %ld elapsed time = %lld us ipTimeAvgPerImage_us = %lld ms  images=%d InRate=%5.2f FPS DispRate=%5.2f FPS",
                            activeCameraIndex, imageCounter, callback_diff_us, ipTimeSum_us / (1000 * (imageCounter - lastImageCounter)),
                            imageCounter - lastImageCounter, fi, f);

            char rateBuffer[256];

            if (!contShotRunning)
                fi = f = 0;
            memset(rateBuffer, '\0', sizeof(rateBuffer));
            sprintf(rateBuffer, "InRate: %.1f fps DispRate: %.1f fps", fi, f);
            rateStr = rateBuffer;

            emit updateTransferRate(rateStr);

            callback_start_us = MCamUtil::getTimeMiroseconds();
            lastImageCounter = imageCounter;
            incomingLastImageCounter = incomingImageCounter;
            ipTimeSum_us = 0;
            setimage_us = 0;
            qimage_us = 0;
            paintCounter = 0;
        }
    }
    return true;
}

// incoming images from "MCamImage::imageCallback"
void *MCamImage::imageProcMain(void *parm)
{
    MCAM_LOG_INIT("MCamImage::imageProcMain")
    int i;
    int found = 0;
    long oldest_seqno = 0;

    threadStarted = true;
    lastQis = 0;

    while (!stopProcessing) {
        sem_wait(&psem);
        if (cameraChangePending) {
            // a camera change was triggered --> check if idle and then allocate new buffers (different size!)
           MCAM_LOGF_INFO("cameraChangePending detected, pendingCameraIndex=%ld", pendingCameraIndex);
           bool found = false;
           lockBufferMutex();
           for (int i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
             if (bufinfo[i].state != ACQ_IMGBUF_UNUSED)
               found=true;
           }

           if (found) {
               MCAM_LOGF_INFO("used buffer found ...");
           } else {
             for (int i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
               if (bufinfo[i].buffer != NULL) {
                 delete [] bufinfo[i].buffer;
                 bufinfo[i].bufferSize = 0;
               }
               bufinfo[i].buffer = new unsigned char [pendingSize];
               bufinfo[i].bufferSize = pendingSize;
               MCAM_LOGF_INFO("allocated buffer #%d, size=%ld bytes", i, pendingSize);
             }
             activeCameraIndex = pendingCameraIndex;
             cameraChangePending = false;
             // confirm to Application
             emit cameraSelected(activeCameraIndex);
           }
           unlockBufferMutex();
        }
        found = -1;
        if (!stopProcessing) {
            lockBufferMutex();

            oldest_seqno = LONG_MAX;
            for (i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
                if (bufinfo[i].state == ACQ_IMGBUF_USED) {
                    if (bufinfo[i].seqno < oldest_seqno) {
                        oldest_seqno = bufinfo[i].seqno;
                        found = i;
                    }
                }
            }
            if (found >= 0) {
                bufinfo[found].state = ACQ_IMGBUF_READING;
                unlockBufferMutex();
            } else {
                unlockBufferMutex();
                continue;
            }

            if (found >= 0) {
                // process the image
                // call processing if applicable
//                if (processRawImage) {
//                    long result = McammExecuteIPFunction(cameraContext, tmpProcessedImageData, (unsigned short*) bufinfo[found].buffer);
//                    if (result == NOERR)
//                      processImage(tmpProcessedImageData, (long) bufinfo[found].size, bufinfo[found].timeStamp,
//                                  bufinfo[found].userParam);
//                } else

                  processImage((unsigned short*) bufinfo[found].buffer, (long) bufinfo[found].size, bufinfo[found].timeStamp,
                                                  bufinfo[found].userParam);

            }
            lockBufferMutex();

            // free this image buffer
            if (found >= 0 && found < NUMBER_OF_IMG_BUFFERS)
                bufinfo[found].state = ACQ_IMGBUF_UNUSED;

            unlockBufferMutex();
        }
    }
    threadStarted = false;
    MCAM_LOGF_STATUS("Stopped image thread");
    return NULL;
}

// display image for single shots
// runs ALWAYS in context of GUI thread
// for ContShot -> via Application::executePaintImage
// frees QImage!
bool MCamImage::setImage(const QImage* image)
{
    bool ret = false;
    if (!disableQtPainting) {
        uint64_t fts64, fte64;
        fts64 = MCamUtil::getTimeMiroseconds();

        applicationPtr->getUi()->imageLabel->setPixmap(QPixmap::fromImage(*image));

        applicationPtr->getUi()->actionFitToWindow->setEnabled(true);

        if (!applicationPtr->getUi()->actionFitToWindow->isChecked()) {
            applicationPtr->getUi()->imageLabel->adjustSize();
        }

        fte64 = MCamUtil::getTimeMiroseconds();
        setimage_us += fte64 - fts64;
        ret = true;
    }
    delete image;
    return ret;
}

// execute single shot
long MCamImage::doSingleShot(long cameraIndex)
{
    MCAM_LOG_INIT("MCamImage::doSingleShot")
    long result = NOERR;
    long imageSize;
    BOOL benabled = false;

    if (cameraIndex < 0) {
        MCAM_LOGF_ERROR("no camera available -> abort!");
        return NODEVICEFOUND;
    }

    if (activeCameraIndex != cameraIndex) {
      MCAM_LOGF_ERROR("invalid activeCameraIndex=%d, cameraIndex=%d", activeCameraIndex, cameraIndex);
      return PARAMERR;
    }
    result = McammGetSoftwareTrigger(cameraIndex, &benabled);
    if (result != 0) {
       MCAM_LOGF_ERROR("McammGetSoftwareTrigger failed result=%d", result);
       return result;
    }
    if (benabled) {
    	result = McammSetSoftwareTrigger(cameraIndex, FALSE);
    	if (result != 0) {
    		MCAM_LOGF_ERROR("McammSetSoftwareTrigger failed result=%d", result);
    	    return result;
        }
    }

    if ((result = McammGetCurrentImageDataSize(cameraIndex, &imageSize)) != 0) {
        MCAM_LOGF_ERROR("McammGetCurrentImageDataSize failed, ret%d", result);
        return result;
    }
    // 16 bit image size
    imageSize /= 2;

    // prepare image data
    unsigned short* imageData = new unsigned short[imageSize];

    // acquire image - image allocated by user
    uint64_t qstart = MCamUtil::getTimeMiroseconds();
    result = McammAcquisitionEx(cameraIndex, imageData, imageSize, NULL, NULL);
    if (result != 0) {
        MCAM_LOGF_ERROR("McammAcquisitionEx failed result=%d", result);
    } else {
        uint64_t qend = MCamUtil::getTimeMiroseconds();
        MCAM_LOGF_INFO("McammAcquisitionEx image acquired time= %lld ms", (qend - qstart) / 1000);
    }
    if (result == 0) {
        // paint image
        QImage* image;
        if (result == NOERR) {
            // check if its a color image
			if (singleShotCount++ % REFRESH_SINGLE_SHOT_RATE == 0)
			{
				IMAGE_HEADER* header = (IMAGE_HEADER*)imageData;
				image = new QImage(header->roiWidth / header->binX, header->roiHeight / header->binY, QImage::Format_RGB32);

				createQImage(imageData, imageSize, image);

				// delegate drawing of image (pixmap) to GUI thread!
				// (called form other thread in stress test case)
				applicationPtr->executePaintImage(image);
			}
			//KBE??
			//printf("Remote Save Image\r\n");
			//applicationPtr->thisMCamRemotePtr->saveImage(image);
			applicationPtr->thisMCamRemotePtr->saveImage(imageData);
		} else {
            MCAM_LOGF_ERROR("Error during image acquisition: result=", result);
        }
    }
    delete[] imageData;

    // re-enable SW-Trigger if necessary
    if (benabled) {
    	result = McammSetSoftwareTrigger(cameraIndex, benabled);
    	if (result != 0) {
    		MCAM_LOGF_ERROR("McammSetSoftwareTrigger failed o restore result=%d", result);
    		return result;
    	}
    }
    return result;
}

// a new image arrived from DLL (via ::imageCallback global function)
// fetch a buffer and copy image to buffer
// inform image thread "imageProcMain" above --> continue processing in this thread
bool MCamImage::imageCallback(unsigned short* data, long byteSize, long bufferNumber, LONGLONG timeStamp, void* userParam)
{
    MCAM_LOG_INIT("MCamImage::imageCallback")
    int i = 0;
    int found = -1;
    long last_seqno = -1;
    long oldest_seqno = LONG_MAX;
    int ret = 0;

    if (activeCameraIndex < 0) {
      MCAM_LOGF_ERROR("invalid cameraIndex = %ld",activeCameraIndex);
      return false;
    }

    lockBufferMutex();
    incomingImageCounter++;

    for (i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
        if ((found < 0) && (bufinfo[i].state == ACQ_IMGBUF_UNUSED)) {
            found = i;
        }
        if (bufinfo[i].seqno > last_seqno) {
            last_seqno = bufinfo[i].seqno;
        }
    }

    if (found < 0) {
        // no free buffer found -> override the oldest used used buffer
        // - never overwrite a buffer in state WRITING or READING
        for (i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
            if (bufinfo[i].state == ACQ_IMGBUF_USED) {
                if (bufinfo[i].seqno < oldest_seqno) {
                    oldest_seqno = bufinfo[i].seqno;  // fixme: handle wrap around -> long
                    found = i;
                }
            }
        }
    }

    if (found < 0) {
        // should never happen: no buffer found at all!
        MCAM_LOGF_ERROR("ERROR: no free image buffer!");
        for (i = 0; i < NUMBER_OF_IMG_BUFFERS; i++) {
            MCAM_LOGF_ERROR("DUMP bufinfo.state[%d]=%d seqno=%ld", i, bufinfo[i].state, bufinfo[i].seqno);
        }
    } else {
        // use buffer
        memcpy(bufinfo[found].buffer, data, byteSize);
        bufinfo[found].size = byteSize;
        bufinfo[found].timeStamp = timeStamp;
        bufinfo[found].userParam = userParam;
        bufferId = found;
        bufinfo[bufferId].state = ACQ_IMGBUF_USED;
        bufinfo[found].seqno = last_seqno + 1;
        sem_post(&psem);
    }
    unlockBufferMutex();
    return true;
}

bool MCamImage::isContShotRunning()
{
    return contShotRunning;
}

long MCamImage::getActiveCameraIndex() {
  return activeCameraIndex;
}
// start/stop contShot processing
// after successful start images will arrive at "MCamImage::imageCallback"
long MCamImage::doContinuousShot(long cameraIndex, bool start)
{
    MCAM_LOG_INIT("MCamImage::doContinuousShot")
    long result = NOERR;

    if (cameraIndex < 0) {
        contShotRunning = false;
        rateStr = tr("InRate: 0 fps DispRate: 0 fps");
        emit updateTransferRate(rateStr);
        return NODEVICEFOUND;
    }

    if (activeCameraIndex != cameraIndex) {
       MCAM_LOGF_ERROR("invalid activeCameraIndex=%d, cameraIndex=%d", activeCameraIndex, cameraIndex);
       return PARAMERR;
    }
    if (start) {
        // set continuous shot callback function
        result = McammGetIPInfo(cameraIndex, &cameraContext, NULL, &::imageCallback, NULL);
        if (result == NOERR) {
            contShotRunning = true;
            emit contShotStart(contShotRunning);
            contShotRunningCameraIndex = cameraIndex;

            // start continuous shot
            lastImageCounter = -1;
            currentFrameTime = 0;
            result = McammStartContinuousAcquisition(cameraIndex, THREAD_PRIORITY_ABOVE_NORMAL, NULL);
            if (result == NOERR) {
                MCAM_LOGF_INFO("Continuous shot started");
                result = McammGetCurrentFrameTime(cameraIndex, &(currentFrameTime));
                if (result == NOERR) {
                    float rate = 1000000.0 / currentFrameTime;
                    MCAM_LOGF_INFO("Continuous shot started frameTime= %ld ms MaxRate= %ld", currentFrameTime, rate);
                    if (swTriggerEnabled) {
                        stopTriggerProc = false;
                        if (pthread_create(&iTriggerTid, NULL, &trigger_proc_main, NULL) != 0) {
                            MCAM_LOGF_ERROR("error creating stress_proc_main thread");
                        }
                    }
                } else {
                    MCAM_LOGF_ERROR("Retrieving of FrameTime failed");
                }
            } else {
                MCAM_LOGF_ERROR("Setting continuous start failed, result=%ld", result);
            }

        } else {
            MCAM_LOGFI_ERROR("Setting continuous shot callback function failed");
        }
    } else { // start
        // stop ContShot
        contShotRunning = false;
        contShotRunningCameraIndex = -1;
        result = McammStopContinuousAcquisition(cameraIndex);
        if (result == NOERR) {
            MCAM_LOGF_INFO("Continuous shot stopped");
        } else {
            MCAM_LOGF_WARN("Setting continuous stop failed. ret=%ld", result);
        }
        rateStr = tr("InRate: 0 fps DispRate: 0 fps");
        emit updateTransferRate(rateStr);
        contShotStart(contShotRunning);
        if (swTriggerEnabled) {
            int count = 0;
            stopTriggerProc = true;
            do {
                MCamUtil::sleep(20);
            } while (triggerProcMainRunning && count++ < 10);
        }
    }
    return result;
}

// only changes transient values in camera
void MCamImage::colorTemperatureChanged(long cameraIndex, int value)
{ // 0 .. 100
    MCAM_LOG_INIT("MCamImage::colorTemperatureChanged")

    MCAM_LOGF_INFO("colorTemperatureChanged: value=%d", value);
    double red, green, blue;
    double newRed, newGreen, newBlue;

    double dval = value; // 0 - 100
    double temp = 2000 + dval * 60;  // 2000 .. 8000
    double indexNotRounded = (temp - 2000) * 24 / 6000;

    int index1 = floor(indexNotRounded);
    int index2;

    if (cameraIndex < 0)
        return;

    if (index1 + 1 > 24)
        index2 = 24;
    else
        index2 = index1;

    double d = indexNotRounded - index1;

    red = RgbTab[index1][0] + d * (RgbTab[index2][0] - RgbTab[index1][0]);
    green = RgbTab[index1][1] + d * (RgbTab[index2][1] - RgbTab[index1][1]);
    blue = RgbTab[index1][2] + d * (RgbTab[index2][2] - RgbTab[index1][2]);
    MCAM_LOGF_DEBUG("d=%lf index1=%d index2=%d Result= %lf %lf %lf", d, index1, index2, red, green, blue);

    double whitePointRed, whitePointGreen, whitePointBlue;
    applicationPtr->thisCameraIFPtr->getWhitePoint(&whitePointRed, &whitePointGreen, &whitePointBlue);

    MCAM_LOGF_DEBUG("getWhitePoint: Red=%lf Green=%lf Blue=%lf", whitePointRed, whitePointGreen, whitePointBlue);

    double factorRed = whitePointRed;
    factorRed /= red;
    double factorGreen = whitePointGreen;
    factorGreen /= green;
    double factorBlue = whitePointBlue;
    factorBlue /= blue;
    MCAM_LOGF_DEBUG("factorRed=%lf factorGreen=%lf factorBlue=%lf", factorRed, factorGreen, factorBlue);

    newRed = factorRed;
    newGreen = factorGreen;
    newBlue = factorBlue;
    MCAM_LOGF_DEBUG("newRed=%lf newGreen=%lf newBlue=%lf", newRed, newGreen, newBlue);
    applicationPtr->thisCameraIFPtr->setWhiteBalance(cameraIndex, newRed, newGreen, newBlue);
}

