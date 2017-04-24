/**
 * @file ccam.c
 * @author ggraf
 * @date 12.05.2015
 *
 * @brief  ccam 
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#define CCAM_VERSION "1.62"
#define CCAM_GRAB_IMAGE "--grab"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tiffio.h>

#ifndef _WIN32
  #include <sys/utsname.h>
  #include <unistd.h>
#endif


#include "ccam.h"


long availableCameras = 0;
long cameraindex = 0;
char *command = NULL;

int main(int argc, char **argv);
int writeTiffImage(unsigned char *buffer, int size, int color, int sx, int sy, int bitsPerSample, char *basename);
long initialize();
long deinitialize();
void printHelp();


int writeTiffImage(unsigned char *buffer, int size, int color, int sx, int sy, int bitsPerSample, char *basename) {
  int error=0;
  char dumpfileStr[1024];
  int c;
  int colorFactor=1;
  static int ctr;
  TIFF *image;
  int y;
  
  if (basename == NULL) {
      ctr++;
      memset(dumpfileStr,'\0', sizeof(dumpfileStr));
      sprintf(dumpfileStr,"axiftest_image_%04d.tif",ctr);
  } else {
      dumpfileStr[1023] = '\0';
      strncpy(dumpfileStr, basename, 1023-4);
      strncat(dumpfileStr, ".tif", 4);
  }

  if (bitsPerSample != 8 && bitsPerSample != 16) {
    printf("= TIFF depth %d not supported\n",bitsPerSample);
    return 1;
  }

  // Open the TIFF file
  if ((image = TIFFOpen(dumpfileStr, "w")) == NULL) {
    printf("= Could not open %s for writing\n",dumpfileStr);
    return 2;
  }

   // We need to set some values for basic tags before we can add any data
  if (color)
    colorFactor=3;
  TIFFSetField(image, TIFFTAG_IMAGEWIDTH, sx);
  TIFFSetField(image, TIFFTAG_IMAGELENGTH, sy);
  TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
  if (color)
    TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 3);
  else
    TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(image, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  if (color)
    TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 2);
  else
    TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);
  TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(image, sx*1));

  for (y = 0; y < sy; y++) {
      uint32_t lineBuffer[16384*2];
      uint16_t *pSrc = (uint16_t*) (buffer + y*sx*2*colorFactor);
      unsigned char *checkPSrcPtr = (unsigned char*) pSrc;
      if (checkPSrcPtr + 2*sx*colorFactor > buffer + size) {
        printf("= Error: writeTiffImage: pointer out of range %p > %p \n",checkPSrcPtr + 2*sx*colorFactor, buffer + size);
        error=1;
        break;
      }
      if (bitsPerSample == 8) {
          uint8_t *pDst = (uint8_t *)lineBuffer;
          for (c= 0; c < sx*colorFactor; c++) {
              // display 8 LSBs
              uint16_t src = (*(pSrc+c) < 0xFF) ? *(pSrc+c) : 0xFF;
              *(pDst+c) = (uint8_t) src;
          }
      } else {
          uint16_t *pDst = (uint16_t *)lineBuffer;
          for (c= 0; c < sx*colorFactor; c++) {
              // display 14 LSBs
              uint16_t src = (*(pSrc+c) < 0x3FFF) ? *(pSrc+c) : 0x3FFF;
              *(pDst+c) = src << 2;
          }
      }
      if (TIFFWriteScanline(image, lineBuffer, y, 0) != 1) {
          TIFFError("= write_image", "error");
          error=1;
          break;
      }
  }
  TIFFClose(image);
  return error;
}


long initialize() {
  long result = NOERR;
  long lastindex = -1;
  long cindex = 0;
  if (availableCameras > 0) {
    printf("- about to initialize %d camera(s) ...\n", availableCameras);
    result = McammLibInit(false);
    if (result != NOERR) {
      printf("= Initialization failed, result = %d\n", result);
      McammLibTerm();
    } else {
      availableCameras = McamGetNumberofCameras();
      for (cindex = 0; cindex < availableCameras; cindex++) {
        result = McammInit(cindex);
        if (result != NOERR) {
          printf("= Initialization of camera #%d failed, result = %d\n", cindex, result);
          break;
        } else {
          lastindex = cindex;
        }
      }
    }
    if (result != NOERR) {
      for (cindex = 0; cindex <= lastindex; cindex++) 
        McammClose(cindex);
    }
  }
  return result;
}

long deinitialize() {
  long result = NOERR;
  long cindex = 0;
  availableCameras = McamGetNumberofCameras();
  for (cindex = 0; cindex < availableCameras; cindex++) 
    McammClose(cindex);
  McammLibTerm();
  return result;
}

void printHelp() {
  printf("\nccam supported commands:\n");
  printf("%s [ exposure_time_ms ]   : grab image and store to file\n", CCAM_GRAB_IMAGE);
  printf("\n");  
}

/*****************************************************************************/
/**
* main
*
* @return 0 : ok   1: error
*
* @note
*
******************************************************************************/
int main(int argc, char **argv) {
  long result = NOERR;
  long exposureTime= 0;
  printf("- ccam version %s starting\n", CCAM_VERSION);
  
  if (argc > 1) {
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "/?") == 0 || strcmp(argv[1], "-?") == 0) {
        printHelp();
        printf("- ccam ends\n");
        return 2;
    }
  }
  
  printf("- searching for Axiocam NG cameras ...\n");
  availableCameras = McamGetNumberofCameras();
  printf("- %d cameras found\n", availableCameras);
  
  result = initialize();
  
  if (result == NOERR) {
    if (argc > 1)
      command = argv[1];
    if (availableCameras > 0) {
      if (command != NULL) {
      
        /////////// Grab a single image //////////////
        if (strcmp(command, CCAM_GRAB_IMAGE) == 0) {
          long imageSize = 0;
          unsigned short *image = NULL;
          
          result = McammGetCurrentExposure(cameraindex, &exposureTime);
          if (result != NOERR) {
            exposureTime = 0;
            printf("= McammGetCurrentExposure failed, ret=%ld\n", result);
          }
          if (argc > 2)  {
            exposureTime = atoi(argv[2]) * 1000;
            result = McammSetExposure(cameraindex, exposureTime);
            if (result != NOERR) {
              exposureTime = 0;
              printf("= McammSetExposure failed, ret=%ld\n", result);
            }
          }
          
          printf("- grab Image - exposureTime=%ld ms\n", exposureTime/1000);
          
          result = McammGetCurrentImageDataSize(cameraindex, &imageSize);
          if (result == NOERR) {
            image = (unsigned short *) malloc(imageSize*2);
            if (image != NULL) {
            
              result = McammAcquisitionEx(cameraindex, image, imageSize/2, NULL, NULL);
              if (result == NOERR) {
                IMAGE_HEADER *header = (IMAGE_HEADER *) image;
                unsigned char *imageData = (unsigned char *) image;
                int isColorImage = 0;
                imageData += header->headerSize; // skip image header
                
                isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;
                
                printf("- Got image %dx%d pixel\n", header->roiLeftWidth + header->roiRightWidth, header->roiTopHeight + header->roiBottomHeight);
                
                int ret = writeTiffImage((unsigned char *) imageData, imageSize, isColorImage, 
                                header->roiLeftWidth + header->roiRightWidth, header->roiTopHeight + header->roiBottomHeight, 16, (char*) "ccam"); 
                if (ret != 0)
                  printf("= Write image to file 'ccam.tif' failed!\n");
                else  
                  printf("= Wrote image to 'ccam.tif'\n");
              } else {
                printf("- McammAcquisitionEx failed result=%ld\n", result);
              }
            } else {
              printf("Memory allocation error\n");
            }
            free(image);
          }
        } else {
            printf("= unknown command='%s', use --help for a list of commands\n", command);
        }
      } else {
          printf("= no command specified. Use --help for a list of commands\n");
      }
    }
    deinitialize();
  }
  printf("- ccam ends\n");
  if (result == NOERR)
    return 0;
  else
    return 1;
}