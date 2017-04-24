/**
 * @file mcamloger.c
 * @author ggraf
 * @date 17.02.2015
 *
 * @brief mcam logger implementation
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

/***************************** Include Files *********************************/
extern "C" {

#ifndef _WIN32
#include <unistd.h>
#include <stdbool.h>
#endif

#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <stdlib.h>
#ifndef _WIN32
  #include <unistd.h>
  #include <sys/syscall.h>
  #include <sys/time.h>
#endif

#include "mcamlogger.h"

/***************************** Constants *************************************/

static char filename[1024];

#define MCAM_LOGGING_LOCK pthread_mutex_lock(&mcamLoggingMutex);

#define MCAM_LOGGING_UNLOCK pthread_mutex_unlock(&mcamLoggingMutex);

#define LOG_LOGGER(LOG_FUNCTION_NAME, LOGVAR, LEVEL, LEVELSTR) \
  int LOG_FUNCTION_NAME(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...) { \
    char buffer[2048]; \
    char timestampBuf[32]; \
    va_list arguments; \
    if (!loggingEnabled)  \
      return 1;   \
    if ((LOGVAR & LEVEL) == 0)  \
      return 1;  \
    va_start(arguments, message);  \
    vsprintf(buffer, message, arguments);  \
    \
    MCAM_LOGGING_LOCK  \
    \
    if (cameraindex == 0xfffe) \
      fprintf(logfp, "%s [%06d] %s: %s %s\n", getTimeStamp((unsigned char*) timestampBuf,32), mcamGetThreadId(), LEVELSTR, log_fkt_name, buffer);  \
    else \
      fprintf(logfp, "%s [%06d] %s: CAM#%d %s %s\n", getTimeStamp((unsigned char*) timestampBuf,32), mcamGetThreadId(), LEVELSTR, cameraindex, log_fkt_name, buffer);  \
    mcamLogCheckAndSwitch();  \
    \
    MCAM_LOGGING_UNLOCK  \
    \
    va_end(arguments);  \
    return 0;  \
  }

/************************** Variable Definitions *****************************/

pthread_mutex_t mcamLoggingMutex;

unsigned long mcamLogLevel = 0;
unsigned char loggingEnabled = 1;

static FILE *logfp = NULL;

#ifdef _WIN32
  #define DELIM '\\'
#else
  #define DELIM '/'
#endif

static long mcamLoggerDebugMaxFileSize;
static int mcamLoggerAppend = 0;

void removePath(char *path) {
  char *ptr;
  if (strlen(path) < 2)
    return;
  ptr = path + strlen(path) - 1;
  while (ptr > path) {
    if (*ptr == DELIM)
      break;
    ptr--;
  }
  if (*ptr != DELIM)
    return;

  ptr++;

  while (*ptr != '\0') {
    *path = *ptr;
    path++;
    ptr++;
  }
   *path='\0';
}


/*************************** Function Definions *****************************/

int mcamGetThreadId() {
#ifdef _WIN32
  return GetCurrentThreadId();
#else
  return syscall(SYS_gettid);
#endif
}

unsigned char *getTimeStamp(unsigned char *result, int size) {
#ifdef WIN32
  SYSTEMTIME tm;
  GetLocalTime(&tm);
   sprintf((char*) result, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
      tm.wYear, tm.wMonth, tm.wDay, tm.wHour, tm.wMinute, tm.wSecond,
      tm.wMilliseconds);
#else
  struct timeval  tv;
  struct tm      *tm;
  gettimeofday(&tv, NULL);
  tm = localtime(&tv.tv_sec);
   sprintf((char*) result, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
         tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday,
         tm->tm_hour, tm->tm_min, (int)tm->tm_sec,(int) tv.tv_usec/1000);
#endif
  result[size-1] = '\0';
  return result;
}

int mcamLoggingInit(int append, const char *pfilename, long debugMaxFileSize) {
  mcamLoggerAppend = append;
  mcamLoggerDebugMaxFileSize = debugMaxFileSize;

  strcpy(filename, pfilename);
  if (pthread_mutex_init(&mcamLoggingMutex, NULL) != 0) {
    printf("mcamLoggingInit: cannot create mcamLoggingMutex\n");
    return 1;
  }
  if (logfp != NULL) {
    loggingEnabled = 0;
    return 1;
  }
  if (mcamLoggerAppend) {
    logfp = fopen(filename, "a");
    if (logfp == NULL) {
      removePath(filename);
      logfp = fopen(filename, "a");
    }
  } else {
    logfp = fopen(filename, "w");
    if (logfp == NULL) {
      removePath(filename);
      logfp = fopen(filename, "w");
    }
  }

  if (logfp != NULL)
    setvbuf(logfp, NULL, _IONBF, 0); // no buffering
  else
    loggingEnabled=0;
  return 0;
}

int mcamLoggingDeInit() {
  if (!loggingEnabled)
    return 1;
  if (logfp != NULL)
    fclose(logfp);
  logfp = NULL;
  if (pthread_mutex_destroy(&mcamLoggingMutex) != 0)
    return 1;
  return 0;
}

// rolling file appender
int mcamLogCheckAndSwitch() {
  int ret = 0;
  if ((ftell(logfp) / (1024*1024)) >= mcamLoggerDebugMaxFileSize) {
    char filenamebak[1024];

    strcpy(filenamebak, filename);
    strcat(filenamebak, ".1");
    fclose(logfp);
    logfp= NULL;
#ifdef _WIN32
    ret = remove (filenamebak);
#else
    ret = unlink (filenamebak);
#endif
    ret = rename(filename, filenamebak);
    // if unlink and/or rename did not work -> just overwrite
    logfp = fopen(filename, "w");
    if (logfp != NULL) {
      ret = setvbuf(logfp, NULL, _IONBF,0);
    }
    else {
      ret = 1;
    }
  }
  if (ret != 0)
    loggingEnabled=0;
  return ret;
}

int mcamSetLogLevel(uint32_t parmAxcamLevel) {
  mcamLogLevel = parmAxcamLevel;
  return 0;
}

LOG_LOGGER(mcam_log_error,  mcamLogLevel, MCAM_DEBUG_ERROR,    "ERROR ")
LOG_LOGGER(mcam_log_warn,   mcamLogLevel, MCAM_DEBUG_WARN,     "WARN  ")
LOG_LOGGER(mcam_log_info,   mcamLogLevel, MCAM_DEBUG_INFO,     "INFO  ")
LOG_LOGGER(mcam_log_debug,  mcamLogLevel, MCAM_DEBUG_DEBUG,    "DEBUG ")
LOG_LOGGER(mcam_log_status, mcamLogLevel, MCAM_DEBUG_STATUS,   "STATUS")

};

