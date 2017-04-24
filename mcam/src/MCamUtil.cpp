/**
 * @file MCamUtil.cpp

 * @author ggraf
 * @date 14.02.2015
 *
 * @brief //TODO add brief file description.
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
  #include <unistd.h>
  #include <sys/time.h>
#endif
#include "MCamUtil.hpp"


#ifdef _WIN32
 #include <windows.h>
#endif

MCamUtil::MCamUtil()
{
}

MCamUtil::~MCamUtil()
{
}

extern "C" {
  /*****************************************************************************/
  /**
  * Get Timestamp
  *
  * @param  pointer to buffer
  * @param  size of buffer
  *
  * @return pointer to result
  *
  * @note   None.
  *
  ******************************************************************************/
  unsigned char *MCamUtil::getTimeStamp(unsigned char *result, int size) {
  #ifdef _WIN32
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

#ifdef _WIN32

uint64_t MCamUtil::getTimeMiroseconds() {
    uint64_t ft64;
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ft64 = ft.dwHighDateTime;
    ft64 = (ft64 << 32) | ft.dwLowDateTime;
    return ft64 / 10; // microseconds
}
#else
uint64_t MCamUtil::getTimeMiroseconds() {
    struct timespec ts;
    unsigned long long t;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    t = ts.tv_sec * 1000000;
    t += ts.tv_nsec/1000;
    return (uint64_t) t;
}

#endif

}


void MCamUtil::sleep(long msec) {
#ifdef _WIN32
  Sleep(msec);
 #else
  usleep(msec*1000);
#endif
}
