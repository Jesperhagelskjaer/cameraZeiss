/**
 * @file MCamUtil.h
 * @author ggraf
 * @date 14.02.2015
 *
 * @brief header file for mcam utilities
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */
#ifndef MCAMUTIL_H_
#define MCAMUTIL_H_

#include <stdint.h>

class MCamUtil
{
public:
    MCamUtil();
    virtual ~MCamUtil();
    static unsigned char *getTimeStamp(unsigned char *result, int size);
    static uint64_t getTimeMiroseconds();
    static void sleep(long msec);
};

#endif /* MCAMUTIL_H_ */
