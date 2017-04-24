/*
 * mcam.h
 *
 *  Common header files fop mcam
 *
 *  Created on: 25.02.2015
 *      Author: ggraf
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#ifndef MCAM_H_
#define MCAM_H_

#include <stdint.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>
#define MAX_PATH PATH_MAX
#endif

extern "C" {
#include <pthread.h>
#include <semaphore.h>
#include "mcamlogger.h"
}

#include "mcam_zei.h"
#include "mcam_zei_ex.h"

#include <ui_mcam.h>

#endif /* MCAM_H_ */
