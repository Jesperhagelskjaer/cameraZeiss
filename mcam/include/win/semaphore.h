/* ---------------------------------------------------------------------
 *
 *   COPYRIGHT:
 *       Copyright (c) by Ing. Buero Graf 2012. All rights reserved.
 *
 *   NAME:
 *      pthread.h
 *
 *   DESCRIPTION:
 *       Lightweight Emulation of POSIX thread for windows
 *
 *
 *   HISTORY:
 *
 *   Vers. |  Date  | Change                                  | System
 * --------|--------|-----------------------------------------|----------
 *    1.00 | 011112 | first created                           | Windows 7
 * --------------------------------------------------------------------- */


#ifndef SEMAPHORE_H_
#define SEMAPHORE_H_

#include <io.h>
//#define _WIN32_WINNT 0x502 /* Require Windows NT5 (2K, XP, 2K3) */
#ifndef _WIN32_WINNT
  #define _WIN32_WINNT NTDDI_WIN7 /* Require Windows 7 */
#endif  
#include <process.h>
#include <windows.h>
#include <time.h>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <pthread.h>


#define SEM_VALUE_MAX     INT_MAX

struct sem_t_
{
  HANDLE sem;
};

typedef struct sem_t_ sem_t;

int sem_init(sem_t *sem, int pshared, unsigned int value);
int sem_destroy(sem_t *sem);

int sem_wait(sem_t *sem);
int sem_post(sem_t *sem);


#endif
