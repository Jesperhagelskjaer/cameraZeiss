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
 *    1.00 | 290612 | first created                           | SuSE 12.1
 *    1.01 | 070712 | added cond_timedwait                    | SuSE 12.1
 * --------------------------------------------------------------------- */

/***************************** Include Files *********************************/

#ifndef PTHREAD_H_
#define PTHREAD_H_

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

/************************** Variable Definitions *****************************/

//
// Threads
//

struct pthread_attr
{
  DWORD attr;
};

typedef struct pthread_attr pthread_attr_t;

typedef struct {
  HANDLE h;
} pthread_t;

#if 0
typedef struct timespec {
  time_t tv_sec; /* Seconds since 00:00:00 GMT, */
  /* 1 January 1970 */
  long tv_nsec; /* Additional nanoseconds since */
  /* tv_sec */
} timespec_t; 
#endif

#define THREAD_FUNCTION unsigned (__stdcall *)(void *)
//#define THREAD_FUNCTION  unsigned int (__cdecl *)(void *)
#define THREAD_FUNCTION_RETURN void *
#define THREAD_SPECIFIC_INDEX pthread_key_t


//
// Mutex
//

typedef struct {
  int dummy;
} pthread_mutexattr_t;

typedef struct {
  HANDLE h;
} pthread_mutex_t;

#define PTHREAD_MUTEX_INITIALIZER {NULL}

//
// Condition variables
//

typedef struct {
  HANDLE h;
} pthread_cond_t;

#define PTHREAD_COND_INITIALIZER { NULL }

typedef struct {
 int duumy;
} pthread_condattr_t;

extern int pthread_debug_level;

/************************** Functions   *************************************/

// thread
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg);
int pthread_join(pthread_t thread, void **value_ptr);

// mutex
int pthread_mutex_init(pthread_mutex_t *mutex, pthread_mutexattr_t *attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);

// cond
int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *attr);
int pthread_cond_destroy(pthread_cond_t *cond);
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_wait(pthread_cond_t *pcond, pthread_mutex_t *pmutex);
int pthread_cond_timedwait(pthread_cond_t *pcond, pthread_mutex_t *pmutex, const struct timespec *timeout);

// other
int pthread_setname_np(pthread_t thread, const char *name);

#endif

