/* ---------------------------------------------------------------------
 *
 *   COPYRIGHT:
 *       Copyright (c) by Ing. Buero Graf 2012-2013. All rights reserved.
 *
 *   NAME:
 *      pthread.c
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
 *    1.02 | 010613 | pthread_setname_np implementation       | Windows 7
 * --------------------------------------------------------------------- */

/***************************** Include Files *********************************/
#include "pthread.h"
#include <time.h>
   
/***************************** Constants *************************************/

/************************** Variable Definitions *****************************/

int pthread_debug_level = 0;   // 0:off 1: lock unlock 3: all



/************************** Function Prototypes ******************************/

int pthread_cond_wait_internal(pthread_cond_t *pcond, pthread_mutex_t *pmutex, DWORD waittime);

extern unsigned char *getTimeStamp(unsigned char *result);


// unused attr ...
#pragma warning( disable : 4100)

//'type cast' : truncation from 'HANDLE' to 'int'
#pragma warning( disable : 4305)


/************************** Functions   **************************************/

//
// thread create
//
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg){
  thread->h =  (HANDLE) _beginthreadex(NULL, 0, (THREAD_FUNCTION) start, arg, 0, NULL);
  if (thread->h != NULL) {
    if (pthread_debug_level>0) {
      unsigned char tbuf[32];
      printf("%s ##### created thread id=%08x\n",getTimeStamp(tbuf), (int) thread->h);
    }
    return 0;
  }
  else
    return 1;
}

//
// thread join
//
int pthread_join(pthread_t thread, void **value_ptr){
  unsigned int rc;
  if (pthread_debug_level > 0) {
      unsigned char tbuf[32];
      printf("%s ##### join thread id=%08x\n",getTimeStamp(tbuf), (int) thread.h);
  }
  if ((rc=WaitForSingleObject(thread.h,INFINITE)) != WAIT_OBJECT_0)
    return 1;
  if (pthread_debug_level >1) {
    unsigned char tbuf[32];
    printf("%s ##### close handle for thread id=%08x\n",getTimeStamp(tbuf), (int)thread.h);
  }
  return !CloseHandle(thread.h);
}


//
// mutex init
//
int pthread_mutex_init(pthread_mutex_t *mutex, pthread_mutexattr_t *attr) {
  mutex->h = CreateMutex(NULL,FALSE,NULL);
  if (mutex->h != NULL)
    return 0;
  else
    return 1;
}

//
// mutex destroy
//
int pthread_mutex_destroy(pthread_mutex_t *mutex) {
  if (mutex->h != NULL)
    CloseHandle(mutex->h);
  return 0;
}

//
// mutex lock
//
int pthread_mutex_lock(pthread_mutex_t *pmutex) {
  int rc;
  unsigned char tbuf[32];
  
  if (pthread_debug_level > 1)
    printf("%s ##### thread=%08x: BEFORE LOCK mutex=%08x\n",getTimeStamp(tbuf), (int)GetCurrentThreadId(), (int)pmutex->h);
  rc= WaitForSingleObject(pmutex->h,INFINITE);
  if (pthread_debug_level >1)
      printf("%s ##### thread=%08x: LOCKED mutex=%08x rc=%d\n",getTimeStamp(tbuf), (int)GetCurrentThreadId(), (int)pmutex->h,rc);
  if (rc == 0)
    return 0;
  else
    return 1;
}

//
// mutex unlock
//
int pthread_mutex_unlock(pthread_mutex_t *pmutex) {
  unsigned char tbuf[32];
  int rc;

  rc = ReleaseMutex(pmutex->h);
  
  if (pthread_debug_level >1)
    printf("%s ##### thread=%08x: UNLOCK mutex=%08x\n",getTimeStamp(tbuf), (int)GetCurrentThreadId(), (int)pmutex->h);
  if (rc != 0)
    return 0;
  else
    return 1;
}


//
// cond init
//
int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *attr) {
  HANDLE h = CreateEvent(NULL,FALSE,FALSE,NULL);
  if (h != NULL) {
    cond->h = h;
    return 0;
  }
  else
    return 1;
}

//
// cond destroy
//
int pthread_cond_destroy(pthread_cond_t *cond) {
  if (cond->h != NULL)
    CloseHandle(cond->h);
  return 0;
}

//
// cond_wait ***INTERNAL***  release mutex and wait for cond variable to be signaled
//
int pthread_cond_wait_internal(pthread_cond_t *pcond, pthread_mutex_t *pmutex, DWORD waittime) {
  unsigned int rc;
  unsigned char tbuf[32];
  
  if (pthread_debug_level > 1)
    printf("%s ##### thread=%08x: WAIT on cond=%08x UNLOCK mutex=%08x\n",getTimeStamp(tbuf), (int)GetCurrentThreadId(), (int)pcond->h, (int)pmutex->h);
  
  // signal mutex and wait for conditional variable to be signaled
  if ((rc = SignalObjectAndWait(pmutex->h, pcond->h, waittime, FALSE)) != WAIT_OBJECT_0) {
    DWORD rc2=GetLastError();
    if (pthread_debug_level)
      printf("### error cond wait rc=%d  lastError=%d\n",rc,rc2);
    return 1; // fixme error code mapping
  }  
  if ((rc = WaitForSingleObject(pmutex->h,waittime)) != WAIT_OBJECT_0) {
    if (pthread_debug_level)
      printf("### error cond wait WaitForSingleObject rc=%d\n",rc);
    return 1; // fixme error code mapping
  }  
  
  if (pthread_debug_level > 0)
    printf("%s ##### thread=%08x: SIGNALED on cond=%08x LOCK mutex=%08x\n",getTimeStamp(tbuf), (int)GetCurrentThreadId(), (int)pcond->h, (int)pmutex->h);
  return 0;
}


//
// cond_wait release mutex and wait for cond variable to be signaled
//
int pthread_cond_wait(pthread_cond_t *pcond, pthread_mutex_t *pmutex) {
  return pthread_cond_wait_internal(pcond, pmutex, INFINITE);
}


//
// release mutex and wait for cond variable to be signaled with timeout
// Warning: timeout is ABSOLUTE TIME
// One second granularity only!
//
int pthread_cond_timedwait(pthread_cond_t *pcond, pthread_mutex_t *pmutex, const struct timespec *timeout) {
  time_t t = time(NULL);
  time_t wtime; // timeout in ms
  
  if (timeout->tv_sec > t) 
    wtime = (timeout->tv_sec - t) * 1000;
  else
    wtime=1000;
  if (pthread_debug_level > 2)
    printf("### pthread_cond_timedwait TIME=%d ms\n",wtime);
  return pthread_cond_wait_internal(pcond, pmutex, (DWORD) wtime);
}

//
// signal cont variable
//
int pthread_cond_signal(pthread_cond_t *pcond) {
  unsigned char tbuf[32];
  if (pthread_debug_level >0)
    printf("%s ##### thread=%08x: SIGNAL cond=%08x\n",getTimeStamp(tbuf), GetCurrentThreadId(), (int) pcond->h);
  return !SetEvent(pcond->h);
}

const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
   DWORD dwType; // Must be 0x1000.
   LPCSTR szName; // Pointer to name (in user addr space).
   DWORD dwThreadID; // Thread ID (-1=caller thread).
   DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)


// name threads
//
// In Windows: Only visible in Visual Studio Debugging
//
int pthread_setname_np(pthread_t thread, const char *name) {
  int ret= 0;

  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name;
  
  info.dwThreadID = GetThreadId(thread.h);
  info.dwFlags = 0;
  //printf("##### thread=%08x set to %s\n", (int)info.dwThreadID,  name);
  
   __try
  {
    RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
  }
  __except(EXCEPTION_EXECUTE_HANDLER)
  {
  }
  
  return ret;
}