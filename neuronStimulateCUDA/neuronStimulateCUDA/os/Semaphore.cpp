#include "Semaphore.h"

//--------------------------------------------------------
// NAME:  Semaphore
// FUNC:  
//--------------------------------------------------------
Semaphore::Semaphore(unsigned max, unsigned init, char *name)
{ 
  semaphoreHandle = CreateSemaphore(NULL, (LONG)init, (LONG)max, (LPCWSTR)name); 
}

//--------------------------------------------------------
// NAME:  ~Semaphore
// FUNC:  
//--------------------------------------------------------
Semaphore::~Semaphore()
{
  CloseHandle(semaphoreHandle);                    
}

//--------------------------------------------------------
// NAME:  wait
// FUNC:  Attempts to take the semaphore. See
//        ms-help://MS.VSCC.2003/MS.MSDNQTR.2004APR.1033/dllproc/base/waitforsingleobject.htm
//--------------------------------------------------------
void Semaphore::wait()
{ 
  WaitForSingleObject(semaphoreHandle, INFINITE);
}

//--------------------------------------------------------
// NAME:  signal
// FUNC:  Signals the semaphore. See 
//        ms-help://MS.VSCC.2003/MS.MSDNQTR.2004APR.1033/dllproc/base/releasesemaphore.htm
//--------------------------------------------------------
void Semaphore::signal()   
{ 
  ReleaseSemaphore(semaphoreHandle, 1, NULL);
}
