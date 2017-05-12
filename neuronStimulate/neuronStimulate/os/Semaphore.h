#pragma once
#include<windows.h>
#include<cstdlib>


//--------------------------------------------------
// S E M A P H O R E
//--------------------------------------------------
class Semaphore
{
public:
  Semaphore(unsigned _maxCount, unsigned _initCount);
  virtual ~Semaphore();
  virtual void wait();
  virtual void signal();

protected:
  HANDLE semaphoreHandle;
};

