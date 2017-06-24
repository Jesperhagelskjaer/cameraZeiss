#pragma once
#include<windows.h>


//--------------------------------------------------
// T I M E R
//--------------------------------------------------
class SleepTimer
{
public:
  SleepTimer()               {}
  ~SleepTimer()              {}
  void sleep(DWORD ms)
  {
    if(ms == 0) return;
    else Sleep(ms);
  }
};