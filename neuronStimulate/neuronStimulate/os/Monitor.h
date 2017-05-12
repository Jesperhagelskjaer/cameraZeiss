#pragma once
#include<windows.h>
#include<cstdlib>
#include "Semaphore.h"


//--------------------------------------------------
// M O N I T O R
//--------------------------------------------------
class Monitor : public Semaphore
{ 
public:
  Monitor() : Semaphore(1, 1) {}
  void enter()  { wait();   }
  void exit()   { signal(); }
};
