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
  Monitor(char *name) : Semaphore(1, 1, name) {}
  void enter()  { wait();   }
  void exit()   { signal(); }
};
