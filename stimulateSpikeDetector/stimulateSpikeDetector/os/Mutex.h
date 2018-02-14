#pragma once
#include<windows.h>
#include<cstdlib>
#include "Semaphore.h"


//--------------------------------------------------
// M U T E X
//--------------------------------------------------
class Mutex : public Semaphore
{ 
public:
  Mutex() : Semaphore(1, 1) {}
};
