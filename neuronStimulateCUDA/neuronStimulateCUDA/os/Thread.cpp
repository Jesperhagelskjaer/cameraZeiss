#include "Thread.h"

//--------------------------------------------------------
// NAME:  Thread
// FUNC: 
//--------------------------------------------------------
Thread::Thread()
{
}

void Thread::runThread(ThreadPriority pri, string nme)
{
	priority = pri;
	name = nme;
	handle = CreateThread(NULL, 0, threadMapper, (void*) this, 0, NULL);
	if (handle != 0)
		SetThreadPriority(handle, priority);
}

Thread::Thread(ThreadPriority pri, string nme) 
{
	runThread(pri, nme);
}

//--------------------------------------------------------
// NAME:  ~Thread()
// FUNC: 
//--------------------------------------------------------
Thread::~Thread()
{
}

//--------------------------------------------------------
// NAME:  getName()
// FUNC: 
//--------------------------------------------------------
string Thread::getName()
{
  return name;
}

//--------------------------------------------------------
// NAME:  setPriority
// FUNC:  Attempts to change the Thread's priority to pri.
// RET:   true if successful, false otherwise
//--------------------------------------------------------
bool Thread::setPriority(ThreadPriority pri) 
{ 
  priority = pri;
  if(SetThreadPriority(handle, priority) == 0) return true;
  else return false;
}

//--------------------------------------------------------
// NAME:  getPriority
// FUNC:  
//--------------------------------------------------------
Thread::ThreadPriority Thread::getPriority()
{
  return priority;
}