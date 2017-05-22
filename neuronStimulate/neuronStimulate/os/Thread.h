#pragma once
#include<windows.h>
#include<string>

using namespace std;

//--------------------------------------------------
// T H R E A D
//--------------------------------------------------
class Thread
{
public:
  // Thread priorities
  enum ThreadPriority{
    PRIORITY_LOW          = THREAD_PRIORITY_LOWEST,
    PRIORITY_BELOW_NORMAL = THREAD_PRIORITY_BELOW_NORMAL,
    PRIORITY_NORMAL       = THREAD_PRIORITY_NORMAL,
    PRIORITY_ABOVE_NORMAL = THREAD_PRIORITY_ABOVE_NORMAL,
    PRIORITY_HIGH         = THREAD_PRIORITY_HIGHEST,
  };

  Thread();
  Thread(ThreadPriority pri, string _name);
  ~Thread();

  void runThread(ThreadPriority pri, string nme);
  bool setPriority(ThreadPriority pri); 
  ThreadPriority getPriority();
  string getName();
  virtual void run() = 0;

private:
  ThreadPriority priority;
  string name;
  HANDLE handle;
  static DWORD WINAPI threadMapper(void* p) 
  { 
    ((Thread*)p)->run();
	ExitThread(0);
	return 0;
  }

};