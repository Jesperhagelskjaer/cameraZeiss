#ifndef DATAFILETHREAD_H
#define DATAFILETHREAD_H

#include "Thread.h"
#include "LynxRecord.h"

class DataFileThread : public Thread
{
public:

	DataFileThread(ThreadPriority pri, string _name, LynxRecord *plxRecord, bool createSingle) :
		Thread(pri, _name),
		m_semaStop(1, 0)
	{
		pLynxRecord = plxRecord;
		m_createSingleFile = createSingle;
		m_running = true;
		m_count = 0;
	}

	virtual void run() 
	{
		while (m_running) {
			if (m_createSingleFile)
				pLynxRecord->AppendMemPoolIntToFile();
			else
				pLynxRecord->AppendMemPoolIntToChFiles();
			printf("%d\r", ++m_count);
			Yield();
		}
		//m_semaStop.signal();
	}

	void Stop(void) 
	{
		if (m_running) {
			m_running = false;
			//pLynxRecord->SignalNewData();
			Sleep(1000);
			//m_semaStop.wait();
		}
	}

private:
	LynxRecord *pLynxRecord;
	int m_count;
	bool m_running;
	bool m_createSingleFile;
	Semaphore m_semaStop;
};

#endif
