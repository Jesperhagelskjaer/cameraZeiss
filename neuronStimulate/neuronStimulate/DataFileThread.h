///////////////////////////////////////////////////////////
//  DataFileThread.h
//  Implementation of the Class Configuration
//  Created on:      20-maj-2017 08:53:06
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#ifndef DATAFILETHREAD_H
#define DATAFILETHREAD_H

#include "Thread.h"
#include "LynxRecord.h"

class DataFileThread : public Thread
{
public:

	DataFileThread(LynxRecord *plxRecord, bool createSingle) :
		Thread(),
		m_semStop(1, 0, "SemaFileThread")
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
			//printf("%d\r", ++m_count);
			Yield();
		}
		printf("DataFileThread stopped\r\n");
		m_semStop.signal();
	}

	void Stop(void) 
	{
		if (m_running) {
			m_running = false;
			pLynxRecord->SignalNewData();
			m_semStop.wait();
		}
	}

private:
	Semaphore m_semStop;
	LynxRecord *pLynxRecord;
	int m_count;
	bool m_running;
	bool m_createSingleFile;
};

#endif
