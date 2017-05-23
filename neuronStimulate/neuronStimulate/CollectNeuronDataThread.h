///////////////////////////////////////////////////////////
//  CollectNeuronDataThread.h
//  Implementation of the Class CollectNeuronDataThread
//  Created on:      19-maj-2017 22:44:32
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_)
#define EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_

#include "SOCK_WrapperFacade.h"
#include "LynxRecord.h"
#include "DataFileThread.h"
#include "AnalyseNeuronData.h"

#define CREATE_SINGLE_FILE  false	 // Store Lynx records in one file og a file for each 32 channels

//#define NUM_RECORDS_TOTAL	180000 // Number of Lynx records to receive in 0.1 minutes (30 kHz)
//#define NUM_RECORDS_TOTAL	18000000 // Number of Lynx records to receive in 10 minutes (30 kHz)
//#define NUM_RECORDS_TOTAL	8400000 // Number of Lynx records to receive in 4 minutes (30 kHz)

//#define NUM_RECORDS			83886 // Number of Lynx records to receive in 4 minutes (30 kHz) - Max heap space 1073741842 = 1 GByte Heap
#define NUM_RECORDS			8388608 // Number of Lynx records to receive in 4 minutes (30 kHz) - Max heap space 1073741842 = 1 GByte Heap
#define UDP_PORT			26090    // 31000, 26090, 
#define RECORD_DELAY		200      // 5 Hz every 200 ms
// Needed size of memory pool including some overhead
#define MEM_POOL_SIZE  ((NUM_RECORDS+1)*(NUM_BOARDS*NUM_CHANNELS))


class CollectNeuronDataThread : public Thread
{

public:
	CollectNeuronDataThread();
	virtual ~CollectNeuronDataThread();

	virtual void run();
	void Start(ThreadPriority pri, string _name, AnalyseNeuronData *pAnalyseNeuronData);
	void Stop();

private:
	WSASession m_Session;
	SOCK_UDP m_Socket;
	Semaphore m_semaStop;
	bool m_Running;
	char *m_pBuffer;
	int m_bufSize;

	AnalyseNeuronData *m_AnalyseNeuronData;
	LynxRecord *m_LynxRecord;
	DataFileThread *m_DataFileThread;

};
#endif // !defined(EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_)
