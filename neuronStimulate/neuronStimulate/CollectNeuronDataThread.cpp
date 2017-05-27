///////////////////////////////////////////////////////////
//  CollectNeuronDataThread.cpp
//  Implementation of the Class CollectNeuronDataThread
//  Created on:      19-maj-2017 22:44:32
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#include <ctime>
#include <iostream>
using namespace std;
#include "CollectNeuronDataThread.h"

// To be defined for testing
#define TEST_GENERATOR_

CollectNeuronDataThread::CollectNeuronDataThread() :
	Thread(),
	m_semaStop(1, 0, "SemaNeuronDataThread"),
	m_Socket(),
	m_Session()
{
	m_Running = false;
	m_bufSize = 0;
	m_pBuffer = 0;

	m_LynxRecord = 0;
	m_DataFileThread = 0;
	m_AnalyseNeuronData = 0;
	m_TestDataGenerator = new TestDataGenerator();
}

CollectNeuronDataThread::~CollectNeuronDataThread()
{
	delete m_TestDataGenerator;
}

void CollectNeuronDataThread::Start(ThreadPriority pri, string _name, AnalyseNeuronData *pAnalyseNeuronData)
{
	char dataFileNames[50];
	char dataFileName[50];
	char headerFileName[50];
	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);

	m_AnalyseNeuronData = pAnalyseNeuronData;
	m_LynxRecord = new LynxRecord();

	// Naming cost, data and header text files
	sprintf(dataFileNames, "data\\LX_%d%02d%02d_%02d%02d%02d_D",
		now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
		now->tm_hour, now->tm_min, now->tm_sec);
	sprintf(dataFileName, "%s.txt", dataFileNames);
	sprintf(headerFileName, "data\\LX_%d%02d%02d_%02d%02d%02d_H.txt",
		now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
		now->tm_hour, now->tm_min, now->tm_sec);
	sprintf(m_costFileName, "data\\LX_%d%02d%02d_%02d%02d%02d_C.txt",
		now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
		now->tm_hour, now->tm_min, now->tm_sec);

	m_bufSize = m_LynxRecord->GetBuffer(&m_pBuffer);
	// Create memory pool
	if (!m_LynxRecord->CreateMemPool(MEM_POOL_SIZE)) {
		cout << "Could not allocate memory of size " << MEM_POOL_SIZE * sizeof(int) << endl;
		exit(0);
	}

	m_LynxRecord->OpenHeaderFile(headerFileName);

	if (!CREATE_SINGLE_FILE)
	{
		if (!m_LynxRecord->OpenChannelDataFiles(dataFileNames))
		{
			cout << "Could not open data files " << dataFileNames << endl;
			exit(0);
		}
	}
	else
		m_LynxRecord->OpenDataFile(dataFileName);

	// Start data file theread to save captured neuron data in files
	m_DataFileThread = new DataFileThread(m_LynxRecord, CREATE_SINGLE_FILE);

	m_DataFileThread->runThread(Thread::PRIORITY_NORMAL, "FileThread");
	m_Running = true;
	runThread(pri, _name);

}

void CollectNeuronDataThread::run()
{
	unsigned int num = 0;
	bool running = true;
	
	try
	{ 

#ifdef TEST_GENERATOR_
		m_LynxRecord->CreatTestData(0);

		while (running)
		{
			//m_LynxRecord->CreatTestData(num);
			
			if (m_AnalyseNeuronData->GetMode() == AnalyseNeuronData::MODE_ANALYSE)
				m_TestDataGenerator->SetPulseActive(true);
			else
				m_TestDataGenerator->SetPulseActive(false);
			m_TestDataGenerator->GenerateSampleRecord((LRECORD *)m_pBuffer);

			if (m_LynxRecord->AppendDataToMemPool())
			{
				m_AnalyseNeuronData->AnalyzeData(m_LynxRecord->GetLxRecord());
				m_LynxRecord->AppendHeaderToFile(m_AnalyseNeuronData->GetMode());
				if (!m_Running) {
					//cout << "Stopping Data File Thread" << endl;
					m_DataFileThread->Stop();
					running = false;
				}
				--num;
				if (num % 60 == 0)
					Sleep(1);
			}
			else
			{
				Sleep(500);
				cout << ".";
			}
			//printf("%d\r", --num);
			Yield();


		}
#else
		// Receive UPD packages from port
		m_Socket.Bind(UDP_PORT);
		while (running)
		{
			// Blocking until new record received
			sockaddr_in add = m_Socket.RecvFrom(m_pBuffer, m_bufSize);
			if (m_LynxRecord->CheckSumOK()) // Verify using xor checksum of record data
			{
				//m_LynxRecord->AppendDataFloatToFile();
				m_AnalyseNeuronData->AnalyzeData(m_LynxRecord->GetLxRecord());
				if (m_LynxRecord->AppendDataToMemPool()) {
					m_LynxRecord->AppendHeaderToFile(m_AnalyseNeuronData->GetMode());
					num++;
				}
				else
					cout << "Lynx record memory space error" << endl;
			}
			else
			{
				cout << "Lynx record checksum error" << endl;
			}
			printf("%d\r", num);
		
			if (!m_Running) {
				cout << "Stopping Data File Thread" << endl;
				m_DataFileThread->Stop();
				running = false;
			}

		}
		//cout << "Writing data to file: " << dataFileName;
		//m_LynxRecord->AppendMemPoolIntToFile();
#endif
		/*
		while (!m_LynxRecord->isMemPoolEmpty(true))
		{
			cout << "Wait for empty memory pool" << endl;
			Sleep(500);
		}
		*/
		

	}
	catch (std::system_error& e)
	{
		std::cout << e.what();
	}

	// Stop thread storing data in files
	m_LynxRecord->CloseFiles();
	cout << "CollectNeuronDataThread stopped, Files closed" << endl;
	m_semaStop.signal();
}

void CollectNeuronDataThread::Stop()
{
	if (m_Running == true) 
	{
		m_Running = false;
		// Wait for collection thread to complete
		m_semaStop.wait(); 

		delete m_DataFileThread;
		delete m_LynxRecord;
		m_DataFileThread = 0;
		m_LynxRecord = 0;
	}
}