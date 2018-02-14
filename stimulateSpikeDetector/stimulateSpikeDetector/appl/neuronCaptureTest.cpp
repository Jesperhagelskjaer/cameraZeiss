// neuronStimulate.cpp : Defines the entry point for the console application.
//
/*********************************************
** Digital Lynx SX, Client UDP Demo program
** Created: 3/4 2017 by Kim Bjerge, AU
** Modified:
** 4/4 2017 Added header and data/time file naming
**********************************************/
#include <ctime>
#include "SOCK_WrapperFacade.h"
#include "LynxRecord.h"
#include "Serial.h"
#include "BaseTimer.h"
#include "DataFileThread.h"

#define PORT_NR				8
#define BAUD_RATE			115200
#define CREATE_SINGLE_FILE  false	 // Store Lynx records in one file og a file for each 32 channels

#define NUM_RECORDS_TOTAL	180000 // Number of Lynx records to receive in 10 minutes (30 kHz)
//#define NUM_RECORDS_TOTAL	18000000 // Number of Lynx records to receive in 10 minutes (30 kHz)
//#define NUM_RECORDS_TOTAL	8400000 // Number of Lynx records to receive in 4 minutes (30 kHz)

//#define NUM_RECORDS			83886 // Number of Lynx records to receive in 4 minutes (30 kHz) - Max heap space 1073741842 = 1 GByte Heap
#define NUM_RECORDS			8388608 // Number of Lynx records to receive in 4 minutes (30 kHz) - Max heap space 1073741842 = 1 GByte Heap
#define UDP_PORT			26090    // 31000, 26090, 
#define RECORD_DELAY		200      // 5 Hz every 200 ms
// Needed size of memory pool including some overhead
#define MEM_POOL_SIZE  ((NUM_RECORDS+1)*(NUM_BOARDS*NUM_CHANNELS))

//----------------------------------------------------------------------------------------------
// Only for test
//----------------------------------------------------------------------------------------------
void TestLynxRecord(LynxRecord &lynxRecord)
{
	/* For testing only */
	lynxRecord.CreatTestData(0);
	lynxRecord.AppendHeaderToFile();
	lynxRecord.AppendDataFloatToFile();
	lynxRecord.CreatTestData(100);
	lynxRecord.AppendHeaderToFile();
	lynxRecord.AppendDataFloatToFile();
	lynxRecord.CreatTestData(1000);
	lynxRecord.AppendHeaderToFile();
	lynxRecord.AppendDataFloatToFile();
	lynxRecord.CloseFiles();
}

// Function to test the serial port
bool TestSerialPort(int port, int baudRate)
{
	//char data[5000];
	//int num;

	CSerial* s = new CSerial();

	if (!s->Open(port, baudRate))
	{
		std::cout << "Could not open COM" << port << endl;
		return false;
	}

	// Sending a string on port
	//sprintf(data,"?help\r\n"); //>laserPM 0.0
	//sprintf(data, ">laserPM 0.2\r\n"); //>laserPM 0.0
	//s->SendData(data, strlen(data)+1);
	//Sleep(2000);
	//while (s->ReadDataWaiting() == 0) Sleep(100);
	//num = s->ReadData(data, 5000);
	//data[num] = 0;
	//cout << data;
	s->Close();

	delete s;
	return true;
}

//----------------------------------------------------------------------------------------------
// Functions to control laser on the USB serial port
//----------------------------------------------------------------------------------------------
bool turnLaserOn(CSerial *s)
{
	char data[20];
	int num;

	// Sending a string on port - laser on
	sprintf(data, ">laserPM 0.2\r\n");
	num = s->SendData(data, (int)strlen(data) + 1);
	if (num != strlen(data) + 1) {
		cout << "Could not turn laser on" << endl;
		return false;
	}
	return true;
}

bool turnLaserOff(CSerial *s)
{
	char data[20];
	int num;

	// Sending a string on port - laser off
	sprintf(data, ">laserPM 0.0\r\n");
	num = s->SendData(data, (int)strlen(data) + 1);
	if (num != strlen(data) + 1) {
		cout << "Could not turn laser off" << endl;
		return false;
	}
	return true;
}

//----------------------------------------------------------------------------------------------
// Timer to control laser
//------------------------------------------ derived from the base class
class CMsgTickTimer : public CBaseTimer {
public:
	CMsgTickTimer(int nIntervalMs, LPCSTR szMsg) {
		m_nIntervalMs = nIntervalMs;
		strncpy(m_szMsg, szMsg, sizeof(m_szMsg));
	}
	void OnTimer() {
		printf("---------- %s ----------- \r\n", m_szMsg);
	}

private:
	char m_szMsg[100];
};

//----------------------------------------------------------------------------------------------

int neuronCaptureTest(void)
{

	try
	{
		WSASession Session;
		SOCK_UDP Socket;
		LynxRecord lynxRecord;
		CMsgTickTimer cTimer(RECORD_DELAY, "200ms laser on");
		/* Not used now
		CSerial* serialPort = new CSerial();

		if (!serialPort->Open(PORT_NR, BAUD_RATE))
		{
		std::cout << "Could not open COM" << PORT_NR << endl;
		exit(-1);
		}
		*/

		char *pBuffer;
		int bufSize;
		unsigned int num = NUM_RECORDS_TOTAL;
		char dataFileNames[50];
		char dataFileName[50];
		char headerFileName[50];
		time_t t = time(0);   // get time now
		struct tm * now = localtime(&t);

		// Naming data and header text files
		sprintf(dataFileNames, "LX_%d%02d%02d_%02d%02d%02d_D",
			now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
			now->tm_hour, now->tm_min, now->tm_sec);
		sprintf(dataFileName, "%s.txt", dataFileNames);
		sprintf(headerFileName, "LX_%d%02d%02d_%02d%02d%02d_H.txt",
			now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
			now->tm_hour, now->tm_min, now->tm_sec);

		/* For test only
		cTimer.Start();
		for (int j=0; j< 40; j++) {
		Sleep(100);
		cout << "Hi there" << endl;
		}
		exit(0);
		*/

		// Initialization
		bufSize = lynxRecord.GetBuffer(&pBuffer);
		// Create memory pool
		if (!lynxRecord.CreateMemPool(MEM_POOL_SIZE)) {
			cout << "Could not allocate memory of size " << MEM_POOL_SIZE * sizeof(int) << endl;
			exit(0);
		}

		lynxRecord.OpenHeaderFile(headerFileName);

		if (!CREATE_SINGLE_FILE)
		{
			if (!lynxRecord.OpenChannelDataFiles(dataFileNames))
			{
				cout << "Could not open data files " << dataFileNames << endl;
				exit(0);
			}
		}
		else
			lynxRecord.OpenDataFile(dataFileName);

		DataFileThread fileThread(&lynxRecord, CREATE_SINGLE_FILE);
		fileThread.runThread(Thread::PRIORITY_NORMAL, "FileThread");

		//TestLynxRecord(lynxRecord);
		//TestSerialPort(8, 115200);
		//turnLaserOn(serialPort);
#if 1
		lynxRecord.CreatTestData(0);

		while (num > 0)
		{
			if (lynxRecord.AppendDataToMemPool())
			{
				lynxRecord.AppendHeaderToFile();
				--num;
				if (num % 60 == 0)
					//if (num%60 == 0)
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
		Socket.Bind(UDP_PORT);
		while (num > 0)
		{
			// Blocking until new record received
			sockaddr_in add = Socket.RecvFrom(pBuffer, bufSize);
			if (lynxRecord.CheckSumOK()) // Verify using xor checksum of record data
			{
				//lynxRecord.AppendDataFloatToFile();
				if (lynxRecord.AppendDataToMemPool()) {
					lynxRecord.AppendHeaderToFile();
					--num;
				}
				else
					cout << "Lynx record memory space error" << endl;
			}
			else
			{
				cout << "Lynx record checksum error" << endl;
			}
			printf("%d\r", num);
		}
		//cout << "Writing data to file: " << dataFileName;
		//lynxRecord.AppendMemPoolIntToFile();
#endif
		while (!lynxRecord.isMemPoolEmpty(true))
		{
			cout << "Wait for empty memory pool" << endl;
			Sleep(500);
		}
		cout << "Closing file: " << dataFileName;
		lynxRecord.CloseFiles();

		//turnLaserOff(serialPort);
		//serialPort->Close();
		//delete serialPort;

	}
	catch (std::system_error& e)
	{
		std::cout << e.what();
	}

	return 0;
}
