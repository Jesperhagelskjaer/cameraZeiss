#include "Thread.h"
#include "LynxRecordFast.h"

class DataFileThread : public Thread
{
public:

	DataFileThread(ThreadPriority pri, string _name, LynxRecord *plxRecord, bool createSingle) :
		Thread(pri, _name)
	{
		pLynxRecord = plxRecord;
		count = 0;
		createSingleFile = createSingle;
	}

	virtual void run() 
	{
		while (1) {
			if (createSingleFile)
				pLynxRecord->AppendMemPoolIntToFile();
			else
				pLynxRecord->AppendMemPoolIntToChFiles();
			//printf("%d\r", ++count);
			Yield();
		}
	}

private:
	LynxRecord *pLynxRecord;
	int count;
	bool createSingleFile;
};
