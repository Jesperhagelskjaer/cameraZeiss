#pragma once
#include <windows.h>
#include <cstdint>
#include <stdio.h>

class TimeMeasure
{
public:
	TimeMeasure() 
	{
		startTime_ = getTimeMicroSec();
	};

	uint64_t getTimeMicroSec()
	{
		uint64_t ft64;
		FILETIME ft;
		GetSystemTimeAsFileTime(&ft);
		ft64 = ft.dwHighDateTime;
		ft64 = (ft64 << 32) | ft.dwLowDateTime;
		return ft64 / 10; // microseconds
	}

	unsigned char *getTimeStamp(unsigned char *result, int size) 
	{
		SYSTEMTIME tm;
		GetLocalTime(&tm);
		sprintf((char*)result, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
			tm.wYear, tm.wMonth, tm.wDay, tm.wHour, tm.wMinute, tm.wSecond,
			tm.wMilliseconds);
		result[size - 1] = '\0';
		return result;
	}
	
	void setStartTime(void)
	{
		startTime_ = getTimeMicroSec();
	}

	void printDuration(char *text)
	{
		uint64_t endTime = getTimeMicroSec();
		printf("%s duration %llu us\r\n", text, endTime - startTime_);
	}

private:
	uint64_t startTime_; 

};
