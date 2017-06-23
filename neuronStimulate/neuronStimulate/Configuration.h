///////////////////////////////////////////////////////////
//  Configuration.h
//  Implementation of the Class Configuration
//  Created on:      20-maj-2017 08:53:06
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
#define EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_
#include "FirFilter.h"

class Configuration
{

public:
	Configuration(int numIterations, int activeCh, int port, int delay, int pause, 
		          float intensity, FirFilter::TYPES type, int numParents, int numBindings)
	{
		m_NumIterations = numIterations;
		m_ActiveChannel = activeCh;
		m_LaserPort = port;
		m_DelayMS = delay;
		m_PauseMS = pause;
		m_LaserIntensity = intensity;
		m_FilterType = type;
		m_NumParents = numParents;
		m_NumBindings = numBindings;
	}

	void Print(void)
	{
		printf("Bindings : %d\r\n", m_NumBindings);
		printf("Parents : %d\r\n", m_NumParents);
		printf("Iterations : %d\r\n", m_NumIterations);
		printf("Port : %d\r\n", m_LaserPort);
		printf("Intensity : %.1f\r\n", m_LaserIntensity);
		printf("On : %d\r\n", m_DelayMS);
		printf("Off : %d\r\n", m_PauseMS);
		printf("Channel : %d\r\n", m_ActiveChannel);
		printf("Filter : %s\r\n\r\n", FilerTypesText[m_FilterType]);
	}
	
	void ReadConfiguration(char *fileName)
	{
		char configName[50];
		float value;
		char seperator;
	    int valInt;
		FILE *hConfigFile;
		hConfigFile = fopen(fileName, "r");
		if (hConfigFile != NULL) {
			printf("Reading configuration file: %s\r\n\r\n", fileName);
			while (fscanf(hConfigFile, "%s %c %f", configName, &seperator, &value) != EOF)
			{
				if (!strcmp(configName, "Bindings"))
					m_NumBindings = (int)value;
				if (!strcmp(configName, "Parents"))
					m_NumParents = (int)value;
				if (!strcmp(configName, "Iterations"))
					m_NumIterations = (int)value;
				if (!strcmp(configName, "Port"))
					m_LaserPort = (int)value;
				if (!strcmp(configName, "Intensity"))
					m_LaserIntensity = value;
				if (!strcmp(configName, "On"))
					m_DelayMS = (int)value;
				if (!strcmp(configName, "Off"))
					m_PauseMS = (int)value;
				if (!strcmp(configName, "Channel"))
					m_ActiveChannel = (int)value;
				if (!strcmp(configName, "Filter")) {
					valInt = (int)value;
					m_FilterType = (FirFilter::TYPES)valInt;
				}
			}
			fclose(hConfigFile);
		}
		else
		{
			printf("Could not open file: %s\r\n", fileName);
		}
	}

	int m_NumBindings;
	int m_NumParents;
	int m_NumIterations;
	int m_LaserPort;
	float m_LaserIntensity;
	int m_DelayMS;
	int m_PauseMS;
	int m_ActiveChannel;
	FirFilter::TYPES m_FilterType;
};
#endif // !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
