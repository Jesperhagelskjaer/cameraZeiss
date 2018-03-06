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
	Configuration(void)
	{
		// Using default values defined in defs.h
		m_NumIterations = GEN_ITERATIONS;
		m_ActiveChannel = ACTIVE_CHANNEL;
		m_ActiveTemplate = ACTIVE_TEMPLATE;
		m_LaserPort = LASER_PORT;
		m_DelayMS = (int)DELAY_MS;
		m_PauseMS = PAUSE_MS;
		m_LaserIntensity = LASER_INTENSITY;
		m_FilterType = FirFilter::FILTER_TYPE;
		m_NumParents = NUM_PARENTS;
		m_NumBindings = NUM_BINDINGS;
		m_CommonAvgRef = COMMON_REF;
		m_RandIterations = NUM_RAND_ITERATIONS;
		m_RandTemplates = NUM_RAND_TEMPLATES;
		m_EndIterations = NUM_END_ITERATIONS;
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
		printf("Template : %d\r\n", m_ActiveTemplate);
		printf("Filter : %s\r\n", FilerTypesText[m_FilterType]);
		printf("CommonRef : %d\r\n", m_CommonAvgRef);
		printf("RandIterations : %d\r\n", m_RandIterations);
		printf("RandTemplates : %d\r\n", m_RandTemplates);
		printf("EndIterations : %d\r\n", m_EndIterations);
		printf("\r\n");
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
				if (!strcmp(configName, "Template"))
					m_ActiveTemplate = (int)value;
				if (!strcmp(configName, "Filter")) {
					valInt = (int)value;
					m_FilterType = (FirFilter::TYPES)valInt;
				}
				if (!strcmp(configName, "CommonRef"))
					m_CommonAvgRef = (int)value;
				if (!strcmp(configName, "RandIterations"))
					m_RandIterations = (int)value;
				if (!strcmp(configName, "RandTemplates"))
					m_RandTemplates = (int)value;
				if (!strcmp(configName, "EndIterations"))
					m_EndIterations = (int)value;
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
	int m_ActiveTemplate;
	FirFilter::TYPES m_FilterType;
	int m_CommonAvgRef;
	int m_RandIterations;
	int m_RandTemplates;
	int m_EndIterations;
};
#endif // !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
