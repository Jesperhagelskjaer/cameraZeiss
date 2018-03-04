///////////////////////////////////////////////////////////
//  SpikeDataGenerator.cpp
//  Implementation of the Class SpikeDataGenerator
//  Created on:      04-marts-2017 12:33:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include "SpikeDataGenerator.h"
#include "ProjectInfo.h"

#define NOISE  40  //amplitude

SpikeDataGenerator::SpikeDataGenerator()
{
	m_generatePulse = false;
	m_sampleOffset = 0;
	m_projectInfo = 0;
}

SpikeDataGenerator::~SpikeDataGenerator()
{

}

void SpikeDataGenerator::SetProjectInfo(ProjectInfo *pProjectInfo)
{
	m_projectInfo = pProjectInfo;
}


void SpikeDataGenerator::GenerateSampleRecord(LRECORD *pLxRecord)
{
	USED_DATATYPE *currentDataPointer;

	pLxRecord->header.packetId = 1;
	pLxRecord->header.timestampHigh = 1;
	pLxRecord->header.timestampLow = 2;
	pLxRecord->header.ttlIO = 0;
	pLxRecord->header.systemStatus = 0;

	if (m_generatePulse && m_projectInfo != 0) {
		currentDataPointer = m_projectInfo->getPredictionData();
		currentDataPointer += m_sampleOffset;
		// Generate pulse based on predition data, only 32 channels
		for (int i = 0; i < DATA_CHANNELS; i++)
			pLxRecord->board[0].data[i] = (int32_t)currentDataPointer[i];
		// Next sample to generate
		if (m_sampleOffset < (DATA_CHANNELS*(RUNTIME_DATA_LENGTH - 1)) ) 
			m_sampleOffset += (int)(DATA_CHANNELS);
	}
	else {
		// Generate random
		for (int j = 0; j < NUM_BOARDS; j++)
			for (int i = 0; i < NUM_CHANNELS; i++)
				pLxRecord->board[j].data[i] = NOISE * rand() / RAND_MAX;
				//pLxRecord->board[j].data[i] = 10 * rand() / RAND_MAX;
	}

}