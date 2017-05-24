///////////////////////////////////////////////////////////
//  TestDataGenerator.cpp
//  Implementation of the Class TestDataGenerator
//  Created on:      24-maj-2017 11:01:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include "TestDataGenerator.h"


TestDataGenerator::TestDataGenerator()
{
	m_generatePulse = false;
}


TestDataGenerator::~TestDataGenerator()
{

}

void TestDataGenerator::GenerateSampleRecord(LRECORD *pLxRecord)
{
	pLxRecord->header.packetId = 1;
	pLxRecord->header.timestampHigh = 1;
	pLxRecord->header.timestampLow = 2;
	pLxRecord->header.ttlIO = 0;
	pLxRecord->header.systemStatus = 0;

	if (m_generatePulse) {

		// Generate pulse
		for (int j = 0; j < NUM_BOARDS; j++)
			for (int i = 0; i < NUM_CHANNELS; i++)
				pLxRecord->board[j].data[i] = 5000 * (j + 1) + i;
	}
	else {
		// Generate random
		for (int j = 0; j < NUM_BOARDS; j++)
			for (int i = 0; i < NUM_CHANNELS; i++)
				pLxRecord->board[j].data[i] = i;
	}

}