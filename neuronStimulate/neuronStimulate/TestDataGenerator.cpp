///////////////////////////////////////////////////////////
//  TestDataGenerator.cpp
//  Implementation of the Class TestDataGenerator
//  Created on:      24-maj-2017 11:01:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include "TestDataGenerator.h"



TestDataGenerator::TestDataGenerator()
{
	const double fs = 30000;
	const double fc = 300;
	m_omega = 2 * 3.14159265359 * fc / fs; 
	m_generatePulse = false;
	m_n = 0;
}


TestDataGenerator::~TestDataGenerator()
{

}

int32_t TestDataGenerator::GenSine(int channel)
{
	double sample = (100+(channel*10))*sin(m_omega*m_n);
	return int32_t(round(sample));
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
				pLxRecord->board[j].data[i] = GenSine(NUM_CHANNELS*j + i);
		m_n++;
	}
	else {
		m_n = 0;
		// Generate random
		for (int j = 0; j < NUM_BOARDS; j++)
			for (int i = 0; i < NUM_CHANNELS; i++)
				pLxRecord->board[j].data[i] = 100*rand()/RAND_MAX;
	}

}