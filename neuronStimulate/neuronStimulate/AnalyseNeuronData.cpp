///////////////////////////////////////////////////////////
//  AnalyseNeuronData.cpp
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: au288681
///////////////////////////////////////////////////////////

#include "AnalyseNeuronData.h"


AnalyseNeuronData::AnalyseNeuronData() : Monitor("AnalyseData")
{
	memset(m_maximum, 0, sizeof(m_maximum));
	memset(m_averageDly, 0, sizeof(m_averageDly));
	memset(m_average, 0, sizeof(m_average));
	m_avgIdx = 0;
	m_mode = MODE_STOP;
	m_modeLast = MODE_STOP;
	m_activeChannel = DEFAULT_ACTIVE_CH;
}

AnalyseNeuronData::~AnalyseNeuronData()
{

}

void AnalyseNeuronData::AnalyzeData(LxRecord * pLxRecord)
{
	switch (m_mode) 
	{
		case MODE_AVERAGE:
			if (m_modeLast != MODE_AVERAGE) {
				memset(m_averageDly, 0, sizeof(m_averageDly));
				memset(m_average, 0, sizeof(m_average));
				m_avgIdx = 0;
			}
			RecursiveAverage(pLxRecord);
			break;
		case MODE_ANALYSE:
			if (m_modeLast != MODE_ANALYSE)
				memset(m_maximum, 0, sizeof(m_maximum));
			SearchPattern(pLxRecord);
			break;
		default:
			// Do nothing
			break;
	}
	m_modeLast = m_mode;

}

// Compute cost as difference between maximum of active channel 
// and higest maximum of all other channels substracting measured average
double AnalyseNeuronData::CalculateCost()
{
	double highestMax = 0;
	double maximum;

	for (int channel = 0; channel < NUM_CHANNELS*NUM_BOARDS; channel++)
	{
		maximum = m_maximum[channel] - m_average[channel];
		if (maximum > highestMax && channel != m_activeChannel)
			highestMax = maximum;
	}
	return (m_maximum[m_activeChannel] - m_average[m_activeChannel] - highestMax);
}

// Search for maximum in all samples
void AnalyseNeuronData::SearchPattern(LxRecord * pLxRecord)
{
	int channel;
	int32_t absSample;
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			absSample = (int32_t)abs(pLxRecord->board[j].data[i]);
			channel = j*NUM_BOARDS + i;
			if (absSample > m_maximum[channel])
				m_maximum[channel] = absSample;
		}
}

// Computes recursive average
void AnalyseNeuronData::RecursiveAverage(LxRecord * pLxRecord)
{
	int channel;
	int32_t sample;
	int newIdx = (m_avgIdx + 1) % AVG_DELAY;
	for (int j = 0; j < NUM_BOARDS; j++) {
		for (int i = 0; i < NUM_CHANNELS; i++) {
			sample = pLxRecord->board[j].data[i];
			channel = j*NUM_BOARDS + i;
			// Computes recursive average
			m_average[channel] = ((double)(sample - m_averageDly[channel][m_avgIdx]) / AVG_DELAY) + m_average[channel];
			// Save newest sample
			m_averageDly[channel][newIdx] = sample;
		}
	}
	m_avgIdx = newIdx;
}
