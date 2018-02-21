///////////////////////////////////////////////////////////
//  AnalyseNeuronData.cpp
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include "AnalyseNeuronData.h"


AnalyseNeuronData::AnalyseNeuronData() : 
    Monitor("AnalyseData"), 
	m_semaAnalyseComplete(1, 0, "SemaAnalyse")
{
	memset(m_maximum, 0, sizeof(m_maximum));
	memset(m_averageDly, 0, sizeof(m_averageDly));
	memset(m_average, 0, sizeof(m_average));
	m_avgIdx = 0;
	m_mode = MODE_STOP;
	m_modeLast = MODE_STOP;
	m_activeChannel = DEFAULT_ACTIVE_CH;
	m_analyseSamples = ANALYSE_SAMPLES;
	m_countSamples = m_analyseSamples;
	costStream = NULL;
	filterEnabled = false;
}

AnalyseNeuronData::~AnalyseNeuronData()
{
	CloseCostFile();
}

int AnalyseNeuronData::OpenCostFile(char *fileName)
{
	bool result = true;
	CloseCostFile(); // Close file if already opened
	costStream = fopen(fileName, "w"); // Truncate file 
	if (costStream == NULL)
	{
		printf("Unable to open file: %s\r\n", fileName);
		result = false;
	}
	return result;
}

void AnalyseNeuronData::CloseCostFile(void)
{
	if (costStream != NULL) {
		fflush(costStream);
		fclose(costStream);
	}
	costStream = NULL;
}

int AnalyseNeuronData::AppendCostToFile(double cost)
{
	int i, res = -1;
	if (costStream != NULL) {
		res = fprintf(costStream, "%.1f ", cost);
		res += fprintf(costStream, "%d ", m_activeChannel);
		for (i = 0; i < NUM_BOARDS*NUM_CHANNELS; i++)
			res += fprintf(costStream, "%d ", m_maximum[i]);
		for (i = 0; i < NUM_BOARDS*NUM_CHANNELS; i++)
			res += fprintf(costStream, "%.1f ", m_average[i]);
		res += fprintf(costStream, "\r\n");
	}
	return res;
}

LxRecord *AnalyseNeuronData::FilterData(LxRecord * pLxRecord)
{
	/* Output to filtLxRecord
	LxRecord filtLxRecord;
	if (filterEnabled) {
		for (int j = 0; j < NUM_BOARDS; j++) {
			firFilter[j].filter(pLxRecord->board[j].data, filtLxRecord.board[j].data);
		}
		return &filtLxRecord;
	}
	*/
	// Overwrites input data
	if (filterEnabled) {
		for (int j = 0; j < NUM_BOARDS; j++) {
			firFilter[j].filter(pLxRecord->board[j].data, pLxRecord->board[j].data);
		}
	}
	return pLxRecord;
}

void AnalyseNeuronData::AnalyzeData(LxRecord * pLxRecord)
{
	enter();

	//pLxRecord = FilterData(pLxRecord);
	
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
			if (m_modeLast != MODE_ANALYSE) {
				memset(m_maximum, 0, sizeof(m_maximum));
				m_countSamples = m_analyseSamples;
			}
			SearchPattern(pLxRecord);
			m_countSamples--;
			if (m_countSamples == 0) {
				// Signal that m_analyseSamples has been searched
				m_mode = MODE_STOP;
				m_semaAnalyseComplete.signal();
			}
			break;
		default:
			// Do nothing
			break;
	}
	m_modeLast = m_mode;

	exit();

}

// Compute cost as difference between maximum of active channel 
// and higest maximum of all other channels
// where measured average already is substracted
double AnalyseNeuronData::CalculateCost()
{
	enter();

	double highestMax = 0;
	double maximum;
	double cost;

	for (int channel = 0; channel < NUM_CHANNELS*NUM_BOARDS; channel++)
	{
		maximum = m_maximum[channel];
		if (maximum > highestMax && channel != m_activeChannel)
			highestMax = maximum;
	}
	cost = m_maximum[m_activeChannel] - highestMax;
	
	exit();

	return cost;
}

// Search for maximum in all samples substracting measured average
void AnalyseNeuronData::SearchPattern(LxRecord * pLxRecord)
{
	int channel;
	int32_t absSample;
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			channel = j*NUM_BOARDS + i;
			// Calculate absolute value as sample minus measured average
			absSample = abs((int32_t)round(pLxRecord->board[j].data[i] - m_average[channel])); 
			if (absSample > m_maximum[channel])
				m_maximum[channel] = absSample;
		}
}

// Computes recursive average
void AnalyseNeuronData::RecursiveAverage(LxRecord * pLxRecord)
{
	int channel;
	int32_t sample;
	for (int j = 0; j < NUM_BOARDS; j++) {
		for (int i = 0; i < NUM_CHANNELS; i++) {
			sample = pLxRecord->board[j].data[i];
			channel = j*NUM_BOARDS + i;
			// Computes recursive average
			m_average[channel] = ((double)(sample - m_averageDly[channel][m_avgIdx]) / AVG_DELAY) + m_average[channel];
			// Save newest sample on oldest place
			m_averageDly[channel][m_avgIdx] = sample;
		}
	}
	m_avgIdx = (m_avgIdx + 1) % AVG_DELAY;
}
