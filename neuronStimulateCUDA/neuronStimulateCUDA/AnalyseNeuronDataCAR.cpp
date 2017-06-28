#include "AnalyseNeuronDataCAR.h"

AnalyseNeuronDataCAR::AnalyseNeuronDataCAR() : AnalyseNeuronData()
{
}

// Search for maximum in all samples substracting measured average and common average reference
void AnalyseNeuronDataCAR::SearchPattern(LxRecord * pLxRecord)
{
	int channel;
	double avgSample[NUM_BOARDS*NUM_CHANNELS];
	double commonAverage = 0;
	int32_t absSample;

	// Computes sample value based on neouron sample substrating channel average
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			channel = j*NUM_BOARDS + i;
			// Calculate avg value as sample value minus measured channel average
			avgSample[channel] = pLxRecord->board[j].data[i] - m_average[channel];
			commonAverage += avgSample[channel];
		}

	// Calculate common average
	commonAverage = commonAverage / (NUM_BOARDS*NUM_CHANNELS);

	// Finds maximum channel peak supstracted common average reference
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			channel = j*NUM_BOARDS + i;
			absSample = (int32_t)abs((int)floor(avgSample[channel] - commonAverage + 0.5));
			if (absSample > m_maximum[channel])
				m_maximum[channel] = absSample;
		}
}