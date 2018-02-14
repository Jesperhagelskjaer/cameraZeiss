#include "AnalyseNeuronDataCMR.h"
#include <algorithm>
#include <vector>

AnalyseNeuronDataCMR::AnalyseNeuronDataCMR() : AnalyseNeuronData()
{
}

double AnalyseNeuronDataCMR::findMedian(double *pVaules, int size)
{
	int mid = (int)floor((size - 1) / 2);
	std::vector<double> vect(pVaules, pVaules + size);

	// Sort vector by increasing numbers
	std::sort(vect.begin(), vect.end());

	// Return median value in vector
	return vect[mid];
}

// Search for maximum in all samples substracting measured average and common median reference
void AnalyseNeuronDataCMR::SearchPattern(LxRecord * pLxRecord)
{
	int channel;
	double avgSample[NUM_BOARDS*NUM_CHANNELS];
	double commonMedian = 0;
	int32_t absSample;

	// Computes sample value based on neouron sample substrating channel average
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			channel = j*NUM_BOARDS + i;
			// Calculate avg value as sample value minus measured channel average
			avgSample[channel] = pLxRecord->board[j].data[i] - m_average[channel];
		}

	// Calculate common median
	commonMedian = findMedian(avgSample, NUM_BOARDS*NUM_CHANNELS);

	// Finds maximum channel peak supstracted common average reference
	for (int j = 0; j < NUM_BOARDS; j++)
		for (int i = 0; i < NUM_CHANNELS; i++)
		{
			channel = j*NUM_BOARDS + i;
			absSample = (int32_t)round(abs(avgSample[channel] - commonMedian));
			if (absSample > m_maximum[channel])
				m_maximum[channel] = absSample;
		}
}