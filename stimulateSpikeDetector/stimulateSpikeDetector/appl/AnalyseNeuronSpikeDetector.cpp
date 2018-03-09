///////////////////////////////////////////////////////////
//  AnalyseNeuronSpikeDetetor.cpp
//  Implementation of the Class AnalyseNeuronSpikeDetetor
//  Created on:      1-mar-2018 15:54:35
//  Original author: Kim Bjerge
//  Analyses neuron data by using neuron spike detector 
//                     to search for individual neurons
///////////////////////////////////////////////////////////
#include <algorithm>
#include <vector>
#include "AnalyseNeuronSpikeDetector.h"


AnalyseNeuronSpikeDetector::AnalyseNeuronSpikeDetector()
{
	m_pNeuronSpikeDetector = 0;
	m_TotalSpikesFound = 0;
}

int AnalyseNeuronSpikeDetector::AppendCostToFile(double cost)
{
	uint32_t *pTotalSpikeCounters;
	int i, res = -1;
	if (costStream != NULL) {
		pTotalSpikeCounters = m_pNeuronSpikeDetector->GetTotalSpikeCounters();
		res = fprintf(costStream, "%.1f ", cost);
		res += fprintf(costStream, "%d ", m_activeChannelOrTemplate);
		for (i = 0; i < MAXIMUM_NUMBER_OF_TEMPLATES; i++)
			res += fprintf(costStream, "%d ", pTotalSpikeCounters[i]);
		res += fprintf(costStream, "\r\n");
	}
	return res;
}


void AnalyseNeuronSpikeDetector::AddSpikeDetector(NeuronSpikeDetector *pNeuronSpikeDetector)
{
	pNeuronSpikeDetector->SetSampleSize(GetNumSamplesToAnalyse());
	m_pNeuronSpikeDetector = pNeuronSpikeDetector;
	m_useNeuronSpikeDetector = true;
}

double AnalyseNeuronSpikeDetector::CalculateCost()
{
	enter();

	double cost = 0;
	if (m_pNeuronSpikeDetector != 0) {
		cost = m_pNeuronSpikeDetector->RealtimePredict();
		m_TotalSpikesFound += (int)cost;
		//cost = rand();
	}

	exit();
	return cost;
}

void AnalyseNeuronSpikeDetector::SearchPattern(LxRecord * pLxRecord)
{
	if (m_pNeuronSpikeDetector != 0)
		m_pNeuronSpikeDetector->AddSampleBlock(pLxRecord->board[0].data);
}