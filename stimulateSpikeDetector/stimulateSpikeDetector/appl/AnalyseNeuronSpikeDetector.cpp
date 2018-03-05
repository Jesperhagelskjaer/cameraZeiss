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

void AnalyseNeuronSpikeDetector::AddSpikeDetector(NeuronSpikeDetector *pNeuronSpikeDetector)
{
	pNeuronSpikeDetector->SetSampleSize(GetNumSamplesToAnalyse());
	m_pNeuronSpikeDetector = pNeuronSpikeDetector;
}

double AnalyseNeuronSpikeDetector::CalculateCost()
{
	double cost = 0;
	if (m_pNeuronSpikeDetector != 0) {
		cost = m_pNeuronSpikeDetector->RealtimePredict();
		m_TotalSpikesFound += (int)cost;
		cost = 0; // rand();
	}
	return cost;
}

void AnalyseNeuronSpikeDetector::SearchPattern(LxRecord * pLxRecord)
{
	if (m_pNeuronSpikeDetector != 0)
		m_pNeuronSpikeDetector->AddSampleBlock(pLxRecord->board[0].data);
}