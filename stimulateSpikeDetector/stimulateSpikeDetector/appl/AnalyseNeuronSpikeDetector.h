///////////////////////////////////////////////////////////
//  AnalyseNeuronSpikeDetetor.h
//  Implementation of the Class AnalyseNeuronSpikeDetetor
//  Created on:      1-mar-2018 15:54:35
//  Original author: Kim Bjerge
//  Analyses neuron data by using neuron spike detector 
//                     to search for individual neurons
///////////////////////////////////////////////////////////
#pragma once
#include "AnalyseNeuronData.h"
#include "NeuronSpikeDetector.h"

// Class to analyse individual neurons using spike detector
class AnalyseNeuronSpikeDetector : public AnalyseNeuronData
{
public:
	AnalyseNeuronSpikeDetector();
	void AddSpikeDetector(NeuronSpikeDetector *pNeuronSpikeDetector);
	virtual double CalculateCost();
	virtual int AppendCostToFile(double cost);
	NeuronSpikeDetector *GetNeuronSpikeDetector(void) { return m_pNeuronSpikeDetector; };
	int GetTotalSpikesFound(void) { return m_TotalSpikesFound; };
	void PrintTotalSpikeCounters(void) {
		uint32_t *TotalSpikeCounters = m_pNeuronSpikeDetector->GetTotalSpikeCounters();
		for (int j = 0; j < MAXIMUM_NUMBER_OF_TEMPLATES; j++) {
			printf("Template: %2d spike counts: %5d\r\n", j + 1, TotalSpikeCounters[j]);
		}
	}

protected:
	virtual void SearchPattern(LxRecord * pLxRecord);
private:
	NeuronSpikeDetector * m_pNeuronSpikeDetector;
	int m_TotalSpikesFound;
};

