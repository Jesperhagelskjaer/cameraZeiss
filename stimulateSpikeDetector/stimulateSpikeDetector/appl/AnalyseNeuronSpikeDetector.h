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
protected:
	virtual void SearchPattern(LxRecord * pLxRecord);
private:
	NeuronSpikeDetector * m_pNeuronSpikeDetector;
};

