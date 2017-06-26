///////////////////////////////////////////////////////////
//  AnalyseNeuronDataCMR.h
//  Implementation of the Class AnalyseNeuronDataCMR
//  Created on:      26-june-2017 13:54:35
//  Original author: Kim Bjerge
//  Analyses neuron data adding common average reference filter
///////////////////////////////////////////////////////////
#pragma once
#include "AnalyseNeuronData.h"

class AnalyseNeuronDataCMR : public AnalyseNeuronData
{
public:
	AnalyseNeuronDataCMR();
protected:
	virtual void SearchPattern(LxRecord * pLxRecord);
	double findMedian(double *pVaules, int size);
};

