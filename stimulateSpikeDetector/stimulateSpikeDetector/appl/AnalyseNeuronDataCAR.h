///////////////////////////////////////////////////////////
//  AnalyseNeuronDataCAR.h
//  Implementation of the Class AnalyseNeuronDataCAR
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
//  Analyses neuron data adding common average reference filter
///////////////////////////////////////////////////////////
#pragma once
#include "AnalyseNeuronData.h"

class AnalyseNeuronDataCAR : public AnalyseNeuronData
{
public:
	AnalyseNeuronDataCAR();
protected:
	virtual void SearchPattern(LxRecord * pLxRecord);
};

