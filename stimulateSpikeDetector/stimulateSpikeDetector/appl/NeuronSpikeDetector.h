#pragma once
///////////////////////////////////////////////////////////
//  NeuronSpikeDetector.h
//  Implementation of the Class NeuronSpikeDetector
//  Created on:      15-feb-2018 17:49:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#include "ProjectDefinitions.h"

template <class T> class SpikeDetect;

class NeuronSpikeDetector
{

public:
	NeuronSpikeDetector();
	void Create(void);
	void Train(void);
	void Predict(void);
	void Terminate(void);
private:
	SpikeDetect<USED_DATATYPE> *pSpikeDetector;
};
