///////////////////////////////////////////////////////////
//  NeuronSpikeDetector.cpp
//  Implementation of the Class NeuronSpikeDetector
//  Created on:      15-feb-2018 17:49:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#include "NeuronSpikeDetector.h"

#ifdef USE_CUDA
//#include "SpikeDetectCUDA.h"
#include "SpikeDetectCUDA_RTP.h"
#else
#include "SpikeDetect.h"
#endif

NeuronSpikeDetector::NeuronSpikeDetector()
{
	pSpikeDetector = 0;
}

void NeuronSpikeDetector::Create(void)
{
#ifdef USE_CUDA
	//SpikeDetectCUDA<USED_DATATYPE> *spikeDetector;
	//spikeDetector = new SpikeDetectCUDA<USED_DATATYPE>();
	SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector;
	pSpikeDetector = new SpikeDetectCUDA_RTP<USED_DATATYPE>();
#else
	SpikeDetect<USED_DATATYPE> *spikeDetector;
	pSpikeDetector = new SpikeDetect<USED_DATATYPE>();
#endif
}

void NeuronSpikeDetector::Train(void)
{
#ifdef USE_CUDA
	SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector = (SpikeDetectCUDA_RTP<USED_DATATYPE> *)pSpikeDetector;
	//spikeDetector->runTraining(); // Training not using CUDA
	spikeDetector->runTrainingCUDA();
#else
	pSpikeDetector->runTraining();
#endif
}

void NeuronSpikeDetector::Predict(void)
{
	pSpikeDetector->runPrediction();
}

void NeuronSpikeDetector::Terminate(void)
{
	delete pSpikeDetector;
}
