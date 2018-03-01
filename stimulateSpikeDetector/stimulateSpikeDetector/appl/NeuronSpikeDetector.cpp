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
	m_pSpikeDetector = 0;
	m_pSampleData = 0;
	m_pData = 0;
}

NeuronSpikeDetector::~NeuronSpikeDetector()
{
	if (m_pSampleData != 0)
		delete m_pSampleData;
}

void NeuronSpikeDetector::SetSampleSize(int size)
{
	if (m_pSampleData != 0) 
		delete m_pSampleData;

	m_SampleDataSize = size;
	m_pSampleData = new USED_DATATYPE[m_SampleDataSize*DATA_CHANNELS];
}

void NeuronSpikeDetector::AddSampleBlock(int32_t *pSamples)
{

	if (m_pData > 0)  // Check more space in sample block
	{
		// Convert and insert sample data in block to analyse
		for (int i = 0; i < DATA_CHANNELS; i++)
			m_pData[i] = (float)pSamples[i]; 

		// Increment position in block to insert samples
		m_pData += DATA_CHANNELS;

		if (m_pData >= m_pSampleData + DATA_CHANNELS)
			m_pData = 0; // Mark end of block, no more space in block
	}
}

void NeuronSpikeDetector::Create(void)
{
#ifdef USE_CUDA
	//SpikeDetectCUDA<USED_DATATYPE> *spikeDetector;
	//spikeDetector = new SpikeDetectCUDA<USED_DATATYPE>();
	//SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector;
	m_pSpikeDetector = new SpikeDetectCUDA_RTP<USED_DATATYPE>();
#else
	SpikeDetect<USED_DATATYPE> *spikeDetector;
	m_pSpikeDetector = new SpikeDetect<USED_DATATYPE>();
#endif
}

void NeuronSpikeDetector::Train(void)
{
#ifdef USE_CUDA
	SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector = (SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector;
	//spikeDetector->runTraining(); // Training not using CUDA
	spikeDetector->runTrainingCUDA();
#else
	pSpikeDetector->runTraining();
#endif
}

void NeuronSpikeDetector::Predict(void)
{
	m_pSpikeDetector->runPrediction();
}

void NeuronSpikeDetector::Terminate(void)
{
	delete m_pSpikeDetector;
}

double NeuronSpikeDetector::RealtimePredict(void) // Predict on realtime data collected in sample block (m_pSampleData)
{
	// TODO needs to be rewritten using sample block buffer
	m_pSpikeDetector->runPrediction();
	return 0;
}
