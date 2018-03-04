///////////////////////////////////////////////////////////
//  NeuronSpikeDetector.cpp
//  Implementation of the Class NeuronSpikeDetector
//  Created on:      15-feb-2018 17:49:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#include "NeuronSpikeDetector.h"

#ifdef USE_CUDA
#include "SpikeDetectCUDA_RTP.h"
#else
#include "SpikeDetect.h"
#endif

NeuronSpikeDetector::NeuronSpikeDetector()
{
	m_pSpikeDetector = 0;
	m_pSampleData = 0;
	m_pData = 0;
	m_predictInitialized = false;
}

NeuronSpikeDetector::~NeuronSpikeDetector()
{
#ifdef USE_CUDA
	if (m_predictInitialized)
		((SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector)->CUDACleanUpPrediction();
#endif
	if (m_pSampleData != 0)
		delete m_pSampleData;
}

void NeuronSpikeDetector::SetSampleSize(int size)
{
	if (m_pSampleData != 0) 
		delete m_pSampleData;

	m_SampleDataSize = size;
	m_pSampleData = new USED_DATATYPE[m_SampleDataSize*DATA_CHANNELS];
	
#ifdef USE_CUDA
	if (!m_predictInitialized) {
		if (((SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector)->prepareCUDAPrediction() == 0)
			m_predictInitialized = true;
	}
#endif
}

ProjectInfo *NeuronSpikeDetector::GetProjectInfo(void)
{
	return m_pSpikeDetector->getProjectInfo();
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

#ifdef	USE_CUDA_TRAIN
	spikeDetector->runTrainingCUDA();
#else
	spikeDetector->runTraining(); // Training not using CUDA
#endif

#endif
}

void NeuronSpikeDetector::Predict(void)
{
	m_pSpikeDetector->runPrediction();
}

void NeuronSpikeDetector::Terminate(void)
{
#ifdef USE_CUDA
	if (m_predictInitialized)
		((SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector)->CUDACleanUpPrediction();
	m_predictInitialized = false;
#endif
	delete m_pSpikeDetector;
}

double NeuronSpikeDetector::RealtimePredict(void) // Predict on realtime data collected in sample block (m_pSampleData)
{
	// TODO needs to be rewritten calculating af cost function
	int spikesFound = 0;

#ifdef USE_CUDA
	if (m_predictInitialized) {
		SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector = (SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector;
		if (spikeDetector->runPredictionRTP(m_pSampleData) == 0) { // Perform prediction
			uint32_t *TemplateFoundCounters = spikeDetector->getFoundTimesCounters();
			for (int j = 0; j < MAXIMUM_NUMBER_OF_TEMPLATES; j++)
				if (TemplateFoundCounters[j] > 0) { // Read number of spikes for each template
					std::cout << "  T" << (j + 1) << " spikes: " << TemplateFoundCounters[j] << std::endl;
					spikesFound += TemplateFoundCounters[j];
				}
		}
	}
#endif

	return spikesFound;
}
