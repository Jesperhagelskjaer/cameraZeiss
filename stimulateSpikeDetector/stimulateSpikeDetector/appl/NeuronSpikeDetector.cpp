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

#define PRINT_ITERATIONS		5

NeuronSpikeDetector::NeuronSpikeDetector()
{
	m_pSpikeDetector = 0;
	m_pData = 0;
	m_predictInitialized = false;
	m_SampleDataCollected = 0;
	m_Iterations = 1;
	m_PredictTime = 0;
}

NeuronSpikeDetector::~NeuronSpikeDetector()
{
#ifdef USE_CUDA
	if (m_predictInitialized)
		((SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector)->CUDACleanUpPrediction();
#endif
}

ProjectInfo *NeuronSpikeDetector::GetProjectInfo(void)
{
	return m_pSpikeDetector->getProjectInfo();
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

void NeuronSpikeDetector::SetSampleSize(int size)
{
	m_SampleDataSize = DATA_CHANNELS * size;
	if (m_SampleDataSize > DATA_CHANNELS * RTP_DATA_LENGTH)
		m_SampleDataSize = (int)(DATA_CHANNELS * RTP_DATA_LENGTH);
	memset(m_pSampleData, 0, (int)(DATA_CHANNELS * RTP_DATA_LENGTH * sizeof(USED_DATATYPE)));
	m_pData = m_pSampleData;
	m_SampleDataCollected = 0;
	m_Iterations = PRINT_ITERATIONS;
	m_PredictTime = 0;

	for (int i = 0; i < MAXIMUM_NUMBER_OF_TEMPLATES; i++) {
		m_TotalSpikeCounters[i] = 0;
		m_LastTotalSpikeCounters[i] = 0;
	}

#ifdef USE_CUDA
	if (!m_predictInitialized) {
		if (((SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector)->prepareCUDAPrediction() == 0)
			m_predictInitialized = true;
	}
	printf("\r\n------------------------------------------------------------ TEMPLATES USED --------------------------------------------------------------------------------------\r\n");
	for (int i = 0; i < MAXIMUM_NUMBER_OF_TEMPLATES; i++) {
		m_TotalSpikeCounters[i] = 0;
		if (GetProjectInfo()->isTemplateUsedTraining(i + 1))
			printf("%02d ", i + 1);
	}
#endif
}


void NeuronSpikeDetector::AddSampleBlock(int32_t *pSamples)
{
	//bool allZero = true;

	if (m_pData > 0)  // Check more space in sample block
	{
		// Convert and insert sample data in block to analyse
		for (int i = 0; i < DATA_CHANNELS; i++) {
			m_pData[i] = (float)pSamples[i];
			//if (m_pData[i] != 0) allZero = false;
		}

		// Increment position in block to insert samples
		m_pData += DATA_CHANNELS;
		m_SampleDataCollected += DATA_CHANNELS;
		if (m_SampleDataCollected >= m_SampleDataSize)
			m_pData = 0; // Mark end of block, no more space in block
	}
	//else // Used for testing
	//	printf("NeuronSpikeDetector::AddSampleBlock block full\r\n");
	//if (allZero)
	//	printf("All channels zero value at %d\r\n", m_SampleDataCollected);
}

double NeuronSpikeDetector::RealtimePredict(void) // Predict on realtime data collected in sample block (m_pSampleData)
{
	// TODO needs to be rewritten calculating af cost function
	int spikesFound = 0;

	//if (m_pData != 0)
	//	printf("NeuronSpikeDetector::RealtimePredict block not full\r\n");

#ifdef USE_CUDA
	if (m_predictInitialized) {
		SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector = (SpikeDetectCUDA_RTP<USED_DATATYPE> *)m_pSpikeDetector;
		if (spikeDetector->runPredictionRTP(m_pSampleData) == 0) { // Perform prediction
			m_Iterations--;
			uint32_t *TemplateFoundCounters = spikeDetector->getFoundTimesCounters();
			for (int i = 0; i < MAXIMUM_NUMBER_OF_TEMPLATES; i++) {
				if (TemplateFoundCounters[i] > 0) { // Read number of spikes for each template
					//std::cout << "  T" << (i + 1) << " spikes: " << TemplateFoundCounters[i] << std::endl;
					spikesFound += TemplateFoundCounters[i];
					m_TotalSpikeCounters[i] += TemplateFoundCounters[i];
				}
				if (m_Iterations == 0) {
					if (GetProjectInfo()->isTemplateUsedTraining(i + 1)) // && m_TotalSpikeCounters[i] > 0)
						//printf("%2d ", m_TotalSpikeCounters[i] - m_LastTotalSpikeCounters[i]); // Print difference
						printf("%2d ", m_TotalSpikeCounters[i]%100); // Print total counters
				}
				m_LastTotalSpikeCounters[i] = m_TotalSpikeCounters[i];
			}
			//m_PredictTime += spikeDetector->getLatestExecutionTime();
			if (m_Iterations == 0) {
				// Average prediction time
				//printf(" predict %0.2f ms ", m_PredictTime/(1000*PRINT_ITERATIONS));
				//printf("\r");
				//m_PredictTime = 0;
				m_Iterations = PRINT_ITERATIONS;
			}
		}
	}
#endif
	m_pData = m_pSampleData;
	m_SampleDataCollected = 0;

	return spikesFound;
}
