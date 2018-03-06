#pragma once
///////////////////////////////////////////////////////////
//  NeuronSpikeDetector.h
//  Implementation of the Class NeuronSpikeDetector
//  Created on:      15-feb-2018 17:49:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#include "ProjectDefinitions.h"
#include <cstdint>

template <class T> class SpikeDetect;
class ProjectInfo;

class NeuronSpikeDetector
{

public:
	NeuronSpikeDetector();
	~NeuronSpikeDetector();
	 
	void Create(void); // Creates spike detector and loads train data and templates
	void Train(void); // Trains template thredsholds based on loaded train data and templates
	void Predict(void); // Predict on data from loaded files
	void Terminate(void); // Terminates spike detector and closing all files

	// Handling collecting samples
	void SetSampleSize(int size); // Set size and allocate sample block
	void AddSampleBlock(int32_t *pSamples); // Add samples from DATA_CHANNELS to block
	double RealtimePredict(void); // Predict on realtime data collected in sample block (m_pSampleData)
	uint32_t *GetTotalSpikeCounters(void) {
		return m_TotalSpikeCounters;
	}

	ProjectInfo *GetProjectInfo(void);

private:
	SpikeDetect<USED_DATATYPE> *m_pSpikeDetector;
	uint32_t m_TotalSpikeCounters[MAXIMUM_NUMBER_OF_TEMPLATES];
	USED_DATATYPE m_pSampleData[(int)(DATA_CHANNELS*RTP_DATA_LENGTH)]; // Pointer to start of sample block
	USED_DATATYPE *m_pData; // Pointer to current position in sample block
	int m_SampleDataSize; // Size of sample block to analyse
	int m_SampleDataCollected; // Number of samples in sample block
	bool m_predictInitialized; // Is predicion of spike detector initialized
	int m_Iterations;
	float m_PredictTime;
};
