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
	void ResetBlock(void) { // Reset current position to start of sample block
		m_pData = m_pSampleData;
	}; 
	void AddSampleBlock(int32_t *pSamples); // Add samples from DATA_CHANNELS to block
	double RealtimePredict(void); // Predict on realtime data collected in sample block (m_pSampleData)

	ProjectInfo *GetProjectInfo(void);

private:
	SpikeDetect<USED_DATATYPE> *m_pSpikeDetector;
	USED_DATATYPE *m_pSampleData; // Pointer to start of sample block
	USED_DATATYPE *m_pData; // Pointer to current position in sample block
	int m_SampleDataSize; // Size of sample block to analyse
	bool m_predictInitialized; // Is predicion of spike detector initialized
};
