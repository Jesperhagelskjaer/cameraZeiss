///////////////////////////////////////////////////////////
//  AnalyseNeuronData.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
#define EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_

#include <stdio.h>
#include <math.h>
#include "Monitor.h"
#include "LxRecord.h"
#include "FirFilter.h"


#define AVG_DELAY			50		// Length of average delay line, at 30 KHz equal to 1.66 ms
#define DEFAULT_ACTIVE_CH	2		// Active channel for where cost is computed
#define ANALYSE_SAMPLES     120     // Default number of samples to analyse

class AnalyseNeuronData : public Monitor
{

public:
	// Analysing modes
	enum MODES {
		MODE_STOP = 0,
		MODE_AVERAGE = 1,
		MODE_ANALYSE = 2
	};

	AnalyseNeuronData();
	virtual ~AnalyseNeuronData();

	MODES GetMode(void)
	{
		return m_mode;
	}

	int OpenCostFile(char *fileName);

	virtual int AppendCostToFile(double cost);

	void CloseCostFile(void);

	void SetDelaySamples(int delay)
	{
		enter();
		m_analyseSamples = delay;
		m_countSamples = delay;
		exit();
	}

	void SetMode(MODES mode) 
	{
		enter();
		m_mode = mode;
		exit();
	}

	void SetActiveChannelOrTemplate(int ch)
	{
		enter();
		m_activeChannelOrTemplate = ch;
		exit();
	}

	void WaitAnalyseSamples(void)
	{
		m_semaAnalyseComplete.wait();
	}

	virtual void AnalyzeData(LxRecord * pLxRecord);
	virtual double CalculateCost();
	LxRecord *FilterData(LxRecord * pLxRecord);

	void SetFilterType(FirFilter::TYPES type)
	{
		for (int j = 0; j < NUM_BOARDS; j++) {
			firFilter[j].setType(type);
		}
		if (type == FirFilter::BYPASS)
			filterEnabled = false;
		else
			filterEnabled = true;
	}

	int GetNumSamplesToAnalyse(void) {
		return m_analyseSamples;
	}

	bool IsNeuronSpikeDetector(void) {
		return m_useNeuronSpikeDetector;
	}

protected:
	virtual void SearchPattern(LxRecord * pLxRecord);
	void RecursiveAverage(LxRecord * pLxRecord);

	int32_t m_averageDly[NUM_BOARDS*NUM_CHANNELS][AVG_DELAY];
	int m_avgIdx;
	double m_average[NUM_BOARDS*NUM_CHANNELS];
	int32_t m_maximum[NUM_BOARDS*NUM_CHANNELS];
	MODES m_mode;
	MODES m_modeLast;
	int m_activeChannelOrTemplate; // Channel or template to stimulate
	int m_analyseSamples; // Number of samples to analyse
	int m_countSamples;
	Semaphore m_semaAnalyseComplete;
	FILE *costStream;
	bool filterEnabled;
	FirFilter firFilter[NUM_BOARDS];
	bool m_useNeuronSpikeDetector; // Set true when neuron spike detector is active
};
#endif // !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
