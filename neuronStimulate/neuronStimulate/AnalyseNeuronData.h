///////////////////////////////////////////////////////////
//  AnalyseNeuronData.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
#define EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_

#include "Monitor.h"
#include "LxRecord.h"


#define AVG_DELAY			100		// Length of average delay line
#define DEFAULT_ACTIVE_CH	2		// Active channel for where cost is computed

class AnalyseNeuronData : public Monitor
{

public:
	// Analysing modes
	enum MODES {
		MODE_AVERAGE,
		MODE_ANALYSE,
		MODE_STOP
	};

	AnalyseNeuronData();
	virtual ~AnalyseNeuronData();

	void SetMode(MODES mode) 
	{
		m_mode = mode;
	}
	void SetActiveChannel(int ch)
	{
		m_activeChannel = ch;
	}

	virtual void AnalyzeData(LxRecord * pLxRecord);
	double CalculateCost();

private:
	virtual void SearchPattern(LxRecord * pLxRecord);
	void RecursiveAverage(LxRecord * pLxRecord);

	int32_t m_averageDly[NUM_BOARDS*NUM_CHANNELS][AVG_DELAY];
	int m_avgIdx;
	double m_average[NUM_BOARDS*NUM_CHANNELS];
	int32_t m_maximum[NUM_BOARDS*NUM_CHANNELS];
	MODES m_mode;
	MODES m_modeLast;
	int m_activeChannel;
};
#endif // !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
