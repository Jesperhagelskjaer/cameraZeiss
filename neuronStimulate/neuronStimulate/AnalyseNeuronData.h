///////////////////////////////////////////////////////////
//  AnalyseNeuronData.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
#define EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_

#include "Monitor.h"
#include "LxRecord.h"


#define AVG_DELAY			50		// Length of average delay line, at 30 KHz equal to 1.66 ms
#define DEFAULT_ACTIVE_CH	2		// Active channel for where cost is computed

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
