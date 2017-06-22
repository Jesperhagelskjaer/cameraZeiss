///////////////////////////////////////////////////////////
//  StimulateNeuronThread.h
//  Implementation of the Class StimulateNeuronThread
//  Created on:      19-maj-2017 22:44:28
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_)
#define EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_

#include "Thread.h"
#include "mcam_zei_ex.h"
#include "GenericAlgo.h"
#include "AnalyseNeuronData.h"

#define SAMPLE_FREQUENCY  30000 // Hz

class StimulateNeuronThread : public Thread
{

public:
	StimulateNeuronThread();
	virtual ~StimulateNeuronThread();

	virtual void run();
	void Start(ThreadPriority pri, string _name, AnalyseNeuronData *pAnalyseNeuronData, 
		       GenericAlgo *pGenericAlgo, int iterations, int pause);
	void WaitForCompletion();
	void SetDelay(int ms)
	{
		m_delayms = ms;
		if (m_AnalyseNeuronData != 0)
			m_AnalyseNeuronData->SetDelaySamples((int)ceil(SAMPLE_FREQUENCY / 1000 * ms));
	}

private:
	GenericAlgo *m_GenericAlgo;
	AnalyseNeuronData *m_AnalyseNeuronData;
	Semaphore m_semaComplete;
	int m_iterations;
	int m_delayms;
	int m_pausems;
	TimeMeasure timeMeas;
};
#endif // !defined(EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_)
