///////////////////////////////////////////////////////////
//  StimulateNeuronThread.h
//  Implementation of the Class StimulateNeuronThread
//  Created on:      19-maj-2017 22:44:28
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_)
#define EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_

#include "Thread.h"
#include "mcam_zei_ex.h"
#include "GenericAlgo.h"
#include "AnalyseNeuronData.h"

class StimulateNeuronThread : public Thread
{

public:
	StimulateNeuronThread();
	virtual ~StimulateNeuronThread();

	virtual void run();
	void Start();
	void Stop();

private:
	GenericAlgo *m_GenericAlgo;
	AnalyseNeuronData *m_AnalyseNeuronData;

};
#endif // !defined(EA_C34C7A4D_067D_4aaf_B24A_B39E09E63F27__INCLUDED_)
