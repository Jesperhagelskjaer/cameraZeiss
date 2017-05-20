///////////////////////////////////////////////////////////
//  CollectNeuronDataThread.h
//  Implementation of the Class CollectNeuronDataThread
//  Created on:      19-maj-2017 22:44:32
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_)
#define EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_

#include "Thread.h"
#include "SOCK_UDP.h"
#include "AnalyseNeuronData.h"
#include "LynxRecord.h"

class CollectNeuronDataThread : public Thread
{

public:
	CollectNeuronDataThread();
	virtual ~CollectNeuronDataThread();

	void run();
	void Start();
	void Stop();

private:
	SOCK_UDP *m_SOCK_UDP;
	AnalyseNeuronData *m_AnalyseNeuronData;
	LynxRecord *m_LynxRecord;

};
#endif // !defined(EA_88D8D966_880B_41d9_8F04_02C346D2C585__INCLUDED_)
