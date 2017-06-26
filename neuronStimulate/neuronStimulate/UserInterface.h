///////////////////////////////////////////////////////////
//  UserInteface.h
//  Implementation of the Class UserInteface
//  Created on:      20-maj-2017 08:53:04
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_)
#define EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_

#include "CollectNeuronDataThread.h"
#include "Configuration.h"
#include "StimulateNeuronThread.h"
#include "AnalyseNeuronData.h"
#include "GenericAlgo.h"
#include "defs.h"

#define DEFAULT_CONFIG "config.txt"

class UserInterface
{

public:
	UserInterface();
	virtual ~UserInterface();
	void run();

private:
	void init(void);
	Configuration *m_Configuration;
	AnalyseNeuronData *m_AnalyseNeuronData;
	StimulateNeuronThread *m_StimulateNeuronThread;
	CollectNeuronDataThread *m_CollectNeuronDataThread;
	GenericAlgo *m_GenericAlgo;

	// Tests
	void testCollectNeuronData(void);
	// Run stimulation of neuron maximizing light intensity for channel
	void runStimulateNeuron(Configuration *config);

};
#endif // !defined(EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_)
