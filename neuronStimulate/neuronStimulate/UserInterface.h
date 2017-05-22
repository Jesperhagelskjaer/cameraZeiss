///////////////////////////////////////////////////////////
//  UserInteface.h
//  Implementation of the Class UserInteface
//  Created on:      20-maj-2017 08:53:04
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_)
#define EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_

#include "Configuration.h"
#include "StimulateNeuronThread.h"
//#include "CollectNeuronDataThread.h"

class UserInterface
{

public:
	UserInterface();
	virtual ~UserInterface();
	void Create();
	void run();

private:
	Configuration *m_Configuration;
	StimulateNeuronThread *m_StimulateNeuronThread;
	//CollectNeuronDataThread *m_CollectNeuronDataThread;

};
#endif // !defined(EA_3929C698_D918_4faa_A2C9_25238069C926__INCLUDED_)
