///////////////////////////////////////////////////////////
//  Configuration.h
//  Implementation of the Class Configuration
//  Created on:      20-maj-2017 08:53:06
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
#define EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_

class Configuration
{

public:
	Configuration(int num, int act, int port, int delay) 
	{
		m_NumIterations = num;
		m_ActiveChannel = act;
		m_LaserPort = port;
		m_DelayMS = delay;
	}

	int m_NumIterations;
	int m_ActiveChannel;
	int m_LaserPort;
	int m_DelayMS;
};
#endif // !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
