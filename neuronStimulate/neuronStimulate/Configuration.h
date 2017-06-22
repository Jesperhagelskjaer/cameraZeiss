///////////////////////////////////////////////////////////
//  Configuration.h
//  Implementation of the Class Configuration
//  Created on:      20-maj-2017 08:53:06
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
#define EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_
#include "FirFilter.h"

class Configuration
{

public:
	Configuration(int numIterations, int activeCh, int port, int delay, int pause, 
		          float intensity, FirFilter::TYPES type, int numParents, int numBindings)
	{
		m_NumIterations = numIterations;
		m_ActiveChannel = activeCh;
		m_LaserPort = port;
		m_DelayMS = delay;
		m_PauseMS = pause;
		m_LaserIntensity = intensity;
		m_FilterType = type;
		m_NumParents = numParents;
		m_NumBindings = numBindings;
	}

	int m_NumIterations;
	int m_ActiveChannel;
	int m_LaserPort;
	int m_DelayMS;
	int m_PauseMS;
	float m_LaserIntensity;
	FirFilter::TYPES m_FilterType;
	int m_NumParents;
	int m_NumBindings;
};
#endif // !defined(EA_F26F424A_2953_466e_976E_2D9054E58697__INCLUDED_)
