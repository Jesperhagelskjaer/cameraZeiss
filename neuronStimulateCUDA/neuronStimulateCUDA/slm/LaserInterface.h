///////////////////////////////////////////////////////////
//  LaserInterface.h
//  Implementation of the Class LaserInterface
//  Created on:      23-maj-2017 12:55:05
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_D4E13C2A_0ECA_463c_BC2E_DA49A8EC8CD5__INCLUDED_)
#define EA_D4E13C2A_0ECA_463c_BC2E_DA49A8EC8CD5__INCLUDED_

#include "Serial.h"

class LaserInterface
{

public:
	LaserInterface(int rate);
	~LaserInterface();
	bool OpenPort(int port);
	bool TurnOn(float intensity);
	bool TurnOff();

private:
	CSerial *m_CSerial;
	int m_baudRate;

};
#endif // !defined(EA_D4E13C2A_0ECA_463c_BC2E_DA49A8EC8CD5__INCLUDED_)
