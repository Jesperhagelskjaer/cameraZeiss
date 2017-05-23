///////////////////////////////////////////////////////////
//  LaserInterface.cpp
//  Implementation of the Class LaserInterface
//  Created on:      23-maj-2017 12:55:05
//  Original author: au288681
///////////////////////////////////////////////////////////
#include <iostream>
using namespace std;
#include "LaserInterface.h"


LaserInterface::LaserInterface(int rate)
{
	m_CSerial = new CSerial();
	m_baudRate = rate;
}

LaserInterface::~LaserInterface()
{
	if (m_CSerial->IsOpened())
		m_CSerial->Close();
}

bool LaserInterface::OpenPort(int port)
{
	if (!m_CSerial->Open(port, m_baudRate))
	{
		cout << "Could not open laser COM" << port << endl;
		return false;
	}
	return true;
}

bool LaserInterface::TurnOn()
{
	char data[20];
	int num;
	if (!m_CSerial->IsOpened()) return false;

	// Sending a string on port - laser on
	sprintf(data, ">laserPM 0.2\r\n");
	num = m_CSerial->SendData(data, (int)strlen(data) + 1);
	if (num != strlen(data) + 1) {
		cout << "Could not turn laser on" << endl;
		return false;
	}
	return true;
}


bool LaserInterface::TurnOff()
{
	char data[20];
	int num;
	if (!m_CSerial->IsOpened()) return false;

	// Sending a string on port - laser off
	sprintf(data, ">laserPM 0.0\r\n");
	num = m_CSerial->SendData(data, (int)strlen(data) + 1);
	if (num != strlen(data) + 1) {
		cout << "Could not turn laser off" << endl;
		return false;
	}
	return true;
}