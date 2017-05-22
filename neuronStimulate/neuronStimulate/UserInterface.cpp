///////////////////////////////////////////////////////////
//  UserInteface.cpp
//  Implementation of the Class UserInteface
//  Created on:      20-maj-2017 08:53:04
//  Original author: au288681
///////////////////////////////////////////////////////////

#include "UserInterface.h"

UserInterface::UserInterface()
{
	m_CollectNeuronDataThread = 0;
	m_Configuration = 0;
	m_CollectNeuronDataThread = 0;
}

UserInterface::~UserInterface()
{

}

void UserInterface::init(void)
{
	m_Configuration = new Configuration();
}

void UserInterface::testCollectNeuronData(void)
{
	m_CollectNeuronDataThread = new CollectNeuronDataThread();
	m_CollectNeuronDataThread->Start(Thread::PRIORITY_ABOVE_NORMAL, "NeuronDataThread");

	// Wait 5 seconds
	Sleep(5000);
	m_CollectNeuronDataThread->Stop();
	delete m_CollectNeuronDataThread;
	m_CollectNeuronDataThread = 0;
}

void UserInterface::run()
{
	bool running = true;
	char choise, ret;
	init();

	while (running) {
		printf("Select menu: \r\n");
		printf("1. Test collect neuron data\r\n");
		printf("e. Exit\r\n");
		printf("\r\n> ");
		scanf("%c%c", &choise, &ret);

		switch (choise) 
		{
			case '1':
				testCollectNeuronData();
				break;
			case 'e':
				running = false;
				break;
		}

	}

}
