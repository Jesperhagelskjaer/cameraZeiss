///////////////////////////////////////////////////////////
//  UserInteface.cpp
//  Implementation of the Class UserInteface
//  Created on:      20-maj-2017 08:53:04
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include "UserInterface.h"
#include "AnalyseNeuronDataCAR.h"
#include "AnalyseNeuronDataCMR.h"


UserInterface::UserInterface()
{
	m_Configuration = 0;
	m_AnalyseNeuronData = 0;
	m_CollectNeuronDataThread = 0;
	m_StimulateNeuronThread = 0;
}

UserInterface::~UserInterface()
{

}

void UserInterface::init(void)
{
	m_Configuration = new Configuration(GEN_ITERATIONS, ACTIVE_CHANNEL, LASER_PORT, DELAY_MS, 
		                                PAUSE_MS, LASER_INTENSITY, FirFilter::FILTER_TYPE,
										NUM_PARENTS, NUM_BINDINGS, COMMON_REF);
}

void UserInterface::testCollectNeuronData(void)
{

	m_AnalyseNeuronData = new AnalyseNeuronData();
	m_CollectNeuronDataThread = new CollectNeuronDataThread();
	m_CollectNeuronDataThread->Start(Thread::PRIORITY_ABOVE_NORMAL, "NeuronDataThread", m_AnalyseNeuronData);

	// Wait seconds
	for (int loop = 0; loop < 10; loop++) {
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_AVERAGE);
		Sleep(500);
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_ANALYSE);
		Sleep(500);
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_STOP);
		printf("Cost %f \r\n", m_AnalyseNeuronData->CalculateCost());
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_AVERAGE);
	}

	m_CollectNeuronDataThread->Stop();
	delete m_CollectNeuronDataThread;
	delete m_AnalyseNeuronData;
	m_CollectNeuronDataThread = 0;
	m_AnalyseNeuronData = 0;
}

void UserInterface::runStimulateNeuron(Configuration *config)
{
	// Create objects
	switch (config->m_CommonAvgRef) {
		case 1:
			m_AnalyseNeuronData = new AnalyseNeuronDataCAR();
			break;
		case 2:
			m_AnalyseNeuronData = new AnalyseNeuronDataCMR();
			break;
		default:
			m_AnalyseNeuronData = new AnalyseNeuronData();
			break;
	}
	m_CollectNeuronDataThread = new CollectNeuronDataThread();
	m_StimulateNeuronThread = new StimulateNeuronThread();
	m_GenericAlgo = new GenericAlgo(config->m_NumParents, config->m_NumBindings, config->m_NumIterations);

	// Start threads
	m_AnalyseNeuronData->SetActiveChannel(config->m_ActiveChannel);
	m_AnalyseNeuronData->SetFilterType(config->m_FilterType);
	m_GenericAlgo->OpenLaserPort(config->m_LaserPort, config->m_LaserIntensity);
	m_CollectNeuronDataThread->Start(Thread::PRIORITY_HIGH, "NeuronDataThread", m_AnalyseNeuronData);
	m_StimulateNeuronThread->Start(Thread::PRIORITY_ABOVE_NORMAL, "StimulateNeuronThread", 
		                           m_AnalyseNeuronData, m_GenericAlgo, 
		                           config->m_NumIterations, config->m_PauseMS);
	m_StimulateNeuronThread->SetDelay(config->m_DelayMS);
	m_AnalyseNeuronData->OpenCostFile(m_CollectNeuronDataThread->GetCostFileName());

	// Wait for completion
	m_StimulateNeuronThread->WaitForCompletion();
	m_CollectNeuronDataThread->Stop();

	// Delete objects
	delete m_CollectNeuronDataThread;
	delete m_AnalyseNeuronData;
	delete m_StimulateNeuronThread;
	delete m_GenericAlgo;
	m_CollectNeuronDataThread = 0;
	m_AnalyseNeuronData = 0;
	m_StimulateNeuronThread = 0;
	m_GenericAlgo = 0;
}

void UserInterface::run()
{
	bool running = true;
	char choise, ret;
	char fileName[50];
	int len;
	init();

	while (running) {
		printf("Select menu: \r\n");
		printf("t. Test collect neuron data\r\n");
		printf("p. Print configuration\r\n");
		printf("r. Read configuration file\r\n");
		printf("s. Stimulate neuron\r\n");
		printf("e. Exit\r\n");
		printf("\r\n> ");
		scanf("%c%c", &choise, &ret);

		switch (choise) 
		{
			case 't':
				testCollectNeuronData();
				break;
			case 'p':
				m_Configuration->Print();
				break;
			case 'r':
				printf("Enter configuration file name: ");
				fgets(fileName, sizeof(fileName), stdin);
				len = (int)strlen(fileName);
				if (len > 1)
					fileName[len - 1] = 0; // Remove CR
				else
					strcpy(fileName, DEFAULT_CONFIG); // Default configuration
				m_Configuration->ReadConfiguration(fileName);
				m_Configuration->Print();
				break;
			case 's':
			    runStimulateNeuron(m_Configuration); // Channel (0-31), loops, ms delay
				break;
			case 'e':
				running = false;
				break;
		}

	}

}
