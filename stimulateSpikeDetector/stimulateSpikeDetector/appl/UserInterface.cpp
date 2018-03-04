///////////////////////////////////////////////////////////
//  UserInteface.cpp
//  Implementation of the Class UserInteface
//  Created on:      20-maj-2017 08:53:04
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include "UserInterface.h"
#include "AnalyseNeuronDataCAR.h"
#include "AnalyseNeuronDataCMR.h"
#include "AnalyseNeuronSpikeDetector.h"


UserInterface::UserInterface()
{
	m_Configuration = 0;
	m_AnalyseNeuronData = 0;
	m_CollectNeuronDataThread = 0;
	m_StimulateNeuronThread = 0;
	m_NeuronSpikeDetector = 0;
}

UserInterface::~UserInterface()
{

}

void UserInterface::init(void)
{
	m_Configuration = new Configuration(GEN_ITERATIONS, ACTIVE_CHANNEL, LASER_PORT, (int)DELAY_MS, 
		                                PAUSE_MS, LASER_INTENSITY, FirFilter::FILTER_TYPE,
										NUM_PARENTS, NUM_BINDINGS, COMMON_REF,
										NUM_RAND_ITERATIONS, NUM_RAND_TEMPLATES, NUM_END_ITERATIONS);
}

void UserInterface::testNeuronSpikeDetector(void)
{
	m_NeuronSpikeDetector = new NeuronSpikeDetector();
	m_NeuronSpikeDetector->Create();
	m_NeuronSpikeDetector->Train();
	m_NeuronSpikeDetector->Predict();
	m_NeuronSpikeDetector->Terminate();
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

void UserInterface::runStimulateIndividualNeurons(Configuration *config)
{
	// Create objects
	m_NeuronSpikeDetector = new NeuronSpikeDetector();
	m_NeuronSpikeDetector->Create(); // Loads training files and templates
	m_NeuronSpikeDetector->Train(); // Train spike detector

	m_AnalyseNeuronData = new AnalyseNeuronSpikeDetector();
	m_CollectNeuronDataThread = new CollectNeuronDataThread();
	m_StimulateNeuronThread = new StimulateNeuronThread();
	m_GenericAlgo = new GenericAlgo(config->m_NumParents, config->m_NumBindings, config->m_NumIterations);

	// Initialization of objects
	m_GenericAlgo->OpenLaserPort(config->m_LaserPort, config->m_LaserIntensity);
	m_StimulateNeuronThread->SetDelay(config->m_DelayMS);
	m_AnalyseNeuronData->SetDelaySamples((int)ceil(SAMPLE_FREQUENCY / 1000 * config->m_DelayMS));
	m_AnalyseNeuronData->SetActiveChannel(config->m_ActiveChannel);
	m_AnalyseNeuronData->SetFilterType(config->m_FilterType);
	((AnalyseNeuronSpikeDetector *)m_AnalyseNeuronData)->AddSpikeDetector(m_NeuronSpikeDetector);

	// Start threads
	m_CollectNeuronDataThread->Start(Thread::PRIORITY_HIGH, "NeuronDataThread", m_AnalyseNeuronData);
	m_AnalyseNeuronData->OpenCostFile(m_CollectNeuronDataThread->GetCostFileName());
	m_StimulateNeuronThread->Start(Thread::PRIORITY_ABOVE_NORMAL, "StimulateNeuronThread",
		m_AnalyseNeuronData, m_GenericAlgo,
		config->m_NumIterations, config->m_PauseMS,
		config->m_RandIterations, config->m_RandTemplates, config->m_EndIterations);

	// Wait for completion
	m_StimulateNeuronThread->WaitForCompletion();
	m_CollectNeuronDataThread->Stop();

	printf("Total number of neuron spikes found %d \r\n", ((AnalyseNeuronSpikeDetector *)m_AnalyseNeuronData)->GetTotalSpikesFound());
	m_NeuronSpikeDetector->Terminate();

	// Delete objects
	delete m_CollectNeuronDataThread;
	delete m_AnalyseNeuronData;
	delete m_StimulateNeuronThread;
	delete m_GenericAlgo;
	delete m_NeuronSpikeDetector;
	m_CollectNeuronDataThread = 0;
	m_AnalyseNeuronData = 0;
	m_StimulateNeuronThread = 0;
	m_GenericAlgo = 0;
}


void UserInterface::runStimulateNeurons(Configuration *config)
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
	m_AnalyseNeuronData->OpenCostFile(m_CollectNeuronDataThread->GetCostFileName());
	m_StimulateNeuronThread->Start(Thread::PRIORITY_ABOVE_NORMAL, "StimulateNeuronThread",
		                           m_AnalyseNeuronData, m_GenericAlgo, 
		                           config->m_NumIterations, config->m_PauseMS,
								   config->m_RandIterations, config->m_RandTemplates, config->m_EndIterations);
	m_StimulateNeuronThread->SetDelay(config->m_DelayMS);

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
		printf("n. Test neuron spike detector \r\n");
		printf("p. Print configuration\r\n");
		printf("r. Read configuration file\r\n");
		printf("s. Stimulate neurons\r\n");
		printf("i. Stimulate individual neurons\r\n");
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
			    runStimulateNeurons(m_Configuration); // Channel (0-31), loops, ms delay
				break;
			case 'i':
				runStimulateIndividualNeurons(m_Configuration); 
				break;
			case 'n':
				testNeuronSpikeDetector(); // Test train and predict on simulated test data
				break;
			case 'e':
				running = false;
				break;
		}

	}

}
