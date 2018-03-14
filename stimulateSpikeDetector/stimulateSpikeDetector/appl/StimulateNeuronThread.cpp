///////////////////////////////////////////////////////////
//  StimulateNeuronThread.cpp
//  Implementation of the Class StimulateNeuronThread
//  Created on:      19-maj-2017 22:44:28
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#include "StimulateNeuronThread.h"


StimulateNeuronThread::StimulateNeuronThread() :
	Thread(),
	m_semaComplete(1, 0, "SemaSimulateNeuronThread")
{
	m_iterations = 0;
	m_delayms = 0;
	m_pausems = 0;
	m_AnalyseNeuronData = 0;
}

StimulateNeuronThread::~StimulateNeuronThread()
{

}

void StimulateNeuronThread::finalRun(int iterations)
{
	double cost;
	int iter = iterations;

	m_GenericAlgo->PrintTemplateCost();
	
	while (iter > 0)
	{
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_AVERAGE);
		m_GenericAlgo->SendTemplateToSLM(false); // 6 ms
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_ANALYSE);
		m_GenericAlgo->TurnLaserOn();
		m_AnalyseNeuronData->WaitAnalyseSamples();
		m_GenericAlgo->TurnLaserOff();
		cost = m_AnalyseNeuronData->CalculateCost();
		timeMeas.printDuration("");
		timeMeas.setStartTime();
		m_AnalyseNeuronData->AppendCostToFile(cost);
		printf("\rEND %4d  ", iter);
		iter--;
		if (m_pausems > 0)
			Sleep(m_pausems);
	}
}

void StimulateNeuronThread::run()
{
	double cost;
	int iter = 0;

	timeMeas.setStartTime();
	while (m_iterations > 0)
	{
			//timeMeas.setStartTime();
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_AVERAGE);
		m_GenericAlgo->GenerateParent(); // 1-8 ms
			//timeMeas.printDuration("Generate Parent");
			//timeMeas.setStartTime();
		m_GenericAlgo->SendTemplateToSLM(true); // 6 ms
#ifdef	TEST_GENERATOR_
		Sleep(4); // Simulate SLM delay
#endif
			//timeMeas.printDuration("Send to SLM");
			//timeMeas.setStartTime();
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_ANALYSE);
		m_GenericAlgo->TurnLaserOn();
		//Sleep(m_delayms); // 4 ms
		m_AnalyseNeuronData->WaitAnalyseSamples();
		m_GenericAlgo->TurnLaserOff();
		//m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_STOP);
			//timeMeas.printDuration("Stimulate");
			//timeMeas.setStartTime();
		//Sleep(1); // NB
		cost = m_AnalyseNeuronData->CalculateCost();
		timeMeas.printDuration("");
		timeMeas.setStartTime();
		//printf("cost %f\r\n", cost);
		m_GenericAlgo->CompareCostAndInsertTemplate(cost);
		m_AnalyseNeuronData->AppendCostToFile(cost);
		    //timeMeas.printDuration("Compute Cost");
		printf("\rDO %5d  ", m_iterations);
		m_iterations--;
		if (m_randIterations > 0 && (++iter%m_randIterations == 0)) {
			// Delete num parents each NUM_RAND_ITERATIONS
			m_GenericAlgo->DeleteTemplates(m_randTemplates);
		}
		if (m_pausems > 0)
			Sleep(m_pausems);
	}

	cout << "StimulateNeuronThread final runs using best template" << endl;
	finalRun(m_endIterations);
	cout << "StimulateNeuronThread completed" << endl;
	m_AnalyseNeuronData->CloseCostFile();
	m_semaComplete.signal();
}

void StimulateNeuronThread::Start(ThreadPriority pri, string _name, AnalyseNeuronData *pAnalyseNeuronData, 
	                              GenericAlgo *pGenericAlgo, int iterations, int pause,
								  int randIterations, int randTemplates, int endIterations)
{
	m_endIterations = endIterations;
	m_randTemplates = randTemplates;
	m_randIterations = randIterations;
	m_pausems = pause;
	m_iterations = iterations;
	m_GenericAlgo = pGenericAlgo;
	m_AnalyseNeuronData = pAnalyseNeuronData;
	runThread(pri, _name);
}

void StimulateNeuronThread::WaitForCompletion()
{
	m_semaComplete.wait();
}