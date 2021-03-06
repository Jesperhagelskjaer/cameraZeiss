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

void StimulateNeuronThread::run()
{
	double cost;

	while (m_iterations > 0)
	{
			timeMeas.setStartTime();
		m_AnalyseNeuronData->SetMode(AnalyseNeuronData::MODE_AVERAGE);
		m_GenericAlgo->GenerateParent(); // 1-8 ms
			timeMeas.printDuration("Generate Parent");
			//timeMeas.setStartTime();
		m_GenericAlgo->SendTemplateToSLM(); // 6 ms
		//Sleep(6); // Simulate SLM delay
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
		//printf("cost %f\r\n", cost);
		m_GenericAlgo->CompareCostAndInsertTemplate(cost);
		m_AnalyseNeuronData->AppendCostToFile(cost);
			//timeMeas.printDuration("Compute Cost");
		printf("%d\r", m_iterations);
		m_iterations--;
		if (m_pausems > 0)
			Sleep(m_pausems);
	}
	cout << "StimulateNeuronThread completed" << endl;
	m_AnalyseNeuronData->CloseCostFile();
	m_semaComplete.signal();
}

void StimulateNeuronThread::Start(ThreadPriority pri, string _name, AnalyseNeuronData *pAnalyseNeuronData, 
	                              GenericAlgo *pGenericAlgo, int iterations, int pause)
{

	m_iterations = iterations;
	m_pausems = pause;
	m_AnalyseNeuronData = pAnalyseNeuronData;
	m_GenericAlgo = pGenericAlgo;
	runThread(pri, _name);
}

void StimulateNeuronThread::WaitForCompletion()
{
	m_semaComplete.wait();
}