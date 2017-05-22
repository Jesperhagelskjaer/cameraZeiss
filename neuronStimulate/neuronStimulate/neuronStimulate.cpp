/****************************************************
** Optical Stimulation of Individual Neurons in Brain
** Created: 12/5 2017 by Kim Bjerge, AU
** Modified:
******************************************************/
#include "stdafx.h"
//#include <iostream>
//using namespace std;
//#include "LynxRecord.h"
//#include "AnalyseNeuronData.h"
//#include "Configuration.h"
//#include "DataFileThread.h"
//#include "CollectNeuronDataThread.h"
//#include "StimulateNeuronThread.h"
#include "UserInterface.h"

int neuronCaptureTest(void);
int genericAlgoTest(void);

/** User Interface to Optical Neuron Stimulator  **/
UserInterface userInterface;

int runOpticalNeuronStimulator(void)
{

	userInterface.Create();
	userInterface.run();

	return 0;
}

//----------------------------------------------------------------------------------------------
int main(int argc, _TCHAR* argv[])
{

	runOpticalNeuronStimulator();
	//genericAlgoTest();
	//neuronCaptureTest();

	return 0;

}
