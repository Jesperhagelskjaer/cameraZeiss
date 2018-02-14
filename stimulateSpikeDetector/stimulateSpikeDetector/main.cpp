#if 0

/****************************************************
** Optical Stimulation of Individual Neurons in Brain
** Created: 12/5 2017 by Kim Bjerge, AU
** Modified:
******************************************************/
#include "stdafx.h"
#include "UserInterface.h"

int neuronCaptureTest(void);
int genericAlgoTest(void);

//----------------------------------------------------------------------------------------------
int main(int argc, _TCHAR* argv[])
{
	// User Interface to Optical Neuron 
	UserInterface userInterface;

	userInterface.run();
	//genericAlgoTest();
	//neuronCaptureTest();

	printf("Neuron Stimulate Completed\r\n");
	return 0;

}

#else

/****************************************************
** Spike Detection project
** Created: 28/8 2017 by MB and ATY, AU
** Modified: KBE
******************************************************/

//#include "SpikeDetection.h"

#include "ProjectDefinitions.h"

#ifdef USE_CUDA
//#include "SpikeDetectCUDA.h"
#include "SpikeDetectCUDA_RTP.h"
#else
#include "SpikeDetect.h"
#endif
//----------------------------------------------------------------------------------------------

#ifdef _DEBUG
#ifdef USE_CUDA
//#error ChannelFilter CUDA kernel does not work correctly in debug mode!
#endif
#endif

int main(void)
{
	int returnValue = 0;

#ifdef USE_CUDA
	//SpikeDetectCUDA<USED_DATATYPE> *spikeDetector;
	//spikeDetector = new SpikeDetectCUDA<USED_DATATYPE>();
	SpikeDetectCUDA_RTP<USED_DATATYPE> *spikeDetector;
	spikeDetector = new SpikeDetectCUDA_RTP<USED_DATATYPE>();

	spikeDetector->runTraining(); // Training not using CUDA
	//spikeDetector->runTrainingCUDA();
#else
	SpikeDetect<USED_DATATYPE> *spikeDetector;
	spikeDetector = new SpikeDetect<USED_DATATYPE>();

	spikeDetector->runTraining();
#endif

	spikeDetector->runPrediction();

	delete spikeDetector;
	
	return returnValue;
}

#endif
