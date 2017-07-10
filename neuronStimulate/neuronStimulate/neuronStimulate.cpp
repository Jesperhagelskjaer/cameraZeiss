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
	/** User Interface to Optical Neuron Stimulator  **/
	UserInterface userInterface;

	userInterface.run();
	//genericAlgoTest();
	//neuronCaptureTest();

	printf("Neuron Stimulate Completed\r\n");
	return 0;

}
