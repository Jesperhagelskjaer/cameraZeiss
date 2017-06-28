/****************************************************
** Optical Stimulation of Individual Neurons in Brain
** Created: 12/5 2017 by Kim Bjerge, AU
** Modified:
******************************************************/
#include "stdafx.h"
#include "TimeMeasure.h"
#include "SLMParents.h"
#include "SLMParentsCUDA.h"
#include "TimerCUDA.h"

//----------------------------------------------------------------------------------------------
int genericCUDATest(void)
{
	/** User Interface to Optical Neuron Stimulator  **/
	//UserInterface userInterface;
	SLMParents *pSLMParents = new SLMParents(NUM_PARENTS, NUM_BINDINGS);
	SLMParentsCUDA *pSLMParentsC = new SLMParentsCUDA(NUM_PARENTS, NUM_BINDINGS);

	TimeMeasure timeMeas;
	StopWatchInterface *cudaTimer = 0;

	pSLMParentsC->InitCUDA();
	CreateTimer(&cudaTimer);

	//timeMeas.setStartTime();
	StartTimer(&cudaTimer);
	pSLMParents->GenerateNewParent();
    StopTimer(&cudaTimer);
	printf("New Parent %.2f ms\r\n", GetTimer(&cudaTimer));
	//timeMeas.printDuration("New Parent");
	pSLMParents->CompareCostAndInsertTemplate(4);
	pSLMParents->GenerateNewParent();
	pSLMParents->CompareCostAndInsertTemplate(5);
	if (pSLMParents->IsTemplatesFull()) {
		//timeMeas.setStartTime();
	    RestartTimer(&cudaTimer);
		pSLMParents->GenerateOffspring();
	    StopTimer(&cudaTimer);
		printf("Offspring %.2f ms\r\n", GetTimer(&cudaTimer));
		//timeMeas.printDuration("Offspring");
	}
	pSLMParents->PrintTemplates();

	StartTimer(&cudaTimer);
	//timeMeas.setStartTime();
	RestartTimer(&cudaTimer);
	pSLMParentsC->GenerateNewParent();
	StopTimer(&cudaTimer);
	printf("CUDA New Parent %.2f ms\r\n", GetTimer(&cudaTimer));
	//timeMeas.printDuration("CUDA New Parent");
	pSLMParentsC->CompareCostAndInsertTemplate(4);
	pSLMParentsC->GenerateNewParent();
	pSLMParentsC->CompareCostAndInsertTemplate(5);
	if (pSLMParentsC->IsTemplatesFull()) {
		//timeMeas.setStartTime();
		RestartTimer(&cudaTimer);
		pSLMParentsC->GenerateOffspring();
		StopTimer(&cudaTimer);
		printf("CUDA Offspring %.2f ms\r\n", GetTimer(&cudaTimer));
		RestartTimer(&cudaTimer);
		pSLMParentsC->GenerateOffspring();
		StopTimer(&cudaTimer);
		printf("CUDA Offspring %.2f ms\r\n", GetTimer(&cudaTimer));
		//timeMeas.printDuration("CUDA Offspring");
	}
	pSLMParentsC->ExitCUDA();
	pSLMParentsC->PrintTemplates();


	//timeMeas.setStartTime();
	//kernelTest();
	//timeMeas.printDuration("kernelTest");


	while (1) {};

	//userInterface.run();
	//genericAlgoTest();
	//neuronCaptureTest();

	return 0;

}
