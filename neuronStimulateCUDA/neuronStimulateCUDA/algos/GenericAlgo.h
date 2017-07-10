///////////////////////////////////////////////////////////
//  GenericAlgo.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once

#include <windows.h>
#include "SLMParentsCUDA.h"
#include "TemplateImages.h"
#include "TimeMeasure.h"
#ifdef LASER_INTERFACE_
	#include "LaserInterface.h"
#endif
#ifdef SLM_INTERFACE_
	#include "SLMInterface.h"
#endif

class GenericAlgo {
public:
	
	GenericAlgo(int numParents, int numBindings, int numIterations) 
#ifdef LASER_INTERFACE_
		: laser(115200), laserIntensity_(0)
#endif
	{
		pSLMParents_ = new SLMParentsCUDA(numParents, numBindings);
		pImg_ = new CamImage();
		num_iterations_ = numIterations;
		pSLMParents_->InitCUDA();

#ifdef	SLM_INTERFACE_
	   pSLMInterface_ = new SLMInterface();
	}

	GenericAlgo(Blink_SDK *pSLMsdk)
#ifdef LASER_INTERFACE_
		: laser(115200), laserIntensity_(0)
#endif
	{
		pSLMParents_ = new SLMParentsCUDA(NUM_PARENTS, NUM_BINDINGS);
		pImg_ = new CamImage();
		pSLMParents_->InitCUDA();
		pSLMInterface_ = new SLMInterface(pSLMsdk);
#endif

	}

	void OpenLaserPort(int port, float intensity)
	{
#ifdef LASER_INTERFACE_
		laser.OpenPort(port);
		laserIntensity_ = intensity;
#endif
	}

	void TurnLaserOn(void)
	{
#ifdef LASER_INTERFACE_		
		laser.TurnOn(laserIntensity_);
#endif
	}

	void TurnLaserOff(void)
	{
#ifdef LASER_INTERFACE_
		laser.TurnOff();
#endif
	}
	
	int GetNumIterations(void) 
	{
		return num_iterations_;
	}
		
	void GenerateParent(void) 
	{
		//timeMeas.setStartTime();

		if (pSLMParents_->IsTemplatesFull()) {
			//pSLMParents_->PrintTemplates();
			pSLMParents_->GenerateOffspring();
		} else {
			//timeMeas.printDuration("Generic Offspring");
			pSLMParents_->GenerateNewParent();
			//timeMeas.printDuration("Generic New Parent");
		}
	}

	void SendTemplateToSLM(void) 
	{
#ifdef	SLM_INTERFACE_
		//pSLMInterface_->SendTestPhase(pSLMParents_->GetNewParentMatrixPtr(), M);
		pSLMInterface_->SendPhase(pSLMParents_->GetNewParentMatrixPtr());
#endif
	}
			
	void StartSLM()
	{
		GenerateParent();
		SendTemplateToSLM();
	}
	
	void CompareCostAndInsertTemplate(double cost)
	{
		 pSLMParents_->CompareCostAndInsertTemplate(cost);
	}

	void TestComputeIntencity(double cost)
	{
		pSLMParents_->CompareCostAndInsertTemplate(cost);
		pSLMParents_->PrintTemplates();
	}

	double ComputeIntencity(unsigned short *pImage, RECT rec)
	{
		double cost;
		int width, height;
		IMAGE_HEADER* header = (IMAGE_HEADER*)pImage;

		bool isColorImage = 0;
		// painting raw data to image
		unsigned short* pixel = NULL;

		width = header->roiWidth / header->binX;
		height = header->roiHeight / header->binY;

		isColorImage = header->bitsPerPixel == MCAM_BPP_COLOR;

		// painting raw camera data to image
		pixel = (unsigned short*)pImage + header->headerSize / 2;
#if 0
		// For zoom in image already zoomed
		rec.left = 245;
		rec.top = 245;
		rec.right = 255;
		rec.bottom = 255;
		//printf("Image taken L%d, R%d, T%d, B%d, H%d, W%d\r\n", rec.left, rec.right, rec.top, rec.bottom, height, width);
		pImg_->CopyImage(pixel, height, width, rec);
#else
		pImg_->CopyImage(pixel, height, width, rec);
#endif
		//pImg_->Print(); //KBE??? For debug only
		cost = pImg_->ComputeIntencity();
		CompareCostAndInsertTemplate(cost);
		//printf("ComputeIntencity done\r\n");
		//pSLMParents_->PrintTemplates(); //KBE??? For debug only
		return cost;
	}
	

	~GenericAlgo()
	{
		delete pSLMParents_;
		delete pImg_;

#ifdef	SLM_INTERFACE_
		delete pSLMInterface_;
	}
private:
		SLMInterface *pSLMInterface_;
#else
	}
#endif

private:
	int num_iterations_;
	SLMParentsCUDA *pSLMParents_;
	CamImage *pImg_;
	//TimeMeasure timeMeas;
#ifdef LASER_INTERFACE_
	LaserInterface laser;
	float laserIntensity_;
#endif
};
