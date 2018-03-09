///////////////////////////////////////////////////////////
//  GenericAlgo.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once

#include <windows.h>
#include <stdio.h>

#include "TemplateImages.h"
#include "TimeMeasure.h"
#include "LaserInterface.h"

#ifdef SLM_INTERFACE_
	#include "SLMInterface.h"
#endif

#ifdef USE_CUDA_GEN
#include "SLMParentsCUDA.h"
#else
#include "SLMParents.h"
#endif

class GenericAlgo {
public:
	
	GenericAlgo(int numParents, int numBindings, int numIterations) 
		: laser(115200)
	{
#ifdef USE_CUDA_GEN
		pSLMParents_ = new SLMParentsCUDA(numParents, numBindings);
		pSLMParents_->InitCUDA();
		std::cout << "CUDA Optimized Generic Algorithm, NOT FULLY TESTED" << std::endl;
#else
		pSLMParents_ = new SLMParents(numParents, numBindings);
#endif
		pImg_ = new CamImage();
		num_iterations_ = numIterations;
		laserIntensity_ = 0;

#ifdef	SLM_INTERFACE_
	   pSLMInterface_ = new SLMInterface();
	}

	GenericAlgo(Blink_SDK *pSLMsdk)
		: laser(115200)
	{

#ifdef USE_CUDA_GEN
		pSLMParents_ = new SLMParentsCUDA(numParents, numBindings);
		pSLMParents_->InitCUDA();
#else
		pSLMParents_ = new SLMParents(numParents, numBindings);
#endif
		pImg_ = new CamImage();
		pSLMInterface_ = new SLMInterface(pSLMsdk);
		laserIntensity_ = 0;
#endif

	}

	void OpenLaserPort(int port, float intensity)
	{
		laser.OpenPort(port);
		laserIntensity_ = intensity;
	}

	void TurnLaserOn(void)
	{
		laser.TurnOn(laserIntensity_);
	}

	void TurnLaserOff(void)
	{
		laser.TurnOff();
	}

	float GetLaserIntensity(void)
	{
		return laserIntensity_;
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

	void PrintTemplateCost(void)
	{
		pSLMParents_->PrintTemplatesCost();
	}

	void SendTemplateToSLM(bool trainGenericAlgo)
	{
		unsigned char *pSLMParentMatrix;
		if (trainGenericAlgo)
			pSLMParentMatrix = pSLMParents_->GetNewParentMatrixPtr();
		else
			pSLMParentMatrix = pSLMParents_->GetMaxCostParentMatrixPtr();
#ifdef	SLM_INTERFACE_
		pSLMInterface_->SendPhase(pSLMParentMatrix);
#endif
	}
			
	void StartSLM(bool trainGenericAlgo = true)
	{
		if (trainGenericAlgo) {
			GenerateParent();
		}
		SendTemplateToSLM(trainGenericAlgo);
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

	double ComputeIntencity(unsigned short *pImage, RECT rec, bool trainMode = true)
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
		pImg_->CopyImage(pixel, height, width, rec, COST_FUNCTION);
#endif
		//pImg_->Print(); //KBE??? For debug only
		cost = pImg_->ComputeIntencity();
		
		if (trainMode) {
			// Only in training mode
			CompareCostAndInsertTemplate(cost);

			// Decrease laser intencity if many pixels are saturated
			if (pImg_->getSaturated() > NUM_SATURATED && laserIntensity_ >= LASER_STEP + 0.2)
			{
				laserIntensity_ -= LASER_STEP;
				printf("Decreased laser intensity to %0.2f\r\n", laserIntensity_);
			}
		}

		//printf("ComputeIntencity done\r\n");
		//pSLMParents_->PrintTemplates(); //KBE??? For debug only
		return cost;
	}

	// Generates a number of random templates every num_iterations by deleting lowest cost templates
	void DeleteTemplates(int num_templates)
	{
		pSLMParents_->DeleteTemplates(num_templates);
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

#ifdef USE_CUDA_GEN
	SLMParentsCUDA *pSLMParents_;
#else
	SLMParents *pSLMParents_;
#endif
	
	CamImage *pImg_;
	//TimeMeasure timeMeas;
	float laserIntensity_;
	LaserInterface laser;
};
