///////////////////////////////////////////////////////////
//  GenericAlgo.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once

#include <windows.h>
#include "SLMParents.h"
#include "TemplateImages.h"
#include "TimeMeasure.h"
#include "LaserInterface.h"
//KBE??? 
#include "SLMInterface.h"

class GenericAlgo {
public:
	
	GenericAlgo() : laser(115200)
	{
		pSLMParents_ = new SLMParents(NUM_PARENTS);
		pImg_ = new CamImage();
		num_iterations_ = NUM_ITERATIONS;

#ifdef	SLM_INTERFACE_
	   pSLMInterface_ = new SLMInterface();
	}

	GenericAlgo(Blink_SDK *pSLMsdk) : laser(115200)
	{
		pSLMParents_ = new SLMParents(NUM_PARENTS);
		pImg_ = new CamImage();
		pSLMInterface_ = new SLMInterface(pSLMsdk);
#endif

	}
	
	void OpenLaserPort(int port)
	{
		laser.OpenPort(port);
	}

	void TurnLaserOn(void)
	{
		laser.TurnOn();
	}

	void TurnLaserOff(void)
	{
		laser.TurnOff();
	}
	
	int GetNumIterations(void) 
	{
		return num_iterations_;
	}
		
	void GenerateParent(void) 
	{
		if (pSLMParents_->IsTemplatesFull()) {
			//pSLMParents_->PrintTemplates();
			pSLMParents_->GenerateOffspring(1);
		} else {
			pSLMParents_->GenerateNewParent();
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

	
	void ComputeIntencity(unsigned short *pImage, RECT rec)
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
#if 1
		// For zoom in image already zoomed
		rec.left = 225;
		rec.top = 225;
		rec.right = 275;
		rec.bottom = 275;
		//printf("Image taken L%d, R%d, T%d, B%d, H%d, W%d\r\n", rec.left, rec.right, rec.top, rec.bottom, height, width);
		pImg_->CopyImage(pixel, height, width, rec);
#else
		pImg_->CopyImage(pixel, height, width);
#endif
		//pImg_->Print(); //KBE??? For debug only
		cost = pImg_->ComputeIntencity();
		CompareCostAndInsertTemplate(cost);
		//printf("ComputeIntencity done\r\n");
		//pSLMParents_->PrintTemplates(); //KBE??? For debug only

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
	SLMParents *pSLMParents_;
	CamImage *pImg_;
	LaserInterface laser;
};
