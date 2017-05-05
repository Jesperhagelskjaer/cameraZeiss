// genericAlgo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SLMParents.h"
#include "TemplateImages.h"

SLMParents slmParents(NUM_PARENTS);

int main()
{
	SLMTemplate *pParentNew;
	slmParents.PrintTemplates();
	pParentNew = slmParents.GenerateOffspring(1);
	pParentNew->Print();
    return 0;
}

