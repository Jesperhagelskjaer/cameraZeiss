///////////////////////////////////////////////////////////
//  AnalyseNeuronData.h
//  Implementation of the Class AnalyseNeuronData
//  Created on:      19-maj-2017 22:44:35
//  Original author: au288681
///////////////////////////////////////////////////////////

#if !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
#define EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_

#include "Monitor.h"

class AnalyseNeuronData : public Monitor
{

public:
	AnalyseNeuronData();
	virtual ~AnalyseNeuronData();

	void SetMode(int mode);
	virtual void AnalyzeData(short * pBuffer, int size);
	double CalculateCost();

private:
	virtual void SearchPattern();
	void RecursiveAverage();

};
#endif // !defined(EA_B3564376_2AD6_4bb7_BFF4_ACB51E0312EB__INCLUDED_)
