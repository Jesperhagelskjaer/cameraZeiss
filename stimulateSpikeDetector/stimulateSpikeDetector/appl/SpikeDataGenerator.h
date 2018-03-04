///////////////////////////////////////////////////////////
//  SpikeDataGenerator.h
//  Implementation of the Class SpikeDataGenerator
//  Created on:      04-marts-2018 12:33:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(SPIKE_DATA_GENERATOR_INCLUDED_)
#define SPIKE_DATA_GENERATOR_INCLUDED_

#include "TestDataGenerator.h"
class ProjectInfo;

class SpikeDataGenerator : public TestDataGenerator
{

public:
	SpikeDataGenerator();
	virtual ~SpikeDataGenerator();

	virtual void GenerateSampleRecord(LRECORD *pLxRecord);
	void SetProjectInfo(ProjectInfo *pProjectInfo);

private:
	LRECORD m_LRECORD;
	bool m_generatePulse;
	ProjectInfo *m_projectInfo;
	int m_sampleOffset;
};

#endif // !defined(SPIKE_DATA_GENERATOR_INCLUDED_)
