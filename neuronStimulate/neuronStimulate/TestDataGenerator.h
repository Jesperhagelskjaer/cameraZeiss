///////////////////////////////////////////////////////////
//  TestDataGenerator.h
//  Implementation of the Class TestDataGenerator
//  Created on:      24-maj-2017 11:01:53
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////

#if !defined(EA_6A44561D_E653_418c_AA0E_EB74C9963716__INCLUDED_)
#define EA_6A44561D_E653_418c_AA0E_EB74C9963716__INCLUDED_

#include "LxRecord.h"

class TestDataGenerator
{

public:
	TestDataGenerator();
	virtual ~TestDataGenerator();

	void SetPulseActive(bool generate) {
		m_generatePulse = generate;
	}

	void GenerateSampleRecord(LRECORD *pLxRecord);

private:
	LRECORD m_LRECORD;
	bool m_generatePulse;
};

#endif // !defined(EA_6A44561D_E653_418c_AA0E_EB74C9963716__INCLUDED_)
