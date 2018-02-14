///////////////////////////////////////////////////////////
//  FirFilter.h
//  Implementation of the Class FirFilter
//  Created on:      17-june-2017 22:37:03
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once
#include <cstdint>

#define TAPS			257 // Number of filter taps
#define NUM_CHS			32  // Number of channels to filter

const char FilerTypesText[7][10] =
{
	"bypass   ",
	"custom   ",
	"fc 0.1Hz ",
	"fc 1Hz   ",
	"fc 10Hz  ",
	"fc 100Hz ",
	"fc 300Hz "
};

class FirFilter
{
public:
	enum TYPES {
		BYPASS,
		CUSTOM,
		FC_0_1Hz,
		FC_1Hz,
		FC_10Hz,
		FC_100Hz,
		FC_300Hz
	};
	
	FirFilter() : idx_(0)
	{
		// Clear delay lines and coefficients
		for (int ch = 0; ch < NUM_CHS; ch++)
			for (int i = 0; i < TAPS; i++)
				delayLine_[ch][i] = 0;
		for (int i = 0; i < TAPS; i++)
				coeffs_[i] = 0;
		type_ = BYPASS;
	}

	void filter(int32_t *x, int32_t *y);
	void setCoeffs(const double *pCoeffs, int len);
	void setType(TYPES type);

private:
	TYPES type_;
	int idx_;
	double coeffs_[TAPS];
	int32_t delayLine_[NUM_CHS][TAPS];
};
