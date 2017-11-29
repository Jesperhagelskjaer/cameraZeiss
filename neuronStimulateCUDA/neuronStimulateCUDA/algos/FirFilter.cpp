///////////////////////////////////////////////////////////
//  FirFilter.cpp
//  Implementation of the Class FirFilter
//  Created on:      17-june-2017 22:37:03
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once
#include <math.h>
#include "FirFilter.h"
#include "FilterCoeffs_0_1Hz.h"
#include "FilterCoeffs_1Hz.h"
#include "FilterCoeffs_10Hz.h"
#include "FilterCoeffs_100Hz.h"
#include "FilterCoeffs_300Hz.h"

void FirFilter::setType(TYPES type)
{
	type_ = type;

	switch(type_) {
		case FC_0_1Hz:
			setCoeffs(B0_1Hz, TAPS);
		    break;
		case FC_1Hz:
			setCoeffs(B1Hz, TAPS);
			break;
		case FC_10Hz:
			setCoeffs(B10Hz, TAPS);
			break;
		case FC_100Hz:
			setCoeffs(B100Hz, TAPS);
			break;
		case FC_300Hz:
			setCoeffs(B300Hz, TAPS);
			break;
	}
}

void FirFilter::filter(int32_t *x, int32_t *y)
{
	double yf[NUM_CHS];
	for (int ch = 0; ch < NUM_CHS; ch++) {
		delayLine_[ch][idx_] = x[ch];
		yf[ch] = 0;
	}
	idx_ = (idx_ + 1) % TAPS;
	int idx = idx_; // Oldest value
	for (int i = 0; i < TAPS; i++) {
		for (int ch = 0; ch < NUM_CHS; ch++) {
			yf[ch] += delayLine_[ch][idx] * coeffs_[i];
		}
		idx = (idx + 1) % TAPS;
	}
	for (int ch = 0; ch < NUM_CHS; ch++) {
		y[ch] = (int32_t)round(yf[ch]);
	}
}

void FirFilter::setCoeffs(const double *pCoeffs, int len)
{
	for (int i = 0; i < TAPS; i++) {
		if (i < len)
			coeffs_[i] = pCoeffs[i];
		else
			coeffs_[i] = 0;
	}
}
