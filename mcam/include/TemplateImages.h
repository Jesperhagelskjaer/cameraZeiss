///////////////////////////////////////////////////////////
//  TemplateImages.h
//  Implementation of the Class CamImage, TemplateImages
//  Created on:      19-maj-2017 22:44:35
//  Original author: Kim Bjerge
///////////////////////////////////////////////////////////
#pragma once
#include "defs.h"

#define SATURATE_VAL 16383 // Pixel saturation value - 14 bits

class CamImage
{
public:

	CamImage() : cost_(0), localCost_(0), numSaturated_(0)
	{
		ClearData();
	}

	void CopyImage(unsigned short *pImage, int height, int width)
	{
		ClearData();
		if (width > COLS) {
			printf("CamImage::CopyImage width %d too big\r\n", width);
			return;
		}
		if (height > ROWS) {
			printf("CamImage::CopyImage height %d too big\r\n", height);
			return;
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
					data_[i][j] = pImage[i*width + j];
			}
		}
	}

	void CopyImageOnly(unsigned short *pImage, int height, int width, RECT rec)
	{
		int id, jd;
		ClearData();
		int Wrec = rec.right - rec.left;
		int Hrec = rec.bottom - rec.top;
		if (Wrec > COLS) {
			printf("CamImage::CopyImage width %d too big\r\n", Wrec);
			return;
		}
		if (Hrec > ROWS) {
			printf("CamImage::CopyImage height %d too big\r\n", Hrec);
			return;
		}

		id = 0;
		for (int i = 0; i < height; i++) {
			jd = 0;
			for (int j = 0; j < width; j++) {
				if ((rec.top <= i && i < rec.bottom) &&
					(rec.left <= j && j < rec.right))
					data_[id][jd++] = pImage[i*width + j];
			}
			if (rec.top <= i && i < rec.bottom)
				id++;
		}
		localCost_ = 0;
	}

	// Copies part of image specified by RECT and computes cost for area around local maximum
	void CopyImageFindLocalMax(unsigned short *pImage, int height, int width, RECT rec)
	{
		int id, jd;
		ClearData();
		int Wrec = rec.right - rec.left;
		int Hrec = rec.bottom - rec.top;
		int lj, li, localArea;
		unsigned short data;
		unsigned short localMaximum = 0;
		if (Wrec > COLS) {
			printf("CamImage::CopyImage width %d too big\r\n", Wrec);
			return;
		}
		if (Hrec > ROWS) {
			printf("CamImage::CopyImage height %d too big\r\n", Hrec);
			return;
		}

		// Copy focus area and search for local maximum outside focus area
		id = 0;
		li = 0; 
		lj = 0;
		for (int i = 0; i < height; i++) {
			jd = 0;
			for (int j = 0; j < width; j++) {
				data = pImage[i*width + j];
				if ((rec.top <= i && i < rec.bottom) &&
					(rec.left <= j && j < rec.right))
					data_[id][jd++] = data; // Copies RECT area
				else {
					if (data > localMaximum) { // Finds local maximum outside RECT area
						localMaximum = data;
						lj = j - Wrec / 2;
						li = i - Hrec / 2;
					}
				}
			}
			if (rec.top <= i && i < rec.bottom)
				id++;
		}

		// Computes cost for area around local maximum
		localCost_ = 0;
		localArea = 0;
		if (localMaximum > 0) {
			if (li < 0) { // Adjust vertical corner case
				Hrec += li;
				li = 0;
			}
			if (lj < 0) { // Adjust horisontal corner case
				Wrec += lj; 
				lj = 0;
			}
			for (int i = li; i < Hrec+li && i < height; i++) {
				for (int j = lj; j < Wrec+lj && j < width; j++) {
					data = pImage[i*width + j];
					// Check local maximum area outside focus area
					if ((i < rec.top || i >= rec.bottom) &&
						(j < rec.left || j >= rec.right)) {
						localCost_ += data;
						localArea++;
					}
				}
			}
		}
		if (localArea > 0) 
			localCost_ = localCost_ / localArea; // Normalize local cost as average value
	}

	void CopyImage(unsigned short *pImage, int height, int width, RECT rec, int costFunction = 0)
	{
		if (costFunction == 1)
			CopyImageFindLocalMax(pImage, height, width, rec);
		else {
			CopyImageOnly(pImage, height, width, rec);
		}
	}
	
	double ComputeIntencity(void)
	{
		unsigned short data;
		cost_ = 0;
		numSaturated_ = 0;

		// computes cost as sum of pixels in image focus area
		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++) {
				data = data_[i][j];
				cost_ += data;
				if (data >= SATURATE_VAL) 
					numSaturated_++; // computes number saturated pixels
			}
		}

		// computes cost as sum of pixels divided by mean of local maximum area
		if (localCost_ > 0)
			cost_ = cost_ / localCost_; // Cost function = 1
	
		return cost_;
	}

	int getSaturated(void)
	{
		return numSaturated_;
	}

	void Print(void) 
	{
		printf("CamImage::data_\r\n");
		printf("line1: %d %d %d %d %d %d %d\r\n", data_[0][0], data_[0][1], data_[0][2], data_[0][3], data_[0][4], data_[0][5], data_[0][COLS-1]);
		printf("line2: %d %d %d %d %d %d %d\r\n", data_[1][0], data_[1][1], data_[1][2], data_[1][3], data_[1][4], data_[1][5], data_[1][COLS-1]);
		printf("line3: %d %d %d %d %d %d %d\r\n", data_[2][0], data_[2][1], data_[2][2], data_[2][3], data_[2][4], data_[2][5], data_[2][COLS-1]);
		printf("line4: %d %d %d %d %d %d %d\r\n", data_[3][0], data_[3][1], data_[3][2], data_[3][3], data_[3][4], data_[3][5], data_[3][COLS-1]);
		printf("line5: %d %d %d %d %d %d %d\r\n", data_[4][0], data_[4][1], data_[4][2], data_[4][3], data_[4][4], data_[4][5], data_[4][COLS-1]);
		printf("line6: %d %d %d %d %d %d %d\r\n", data_[5][0], data_[5][1], data_[5][2], data_[5][3], data_[5][4], data_[5][5], data_[5][COLS-1]);
		printf("line49: %d %d %d %d %d %d %d\r\n", data_[ROWS-1][0], data_[ROWS-1][1], data_[ROWS-1][2], data_[ROWS-1][3], 
												   data_[ROWS-1][4], data_[ROWS-1][5], data_[ROWS-1][COLS - 1]);

	}

private:
	unsigned short data_[ROWS][COLS];
	double cost_;
	double localCost_;
	int numSaturated_;

	void ClearData(void) 
	{
		memset(&data_[0][0], 0, sizeof(data_));
		/*
		for (int i = 0; i < ROWS; i++) 
			for (int j = 0; j < COLS; j++)
				data_[i][j] = 0;
		*/
	}
};

