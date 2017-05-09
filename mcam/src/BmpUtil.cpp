/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
**************************************************************************
* \file BmpUtil.cpp
* \brief Contains basic image operations implementation.
*
* This file contains implementation of basic bitmap loading, saving, 
* conversions to different representations and memory management routines.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "BmpUtil.h"


#ifdef _WIN32
#pragma warning( disable : 4996 ) // disable deprecated warning 
#endif


/**
**************************************************************************
*  The routine clamps the input value to integer byte range [0, 255]
*
* \param x			[IN] - Input value
*  
* \return Pointer to the created plane
*/
int clamp_0_255(int x)
{
	return (x < 0) ? 0 : ( (x > 255) ? 255 : x );
}


/**
**************************************************************************
*  Float round to nearest value
*
* \param num			[IN] - Float value to round
*  
* \return The closest to the input float integer value
*/
float round_f(float num) 
{
	float NumAbs = (float)fabs(num);
	int NumAbsI = (int)(NumAbs + 0.5f);
	float sign = num > 0 ? 1.0f : -1.0f;
	return sign * NumAbsI;
}


/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 8bpp pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
byte *MallocPlaneByte(int width, int height, int *pStepBytes)
{
	byte *ptr;
	*pStepBytes = ((int)ceil(width/16.0f))*16;
	ptr = (byte *)malloc(*pStepBytes * height);
	return ptr;
}


/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 16bpp float pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
short *MallocPlaneShort(int width, int height, int *pStepBytes)
{
	short *ptr;
	*pStepBytes = ((int)ceil((width*sizeof(short))/16.0f))*16;
	ptr = (short *)malloc(*pStepBytes * height);
	*pStepBytes = *pStepBytes / sizeof(short);
	return ptr;
}

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 32bpp float pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
float *MallocPlaneFloat(int width, int height, int *pStepBytes)
{
	float *ptr;
	*pStepBytes = ((int)ceil((width*sizeof(float))/16.0f))*16;
	ptr = (float *)malloc(*pStepBytes * height);
	*pStepBytes = *pStepBytes / sizeof(float);
	return ptr;
}


/**
**************************************************************************
*  Copies byte plane to float plane
*
* \param ImgSrc				[IN] - Source byte plane
* \param StrideB			[IN] - Source plane stride
* \param ImgDst				[OUT] - Destination float plane
* \param StrideF			[IN] - Destination plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void CopyByte2Float(byte *ImgSrc, int StrideB, float *ImgDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgDst[i*StrideF+j] = (float)ImgSrc[i*StrideB+j];
		}
	}
}


/**
**************************************************************************
*  Copies float plane to byte plane (with clamp)
*
* \param ImgSrc				[IN] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param ImgDst				[OUT] - Destination byte plane
* \param StrideB			[IN] - Destination plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void CopyFloat2Byte(float *ImgSrc, int StrideF, byte *ImgDst, int StrideB, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgDst[i*StrideB+j] = (byte)clamp_0_255((int)(round_f(ImgSrc[i*StrideF+j])));
		}
	}
}


/**
**************************************************************************
*  Memory deallocator, deletes aligned format frame.
*
* \param ptr			[IN] - Pointer to the plane
*  
* \return None
*/
void FreePlane(void *ptr)
{
	if (ptr) 
	{
		free(ptr);
	}
}


/**
**************************************************************************
*  Performs addition of given value to each pixel in the plane
*
* \param Value				[IN] - Value to add
* \param ImgSrcDst			[IN/OUT] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgSrcDst[i*StrideF+j] += Value;
		}
	}
}


/**
**************************************************************************
*  Performs multiplication of given value with each pixel in the plane
*
* \param Value				[IN] - Value for multiplication
* \param ImgSrcDst			[IN/OUT] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgSrcDst[i*StrideF+j] *= Value;
		}
	}
}


/**
**************************************************************************
*  This function performs acquisition of image dimensions
*
* \param FileName		[IN] - Image name to load
* \param Width			[OUT] - Image width from file header
* \param Height			[OUT] - Image height from file header
*
* \return Status code
*/
int PreLoadBmp(char *FileName, int *Width, int *Height)
{
	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	FILE *fh;
	
	if (!(fh = fopen(FileName, "rb")))
	{
		return 1; //invalid filename
	}

	fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);

	if (FileHeader._bm_signature != 0x4D42)
	{
		return 2; //invalid file format
	}

	fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);

	if (InfoHeader._bm_color_depth != 24)
	{
		return 3; //invalid color depth
	}

	if(InfoHeader._bm_compressed)
	{
		return 4; //invalid compression property
	}

	*Width  = InfoHeader._bm_image_width;
	*Height = InfoHeader._bm_image_height;

	fclose(fh);
	return 0;
}


/**
**************************************************************************
*  This function performs loading of bitmap luma
*
* \param FileName		[IN] - Image name to load
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
* \param Img			[OUT] - Prepared buffer
*
* \return None
*/

void LoadBmpAsGray(char *FileName, ROI *ImSize, byte *Img)
{
	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	//char *pData;
	FILE *fh;
	int Stride;
	fh = fopen(FileName, "rb");
	if (fh == NULL) {
		printf("Could not open file %s\r\n", FileName);
		ImSize->height = 0;
		ImSize->width = 0;
		return;
	}

	//printf("File %s header %zd, info header %zd\r\n", FileName, sizeof(BMPFileHeader), sizeof(BMPInfoHeader));

	fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);
	fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);
	/*
	printf("Header: 0x%X, %d, %d, 0x%X \r\n",
		FileHeader._bm_signature,
		FileHeader._bm_file_size,
		FileHeader._bm_reserved,
		FileHeader._bm_bitmap_data);

	printf("Info: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d \r\n",
		InfoHeader._bm_bitmap_size,
		InfoHeader._bm_color_depth,
		InfoHeader._bm_compressed,
		InfoHeader._bm_hor_resolution,
		InfoHeader._bm_image_height,
		InfoHeader._bm_image_width,
		InfoHeader._bm_info_header_size,
		InfoHeader._bm_num_colors_used,
		InfoHeader._bm_num_important_colors,
		InfoHeader._bm_num_of_planes,
		InfoHeader._bm_ver_resolution);
	*/
	/*
	pData = (char *)&FileHeader;
	for (int i = 0; i < sizeof(BMPFileHeader); i++)
		printf("%02X ", pData[i]);
	pData = (char *)&InfoHeader;
	for (int i = 0; i < sizeof(BMPInfoHeader); i++)
		printf("%02X ", pData[i]);
	printf("\r\n");
	*/

	ImSize->height = InfoHeader._bm_image_height;
	ImSize->width = InfoHeader._bm_image_width;
	Stride = ImSize->width*3;

	//printf("Height %d, Width %d, Stride %d\r\n", ImSize->height, ImSize->width, Stride);

	for (int i=ImSize->height-1; i>=0; i--)
	{
		for (int j=0; j<ImSize->width; j++)
		{
			int k = j * 3;
			int r = 0, g = 0, b = 0;
			fread(&b, 1, 1, fh);
			fread(&g, 1, 1, fh);
			fread(&r, 1, 1, fh);
			Img[i*Stride + k] = b;
			Img[i*Stride + k+1] = g;
			Img[i*Stride + k+2] = r;

			/*
			int val = ( 313524*r + 615514*g + 119537*b + 524288) >> 20 ;
			Img[i*Stride+j] = (byte)clamp_0_255(val);
			*/
		}
	}

	fclose(fh);
	return;
}

/**
**************************************************************************
*  This function performs dumping of bitmap luma on HDD 
*
* \param FileName		[OUT] - Image name to dump to
* \param Img			[IN] - Image luma to dump
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
*
* \return None
*/
void DumpBmpShortAsGray(char *FileName, unsigned short *Img, ROI ImSize)
{
	FILE *fp = NULL;
	fp = fopen(FileName, "wb");
    if (fp == NULL) 
	{
    	return;
    }

	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	int Stride = ImSize.width;

	//init headers
	FileHeader._bm_signature = 0x4D42;
	FileHeader._bm_file_size = 54 + 2 * ImSize.width * ImSize.height;
	FileHeader._bm_reserved = 0;
	FileHeader._bm_bitmap_data = 0x36;

	InfoHeader._bm_bitmap_size = 0;
	InfoHeader._bm_color_depth = 16;
	InfoHeader._bm_compressed = 0;
	InfoHeader._bm_hor_resolution = 0;
	InfoHeader._bm_image_height = ImSize.height;
	InfoHeader._bm_image_width = ImSize.width;
	InfoHeader._bm_info_header_size = 40;
	InfoHeader._bm_num_colors_used = 0;
	InfoHeader._bm_num_important_colors = 0;
	InfoHeader._bm_num_of_planes = 1;
	InfoHeader._bm_ver_resolution = 0;

	//KBE?? fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
	//KBE?? fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

	printf("Height %d, Width %d, Stride %d, Depth %d\r\n", ImSize.height, ImSize.width, Stride, InfoHeader._bm_color_depth);

	for (int i = ImSize.height - 1; i>=0; i--)
	{
		for (int j=0; j<ImSize.width; j++)
		{
			fwrite(&(Img[i*Stride+j]), 2, 1, fp);
		}
	}

	fclose(fp);
}

/**
**************************************************************************
*  This function performs dumping of bitmap luma on HDD
*
* \param FileName		[OUT] - Image name to dump to
* \param Img			[IN] - Image luma to dump
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
*
* \return None
*/
void DumpBmpAsGray(char *FileName, byte *Img, ROI ImSize)
{
	FILE *fp = NULL;
	fp = fopen(FileName, "wb");
	if (fp == NULL)
	{
		return;
	}

	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	int Stride = ImSize.width * 3;

	//init headers
	FileHeader._bm_signature = 0x4D42;
	FileHeader._bm_file_size = 54 + 3 * ImSize.width * ImSize.height;
	FileHeader._bm_reserved = 0;
	FileHeader._bm_bitmap_data = 0x36;

	InfoHeader._bm_bitmap_size = 0;
	InfoHeader._bm_color_depth = 24;
	InfoHeader._bm_compressed = 0;
	InfoHeader._bm_hor_resolution = 0;
	InfoHeader._bm_image_height = ImSize.height;
	InfoHeader._bm_image_width = ImSize.width;
	InfoHeader._bm_info_header_size = 40;
	InfoHeader._bm_num_colors_used = 0;
	InfoHeader._bm_num_important_colors = 0;
	InfoHeader._bm_num_of_planes = 1;
	InfoHeader._bm_ver_resolution = 0;

	fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
	fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

	printf("Height %d, Width %d, Stride %d\r\n", ImSize.height, ImSize.width, Stride);

	for (int i = ImSize.height - 1; i >= 0; i--)
	{
		for (int j = 0; j<ImSize.width; j++)
		{
			int k = j * 3;
			fwrite(&(Img[i*Stride + k]), 1, 1, fp);
			fwrite(&(Img[i*Stride + k+1]), 1, 1, fp);
			fwrite(&(Img[i*Stride + k+2]), 1, 1, fp);
		}
	}

	fflush(fp);
	fclose(fp);
}


/**
**************************************************************************
*  This function performs dumping of 8x8 block from float plane
*
* \param PlaneF			[IN] - Image plane
* \param StrideF		[IN] - Image stride
* \param Fname			[OUT] - File name to dump to
*
* \return None
*/
void DumpBlockF(float *PlaneF, int StrideF, char *Fname)
{
	FILE *fp = fopen(Fname, "wb");
	for (int i=0; i<8; i++)
	{
		for (int j=0; j<8; j++)
		{
			fprintf(fp, "%.*f  ", 14, PlaneF[i*StrideF+j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


/**
**************************************************************************
*  This function performs dumping of 8x8 block from byte plane
*
* \param Plane			[IN] - Image plane
* \param Stride			[IN] - Image stride
* \param Fname			[OUT] - File name to dump to
*
* \return None
*/
void DumpBlock(byte *Plane, int Stride, char *Fname)
{
	FILE *fp = fopen(Fname, "wb");
	for (int i=0; i<8; i++)
	{
		for (int j=0; j<8; j++)
		{
			fprintf(fp, "%.3d  ", Plane[i*Stride+j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


/**
**************************************************************************
*  This function performs evaluation of Mean Square Error between two images
*
* \param Img1			[IN] - Image 1
* \param Img2			[IN] - Image 2
* \param Stride			[IN] - Image stride
* \param Size			[IN] - Image size
*
* \return Mean Square Error between images
*/
float CalculateMSE(byte *Img1, byte *Img2, int Stride, ROI Size)
{
	uint32 Acc = 0;
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			int TmpDiff = Img1[i*Stride+j] - Img2[i*Stride+j];
			TmpDiff	*= TmpDiff;
			Acc += TmpDiff;
		}
	}
	return ((float)Acc) / (Size.height * Size.width);
}


/**
**************************************************************************
*  This function performs evaluation of Peak Signal to Noise Ratio between 
*  two images
*
* \param Img1			[IN] - Image 1
* \param Img2			[IN] - Image 2
* \param Stride			[IN] - Image stride
* \param Size			[IN] - Image size
*
* \return Peak Signal to Noise Ratio between images
*/
float CalculatePSNR(byte *Img1, byte *Img2, int Stride, ROI Size)
{
	float MSE = CalculateMSE(Img1, Img2, Stride, Size);
	return (float)(10 * log10(255*255 / MSE));
}
