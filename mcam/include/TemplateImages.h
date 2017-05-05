#pragma once

#define ROWS 300 // Hight of image
#define COLS 500 // Width of image

class CamImage
{
public:
	CamImage() {};

private:
	unsigned short data[ROWS][COLS];
};

class TemplateImages
{
public:
	TemplateImages() {};

private:
	std::vector<CamImage*> TemplateImgs_;
};
