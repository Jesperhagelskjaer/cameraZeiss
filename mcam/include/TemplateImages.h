#pragma once

#define ROWS 300 // Hight of image
#define COLS 500 // Width of image

class CamImage
{
public:
	CamImage() {};

	void CopyImage(unsigned short *pImage, int height, int width, RECT rec)
	{
		int id, jd;
		id = 0;
		for (int i = 0; i < height; i++) {
			jd = 0;
			for (int j = 0; j < width; j++)
				if ((rec.top <= i && i < rec.bottom) &&
					(rec.left <= j && j < rec.right))
					data[id][jd++] = pImage[i*width + j];
			if (rec.top <= i && i < rec.bottom)
				id++;
		}
	};

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
