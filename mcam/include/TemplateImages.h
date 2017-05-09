#pragma once

#define ROWS 300 // Hight of image
#define COLS 500 // Width of image

class CamImage
{
public:
	CamImage() : cost_(0) {};

	void CopyImage(unsigned short *pImage, int height, int width, RECT rec)
	{
		int id, jd;
		id = 0;
		for (int i = 0; i < height; i++) {
			jd = 0;
			for (int j = 0; j < width; j++)
				if ((rec.top <= i && i < rec.bottom) &&
					(rec.left <= j && j < rec.right))
					data_[id][jd++] = pImage[i*width + j];
			if (rec.top <= i && i < rec.bottom)
				id++;
		}
	};

	double ComputeIntencity(void)
	{
		cost_ = 0;
		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++)
				cost_ += data_[i][j];
		}
		return cost_;
	}

private:
	unsigned short data_[ROWS][COLS];
	double cost_;
};

class TemplateImages
{
public:
	TemplateImages() {};

private:
	std::vector<CamImage*> TemplateImgs_;
};
