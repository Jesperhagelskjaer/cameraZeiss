#pragma once

#define ROWS 500 // Hight of image section
#define COLS 500 // Width of image section

class CamImage
{
public:

	CamImage() : cost_(0) 
	{
		ClearData();
	};

	void CopyImage(unsigned short *pImage, int height, int width, RECT rec)
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
	};

	void Print(void) 
	{
		printf("CamImage::data_\r\n");
		printf("line1: %d %d %d %d %d %d %d\r\n", data_[0][0], data_[0][1], data_[0][2], data_[0][3], data_[0][4], data_[0][5], data_[0][COLS-1]);
		printf("line2: %d %d %d %d %d %d %d\r\n", data_[1][0], data_[1][1], data_[1][2], data_[1][3], data_[1][4], data_[1][5], data_[1][COLS-1]);
		printf("line3: %d %d %d %d %d %d %d\r\n", data_[2][0], data_[2][1], data_[2][2], data_[2][3], data_[2][4], data_[2][5], data_[2][COLS-1]);
		printf("line4: %d %d %d %d %d %d %d\r\n", data_[3][0], data_[3][1], data_[3][2], data_[3][3], data_[3][4], data_[3][5], data_[3][COLS-1]);
		printf("line5: %d %d %d %d %d %d %d\r\n", data_[4][0], data_[4][1], data_[4][2], data_[4][3], data_[4][4], data_[4][5], data_[4][COLS-1]);
		printf("line6: %d %d %d %d %d %d %d\r\n", data_[5][0], data_[5][1], data_[5][2], data_[5][3], data_[5][4], data_[5][5], data_[5][COLS-1]);
		printf("line499: %d %d %d %d %d %d %d\r\n", data_[ROWS-1][0], data_[ROWS-1][1], data_[ROWS-1][2], data_[ROWS-1][3], 
												    data_[ROWS-1][4], data_[ROWS-1][5], data_[ROWS-1][COLS - 1]);

	};

private:
	unsigned short data_[ROWS][COLS];
	double cost_;

	void ClearData(void) 
	{
		for (int i = 0; i < ROWS; i++) 
			for (int j = 0; j < COLS; j++)
				data_[i][j] = 0;
	};
};

class TemplateImages
{
public:
	TemplateImages() {};

private:
	std::vector<CamImage*> TemplateImgs_;
};
