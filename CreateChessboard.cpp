#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include "opencv2/opencv.hpp"
#define PI 3.14159265358979323846
using namespace std;
using namespace cv;
int img_width = 22000;
int img_height = 22000;
int gray = 0;
int Cb_rows = 7, Cb_cols = 7;
int square_size = 2500;  //mm
int main()
{
	int Cb_size = square_size;
	Mat img_final = Mat::zeros(img_height, img_width, CV_8UC1);
	int org_X = (img_height - Cb_rows * Cb_size) / 2;
	int org_Y = (img_width - Cb_cols * Cb_size) / 2;
	Mat img = Mat::zeros(Cb_rows*Cb_size, Cb_cols*Cb_size, CV_8UC1);
	img = 255, img_final = 255;
	int mark = 0;
	for (size_t i = 0; i < Cb_rows; i++)
	{
		int temp_mark = mark;
		for (size_t j = 0; j < Cb_cols; j++)
		{
			if (mark % 2 == 0)
			{
				img(Range(i*Cb_size, (i + 1)*Cb_size), Range(j*Cb_size, (j + 1)*Cb_size)) = gray;
			}
			mark++;
		}
		mark = temp_mark + 1;
	}
	img_final(Range(org_X, org_X + img.rows), Range(org_Y, org_Y + img.cols)) = img + 0;
	//	namedWindow("pattern", WINDOW_NORMAL);
//	imshow("pattern", img_final);
	imwrite("chess.bmp", img_final);

	waitKey(0);
	return 0;
}