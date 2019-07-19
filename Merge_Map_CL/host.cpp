#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

#define MAP1_SIZE Size(2*1280,2*720)
#define MAP2_SIZE Size(1280,720)
/**************************读取map文件***************************/
/*****   return  mapdata    *****/
Mat  RradMapdata(string maplocate,Size matsize)
{
	ifstream ReadData(maplocate, ios::in | ios::binary);
	Mat mapdata = Mat(matsize, CV_32FC1);

	if (!ReadData.is_open()) {
		cerr << "read map error!" << endl;
		exit(0);
	}
	ReadData.read((char*)mapdata.data, sizeof(float) * matsize.height * matsize.width);
	ReadData.close();

	return mapdata;
}

int main()
{
	Mat Input_Img = imread("input.png"), Output_Img;
	Mat Mapx1 = RradMapdata("map/mapx1.txt", MAP1_SIZE);//定义remap映射mapx
	Mat Mapy1 = RradMapdata("map/mapy1.txt", MAP1_SIZE);//定义remap映射mapy	
	Mat Mapx2 = RradMapdata("map/mapx2.txt", MAP2_SIZE);//定义remap映射mapx
	Mat Mapy2 = RradMapdata("map/mapy2.txt", MAP2_SIZE);//定义remap映射mapy	
	Mat Mapx_Merge(MAP2_SIZE,CV_32FC1), Mapy_Merge(MAP2_SIZE, CV_32FC1);
	Mat Mapx1_pro(MAP1_SIZE, CV_32FC1), Mapy1_pro(MAP1_SIZE, CV_32FC1);
	
	for (int i = 0; i < MAP1_SIZE.height; i++) {
		for (int j = 0; j < MAP1_SIZE.width; j++) {
			Mapx1_pro.ptr<float>(i)[j] = round(Mapx1.ptr<float>(i)[j]);
			Mapy1_pro.ptr<float>(i)[j] = round(Mapy1.ptr<float>(i)[j]);
		}
	}
	
	for (int i = 0; i < MAP2_SIZE.height; i++) {
		for (int j = 0; j < MAP2_SIZE.width; j++) {
			float Mapx2data = Mapx2.ptr<float>(i)[j];
			float Mapy2data = Mapy2.ptr<float>(i)[j];
			int Mapx_down = floor(Mapx2data);
			int Mapx_up = Mapx_down+1;
			int Mapy_down = floor(Mapy2data);
			int Mapy_up = Mapy_down+1;

			float map_weight[4];
			map_weight[0] = (Mapx_up - Mapx2data)*(Mapy_up - Mapy2data);
			map_weight[1] = (Mapx_up - Mapx2data)*(Mapy2data - Mapy_down);
			map_weight[2] = (Mapx2data - Mapx_down)*(Mapy_up - Mapy2data);
			map_weight[3] = (Mapx2data - Mapx_down)*(Mapy2data - Mapy_down);

			Mapx_Merge.ptr<float>(i)[j] = map_weight[0] * Mapx1_pro.ptr<float>(Mapy_down)[Mapx_down]\
										+ map_weight[1] * Mapx1_pro.ptr<float>(Mapy_up)[Mapx_down]\
										+ map_weight[2] * Mapx1_pro.ptr<float>(Mapy_down)[Mapx_up]\
										+ map_weight[3] * Mapx1_pro.ptr<float>(Mapy_up)[Mapx_up];

			Mapy_Merge.ptr<float>(i)[j] = map_weight[0] * Mapy1_pro.ptr<float>(Mapy_down)[Mapx_down]\
										+ map_weight[1] * Mapy1_pro.ptr<float>(Mapy_up)[Mapx_down]\
										+ map_weight[2] * Mapy1_pro.ptr<float>(Mapy_down)[Mapx_up]\
										+ map_weight[3] * Mapy1_pro.ptr<float>(Mapy_up)[Mapx_up];

		}
	}
	
	//Mat mat11,mat22;
	//remap(Input_Img, mat11, Mapx1,Mapy1,0);
	//remap(mat11, mat22, Mapx2, Mapy2, 1);
	//imshow("mat22", mat22);
	
	remap(Input_Img, Output_Img, Mapx_Merge, Mapy_Merge, 1);
	imshow("output", Output_Img);
	waitKey(0);
	getchar();
	return 0;
}



