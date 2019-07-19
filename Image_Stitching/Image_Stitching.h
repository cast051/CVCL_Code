#ifndef _STITCHING_H_
#define _STITCHING_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"


#define DSTIMG_HEIGHT  1200
#define DSTIMG_LENGH   1500
#define IMG_SIZE  Size(1280, 720)
#define VIDEOPATH  "lxg.mkv"
#define IMAGEPATH  "P_Image/"


typedef enum detect_mod { SURF_MODE, SIFT_MODE, ORB_MODE ,FLOWFB_MODE,FLOWLK_MODE};


using namespace cv;
using namespace std;
using namespace xfeatures2d;


Size pattern_size = Size(20, 4);//标定板上每行、每列的角点数；测试图片中的标定板上内角点数
//相机内参
Matx33d CamMatrix = { 408.1083643498845, 0, 676.1162836989628,
						0, 402.6742899926278, 382.2402199913725,
						0, 0, 1 };
//相机畸变
Vec4d DistCoeff = { -0.0499652, 0.0072904, 0.000776633, -0.00278516 };

bool Get_Trans_Matrix(
	Mat SrcImage,
	Matx33d CameraMatrix,
	Vec4d DistCoefficients,
	Mat mapx,
	Mat mapy,
	Mat& Trans_Matrix);
Mat PerspectiveTransform(
	Mat Perspective_SrcImage,
	Mat mapx,
	Mat mapy,
	Mat Trans_Matrix);
Mat RradMapdata(string maplocate);
Mat Alpha_Blending(Mat img1, Mat img2, float moving_distance, Mat img);
Mat Read_VideoToImage(string video_path, const char image_path[]);
void My_warpAffine(Mat src, Mat& dst, Mat M);
Mat ORB_Feature(Mat img1, Mat img2, bool& flag);
Mat SIFT_Feature(Mat img1, Mat img2, bool& flag);
Mat SURF_Feature(Mat img1, Mat img2, bool& flag);
Mat OpticalFlow_FB(Mat img1, Mat img2, bool& flag);
Mat OpticalFlow_LK(Mat img1, Mat img2, bool& flag);
void Ransac_Transform(vector<Point2f> inputpoint1, vector<Point2f> inputpoint2, vector<Point2f>& outputpoint1, vector<Point2f>& outputpoint2, vector<int>& best_inner);
Mat M_ChangeCoordinate(Mat m, Point2f rotate_point);


#endif