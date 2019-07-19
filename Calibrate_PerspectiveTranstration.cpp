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


#define corner_points_rownum   20 //9    //标定图片内角点行数
#define corner_points_colnum    4 //6    //标定图片内角点列数

Size pattern_size = Size(corner_points_rownum, corner_points_colnum);//标定板上每行、每列的角点数；测试图片中的标定板上内角点数为9*6

#define FILE_PATH "imagepath.txt" //图片路径文件
#define IMAGE_PATH "input.png"   //图片存储路径


bool Calibrate(string Calibrate_ImgFilePath,
	string Transform_ImgPath,
	Matx33d& CameraMatrix,
	Vec4d& DistCoefficients,
	Mat&Trans_Matrix,
	Mat& mapx,
	Mat& mapy);

bool Get_Trans_Matrix(
	Mat SrcImage,
	Matx33d CameraMatrix,
	Vec4d DistCoefficients,
	Mat mapx,
	Mat mapy,
	Mat &Trans_Matrix);

Mat PerspectiveTransform(
	Mat Perspective_SrcImage,
	Mat mapx,
	Mat mapy,
	Mat Trans_Matrix);

/****************      代码功能      *********************/
/*    对摄像头进行标定，输出内参，MAP表，并俯视变换      */
/*********************************************************/

int main()
{
	Matx33d cameramatrix;
	Vec4d distcoefficients;
	Mat trans_matrix;
	Mat cal_mapx, cal_mapy;


	if (!Calibrate(FILE_PATH, IMAGE_PATH, cameramatrix, distcoefficients, trans_matrix, cal_mapx, cal_mapy)) {
		cout << "标定成功" << endl;
	}
	else {
		cout << "标定失败" << endl;
	}

	if (!Get_Trans_Matrix(imread(IMAGE_PATH), cameramatrix, distcoefficients, cal_mapx, cal_mapy, trans_matrix)) {
		cout << "use map  to  mapping corne" << endl;
	}
	else {
		cout << "direct  to  find  corner" << endl;
	}

	Mat Transformed_Img = PerspectiveTransform(imread(IMAGE_PATH), cal_mapx, cal_mapy, trans_matrix);

	waitKey(0);
	return 0;


}



/***********************************************************************/
/************************  标定&&透视变换函数  *************************/
/***********************************************************************/
	/*Calibrate_ImgFilePath：标定图片存放指引文件位置*/
	/*Transform_ImgPath：透视变换图片存放路径*/
	/*CameraMatrix：相机内参矩阵*/
	/*DistCoefficients：摄像机的5个畸变系数：p1,p2,,k1,k2,k3*/
	/*Trans_Matrix：透视变换矩阵*/
bool Calibrate(string Calibrate_ImgFilePath,
	string Transform_ImgPath,
	Matx33d& CameraMatrix,
	Vec4d& DistCoefficients,
	Mat& Trans_Matrix,
	Mat& mapx,
	Mat& mapy)
{
	/***********************文件读取************************/
	ifstream inImgPath(Calibrate_ImgFilePath);    //标定所用图像文件的路径
	vector<string> imgList;
	vector<string>::iterator p;
	string temp;
	if (!inImgPath.is_open()) {
		cout << "没有找到文件" << endl;
	}
	//读取文件中保存的图片文件路径，并存放在数组中
	while (getline(inImgPath, temp)) {
		imgList.push_back(temp);
	}
	ofstream fout("caliberation_result.txt");   //保存标定结果的文件


	/***********************开始提取角点************************/
	cout << "开始提取角点......" << endl;
	cv::Size image_size;//保存图片大小
	cv::Size pattern_size = cv::Size(corner_points_rownum, corner_points_colnum);//标定板上每行、每列的角点数；测试图片中的标定板上内角点数为9*6
	vector<cv::Point2f> corner_points_buf;//建一个数组缓存检测到的角点，通常采用Point2f形式
	vector<cv::Point2f>::iterator corner_points_buf_ptr;
	vector<vector<cv::Point2f>> corner_points_of_all_imgs;
	int image_num = 0;
	string filename;
	cout << "image_num=" << imgList.size() << endl;
	while (image_num < imgList.size()) {
		filename = imgList[image_num++];
		cout << "image_num = " << image_num << endl;
		cout << filename.c_str() << endl;
		cv::Mat imageInput = cv::imread(filename.c_str());
		if (image_num == 1) {
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "image_size.width = " << image_size.width << endl;
			cout << "image_size.height = " << image_size.height << endl;
		}

		if (findChessboardCorners(imageInput, pattern_size, corner_points_buf) == 0) {
			cout << "can not find chessboard corners!\n";   //找不到角点
			exit(1);
		}
		else {
			cv::Mat gray;
			cv::cvtColor(imageInput, gray, CV_RGB2GRAY);
			cv::find4QuadCornerSubpix(gray, corner_points_buf, cv::Size(5, 5));//亚像素精确化
			corner_points_of_all_imgs.push_back(corner_points_buf);
			cv::drawChessboardCorners(gray, pattern_size, corner_points_buf, true);
			//cv::imshow("camera calibration", gray);
			cv::waitKey(100);
		}
	}

	int total = corner_points_of_all_imgs.size();
	cout << "total=" << total << endl;
	int cornerNum = pattern_size.width * pattern_size.height;//每张图片上的总的角点数
	for (int i = 0; i < total; i++) {
		cout << "--> 第" << i + 1 << "幅图片的数据 -->:" << endl;
		for (int j = 0; j < cornerNum; j++) {
			cout << "-->" << corner_points_of_all_imgs[i][j].x;
			cout << "-->" << corner_points_of_all_imgs[i][j].y;
			if ((j + 1) % 3 == 0) {
				cout << endl;
			}
			else {
				cout.width(10);
			}
		}
		cout << endl;
	}
	cout << endl << "角点提取完成" << endl;


	/***********************摄像机标定************************/
	cout << "开始标定………………" << endl;
	//Matx33d cameraMatrix;//内参矩阵，H——单应性矩阵
	//cv::Vec4d distCoefficients; // 摄像机的5个畸变系数：p1,p2,,k1,k2,k3
	std::vector<cv::Vec3d> tvecsMat; //每幅图像的平移向量 t
	std::vector<cv::Vec3d> rvecsMat; //每幅图像的旋转向量 R
	vector<vector<cv::Point3f>> objectPoints;//保存所有图片的角点的三维坐标，初始化每一张图片中标定板上角点的三维坐标
	int flags = 0;//标志位
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	int i, j, k;
	//遍历每一张图片
	for (k = 0; k < image_num; k++) {
		vector<cv::Point3f> tempCornerPoints;//每一幅图片对应的角点数组
		//遍历所有的角点
		for (i = 0; i < pattern_size.height; i++) {
			for (j = 0; j < pattern_size.width; j++) {
				cv::Point3f singleRealPoint;//一个角点的坐标
				singleRealPoint.x = i * 10;
				singleRealPoint.y = j * 10;
				singleRealPoint.z = 0;//假设z=0
				tempCornerPoints.push_back(singleRealPoint);
			}
		}
		objectPoints.push_back(tempCornerPoints);
	}
	cv::fisheye::calibrate(objectPoints, corner_points_of_all_imgs, image_size, CameraMatrix, DistCoefficients, rvecsMat, tvecsMat, flags, cv::TermCriteria(3, 20, 1e-6));//函数加入fisheye::为鱼眼专用   |calibrateCameraca为普通摄像头
	cout << "标定完成" << endl;


	/***********************开始保存标定结果************************/
	cout << "开始保存标定结果" << endl;
	cout << endl << "相机相关参数：" << endl;
	fout << "相机相关参数：" << endl;
	cout << "1.内外参数矩阵:" << endl;
	fout << "1.内外参数矩阵:" << endl;
	cout << CameraMatrix << endl;
	fout << CameraMatrix << endl;
	cout << "2.畸变系数：" << endl;
	fout << "2.畸变系数：" << endl;
	cout << DistCoefficients << endl;
	fout << DistCoefficients << endl;
	cout << endl << "图像相关参数：" << endl;
	fout << endl << "图像相关参数：" << endl;
	cv::Mat rotation_Matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));//旋转矩阵
	for (i = 0; i < image_num; i++) {
		cout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		cout << rvecsMat[i] << endl;
		fout << rvecsMat[i] << endl;
		cout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		cv::Rodrigues(rvecsMat[i], rotation_Matrix);//将旋转向量转换为相对应的旋转矩阵
		cout << rotation_Matrix << endl;
		fout << rotation_Matrix << endl;
		cout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		cout << tvecsMat[i] << endl;
		fout << tvecsMat[i] << endl;
	}
	cout << "结果保存完毕" << endl;


	/***********************对标定结果进行评价************************/
	cout << "开始评价标定结果......" << endl;
	//计算每幅图像中的角点数量，假设全部角点都检测到了
	int corner_points_counts;
	corner_points_counts = pattern_size.width * pattern_size.height;
	cout << "每幅图像的标定误差：" << endl;
	fout << "每幅图像的标定误差：" << endl;
	double err = 0;//单张图像的误差
	double total_err = 0;//所有图像的平均误差
	for (i = 0; i < image_num; i++) {
		vector<cv::Point2f> image_points_calculated;//存放新计算出的投影点的坐标
		vector<cv::Point3f> tempPointSet = objectPoints[i];
		cv::projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], CameraMatrix, DistCoefficients, image_points_calculated);

		//计算新的投影点与旧的投影点之间的误差
		vector<cv::Point2f> image_points_old = corner_points_of_all_imgs[i];
		//将两组数据换成Mat格式
		cv::Mat image_points_calculated_mat = cv::Mat(1, image_points_calculated.size(), CV_32FC2);
		cv::Mat image_points_old_mat = cv::Mat(1, image_points_old.size(), CV_32FC2);
		for (j = 0; j < tempPointSet.size(); j++) {
			image_points_calculated_mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points_calculated[j].x, image_points_calculated[j].y);
			image_points_old_mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points_old[j].x, image_points_old[j].y);
		}
		err = cv::norm(image_points_calculated_mat, image_points_old_mat, cv::NORM_L2);
		err /= corner_points_counts;
		total_err += err;
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_num << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_num << "像素" << endl;
	cout << "评价完成" << endl;

	fout.close();
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
	string imageFileName, mapxFilename, mapyFilename;

	std::stringstream StrStm;


	/***********************新相机内参变换************************/
	float Scaling_coefficient = 2;//矫正图像方法倍数
	float coefficient = 0.4;   //0-1
	Mat NewcameraMatrix = (Mat)CameraMatrix;
	NewcameraMatrix.at<double>(0, 0) *= coefficient;//相机等效焦距fx
	NewcameraMatrix.at<double>(1, 1) *= coefficient;//相机等效焦距fy
	NewcameraMatrix.at<double>(0, 2) = 0.5 * 1280* Scaling_coefficient;//主点坐标cx
	NewcameraMatrix.at<double>(1, 2) = 0.5 * 720 * Scaling_coefficient; //主点坐标cy


	/***********************去畸变************************/
	for (int i = 0; i < image_num; i++) {
		cout << "Frame #" << i + 1 << endl;
		fisheye::initUndistortRectifyMap(CameraMatrix, DistCoefficients, R, NewcameraMatrix, Size(Scaling_coefficient *1280, Scaling_coefficient *720), CV_32FC1, mapx, mapy);//计算无畸变和修正转换映射,函数加入fisheye::为鱼眼专用
		Mat src_image = imread(imgList[i].c_str(), 1);
		Mat new_image = src_image.clone();
		remap(src_image, new_image, mapx, mapy, 0);//重映射
		//imshow("原始图像", src_image);
		imshow("矫正后图像", new_image);
		
		//save
		StrStm.clear();
		imageFileName.clear();
		mapxFilename.clear();
		mapyFilename.clear();

		StrStm << i + 1;
		StrStm >> imageFileName;
		StrStm.clear();
		StrStm << i + 1;
		StrStm >> mapxFilename;
		StrStm.clear();
		StrStm << i + 1;
		StrStm >> mapyFilename;
		imageFileName += "_d.jpg";
		mapxFilename += "_mapx.txt";
		mapyFilename += "_mapy.txt";
		imwrite(imageFileName, new_image);

		//写入map.txt
		cout << "map name  " << mapxFilename << endl;
		ofstream outputmapx(mapxFilename, ios::out | ios::binary);
		outputmapx.write((char*)mapx.data, sizeof(float)*mapx.cols*mapx.rows);
		ofstream outputmapy(mapyFilename, ios::out | ios::binary);
		outputmapy.write((char*)mapy.data, sizeof(float)* mapy.cols* mapy.rows);

		waitKey(20);
	}
	cout << "保存结束" << endl;

	return 0;

}

/***********************获取透视变换Matrix************************/
/*****   return 0 : use map  to  mapping corner    *****/
/*****   return 1 : direct  to  find  corner       *****/
bool Get_Trans_Matrix(
	Mat SrcImage,
	Matx33d CameraMatrix,
	Vec4d DistCoefficients,
	Mat mapx,
	Mat mapy,
	Mat &Trans_Matrix)
{
	/***********************提取角点************************/

	Mat RemapImage; //重映射后的图
	remap(SrcImage, RemapImage, mapx, mapy, 1);//重映射
	vector<vector<Point2f>> corner_points_of_all_imgs_Per;
	vector<Point2f>  corner_points_buf_Per;//角点

	//remap后的图中寻找角点
	if (findChessboardCorners(RemapImage, pattern_size, corner_points_buf_Per) == 0 ) {
		cout << "can not find chessboard corners! (PerspectiveTransform)\n";   //找不到角点
	/***********************原图角点映射在标定后的图片中的位置************************/
		/*****选择（0,0）(0,19)(3,0)(3,19)即（0，19，60，79）内角点**********/
		/*****remap()函数中，校正后图片中point(j,i）中像素为原图point（mapx(j,i)，mapy(j,i)）对应值，但mapx,mapy为float型，所以需要方法处理**********/
		/*****选择阈值法寻找map表，精度不高，可使用最邻近搜索发提高精度**********/
		float remap_threshold = 0.9;//阈值
		int  corner_points_afterremap[2][4];
		vector<cv::Point2f> corner_points_buf_bef;//建一个数组缓存检测到的角点，通常采用Point2f形式
		vector<vector<cv::Point2f>> corner_points_of_all_imgs_bef;

		//srcimg中寻找角点
		findChessboardCorners(SrcImage, pattern_size, corner_points_buf_bef);
		Mat Src_gray;
		cvtColor(SrcImage, Src_gray, CV_RGB2GRAY);
		find4QuadCornerSubpix(Src_gray, corner_points_buf_bef, cv::Size(5, 5));//亚像素精确化
		corner_points_of_all_imgs_bef.push_back(corner_points_buf_bef);
		waitKey(100);

		//map表匹配
		for (int j = 0; j < mapx.rows; j++) {
			for (int i = 0; i < mapx.cols; i++) {
				if ((abs(mapx.at<float>(j, i) - corner_points_of_all_imgs_bef[0][0].x) < remap_threshold) && (abs(mapy.at<float>(j, i) - corner_points_of_all_imgs_bef[0][0].y) < remap_threshold)) {
					corner_points_afterremap[0][0] = j;
					corner_points_afterremap[1][0] = i;
				}
				else if ((abs(mapx.at<float>(j, i) - corner_points_of_all_imgs_bef[0][19].x) < remap_threshold) && (abs(mapy.at<float>(j, i) - corner_points_of_all_imgs_bef[0][19].y) < remap_threshold)) {
					corner_points_afterremap[0][1] = j;
					corner_points_afterremap[1][1] = i;
				}
				else if ((abs(mapx.at<float>(j, i) - corner_points_of_all_imgs_bef[0][60].x) < remap_threshold - 0.4) && (abs(mapy.at<float>(j, i) - corner_points_of_all_imgs_bef[0][60].y) < remap_threshold - 0.4)) {
					corner_points_afterremap[0][2] = j;
					corner_points_afterremap[1][2] = i;
				}
				else if ((abs(mapx.at<float>(j, i) - corner_points_of_all_imgs_bef[0][79].x) < remap_threshold - 0.4) && (abs(mapy.at<float>(j, i) - corner_points_of_all_imgs_bef[0][79].y) < remap_threshold - 0.4)) {
					corner_points_afterremap[0][3] = j;
					corner_points_afterremap[1][3] = i;
				}
			}
		}
		/***********************透视变换************************/
		//透视变换原图四点坐标
		Point2f src_points[] = {
			Point2f(corner_points_afterremap[1][0], corner_points_afterremap[0][0]),
			Point2f(corner_points_afterremap[1][1], corner_points_afterremap[0][1]),
			Point2f(corner_points_afterremap[1][2], corner_points_afterremap[0][2]),
			Point2f(corner_points_afterremap[1][3], corner_points_afterremap[0][3])
		};
		//透视变换变换后四点坐标
		Point2f dst_points[] = {
			Point2f(240 + 190 * 4 ,30 * 4 + 60) , //4
			Point2f(240,30 * 4 + 60),//1
			Point2f(240 + 190 * 4 ,60),//2
			Point2f(240,60)//3
		};
		Trans_Matrix = getPerspectiveTransform(src_points, dst_points);//获取透视变换矩阵
		return 0;

	}
	else {
		cvtColor(RemapImage, RemapImage, CV_RGB2GRAY);
		find4QuadCornerSubpix(RemapImage, corner_points_buf_Per, Size(5, 5));//亚像素精确化-利用梯度精确优化角点 也可用Cornerssubpix
		corner_points_of_all_imgs_Per.push_back(corner_points_buf_Per);
		drawChessboardCorners(RemapImage, pattern_size, corner_points_buf_Per, true);//画出棋盘格角点
		waitKey(100);
		/***********************透视变换************************/
		//透视变换原图四点坐标
		Point2f src_points[] = {
			Point2f(corner_points_of_all_imgs_Per[0][0].x, corner_points_of_all_imgs_Per[0][0].y),
			Point2f(corner_points_of_all_imgs_Per[0][19].x, corner_points_of_all_imgs_Per[0][19].y),
			Point2f(corner_points_of_all_imgs_Per[0][60].x, corner_points_of_all_imgs_Per[0][60].y),
			Point2f(corner_points_of_all_imgs_Per[0][79].x, corner_points_of_all_imgs_Per[0][79].y)
		};
		//透视变换变换后四点坐标
		Point2f dst_points[] = {
			Point2f(240+190 * 4 ,30 * 4 + 60) , //4
			Point2f(240,30 * 4 + 60),//1
			Point2f(240 +190 * 4 ,60),//2
			Point2f(240,60)//3
		};

		//Point2f dst_points[] = {
		//	Point2f(190 * 4 + 200,30 * 4 + 10) ,
		//	Point2f(200,30 * 4 + 10),
		//	Point2f(190 * 4 + 200,10),
		//	Point2f(200,10)
		//};
		Trans_Matrix = getPerspectiveTransform(src_points, dst_points);//获取透视变换矩阵

		Mat mapx_2(SrcImage.rows, SrcImage.cols,CV_32FC1), mapy_2(SrcImage.rows, SrcImage.cols, CV_32FC1);
		Mat Trans_Matrix_Inv = Trans_Matrix.inv();
		for (size_t i = 0; i < SrcImage.rows; i++)
		{
			for (size_t j = 0; j < SrcImage.cols; j++)
			{
				double a11 = Trans_Matrix_Inv.at<double>(0, 0);
				double a12 = Trans_Matrix_Inv.at<double>(0, 1);
				double a13 = Trans_Matrix_Inv.at<double>(0, 2);
				double a21 = Trans_Matrix_Inv.at<double>(1, 0);
				double a22 = Trans_Matrix_Inv.at<double>(1, 1);
				double a23 = Trans_Matrix_Inv.at<double>(1, 2);
				double a31 = Trans_Matrix_Inv.at<double>(2, 0);
				double a32 = Trans_Matrix_Inv.at<double>(2, 1);
				double a33 = Trans_Matrix_Inv.at<double>(2, 2);
				float _X = a11 * j + a12 * i + a13;
				float _W = a31 * j + a32 * i + a33;
				float _Y = a21 * j + a22 * i + a23;
				mapx_2.at<float>(i, j) = _X / _W;
				mapy_2.at<float>(i, j) = _Y / _W;
			}
		}
		//写入map2.txt
		cout << "write mapx2  mapy2  "<< endl;
		ofstream outputmapx("mapx2.txt", ios::out | ios::binary);
		outputmapx.write((char*)mapx_2.data, sizeof(float)*mapx_2.cols*mapx_2.rows);
		ofstream outputmapy("mapy2.txt", ios::out | ios::binary);
		outputmapy.write((char*)mapy_2.data, sizeof(float)* mapy_2.cols* mapy_2.rows);

		return 1;
	}

}



//透视变换
Mat PerspectiveTransform(
	Mat Perspective_SrcImage,
	Mat mapx,
	Mat mapy,
	Mat Trans_Matrix)
{
	Mat Perspective_DstImage; //透视变换图
	remap(Perspective_SrcImage, Perspective_SrcImage, mapx, mapy, 1);//重映射
	warpPerspective(Perspective_SrcImage, Perspective_DstImage, Trans_Matrix, Size(1280, 720));//透视变换

	imwrite("透视变换图.jpg", Perspective_DstImage);
	imshow("透视变换结果图", Perspective_DstImage);
	cout << "透视变换结束" << endl;

	return Perspective_DstImage;

}