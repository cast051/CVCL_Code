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


#define corner_points_rownum   20 //9    //�궨ͼƬ�ڽǵ�����
#define corner_points_colnum    4 //6    //�궨ͼƬ�ڽǵ�����

Size pattern_size = Size(corner_points_rownum, corner_points_colnum);//�궨����ÿ�С�ÿ�еĽǵ���������ͼƬ�еı궨�����ڽǵ���Ϊ9*6

#define FILE_PATH "imagepath.txt" //ͼƬ·���ļ�
#define IMAGE_PATH "input.png"   //ͼƬ�洢·��


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

/****************      ���빦��      *********************/
/*    ������ͷ���б궨������ڲΣ�MAP�������ӱ任      */
/*********************************************************/

int main()
{
	Matx33d cameramatrix;
	Vec4d distcoefficients;
	Mat trans_matrix;
	Mat cal_mapx, cal_mapy;


	if (!Calibrate(FILE_PATH, IMAGE_PATH, cameramatrix, distcoefficients, trans_matrix, cal_mapx, cal_mapy)) {
		cout << "�궨�ɹ�" << endl;
	}
	else {
		cout << "�궨ʧ��" << endl;
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
/************************  �궨&&͸�ӱ任����  *************************/
/***********************************************************************/
	/*Calibrate_ImgFilePath���궨ͼƬ���ָ���ļ�λ��*/
	/*Transform_ImgPath��͸�ӱ任ͼƬ���·��*/
	/*CameraMatrix������ڲξ���*/
	/*DistCoefficients���������5������ϵ����p1,p2,,k1,k2,k3*/
	/*Trans_Matrix��͸�ӱ任����*/
bool Calibrate(string Calibrate_ImgFilePath,
	string Transform_ImgPath,
	Matx33d& CameraMatrix,
	Vec4d& DistCoefficients,
	Mat& Trans_Matrix,
	Mat& mapx,
	Mat& mapy)
{
	/***********************�ļ���ȡ************************/
	ifstream inImgPath(Calibrate_ImgFilePath);    //�궨����ͼ���ļ���·��
	vector<string> imgList;
	vector<string>::iterator p;
	string temp;
	if (!inImgPath.is_open()) {
		cout << "û���ҵ��ļ�" << endl;
	}
	//��ȡ�ļ��б����ͼƬ�ļ�·�����������������
	while (getline(inImgPath, temp)) {
		imgList.push_back(temp);
	}
	ofstream fout("caliberation_result.txt");   //����궨������ļ�


	/***********************��ʼ��ȡ�ǵ�************************/
	cout << "��ʼ��ȡ�ǵ�......" << endl;
	cv::Size image_size;//����ͼƬ��С
	cv::Size pattern_size = cv::Size(corner_points_rownum, corner_points_colnum);//�궨����ÿ�С�ÿ�еĽǵ���������ͼƬ�еı궨�����ڽǵ���Ϊ9*6
	vector<cv::Point2f> corner_points_buf;//��һ�����黺���⵽�Ľǵ㣬ͨ������Point2f��ʽ
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
			cout << "can not find chessboard corners!\n";   //�Ҳ����ǵ�
			exit(1);
		}
		else {
			cv::Mat gray;
			cv::cvtColor(imageInput, gray, CV_RGB2GRAY);
			cv::find4QuadCornerSubpix(gray, corner_points_buf, cv::Size(5, 5));//�����ؾ�ȷ��
			corner_points_of_all_imgs.push_back(corner_points_buf);
			cv::drawChessboardCorners(gray, pattern_size, corner_points_buf, true);
			//cv::imshow("camera calibration", gray);
			cv::waitKey(100);
		}
	}

	int total = corner_points_of_all_imgs.size();
	cout << "total=" << total << endl;
	int cornerNum = pattern_size.width * pattern_size.height;//ÿ��ͼƬ�ϵ��ܵĽǵ���
	for (int i = 0; i < total; i++) {
		cout << "--> ��" << i + 1 << "��ͼƬ������ -->:" << endl;
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
	cout << endl << "�ǵ���ȡ���" << endl;


	/***********************������궨************************/
	cout << "��ʼ�궨������������" << endl;
	//Matx33d cameraMatrix;//�ڲξ���H������Ӧ�Ծ���
	//cv::Vec4d distCoefficients; // �������5������ϵ����p1,p2,,k1,k2,k3
	std::vector<cv::Vec3d> tvecsMat; //ÿ��ͼ���ƽ������ t
	std::vector<cv::Vec3d> rvecsMat; //ÿ��ͼ�����ת���� R
	vector<vector<cv::Point3f>> objectPoints;//��������ͼƬ�Ľǵ����ά���꣬��ʼ��ÿһ��ͼƬ�б궨���Ͻǵ����ά����
	int flags = 0;//��־λ
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	int i, j, k;
	//����ÿһ��ͼƬ
	for (k = 0; k < image_num; k++) {
		vector<cv::Point3f> tempCornerPoints;//ÿһ��ͼƬ��Ӧ�Ľǵ�����
		//�������еĽǵ�
		for (i = 0; i < pattern_size.height; i++) {
			for (j = 0; j < pattern_size.width; j++) {
				cv::Point3f singleRealPoint;//һ���ǵ������
				singleRealPoint.x = i * 10;
				singleRealPoint.y = j * 10;
				singleRealPoint.z = 0;//����z=0
				tempCornerPoints.push_back(singleRealPoint);
			}
		}
		objectPoints.push_back(tempCornerPoints);
	}
	cv::fisheye::calibrate(objectPoints, corner_points_of_all_imgs, image_size, CameraMatrix, DistCoefficients, rvecsMat, tvecsMat, flags, cv::TermCriteria(3, 20, 1e-6));//��������fisheye::Ϊ����ר��   |calibrateCameracaΪ��ͨ����ͷ
	cout << "�궨���" << endl;


	/***********************��ʼ����궨���************************/
	cout << "��ʼ����궨���" << endl;
	cout << endl << "�����ز�����" << endl;
	fout << "�����ز�����" << endl;
	cout << "1.�����������:" << endl;
	fout << "1.�����������:" << endl;
	cout << CameraMatrix << endl;
	fout << CameraMatrix << endl;
	cout << "2.����ϵ����" << endl;
	fout << "2.����ϵ����" << endl;
	cout << DistCoefficients << endl;
	fout << DistCoefficients << endl;
	cout << endl << "ͼ����ز�����" << endl;
	fout << endl << "ͼ����ز�����" << endl;
	cv::Mat rotation_Matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));//��ת����
	for (i = 0; i < image_num; i++) {
		cout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		cout << rvecsMat[i] << endl;
		fout << rvecsMat[i] << endl;
		cout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		cv::Rodrigues(rvecsMat[i], rotation_Matrix);//����ת����ת��Ϊ���Ӧ����ת����
		cout << rotation_Matrix << endl;
		fout << rotation_Matrix << endl;
		cout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		cout << tvecsMat[i] << endl;
		fout << tvecsMat[i] << endl;
	}
	cout << "����������" << endl;


	/***********************�Ա궨�����������************************/
	cout << "��ʼ���۱궨���......" << endl;
	//����ÿ��ͼ���еĽǵ�����������ȫ���ǵ㶼��⵽��
	int corner_points_counts;
	corner_points_counts = pattern_size.width * pattern_size.height;
	cout << "ÿ��ͼ��ı궨��" << endl;
	fout << "ÿ��ͼ��ı궨��" << endl;
	double err = 0;//����ͼ������
	double total_err = 0;//����ͼ���ƽ�����
	for (i = 0; i < image_num; i++) {
		vector<cv::Point2f> image_points_calculated;//����¼������ͶӰ�������
		vector<cv::Point3f> tempPointSet = objectPoints[i];
		cv::projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], CameraMatrix, DistCoefficients, image_points_calculated);

		//�����µ�ͶӰ����ɵ�ͶӰ��֮������
		vector<cv::Point2f> image_points_old = corner_points_of_all_imgs[i];
		//���������ݻ���Mat��ʽ
		cv::Mat image_points_calculated_mat = cv::Mat(1, image_points_calculated.size(), CV_32FC2);
		cv::Mat image_points_old_mat = cv::Mat(1, image_points_old.size(), CV_32FC2);
		for (j = 0; j < tempPointSet.size(); j++) {
			image_points_calculated_mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points_calculated[j].x, image_points_calculated[j].y);
			image_points_old_mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points_old[j].x, image_points_old[j].y);
		}
		err = cv::norm(image_points_calculated_mat, image_points_old_mat, cv::NORM_L2);
		err /= corner_points_counts;
		total_err += err;
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	cout << "����ƽ����" << total_err / image_num << "����" << endl;
	fout << "����ƽ����" << total_err / image_num << "����" << endl;
	cout << "�������" << endl;

	fout.close();
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	cout << "�������ͼ��" << endl;
	string imageFileName, mapxFilename, mapyFilename;

	std::stringstream StrStm;


	/***********************������ڲα任************************/
	float Scaling_coefficient = 2;//����ͼ�񷽷�����
	float coefficient = 0.4;   //0-1
	Mat NewcameraMatrix = (Mat)CameraMatrix;
	NewcameraMatrix.at<double>(0, 0) *= coefficient;//�����Ч����fx
	NewcameraMatrix.at<double>(1, 1) *= coefficient;//�����Ч����fy
	NewcameraMatrix.at<double>(0, 2) = 0.5 * 1280* Scaling_coefficient;//��������cx
	NewcameraMatrix.at<double>(1, 2) = 0.5 * 720 * Scaling_coefficient; //��������cy


	/***********************ȥ����************************/
	for (int i = 0; i < image_num; i++) {
		cout << "Frame #" << i + 1 << endl;
		fisheye::initUndistortRectifyMap(CameraMatrix, DistCoefficients, R, NewcameraMatrix, Size(Scaling_coefficient *1280, Scaling_coefficient *720), CV_32FC1, mapx, mapy);//�����޻��������ת��ӳ��,��������fisheye::Ϊ����ר��
		Mat src_image = imread(imgList[i].c_str(), 1);
		Mat new_image = src_image.clone();
		remap(src_image, new_image, mapx, mapy, 0);//��ӳ��
		//imshow("ԭʼͼ��", src_image);
		imshow("������ͼ��", new_image);
		
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

		//д��map.txt
		cout << "map name  " << mapxFilename << endl;
		ofstream outputmapx(mapxFilename, ios::out | ios::binary);
		outputmapx.write((char*)mapx.data, sizeof(float)*mapx.cols*mapx.rows);
		ofstream outputmapy(mapyFilename, ios::out | ios::binary);
		outputmapy.write((char*)mapy.data, sizeof(float)* mapy.cols* mapy.rows);

		waitKey(20);
	}
	cout << "�������" << endl;

	return 0;

}

/***********************��ȡ͸�ӱ任Matrix************************/
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
	/***********************��ȡ�ǵ�************************/

	Mat RemapImage; //��ӳ����ͼ
	remap(SrcImage, RemapImage, mapx, mapy, 1);//��ӳ��
	vector<vector<Point2f>> corner_points_of_all_imgs_Per;
	vector<Point2f>  corner_points_buf_Per;//�ǵ�

	//remap���ͼ��Ѱ�ҽǵ�
	if (findChessboardCorners(RemapImage, pattern_size, corner_points_buf_Per) == 0 ) {
		cout << "can not find chessboard corners! (PerspectiveTransform)\n";   //�Ҳ����ǵ�
	/***********************ԭͼ�ǵ�ӳ���ڱ궨���ͼƬ�е�λ��************************/
		/*****ѡ��0,0��(0,19)(3,0)(3,19)����0��19��60��79���ڽǵ�**********/
		/*****remap()�����У�У����ͼƬ��point(j,i��������Ϊԭͼpoint��mapx(j,i)��mapy(j,i)����Ӧֵ����mapx,mapyΪfloat�ͣ�������Ҫ��������**********/
		/*****ѡ����ֵ��Ѱ��map�����Ȳ��ߣ���ʹ�����ڽ���������߾���**********/
		float remap_threshold = 0.9;//��ֵ
		int  corner_points_afterremap[2][4];
		vector<cv::Point2f> corner_points_buf_bef;//��һ�����黺���⵽�Ľǵ㣬ͨ������Point2f��ʽ
		vector<vector<cv::Point2f>> corner_points_of_all_imgs_bef;

		//srcimg��Ѱ�ҽǵ�
		findChessboardCorners(SrcImage, pattern_size, corner_points_buf_bef);
		Mat Src_gray;
		cvtColor(SrcImage, Src_gray, CV_RGB2GRAY);
		find4QuadCornerSubpix(Src_gray, corner_points_buf_bef, cv::Size(5, 5));//�����ؾ�ȷ��
		corner_points_of_all_imgs_bef.push_back(corner_points_buf_bef);
		waitKey(100);

		//map��ƥ��
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
		/***********************͸�ӱ任************************/
		//͸�ӱ任ԭͼ�ĵ�����
		Point2f src_points[] = {
			Point2f(corner_points_afterremap[1][0], corner_points_afterremap[0][0]),
			Point2f(corner_points_afterremap[1][1], corner_points_afterremap[0][1]),
			Point2f(corner_points_afterremap[1][2], corner_points_afterremap[0][2]),
			Point2f(corner_points_afterremap[1][3], corner_points_afterremap[0][3])
		};
		//͸�ӱ任�任���ĵ�����
		Point2f dst_points[] = {
			Point2f(240 + 190 * 4 ,30 * 4 + 60) , //4
			Point2f(240,30 * 4 + 60),//1
			Point2f(240 + 190 * 4 ,60),//2
			Point2f(240,60)//3
		};
		Trans_Matrix = getPerspectiveTransform(src_points, dst_points);//��ȡ͸�ӱ任����
		return 0;

	}
	else {
		cvtColor(RemapImage, RemapImage, CV_RGB2GRAY);
		find4QuadCornerSubpix(RemapImage, corner_points_buf_Per, Size(5, 5));//�����ؾ�ȷ��-�����ݶȾ�ȷ�Ż��ǵ� Ҳ����Cornerssubpix
		corner_points_of_all_imgs_Per.push_back(corner_points_buf_Per);
		drawChessboardCorners(RemapImage, pattern_size, corner_points_buf_Per, true);//�������̸�ǵ�
		waitKey(100);
		/***********************͸�ӱ任************************/
		//͸�ӱ任ԭͼ�ĵ�����
		Point2f src_points[] = {
			Point2f(corner_points_of_all_imgs_Per[0][0].x, corner_points_of_all_imgs_Per[0][0].y),
			Point2f(corner_points_of_all_imgs_Per[0][19].x, corner_points_of_all_imgs_Per[0][19].y),
			Point2f(corner_points_of_all_imgs_Per[0][60].x, corner_points_of_all_imgs_Per[0][60].y),
			Point2f(corner_points_of_all_imgs_Per[0][79].x, corner_points_of_all_imgs_Per[0][79].y)
		};
		//͸�ӱ任�任���ĵ�����
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
		Trans_Matrix = getPerspectiveTransform(src_points, dst_points);//��ȡ͸�ӱ任����

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
		//д��map2.txt
		cout << "write mapx2  mapy2  "<< endl;
		ofstream outputmapx("mapx2.txt", ios::out | ios::binary);
		outputmapx.write((char*)mapx_2.data, sizeof(float)*mapx_2.cols*mapx_2.rows);
		ofstream outputmapy("mapy2.txt", ios::out | ios::binary);
		outputmapy.write((char*)mapy_2.data, sizeof(float)* mapy_2.cols* mapy_2.rows);

		return 1;
	}

}



//͸�ӱ任
Mat PerspectiveTransform(
	Mat Perspective_SrcImage,
	Mat mapx,
	Mat mapy,
	Mat Trans_Matrix)
{
	Mat Perspective_DstImage; //͸�ӱ任ͼ
	remap(Perspective_SrcImage, Perspective_SrcImage, mapx, mapy, 1);//��ӳ��
	warpPerspective(Perspective_SrcImage, Perspective_DstImage, Trans_Matrix, Size(1280, 720));//͸�ӱ任

	imwrite("͸�ӱ任ͼ.jpg", Perspective_DstImage);
	imshow("͸�ӱ任���ͼ", Perspective_DstImage);
	cout << "͸�ӱ任����" << endl;

	return Perspective_DstImage;

}