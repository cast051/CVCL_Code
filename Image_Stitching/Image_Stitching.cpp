#include "Image_Stitching.h"


detect_mod Detect_Mod = FLOWLK_MODE;//特征提取模式




/**********************************************************/
/**********************   主函数   ************************/
/**********************************************************/
int main(int argc, char** argv)
{
	vector <Mat> srcimage;//输入原图
	//读取视频帧
	srcimage.push_back(Read_VideoToImage(VIDEOPATH, IMAGEPATH));
	//保存到视频
	VideoWriter avi_writer("ORB.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(DSTIMG_LENGH, DSTIMG_HEIGHT), true);
	//caclute PerspectiveTrans Matrix
	Mat PerspectiveTrans_Matrix;		 //透视变换矩阵
	Mat Mapx = RradMapdata("1_mapx.txt");//定义remap映射mapx
	Mat Mapy = RradMapdata("1_mapy.txt");//定义remap映射mapy			

	if (!Get_Trans_Matrix(imread("image1.png"), CamMatrix, DistCoeff, Mapx, Mapy, PerspectiveTrans_Matrix)) 
		cout << "use map to find corner" << endl;
	else
		cout << "firect to find corner " << endl;

	//图片拼接
	Mat Current_Img, Previous_Img;
	Mat Stitching_PreImg(DSTIMG_HEIGHT, DSTIMG_LENGH, CV_8UC3);//上张拼接图
	Mat Stitching_CurImg(DSTIMG_HEIGHT, DSTIMG_LENGH, CV_8UC3);//当前输出图
	for (int i = 0; i < srcimage.size()+1 ;i++)
	{
		if (i == 0){
			//透视变换
			Previous_Img= PerspectiveTransform(srcimage[0], Mapx, Mapy, PerspectiveTrans_Matrix);
			//将上一张图放入上一张拼接图中
			Stitching_PreImg.setTo(0);
			Previous_Img.copyTo(Stitching_PreImg(Rect((Stitching_PreImg.cols-Previous_Img.cols)/2, Stitching_PreImg.rows- Previous_Img.rows, Previous_Img.cols, Previous_Img.rows)));
		}
		else {
			//读取视频帧
			srcimage.push_back(Read_VideoToImage(VIDEOPATH, IMAGEPATH));
			//透视变换
			Current_Img = PerspectiveTransform(srcimage[i], Mapx, Mapy, PerspectiveTrans_Matrix);
			//当前拼接图片置位
			Stitching_CurImg.setTo(0);
			//将当前图放入当前拼接图中
			Current_Img.copyTo(Stitching_CurImg(Rect((Stitching_CurImg.cols - Current_Img.cols) / 2, Stitching_CurImg.rows - Current_Img.rows, Current_Img.cols, Current_Img.rows)));
			//获取放射变换矩阵
			Mat  matrix = Mat::zeros(Size(3, 2), CV_64FC1);
			bool matrix_flag = 0;
			
			//OpticalFlow_LK(Previous_Img, Current_Img, matrix_flag);
			//选择特征提取模式
			switch (Detect_Mod) {
			case SURF_MODE: //SURF特征点提取仿射矩阵
				matrix =SURF_Feature(Previous_Img, Current_Img, matrix_flag);
				break;
			case SIFT_MODE: //SIFT特征点提取仿射矩阵
				matrix =SIFT_Feature(Previous_Img, Current_Img, matrix_flag);
				break;
			case ORB_MODE:  //ORB特征点提取仿射矩阵
				matrix =ORB_Feature(Previous_Img, Current_Img, matrix_flag);
				break;
			case FLOWFB_MODE:
				matrix = OpticalFlow_FB(Previous_Img, Current_Img, matrix_flag);
				break;
			case FLOWLK_MODE:
				matrix = OpticalFlow_LK(Previous_Img, Current_Img, matrix_flag);				
				break;
			}
			//如果标志位置1，matrix获取数据失败，则直接开始处理下一帧图片
			if (matrix_flag) {
				cout << "flag=1，获取M矩阵失败" << endl;
			}
			//提取垂直位移作为alpha blending distance的计算值
			double Mov_Distance = matrix.ptr<double>(1)[2];
			//my仿射变换
			Mat Stitching_PreImg_warp(Stitching_PreImg.size(), Stitching_PreImg.type());			
			My_warpAffine(Stitching_PreImg, Stitching_PreImg_warp, matrix);
			Stitching_PreImg_warp.copyTo(Stitching_PreImg);
			//仿射变换
			//warpAffine(Stitching_PreImg, Stitching_PreImg, matrix, Stitching_PreImg.size(), 1);

			//Alpha Blending 融合图片
			Stitching_PreImg = Alpha_Blending(Stitching_CurImg, Stitching_PreImg, Mov_Distance, Current_Img).clone();
			Previous_Img = Current_Img.clone();
			imshow("拼接图", Stitching_PreImg);
			//保存avi视频
			avi_writer.write(Stitching_PreImg);
			//imwrite("P_Image/Stitching_Image.jpg", Stitching_PreImg);
			waitKey(10);
		}
		
	}
	waitKey(0);
	return 0;

}


/*******************   仿射变换   *********************/
void My_warpAffine(Mat src, Mat& dst, Mat M) {

	double M_arr[3][3] = { {M.ptr<double>(0)[0],M.ptr<double>(0)[1],M.ptr<double>(0)[2]},
						   {M.ptr<double>(1)[0],M.ptr<double>(1)[1],M.ptr<double>(1)[2]},
						   {0,0,1} };
	Mat M_33(3, 3, CV_64FC1, M_arr);//齐次后的3*3的M矩阵
	Mat M_33_invert;//M矩阵的逆矩阵
	invert(M_33, M_33_invert);//求逆矩阵
	//dst.setTo(0);
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			double dst_xy_arr[3][1] = { j,i,1 };
			Mat dst_xy(3, 1, CV_64FC1, dst_xy_arr);
			Mat src_xy = M_33_invert * dst_xy;
			/*********  双线性插值  *********/
			/*   相邻像素坐标   */
			double src_x = src_xy.ptr<double>(1)[0];
			double src_y = src_xy.ptr<double>(0)[0];
			int src_x_front = floor(src_x);
			int src_x_after = src_x_front + 1;
			int src_y_front = floor(src_y);
			int src_y_after = src_y_front + 1;

			/*   各像素权重   */
			double k1 = (src_x - src_x_front) * (src_y - src_y_front);
			double k2 = (src_x - src_x_front) * (src_y_after - src_y);
			double k3 = (src_x_after - src_x) * (src_y - src_y_front);
			double k4 = (src_x_after - src_x) * (src_y_after - src_y);
			if (src_x_front < 0 || src_x_after < 0 || src_y_front < 0 || src_y_after < 0 || src_x_front > dst.rows - 1 || src_x_after > dst.rows - 1 || src_y_front > dst.cols - 1 || src_y_after > dst.cols - 1)
				dst.ptr<Vec3b>(i)[j] = 0;
			else
				dst.ptr<Vec3b>(i)[j] = k1 * src.ptr<Vec3b>(src_x_front)[src_y_front] + k2 * src.ptr<Vec3b>(src_x_front)[src_y_after]
									 + k3 * src.ptr<Vec3b>(src_x_after)[src_y_front] + k4 * src.ptr<Vec3b>(src_x_after)[src_y_after];
		
		}
	}
}




/***********************Alpha Blending************************/
/*****  img1 =current image ; img2 = Previous image *****/
/*****   return 0 : use map  to  mapping  corner    *****/
Mat Alpha_Blending(Mat img1,Mat img2,float moving_distance,Mat img)
{
	int alpha_startrows = DSTIMG_HEIGHT - img1.rows;
	int alpha_stoprows = DSTIMG_HEIGHT + moving_distance;
	double alpha = 1;//前图像素的权重  
	int alpha_length = alpha_stoprows - alpha_startrows;

	//重叠区域融合alpha_blending
	for (int j = (DSTIMG_LENGH- img.cols)/2; j < DSTIMG_LENGH/2+ img.cols; j++) {
		for (int i = alpha_startrows; i < DSTIMG_HEIGHT; i++) {
			if (i > alpha_stoprows) {
				alpha = 1;
			}
			else {
				if (img1.ptr<Vec3b>(i)[j][0] == 0 &&
					img1.ptr<Vec3b>(i)[j][1] == 0 &&
					img1.ptr<Vec3b>(i)[j][2] == 0) {
					alpha = 0;
				}else {
					alpha = (double)(i - alpha_startrows) / alpha_length;
				}
			}
			img2.ptr<Vec3b>(i)[j][0] = img1.ptr<Vec3b>(i)[j][0] * alpha + img2.ptr<Vec3b>(i)[j][0] * (1 - alpha);
			img2.ptr<Vec3b>(i)[j][1] = img1.ptr<Vec3b>(i)[j][1] * alpha + img2.ptr<Vec3b>(i)[j][1] * (1 - alpha);
			img2.ptr<Vec3b>(i)[j][2] = img1.ptr<Vec3b>(i)[j][2] * alpha + img2.ptr<Vec3b>(i)[j][2] * (1 - alpha);
		}
	}
	return  img2;
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
	Mat& Trans_Matrix)
{
	/***********************提取角点************************/

	Mat RemapImage; //重映射后的图
	remap(SrcImage, RemapImage, mapx, mapy, 1);//重映射
	vector<vector<Point2f>> corner_points_of_all_imgs_Per;
	vector<Point2f>  corner_points_buf_Per;//角点

	//remap后的图中寻找角点
	if (findChessboardCorners(RemapImage, pattern_size, corner_points_buf_Per) == 0) {
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
		for (int j = 0; j < SrcImage.rows; j++) {
			for (int i = 0; i < SrcImage.cols; i++) {
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
			Point2f(190 * 4 + 200,30 * 4 + 10) ,
			Point2f(200,30 * 4 + 10),
			Point2f(190 * 4 + 200,10),
			Point2f(200,10)
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
			Point2f(190 * 4 + 200,30 * 4 + 10) ,
			Point2f(200,30 * 4 + 10),
			Point2f(190 * 4 + 200,10),
			Point2f(200,10)
		};
		Trans_Matrix = getPerspectiveTransform(src_points, dst_points);//获取透视变换矩阵
		return 1;
	}

}



/**************************透视变换***************************/
/*****   return  Trans_Matrix    *****/
Mat PerspectiveTransform(
	Mat Perspective_SrcImage,
	Mat mapx,
	Mat mapy,
	Mat Trans_Matrix)
{
	Mat Perspective_DstImage; //透视变换图
	remap(Perspective_SrcImage, Perspective_SrcImage, mapx, mapy, 1);//重映射
	warpPerspective(Perspective_SrcImage, Perspective_DstImage, Trans_Matrix, Size(1200, 500));//透视变换

	imwrite("P_Image/Trans.jpg", Perspective_DstImage);
	//imshow("透视变换结果图", Perspective_DstImage);

	return Perspective_DstImage;
}



/**************************读取map文件***************************/
/*****   return  mapdata    *****/
Mat  RradMapdata(string maplocate)
{
	ifstream ReadData(maplocate, ios::in | ios::binary);
	Mat mapdata = Mat(IMG_SIZE, CV_32FC1);

	if (!ReadData.is_open()) {
		cerr << "read map error!" << endl;
		exit(0);
	}
	ReadData.read((char*)mapdata.data, sizeof(float) * IMG_SIZE.height * IMG_SIZE.width);
	ReadData.close();

	return mapdata;
}


/**************************读取视频帧***************************/
/*****   return  video_to_img    *****/
Mat Read_VideoToImage(string video_path,const char image_path[])
{
	static VideoCapture capture(video_path);
	char image_namepath[80];
	Mat video_to_img;     //图像变量
	static int i = 0;     //图像计数
	int loop_frame = 4;
	int loop_frame_300 = 400;
	//从600帧开始
	if (i == 0){
		while (loop_frame_300--)
			capture.read(video_to_img);    
	}
	//每loop_frame帧取1帧
	while(loop_frame--)
		capture.read(video_to_img);    //读取视频帧

	//sprintf_s(image_namepath, "%s%s%d%s", image_path, "image", i , ".jpg");   //指定保存路径
	//imwrite(image_namepath, video_to_img);  //保存图像
	i++;
	if (video_to_img.data == NULL){
		exit(0);
	}

	return video_to_img;
}



/**************************SURF特征匹配***************************/
Mat SURF_Feature(Mat img1, Mat img2, bool& flag)
{
	//灰度图转换  
	Mat img1_gry, img2_gry;
	cvtColor(img1, img1_gry, CV_RGB2GRAY);
	cvtColor(img2, img2_gry, CV_RGB2GRAY);

	Ptr<SURF> surf; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2d   
	surf = SURF::create();   //越大越少特征点
	BFMatcher matcher;         //实例化一个暴力匹配器
	Mat surf_descriptor1, surf_descriptor2;//surf描述子
	vector<KeyPoint>surf_keypoint1, surf_keypoint2;//surf关键点
	vector<DMatch> matches;    //DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息

	//计算描述子
	surf->detectAndCompute(img1_gry, Mat(), surf_keypoint1, surf_descriptor1);//输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	surf->detectAndCompute(img2_gry, Mat(), surf_keypoint2, surf_descriptor2);//这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）

	//匹配数据来源是特征向量描述子，结果存放在DMatch类型里面  
	matcher.match(surf_descriptor1, surf_descriptor2, matches);            

	//sort函数对匹配点进行升序排列
	sort(matches.begin(), matches.end());     

	vector< DMatch > good_matches;
	int ptsPairs = min(100, (int)(matches.size() * 0.3));
	//cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++){
		good_matches.push_back(matches[i]);//距离最小的50个压入新的DMatch
	}

	Mat match_image;  //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(img1, surf_keypoint1, img2, surf_keypoint2, good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("match_image ", match_image);

	vector<Point2f> goodimagepoints1, goodimagepoints2;

	for (int i = 0; i < good_matches.size(); i++){	
		//这里的queryIdx代表了查询点的目录 , trainIdx代表了在匹配时训练分类器所用的点的目录
		goodimagepoints1.push_back(surf_keypoint1[good_matches[i].queryIdx].pt);
		goodimagepoints2.push_back(surf_keypoint2[good_matches[i].trainIdx].pt);
	}

	//ransac筛选匹配点
	vector<int> ransac_match_i;
	vector<Point2f> Ransac_points1, Ransac_points2;
	Ransac_Transform(goodimagepoints1, goodimagepoints2, Ransac_points1, Ransac_points2, ransac_match_i);

	vector< DMatch > ransac_good_matches;
	for(int i=0;i< ransac_match_i.size();i++)
		ransac_good_matches.push_back(good_matches[ransac_match_i[i]]);

	//drawMatches(img1, surf_keypoint1, img2, surf_keypoint2, ransac_good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("ransac match_image ", match_image);

	//获取图像1到图像2的仿射变换矩阵 尺寸为2*3  
	Mat M = estimateRigidTransform(Ransac_points1, Ransac_points2,false);
	//M = getAffineTransform(goodimagepoints1, goodimagepoints2);cos(theta)
	
	//保存上一次M矩阵
	static Mat pre_M_Coordinate;
	Mat M_Coordinate;
	if (M.cols * M.rows == 0)
	{
		flag = 1;
		return pre_M_Coordinate;
	}
	else
	{
		flag = 0;
		//M矩阵坐标变换
		M_Coordinate = M_ChangeCoordinate(M, Point2f((DSTIMG_LENGH - img2.cols) / 2, DSTIMG_HEIGHT - img2.rows));
		pre_M_Coordinate = M_Coordinate.clone();
		return M_Coordinate;
	}
}



/**************************SIFT特征匹配***************************/
Mat SIFT_Feature(Mat img1, Mat img2, bool& flag)
{
	//灰度图转换  
	Mat img1_gry, img2_gry;
	cvtColor(img1, img1_gry, CV_RGB2GRAY);
	cvtColor(img2, img2_gry, CV_RGB2GRAY);

	//定义SIFT特征检测类对象
	//SiftFeatureDetector siftDetector;
	Ptr <SIFT> siftDetector = SIFT::create();//取值越大，特征点越多
	//定义KeyPoint变量
	vector<KeyPoint>sift_keypoint1;
	vector<KeyPoint>sift_keypoint2;
	Mat sift_descriptor1, sift_descriptor2;
	//sift_keypoint1.resize(200);
	//sift_keypoint2.resize(200);
	//特征点检测
	siftDetector->detectAndCompute(img1_gry, noArray(), sift_keypoint1, sift_descriptor1);
	siftDetector->detectAndCompute(img2_gry, noArray(), sift_keypoint2, sift_descriptor2);

	BFMatcher matcher;    //实例化暴力匹配器
	vector<DMatch>matches;   //定义匹配结果变量
	matcher.match(sift_descriptor1, sift_descriptor2, matches);  //实现描述符之间的匹配

	//sort函数对匹配点进行升序排列
	sort(matches.begin(), matches.end());

	vector< DMatch > good_matches;
	int ptsPairs = min(100, (int)(matches.size() * 0.3));
	//cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++) {
		good_matches.push_back(matches[i]);//距离最小的50个压入新的DMatch
	}

	//Mat match_image;  //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(img1, sift_keypoint1, img2, sift_keypoint2, good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("match_image ", match_image);

	vector<Point2f> goodimagepoints1, goodimagepoints2;

	for (int i = 0; i < good_matches.size(); i++) {
		//这里的queryIdx代表了查询点的目录 , trainIdx代表了在匹配时训练分类器所用的点的目录
		goodimagepoints1.push_back(sift_keypoint1[good_matches[i].queryIdx].pt);
		goodimagepoints2.push_back(sift_keypoint2[good_matches[i].trainIdx].pt);
	}

	//ransac筛选匹配点
	vector<int> ransac_match_i;
	vector<Point2f> Ransac_points1, Ransac_points2;
	Ransac_Transform(goodimagepoints1, goodimagepoints2, Ransac_points1, Ransac_points2, ransac_match_i);

	vector< DMatch > ransac_good_matches;
	for (int i = 0; i < ransac_match_i.size(); i++)
		ransac_good_matches.push_back(good_matches[ransac_match_i[i]]);

	//drawMatches(img1, sift_keypoint1, img2, sift_keypoint2, ransac_good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("ransac match_image ", match_image);

	//获取图像1到图像2的仿射变换矩阵 尺寸为2*3  
	Mat M = estimateRigidTransform(Ransac_points1, Ransac_points2, false);
	//M = getAffineTransform(goodimagepoints1, goodimagepoints2);cos(theta)

	//保存上一次M矩阵
	static Mat pre_M_Coordinate;
	Mat M_Coordinate;
	if (M.cols * M.rows == 0)
	{
		flag = 1;
		return pre_M_Coordinate;
	}
	else
	{
		flag = 0;
		//M矩阵坐标变换
		M_Coordinate = M_ChangeCoordinate(M, Point2f((DSTIMG_LENGH - img2.cols) / 2, DSTIMG_HEIGHT - img2.rows));
		pre_M_Coordinate = M_Coordinate.clone();
		return M_Coordinate;
	}
}



/**************************ORB特征匹配***************************/
/**********   flag=1  ->获取M矩阵失败  返回上一次矩阵   *********/
/**********   flag=0  ->获取M矩阵成功                   *********/
Mat ORB_Feature(Mat img1, Mat img2, bool& flag)
{
	//灰度图转换  
	Mat img1_gry, img2_gry;
	cvtColor(img1, img1_gry, CV_RGB2GRAY);
	cvtColor(img2, img2_gry, CV_RGB2GRAY);

	std::vector<KeyPoint> orb_keypoint1, orb_keypoint2;
	Mat orb_descriptor1, orb_descriptor2;
	Ptr<ORB> orb = ORB::create();//越大特征点越多

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	orb->detectAndCompute(img1_gry, noArray(), orb_keypoint1, orb_descriptor1);
	orb->detectAndCompute(img2_gry, noArray(), orb_keypoint2, orb_descriptor2);

	Mat key_img;
	//drawKeypoints(img1, orb_keypoint1, key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB特征点", key_img);

	//对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> matches;
	//BFMatcher matcher ( NORM_HAMMING );
	matcher->match(orb_descriptor1, orb_descriptor2, matches);

	//匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < orb_descriptor1.rows; i++){
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance < m2.distance; })->distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance < m2.distance; })->distance;


	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	std::vector< DMatch > good_matches;
	for (int i = 0; i < orb_descriptor1.rows; i++){
		if (matches[i].distance <= max(2 * min_dist, 50.0)){
			good_matches.push_back(matches[i]);
		}
	}

	Mat match_image;  //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(img1, orb_keypoint1, img2, orb_keypoint2, good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("match_image ", match_image);

	vector<Point2f> goodimagepoints1, goodimagepoints2;

	for (int i = 0; i < good_matches.size(); i++) {
		//这里的queryIdx代表了查询点的目录 , trainIdx代表了在匹配时训练分类器所用的点的目录
		goodimagepoints1.push_back(orb_keypoint1[good_matches[i].queryIdx].pt);
		goodimagepoints2.push_back(orb_keypoint2[good_matches[i].trainIdx].pt);
	}

	//ransac筛选匹配点
	vector<int> ransac_match_i;
	vector<Point2f> Ransac_points1, Ransac_points2;
	Ransac_Transform(goodimagepoints1, goodimagepoints2, Ransac_points1, Ransac_points2, ransac_match_i);

	vector< DMatch > ransac_good_matches;
	for (int i = 0; i < ransac_match_i.size(); i++)
		ransac_good_matches.push_back(good_matches[ransac_match_i[i]]);

	//drawMatches(img1, orb_keypoint1, img2, orb_keypoint2, ransac_good_matches, match_image, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("ransac match_image ", match_image);

	//获取图像1到图像2的仿射变换矩阵 尺寸为2*3  
	Mat M = estimateRigidTransform(Ransac_points1, Ransac_points2, false);
	//M = getAffineTransform(goodimagepoints1, goodimagepoints2);
	
	//保存上一次M矩阵
	static Mat pre_M_Coordinate;
	Mat M_Coordinate;
	if (M.cols * M.rows == 0)
	{
		flag = 1;
		return pre_M_Coordinate;
	}else
	{
		flag = 0;
		//M矩阵坐标变换
		M_Coordinate = M_ChangeCoordinate(M, Point2f((DSTIMG_LENGH - img2.cols) / 2, DSTIMG_HEIGHT - img2.rows));
		pre_M_Coordinate = M_Coordinate.clone();
		return M_Coordinate;
	}
}

/**************************稠密光流法***************************/
/*            img1  先前图片    ;   img2   之后图片            */
Mat OpticalFlow_LK(Mat img1, Mat img2, bool& flag)
{
	Mat img1_gry, img2_gry;
	cvtColor(img1, img1_gry, CV_RGB2GRAY);//灰度图转换
	cvtColor(img2, img2_gry, CV_RGB2GRAY);//灰度图转换
	vector<uchar>status;//特征点跟踪成功标志位
	vector<float>errors;//跟踪时候区域误差和
	vector<Point2f> pre_keypoint2f, cur_keypoint2f;//整理之前得前图和当前图得特征点
	vector<Point2f> pre_keypoint2f_sort, cur_keypoint2f_sort;//整理之后得前图和当前图得特征点
	
	std::vector<KeyPoint> keypoint1;
	int switch_key_way=2;
	switch (switch_key_way) {
		case 1: {	
			/*******SURF特征点*******/
			Ptr<SURF> surf=SURF::create(); //越大越少特征点
			surf->detect(img1_gry, keypoint1);//计算关键点
			break;
		}
		case 2: {
			/*******SIFT特征点*******/
			Ptr <SIFT> sift = SIFT::create();//取值越大，特征点越多
			sift->detect(img1_gry, keypoint1);//计算关键点
			break;
		}

		case 3: {
			/*******ORB特征点*******/
			Ptr<FeatureDetector> orb = ORB::create();//越大特征点越多
			orb->detect(img1_gry, keypoint1);//检测 Oriented FAST角点位置
			break;	
		}
	}
	//特征点转换为point2f
	KeyPoint::convert(keypoint1, pre_keypoint2f);
	//LK特征点跟踪
	calcOpticalFlowPyrLK(img1_gry, img2_gry, pre_keypoint2f, cur_keypoint2f, status, errors);
	//特征点整理

	for (int i = 0; i < cur_keypoint2f.size(); i++) {

		if (status[i]){
			pre_keypoint2f_sort.push_back(pre_keypoint2f[i]);
			cur_keypoint2f_sort.push_back(cur_keypoint2f[i]);
		}
	}
	//cout << "前帧特征点为：" << pre_keypoint2f.size() << "后帧追踪得特征点为：" << cur_keypoint2f_sort.size() << endl;

	//ransac筛选匹配点
	vector<int> ransac_match_i;
	vector<Point2f> Ransac_points1, Ransac_points2;
	Ransac_Transform(pre_keypoint2f_sort, cur_keypoint2f_sort, Ransac_points1, Ransac_points2, ransac_match_i);

	//获取图像1到图像2的仿射变换矩阵 尺寸为2*3  
	Mat M = estimateRigidTransform(Ransac_points1, Ransac_points2, false);
	//M = getAffineTransform(goodimagepoints1, goodimagepoints2);cos(theta)

	//保存上一次M矩阵
	static Mat pre_M_Coordinate;
	Mat M_Coordinate;
	if (M.cols * M.rows == 0)
	{
		flag = 1;
		return pre_M_Coordinate;
	}
	else
	{
		flag = 0;
		//M矩阵坐标变换
		M_Coordinate = M_ChangeCoordinate(M, Point2f((DSTIMG_LENGH - img2.cols) / 2, DSTIMG_HEIGHT - img2.rows));
		pre_M_Coordinate = M_Coordinate.clone();
		return M_Coordinate;
	}

}


/**************************稠密光流法***************************/
Mat OpticalFlow_FB(Mat img1, Mat img2, bool& flag)
{
	//灰度图转换  
	Mat img1_gry, img2_gry;
	cvtColor(img1, img1_gry, CV_RGB2GRAY);
	cvtColor(img2, img2_gry, CV_RGB2GRAY);

	Mat flow;//光流数据
	calcOpticalFlowFarneback(img1_gry, img2_gry, flow, 0.5, 3, 15, 3, 5, 1.2, 0);//稠密光流
	//calcOpticalFlowPyrLK();//稀疏光流LK法
	float Mov_dixtance_x=0, Mov_dixtance_y=0;
	//原图画出移动点
	for (int i = 0; i < flow.rows; i++) {
		for (int j = 0; j < flow.cols; j++) {
			Mov_dixtance_x += flow.ptr<Vec2f>(i)[j][0];
			Mov_dixtance_y += flow.ptr<Vec2f>(i)[j][1];
			const Point2f fxy = flow.at<Point2f>(i, j);
			if (fxy.x > 1 || fxy.y > 1) {
				circle(img1, Point(j, i), 2, Scalar(0, 255, 0), 2);
			}
		}
	}
	Mov_dixtance_x /= flow.rows * flow.cols;
	Mov_dixtance_y /= flow.rows * flow.cols;
	Mat matrix = Mat::zeros(Size(3,2),CV_64FC1);

	matrix.ptr<double>(0)[0] = 1;
	matrix.ptr<double>(1)[1] = 1;
	matrix.ptr<double>(0)[2] = Mov_dixtance_x;
	matrix.ptr<double>(1)[2] = Mov_dixtance_y;
	
//	imshow("flow", img1);
	return matrix;
}



/************************** M矩阵 --坐标变换RT ***************************/
Mat M_ChangeCoordinate( Mat m, Point2f rotate_point)
{
	double costheta= m.ptr<double>(0)[0];
	double sintheta= m.ptr<double>(1)[0];
	//旋转矩阵
	double r[3][3]= { {costheta,-sintheta,rotate_point.x - rotate_point.x * costheta + rotate_point.y * sintheta},
					  {sintheta, costheta,rotate_point.y - rotate_point.x * sintheta - rotate_point.y * costheta},
					  {0,0,1 }};
	Mat rotate_matrix(3, 3, CV_64FC1, r);

	//平移矩阵
	double t[3][3] = { {1, 0, m.ptr<double>(0)[2]},
					   {0, 1, m.ptr<double>(1)[2]},
					   {0, 0, 1 } };
	Mat translation_matrix(3, 3, CV_64FC1,t);

	//融合后矩阵
	Mat rt_matrix = translation_matrix * rotate_matrix ;

	//2*3矩阵
	double rt[2][3] =  {{rt_matrix.ptr<double>(0)[0],  rt_matrix.ptr<double>(0)[1],  rt_matrix.ptr<double>(0)[2]},
						{rt_matrix.ptr<double>(1)[0],  rt_matrix.ptr<double>(1)[1],  rt_matrix.ptr<double>(1)[2]}};
		
	Mat rt_matrix23;
	rt_matrix23.create(2, 3, CV_64FC1);
	memcpy(rt_matrix23.data, rt ,sizeof(double)*6);
	
	return rt_matrix23;
}



/************************** Ransac算法 ***************************/
void Ransac_Transform(vector<Point2f> inputpoint1, vector<Point2f> inputpoint2, vector<Point2f>& outputpoint1, vector<Point2f>& outputpoint2,vector<int>& best_inner)
{
	int ransac_threshold = 5;         //误差阈值
	const double confidence = 0.995;  //置信度
	const int max_iteration = 2000;   //迭代最大次数
	int input_totalnum = inputpoint1.size();//输入样本总数
	int interation_num = max_iteration;
	const int ransac_n =3;//训练点数目
	int iteration=0;
	int best_inner_count = 0;//最优内点个数

	//循环max_iteration次
	while (interation_num > iteration++)
	{
		vector<Point2f> randpoint1, randpoint2;//随机点
		int rand_num;//随机数
		for (int i = 0; i < ransac_n;i++) {
			rand_num = rand() % input_totalnum;
			randpoint1.push_back(inputpoint1[rand_num]);
			randpoint2.push_back(inputpoint2[rand_num]);		
		}

		//求取M变换矩阵
		/*Mat M = estimateRigidTransform(randpoint1, randpoint2, false);*/
		Mat M = getAffineTransform(randpoint1, randpoint2);
		if (M.rows * M.cols == 0) {
			iteration--;
			continue;
		}
		//Mat M=getAffineTransform(randpoint1, randpoint2);
		int inner_count = 0;//内点个数
		vector<int> inner;//内点
		for (int i=0; i < input_totalnum; i++) {
			//误差函数
			float error_function = sqrt(pow(inputpoint2[i].x - (M.ptr<double>(0)[0] * inputpoint1[i].x + M.ptr<double>(0)[1] * inputpoint1[i].y + M.ptr<double>(0)[2]), 2) +
								        pow(inputpoint2[i].y - (M.ptr<double>(1)[0] * inputpoint1[i].x + M.ptr<double>(1)[1] * inputpoint1[i].y + M.ptr<double>(1)[2]), 2));
			//若误差小于阈值则加入内点
			if (error_function < ransac_threshold) {
				inner_count++;
				inner.push_back(i);
			}
		}
		////赋值最优内点
		if (inner_count > best_inner_count){
			best_inner_count = inner_count;
			inner.swap(best_inner);
			//更新迭代次数
			interation_num = log(1 - confidence) / log(1 - pow((float)inner_count/input_totalnum, ransac_n));
		}	
	}
	//整理输出ransac后的数据点
	for (int i = 0; i < best_inner.size(); i++){
		outputpoint1.push_back(inputpoint1[best_inner[i]]);
		outputpoint2.push_back(inputpoint2[best_inner[i]]);
	}
	
}



