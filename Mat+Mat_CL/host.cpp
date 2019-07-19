#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
const char *CLFilePath = "merge.cl";// �˺����ļ�
const char *imagePath1 = "1.jpg";
const char *imagePath2 = "2.jpg";


//#ifdef __unix
//#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
//#endif


#define CHECK_ERROR(warnstatus,warnstring)   	if (warnstatus != 0) cout << "line "<<__LINE__-1<<" ;"<<warnstring <<" ; status:"<< status << endl

/***************************************************/
/***************   ��ȡkernel����    ***************/
/***************************************************/
int readSource(const char* kernelPath, char **output)// ��ȡ�ı��ļ����洢Ϊ char *
{
	FILE *fp;
	int size;
	fopen_s(&fp, kernelPath, "rb");
	if (!fp)
	{
		printf("Open kernel file failed\n");
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_END) != 0)
	{
		printf("Seek end of file faildd\n");
		exit(-1);
	}
	if ((size = ftell(fp)) < 0)
	{
		printf("Get file position failed\n");
		exit(-1);
	}
	rewind(fp);
	if ((*output = (char *)malloc(size + 1)) == NULL)
	{
		printf("Allocate space failed\n");
		exit(-1);
	}
	fread((void*)*output, 1, size, fp);
	fclose(fp);
	(*output)[size] = '\0';
	printf("readSource succeed, program file: %s\n", kernelPath);
	return size;
}





/***************************************************/
/***************        ������       ***************/
/***************************************************/
int main()
{
	/*****准备平台，设备，上下文，命令队列部分****/
	cl_int status;
	cl_uint nPlatform;
	cl_uint nDevice = 0;
	//平台
	clGetPlatformIDs(0, NULL, &nPlatform);
	cl_platform_id *ListPlatform = (cl_platform_id*)malloc(nPlatform * sizeof(cl_platform_id));
	clGetPlatformIDs(nPlatform, ListPlatform, NULL);
	//设备
	clGetDeviceIDs(ListPlatform[0], CL_DEVICE_TYPE_ALL, 0, NULL, &nDevice);
	cl_device_id *ListDevice = (cl_device_id*)malloc(nDevice * sizeof(cl_device_id));
	clGetDeviceIDs(ListPlatform[0], CL_DEVICE_TYPE_ALL, nDevice, ListDevice, NULL);
	//上下文
	cl_context context = clCreateContext(NULL, nDevice, ListDevice, NULL, NULL, &status);
	CHECK_ERROR(status, "Error Create Context: ");
	//命令队列
	cl_command_queue queue = clCreateCommandQueue(context, ListDevice[0], NULL, &status);
	CHECK_ERROR(status, "Error Create  Command Queue: ");



	/*****图片相关****/
	Mat image1 = imread(imagePath1);
	Mat image2 = imread(imagePath2);
	cvtColor(image1, image1, COLOR_BGR2BGRA);
	cvtColor(image2, image2, COLOR_BGR2BGRA);

	const int imageWidth = image1.cols;
	const int imageHeight = image1.rows;
	void *imageData1 = (void*)malloc(sizeof(unsigned char) * 4 * imageWidth*imageHeight);
	void *imageData2 = (unsigned char*)malloc(sizeof(unsigned char) * 4 * imageWidth*imageHeight);
	void *dstimageData = (unsigned char*)malloc(sizeof(unsigned char) * 4 * imageWidth*imageHeight);

	cl_image_format image_format;
	image_format.image_channel_order = CL_RGBA;  //四通道
	image_format.image_channel_data_type = CL_UNSIGNED_INT8;//无符号8为整形
	cl_image_desc image_desc;
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;        // 可以 memset(desc,sizeof(cl_image_desc)); 后仅对前三项赋值
	image_desc.image_width = imageWidth;
	image_desc.image_height = imageHeight;
	image_desc.image_depth = 0;
	image_desc.image_array_size = 0;
	image_desc.image_row_pitch = 0;
	image_desc.image_slice_pitch = 0;
	image_desc.num_mip_levels = 0;
	image_desc.num_samples = 0;
	image_desc.buffer = NULL;
	//����CLͼƬ
	cl_mem clInputeImage1 = clCreateImage(context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create Image1: ");
	cl_mem clInputeImage2 = clCreateImage(context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create Image2: ");

	image_format.image_channel_order = CL_RGBA;  //��ͨ��
	cl_mem clOutputImage = clCreateImage(context, CL_MEM_WRITE_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create Output Image: ");
	//采样器
	cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
	CHECK_ERROR(status, "Error Create Sampler: ");


	//if (status != CL_SUCCESS)
	//{
	//	size_t len;
	//	char buffer[8 * 1024];

	//	printf("Error: Failed to build program executable!\n");
	//	clGetProgramBuildInfo(program, ListDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	//	printf("%s\n", buffer);
	//}

	/*****内核和程序****/
	//程序
	char* source = NULL;
	const size_t CLFileLengh = readSource(CLFilePath, &source);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &CLFileLengh, &status);
	CHECK_ERROR(status, "Error Create Program: ");

	clBuildProgram(program, 1, ListDevice, NULL, NULL, NULL);
	//内核
	cl_kernel kernel = clCreateKernel(program, "Image_Merge", &status);
	if (status != 0)
		cout << "Error Create Kernel: " << status << endl;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &clInputeImage1);//依次设置内核参数
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &clInputeImage2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &clOutputImage);
	clSetKernelArg(kernel, 3, sizeof(cl_sampler), &sampler);

	size_t GlobalSize[2] = { imageWidth, imageHeight };
	size_t origin[3] = { 0,0,0 };// 拷贝图片缓冲区时使用的起点参数
	size_t region[3] = { imageWidth, imageHeight, 1 };// 拷贝图片缓冲区时使用的范围参数
	memcpy(imageData1, image1.data, sizeof(unsigned char) * 4 * imageWidth*imageHeight);
	memcpy(imageData2, image2.data, sizeof(unsigned char) * 4 * imageWidth*imageHeight);
	size_t image_pitch = 4 * imageWidth * sizeof(uchar);
	clEnqueueWriteImage(queue, clInputeImage1, CL_TRUE, origin, region, 0, 0, imageData1, 0, NULL, NULL);
	clEnqueueWriteImage(queue, clInputeImage2, CL_TRUE, origin, region, 0, 0, imageData2, 0, NULL, NULL);

	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, GlobalSize, NULL, 0, NULL, NULL);//执行内核

	clEnqueueReadImage(queue, clOutputImage, CL_TRUE, origin, region, 0, 0, dstimageData, 0, NULL, NULL);
	Mat dst_img(imageHeight, imageWidth, CV_8UC4);
	memcpy(dst_img.data, dstimageData, sizeof(unsigned char) * 4 * imageWidth*imageHeight);

	//清理内存
	free(ListPlatform);
	free(ListDevice);
	clReleaseContext(context);
	clReleaseMemObject(clInputeImage1);
	clReleaseMemObject(clInputeImage2);
	clReleaseMemObject(clOutputImage);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

	imshow("融合后的图片", dst_img);
	waitKey(0);
	return 0;
}
