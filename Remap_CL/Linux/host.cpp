#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include<math.h>

using namespace cv;
using namespace std;
const char *CLFilePath = "remap.cl";// �˺����ļ�


#define USE_GPU 0
#define USE_CPU 1
#define IMG_SIZE  Size(1280, 720)
#define MALLOC_SIZE  IMG_SIZE.width*IMG_SIZE.height
const char *imagePath = "input.png";


#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif


#define CHECK_ERROR(warnstatus,warnstring)   	if (warnstatus != 0) cout << "location: "<<__FILE__<<" ;" "line "<<__LINE__-1<<" ;"<<warnstring <<" ; status:"<< status << endl

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


void CheckErrorLog(cl_program program, cl_device_id  device, cl_int status) {
	if (status != CL_SUCCESS)
	{
		size_t len;
		char buffer[8 * 1024];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		//goto FINISH;
	}
}




/***************************************************/
/***************        ������       ***************/
/***************************************************/
int main()
{

	/*****图片相关****/
	Mat Input_Img = imread(imagePath);
	cvtColor(Input_Img, Input_Img, CV_RGB2RGBA);
	Mat Output_Img(IMG_SIZE, Input_Img.type());
	Mat Mapx = RradMapdata("Dedistort_MapRx.txt");//定义remap映射mapx
	Mat Mapy = RradMapdata("Dedistort_MapRy.txt");//定义remap映射mapy	
	

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

	//分配内存
	void *Input_imageData = (unsigned char*)malloc(sizeof(unsigned char) * 4 * MALLOC_SIZE);
	void *Output_ImgData = (unsigned char*)malloc(sizeof(unsigned char) * 4 * MALLOC_SIZE);
	void *MapxData = (float*)malloc(sizeof(float) * MALLOC_SIZE);
	void *MapyData = (float*)malloc(sizeof(float) * MALLOC_SIZE);

	//创建CL图像
	cl_image_format image_format;
	image_format.image_channel_order = CL_RGBA;  //四通道
	image_format.image_channel_data_type = CL_UNSIGNED_INT8;//无符号8为整形
	cl_image_desc image_desc;
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;        // 可以 memset(desc,sizeof(cl_image_desc)); 后仅对前三项赋值
	image_desc.image_width = IMG_SIZE.width;
	image_desc.image_height = IMG_SIZE.height;
	image_desc.image_depth = 0;
	image_desc.image_array_size = 0;
	image_desc.image_row_pitch = 0;
	image_desc.image_slice_pitch = 0;
	image_desc.num_mip_levels = 0;
	image_desc.num_samples = 0;
	image_desc.buffer = NULL;
	cl_mem clInputImage = clCreateImage(context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create InputImage: ");
	cl_mem clOutputImage = clCreateImage(context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create OutputImage: ");
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = CL_FLOAT;
	cl_mem clMapx_Mat = clCreateImage(context, CL_MEM_WRITE_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create Mapx Mat: ");
	cl_mem clMapy_Mat = clCreateImage(context, CL_MEM_WRITE_ONLY || CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc, NULL, &status);
	CHECK_ERROR(status, "Error Create Mapy Mat: ");
	//采样器
	cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
	CHECK_ERROR(status, "Error Create Sampler: ");


	/*****内核和程序****/
	//程序
	char* source = NULL;
	const size_t CLFileLengh = readSource(CLFilePath, &source);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &CLFileLengh, &status);
	CHECK_ERROR(status, "Error Create Program: ");

	clBuildProgram(program, 1, ListDevice, NULL, NULL, NULL);
	//内核
	cl_kernel kernel = clCreateKernel(program, "Image_Remap", &status);
	if (status != 0)
		cout << "Error Create Kernel: " << status << endl;
	
	//CheckErrorLog(program, ListDevice[0],status);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &clInputImage);//依次设置内核参数
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &clMapx_Mat);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &clMapy_Mat);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &clOutputImage);

	size_t GlobalSize[2] = { IMG_SIZE.width, IMG_SIZE.height };
	size_t origin[3] = { 0,0,0 };// 拷贝图片缓冲区时使用的起点参数
	size_t region[3] = { IMG_SIZE.width, IMG_SIZE.height, 1 };// 拷贝图片缓冲区时使用的范围参数
	size_t imageRowPitch_in =  IMG_SIZE.width * sizeof(char) * 4;
	void *mapPtr_out = clEnqueueMapImage(queue, clInputImage, CL_TRUE, CL_MAP_WRITE,
		origin, region, &imageRowPitch_in, NULL, 0, NULL, NULL, &status);
	CHECK_ERROR(status, "Error Create Mapx Mat: ");
	void *mapPtr_out2 = clEnqueueMapImage(queue, clMapx_Mat, CL_TRUE, CL_MAP_WRITE,
		origin, region, &imageRowPitch_in, NULL, 0, NULL, NULL, &status);
	CHECK_ERROR(status, "Error Create Mapx Mat: ");
	void *mapPtr_out3 = clEnqueueMapImage(queue, clMapy_Mat, CL_TRUE, CL_MAP_WRITE,
		origin, region, &imageRowPitch_in, NULL, 0, NULL, NULL, &status);
	CHECK_ERROR(status, "Error Create Mapx Mat: ");
	
	memcpy(mapPtr_out, Input_Img.data, sizeof(unsigned char) * 4 * MALLOC_SIZE);
	memcpy(mapPtr_out2, Mapx.data, sizeof(float) * MALLOC_SIZE);
	memcpy(mapPtr_out3, Mapy.data, sizeof(float) * MALLOC_SIZE);

	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, GlobalSize, NULL, 0, NULL, NULL);//执行内核
	CHECK_ERROR(status, "Error clEnqueueNDRangeKernel: ");

	status = clEnqueueReadImage(queue, clOutputImage, CL_TRUE, origin, region, 0, 0, Output_ImgData, 0, NULL, NULL);
	CHECK_ERROR(status, "Error clEnqueueReadImage: ");
	memcpy(Output_Img.data, Output_ImgData, sizeof(unsigned char) * 4 * MALLOC_SIZE);

	//清理内存
	free(ListPlatform);
	free(ListDevice);
	clReleaseContext(context);
	clReleaseMemObject(clInputImage);
	clReleaseMemObject(clMapx_Mat);
	clReleaseMemObject(clMapy_Mat);
	clReleaseMemObject(clOutputImage);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

   	imwrite("remap.png",Output_Img);
	return 0;
}
