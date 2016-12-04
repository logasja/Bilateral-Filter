#pragma once

#include <opencv2\core\core.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using cv::Mat;
#ifdef _DEBUG
#include <opencv2\highgui\highgui.hpp>
using std::endl;
using cv::imshow;
using cv::waitKey;
#endif

class BilateralFilter
{
public:
	BilateralFilter(int width = 5, float sigd = 3.f, float sigr = 0.1f);
	~BilateralFilter();

	Mat ApplyFilter(Mat img);
	Mat ApplyFilterCUDA(Mat img);
private:
	void GenerateGMat();
	void meshgrid(Mat* X, Mat* Y);
	void ApplyFilterColor(Mat* img, Mat* out);
	void ApplyFilterGray(Mat* img, Mat* out);
	Mat G;
	float *d_kernel = NULL;
	float d, r;
	int w;

#ifdef _DEBUG
	inline std::string type2str(int type) {
		std::string r;

		uchar depth = type & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch (depth) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
		}

		r += "C";
		r += (chans + '0');

		return r;
	}
#endif //_DEBUG
};