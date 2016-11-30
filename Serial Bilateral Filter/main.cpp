#include "stdafx.h"

using cv::Mat;
using cv::imread;

int main(int argc, char **argv)
{
	BilateralFilter *bf = new BilateralFilter();
	Mat input = imread(inputPath, CV_LOAD_IMAGE_ANYDEPTH);
}