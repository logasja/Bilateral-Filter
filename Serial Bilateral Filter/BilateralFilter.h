#pragma once

#include <opencv2\core\core.hpp>
#include <iostream>

using cv::Mat;

class BilateralFilter
{
public:
	BilateralFilter(float width = 5.f, float sigd = 3.f, float sigr = 0.1f);
	~BilateralFilter();

	Mat ApplyFilter(Mat img);
private:
	void GenerateGMat();
	void meshgrid(Mat* X, Mat* Y);
	Mat G;
	int w, d, r;
};