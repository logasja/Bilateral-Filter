#include "stdafx.h"

using cv::Mat;
using cv::imread;
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	cout << "Choose what type of image to load:\n\t1: Color\n\t2: Grayscale" << endl;

#ifndef _DEBUG
	int selection, mode, width;
	float sigma_d, sigma_r;
	std::cin >> selection;
	cout << "Enter Width:" << endl;
	std::cin >> width;
	cout << "Enter Sigma d:" << endl;
	std::cin >> sigma_d;
	cout << "Enter Sigma r:" << endl;
	std::cin >> sigma_r;

	switch (selection)
	{
	case 2: mode = CV_LOAD_IMAGE_GRAYSCALE; break;
	case 1: 
	default: mode = CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH;
	}

	cout << "Creating Bilateral Filter...\n";
	BilateralFilter *bf = new BilateralFilter(width, sigma_d, sigma_r);
#else
//	int mode = CV_LOAD_IMAGE_GRAYSCALE;
	int mode = CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH;
	cout << "Creating Bilateral Filter...\n";
	BilateralFilter *bf = new BilateralFilter();
#endif

	cout << "Reading in image...\n";
	Mat input = imread(inputPath, mode);
	cout << "Displaying input image...\n";
	cv::imshow("Input",input);
	cv::waitKey(0);

	cout << "Converting to float matrix...\n";
	// Converts the image to float with a domain of [0,1]
	input.convertTo(input, CV_32FC3, 1/255.0);

	cout << "Applying filter with CUDA...\n";
	//Insert Timer
	Mat cudaOut = bf->ApplyFilterCUDA(input.clone());

	cout << "Displaying output image...\n";
	cv::imshow("Output CUDA", cudaOut);
	cv::waitKey(1);

	cout << "Applying filter serially...\n";
	//Insert Timer
	Mat out = bf->ApplyFilter(input.clone());

	cout << "Displaying output image...\n";
	cv::imshow("Output Serial", out);
	cv::waitKey(0);

	delete bf;
	cudaOut.deallocate();
	out.deallocate();
	input.deallocate();
}