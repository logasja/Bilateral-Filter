#include "stdafx.h"

using cv::Mat;
using cv::imread;
using std::cout;

int main(int argc, char **argv)
{
	cout << "Creating Bilateral Filter...\n";
	BilateralFilter *bf = new BilateralFilter();

	cout << "Reading in image...\n";
	Mat input = imread(inputPath, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	cout << "Displaying input image...\n";
	cv::imshow("Input",input);
	cv::waitKey(0);

	cout << "Converting to float matrix...\n";
	// Converts the image to float with a domain of [0,1]
	input.convertTo(input, CV_32FC3, 1/255.0);

	cout << "Applying filter...\n";
	Mat output = bf->ApplyFilter(input);

	cout << "Displaying output image...\n";
	cv::imshow("Output Serial", output);
	cv::waitKey(0);

	cv::imwrite(outputPath, output);
	delete bf;
	output.deallocate();
	input.deallocate();
}