#include "BilateralFilter.h"



BilateralFilter::BilateralFilter(float width, float sigd, float sigr)
{
	w = std::ceilf(width);
	d = std::ceilf(sigd);
	r = sigr;
	GenerateGMat();
}


BilateralFilter::~BilateralFilter()
{
	G.deallocate();
}

Mat BilateralFilter::ApplyFilter(Mat img)
{
	return Mat();
}

void BilateralFilter::GenerateGMat()
{
	// Make meshgrid for X and Y
	Mat X, Y;
	meshgrid(&X, &Y);
#ifdef _DEBUG
	std::cout << "X and Y Matrices:\n";
	std::cout << std::endl << X << std::endl << std::endl;
	std::cout << Y << std::endl;
#endif	//_DEBUG
	
	// Precompute values for square of X, Y, and -(X^2+Y^2)
	X = X.mul(X);
	Y = Y.mul(Y);
	Mat A = -(X + Y);

	// Calcualte G
	G = A / (2.f*powf(d, 2.f));
	cv::exp(G, G);

#ifdef _DEBUG
	std::cout << "Calcuated G Matrix:\n";
	std::cout << std::endl << G << std::endl;
#endif	//_DEBUG

	// Don't forget to free memory
	X.deallocate();
	Y.deallocate();
	A.deallocate();
}

void BilateralFilter::meshgrid(Mat * X, Mat * Y)
{
	// Precalculate the required size of the NxN matrix
	int N = w * 2 + 1;

	// Used to create the X meshgrid matrix
	Mat row(1, N, CV_32F);

	// Creates the pattern of -w:w
	float* p = row.ptr<float>(0);
	for (int i = 0; i < N; i++)
	{
		p[i] = -w + i;
#ifdef _DEBUG
		std::cout << p[i] << "\t";
#endif	//_DEBUG
	}

	// Repeats the calculated row to create the full NxN matrix
	*X = cv::repeat(row, N, 1);
	// Simply transpose X to get Y
	cv::transpose(*X, *Y);

	// Don't forget to free memory
	row.deallocate();
}
