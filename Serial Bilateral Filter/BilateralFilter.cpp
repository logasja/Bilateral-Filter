#include "BilateralFilter.h"

#include "opencv2\imgproc\imgproc.hpp"

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
#ifdef _DEBUG
	std::cout << "Data type of image:" << type2str(img.type()) << endl;
#endif

	Mat out = Mat::zeros(img.rows, img.cols, CV_32FC3);

	if (img.channels() > 1)
		ApplyFilterColor(&img ,&out);
	else
		ApplyFilterGray(&img ,&out);

	return out;
}

void BilateralFilter::GenerateGMat()
{
	// Make meshgrid for X and Y
	Mat X, Y;
	meshgrid(&X, &Y);
#ifdef _DEBUG
	std::cout << "X and Y Matrices:\n";
	std::cout << endl << X << endl << endl;
	std::cout << Y << endl;
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
	std::cout << endl << G << endl;
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
	Mat row(1, N, CV_32FC3);

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

void BilateralFilter::ApplyFilterColor(Mat * img, Mat * out)
{
	int rowBound = img->rows - 1;
	int colBound = img->cols - 1;
	Mat tmp;
	// Convert to Lab color space, makes application easier
	cv::cvtColor(*img, tmp, CV_BGR2Lab);

#ifdef _DEBUG
	std::cout << "The temporary Lab matrix type is: " << type2str(tmp.type()) << endl;
#endif

	int i, j, iMin, iMax, jMin, jMax;
	Mat I, dL, da, db;
	std::vector<Mat> Ich(3);
	for (i = 0; i <= rowBound; i++)
	{
		for (j = 0; j <= colBound; j++)
		{
			// Get local region taking into account the bounds of the image
			iMin = std::max<int>(i - w, 0);
			iMax = std::min<int>(i + w + 1, rowBound);
			jMin = std::max<int>(j - w, 0);
			jMax = std::min<int>(j + w + 1, colBound);

			// Functionally equivalent to iMin:iMax in matlab
			cv::Range iMinMax(iMin, iMax);
			cv::Range jMinMax(jMin, jMax);

			I = tmp(iMinMax,jMinMax);
			
#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "Number of channels in I: " << I.channels() << endl;
				std::cout << I << endl;
			}
#endif

			cv::split(I,Ich);

			cv::Point3_<float>* p = tmp.ptr<cv::Point3_<float>>(i, j);

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "The values of Lab at (0,0) are:\n" <<
					"\tL:\t" << p->x << endl <<
					"\ta:\t" << p->y << endl <<
					"\tb:\t" << p->z << endl;

				cv::Point3_<float>* p = img->ptr<cv::Point3_<float>>(i, j);
				std::cout << "The values of LBGR at (0,0) are:\n" <<
					"\tB:\t" << p->x << endl <<
					"\tG:\t" << p->y << endl <<
					"\tR:\t" << p->z << endl << endl;
			}
#endif //_DEBUG

			dL = Ich.at(0) - p->x;
			da = Ich.at(1) - p->y;
			db = Ich.at(2) - p->z;

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "dL, da, and db values respectively:\n" <<
					dL << endl << endl <<
					da << endl << endl <<
					db << endl << endl;
			}
#endif

			/* Computing range weights */
			// Precompute square of dL, da, and db
			dL = dL.mul(dL);
			da = da.mul(da);
			db = db.mul(db);
			// Calcuate H matrix
			Mat H = -(dL + da + db) / (2 * std::powf(r, 2.f));
			cv::exp(H, H);

			/* Calculating the response */
			Mat F = 
		}
	}

	// Convert back to BGR to allow for writing to file
	cv::cvtColor(tmp, *out, CV_Lab2BGR);
	
	// Don't forget to free memory
	tmp.deallocate();
}

void BilateralFilter::ApplyFilterGray(Mat * img, Mat * out)
{
}
