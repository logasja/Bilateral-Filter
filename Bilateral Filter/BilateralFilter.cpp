#include "BilateralFilter.h"

BilateralFilter::BilateralFilter(int width, float sigd, float sigr)
{
	w = width;
	d = std::ceilf(sigd);
	r = sigr;
	GenerateGMat();
}


Mat BilateralFilter::ApplyFilter(Mat img)
{
#ifdef _DEBUG
	std::cout << "Data type of image:" << type2str(img.type()) << endl;
#endif

	Mat out;

	if (img.channels() > 1)
	{
		out = Mat::zeros(img.rows, img.cols, CV_32FC3);
		ApplyFilterColor(&img, &out);
	}
	else
	{
		out = Mat::zeros(img.rows, img.cols, CV_32FC1);
		ApplyFilterGray(&img, &out);
	}

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
	Mat row(1, N, CV_32FC1);

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
	float rFilt = r * 100;
	// Convert to Lab color space, makes application easier
	cv::cvtColor(*img, tmp, CV_BGR2Lab);

#ifdef _DEBUG
	std::cout << "The temporary Lab matrix type is: " << type2str(tmp.type()) << endl;
#endif

	int i, j;
	float nF;
	Mat I, dL, da, db, F;
	std::vector<Mat> Ich(3);
	for (i = 0; i <= rowBound; i++)
	{
		for (j = 0; j <= colBound; j++)
		{
#ifdef _DEBUG
			// Get local region taking into account the bounds of the image
			int iMin = std::max<int>(i - w, 0);
			int iMax = std::min<int>(i + w + 1, rowBound);
			int jMin = std::max<int>(j - w, 0);
			int jMax = std::min<int>(j + w + 1, colBound);

			// Functionally equivalent to iMin:iMax in matlab
			cv::Range iMinMax(iMin, iMax);
			cv::Range jMinMax(jMin, jMax);
#else
			// Used to get a range of values from the tmp matrix
			cv::Range iMinMax(std::max<int>(i - w, 0), std::min<int>(i + w + 1, rowBound));
			cv::Range jMinMax(std::max<int>(j - w, 0), std::min<int>(j + w + 1, colBound));
#endif

			I = tmp(iMinMax,jMinMax);
			
			cv::split(I,Ich);

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "Number of channels in I: " << I.channels() << endl;
				std::cout << "L mat:\n" << Ich[0] << endl << endl <<
					"a mat:\n" << Ich[1] << endl << endl <<
					"b mat:\n" << Ich[2] << endl << endl;
			}
#endif

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

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "dL, da, and db squared values respectively:\n" <<
					dL << endl << endl <<
					da << endl << endl <<
					db << endl << endl;
			}
#endif

			// Calcuate H matrix
			Mat H = -(dL + da + db) / (2 * std::powf(rFilt, 2.f));

			cv::exp(H, H);

#ifdef _DEBUG
			if (i == 0 && j == 0)
				std::cout << "Channel number of H:\n" <<
					H.channels() << endl
					<< "The matrix H:\n"
					<< H << endl << endl;
#endif

#ifdef _DEBUG

			cv::Range a = iMinMax - i + w;
			cv::Range b = jMinMax - j + w;

			Mat Gsub = G(a, b);

			if (i == 0 && j == 0)
			{
				std::cout << "Gsub is from i = " << a.start << " to " << a.end << endl <<
					"\tj = " << b.start << " to " << b.end << endl <<
					"Gsub:\n" << Gsub << endl << endl;
			}

			/* Calculating the response */
			cv::multiply(H, Gsub, F);

			if (i == 0 && j == 0)
			{
				std::cout << "The response matrix F: \n" <<
					F << endl << "\tWith " << F.channels() << " channels.\n";
			}
#else
			cv::multiply(H, G(iMinMax - i + w, jMinMax - j + w), F);
#endif

			nF = cv::sum(F).val[0];

			/* Apply to output image */
			cv::Point3_<float>* o = out->ptr<cv::Point3_<float>>(i, j);

#ifdef _DEBUG
			// Piecewise calculation of pixel L, a, b values
			Mat FI1 = F.mul(Ich.at(0));
			Mat FI2 = F.mul(Ich.at(1));
			Mat FI3 = F.mul(Ich.at(2));

			float sFI1 = cv::sum(FI1).val[0];
			float sFI2 = cv::sum(FI2).val[0];
			float sFI3 = cv::sum(FI3).val[0];

			float tFI1 = sFI1 / nF;
			float tFI2 = sFI2 / nF;
			float tFI3 = sFI3 / nF;

			if (i == 0 && j == 0)
				std::cout << "Multiplication Results:\n" <<
				"\tF .* Ich(0)\n" << FI1 << endl << endl <<
				"\tF .* Ich(1)\n" << FI2 << endl << endl <<
				"\tF .* Ich(2)\n" << FI3 << endl << endl <<
				"Sum Results:\n" <<
				"\tsum(FI1):" << sFI1 << endl <<
				"\tsum(FI2):" << sFI2 << endl <<
				"\tsum(FI3):" << sFI3 << endl <<
				"Total Results:\n" <<
				"\tL:" << tFI1 << endl <<
				"\ta:" << tFI2 << endl <<
				"\tb:" << tFI3 << endl;

			// Write to output image
			o->x = tFI1;
			o->y = tFI2;
			o->z = tFI3;
#else
			o->x = cv::sum(F.mul(Ich.at(0))).val[0] / nF;
			o->y = cv::sum(F.mul(Ich.at(1))).val[0] / nF;
			o->z = cv::sum(F.mul(Ich.at(2))).val[0] / nF;
#endif
		}
	}

	// Convert back to BGR to allow for writing to file
	cv::cvtColor(*out, *out, CV_Lab2BGR);
	
	// Don't forget to free memory
	tmp.deallocate();
	I.deallocate();
	dL.deallocate();
	da.deallocate();
	db.deallocate();
	F.deallocate();
	Ich.at(0).deallocate();
	Ich.at(1).deallocate();
	Ich.at(2).deallocate();
}

void BilateralFilter::ApplyFilterGray(Mat * img, Mat * out)
{
	int rowBound = img->rows - 1;
	int colBound = img->cols - 1;

	Mat tmp = *img;

	int i, j;
	Mat I, F;
	for (i = 0; i <= rowBound; i++)
	{
		for (j = 0; j <= colBound; j++)
		{
#ifdef _DEBUG
			// Get local region taking into account the bounds of the image
			int iMin = std::max<int>(i - w, 0);
			int iMax = std::min<int>(i + w + 1, rowBound);
			int jMin = std::max<int>(j - w, 0);
			int jMax = std::min<int>(j + w + 1, colBound);

			// Functionally equivalent to iMin:iMax in matlab
			cv::Range iMinMax(iMin, iMax);
			cv::Range jMinMax(jMin, jMax);
#else
			// Used to get a range of values from the tmp matrix
			cv::Range iMinMax(std::max<int>(i - w, 0), std::min<int>(i + w + 1, rowBound));
			cv::Range jMinMax(std::max<int>(j - w, 0), std::min<int>(j + w + 1, colBound));
#endif

			I = tmp(iMinMax, jMinMax);

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "I mat: " << endl <<
					I << endl << endl;
			}
#endif

			/* Compute Gaussian intensity weights. */
			Mat H = (I - img->at<float>(i, j));
			H = -H.mul(H);
			H = H / (2 * std::powf(r, 2.f));
			cv::exp(H, H);

#ifdef _DEBUG
			if (i == 0 && j == 0)
			{
				std::cout << "H mat: " << endl <<
					H << endl << endl;
			}

			cv::Range a = iMinMax - i + w;
			cv::Range b = jMinMax - j + w;

			Mat Gsub = G(a, b);

			if (i == 0 && j == 0)
			{
				std::cout << "Gsub is from i = " << a.start << " to " << a.end << endl <<
					"\tj = " << b.start << " to " << b.end << endl <<
					"Gsub:\n" << Gsub << endl << endl;
			}

			/* Calculating the response */
			cv::multiply(H, Gsub, F);

			if (i == 0 && j == 0)
			{
				std::cout << "The response matrix F: \n" <<
					F << endl << "\tWith " << F.channels() << " channels.\n";
			}

			Mat mult = F.mul(I);

			float sumNum = cv::sum(mult).val[0];
			float sumDenom = cv::sum(F).val[0];

			if (i == 0 && j == 0)
			{
				std::cout << "The matrix multiplication gave:\n" <<
					mult << endl << endl <<
					"with the numerator = " << sumNum << " and denominator = " << sumDenom << endl;
			}

			float* value = out->ptr<float>(i, j);

			*value = sumNum / sumDenom;

#else
			/* Calculate the response matrix */
			cv::multiply(H, G(iMinMax - i + w, jMinMax - j + w), F);
			/* Calculate pixel value */
			*out->ptr<float>(i, j) = cv::sum(F.mul(I)).val[0] / cv::sum(F).val[0];
#endif
		}
#ifdef _DEBUG
		// Displays row after computing
		imshow("Output", *out);
		waitKey(1);
#endif
	}
}
