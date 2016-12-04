#include "BilateralFilter.h"

#include "device_functions.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define CHECK_CUDA_ERROR(call, msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

#define PI 3.14159265

// An implementation of 1d gaussian calcualtion using optimized device functions
inline __device__
float gaussian1d_gpu(float x, float sigma)
{
	float variance = powf(sigma, 2);
	float power = pow(x, 2); //this doesnt work for __powf(x,2.0f) for some reason
	float exponent = -power / (2 * variance);
	return expf(exponent) / sqrt(2 * PI * variance);
}

// Kernel code for bilateral filtering of color image
__global__ 
void color_bilateral_filter(const float* input,
							const float* kernel,
							const float r,
							const int w,
							const int width, 
							const int height,
							const int step,
							const int gstep,
							float* output)
{
	// 2D Index of current thread
	const int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

	//// Only valid threads can perform memory I/O
	if ((xIdx < width) && (yIdx < height))
	{
		const char* inCharPtr = (char*)input;
		const float* srcRow = (const float*)(inCharPtr + yIdx*step);

		float3 cLab = make_float3(srcRow[xIdx], srcRow[xIdx + 1], srcRow[xIdx + 2]);

		// The normalization values that are iteratively solved for
		float3 nLab = make_float3(0.f, 0.f, 0.f);
		float3 rLab = make_float3(0.f, 0.f, 0.f);
		float3 gLab, wLab, pLab;
		for(int i = -w; i <= w; i++)
			for (int j = -w; j <= w; j++)
			{
				int x_sample = xIdx + i;
				int y_sample = yIdx + j;

				// mirror edges
				if (x_sample < 0) x_sample = -x_sample;
				if (y_sample < 0) y_sample = -y_sample;
				if (x_sample > width - 1) x_sample = width - 1 - i;
				if (y_sample > height - 1) y_sample = height - 1 - i;

				const float* aRow = (const float*)(inCharPtr + y_sample*step);
				pLab = make_float3(aRow[x_sample], aRow[x_sample + 1], aRow[x_sample + 2]);

				char* gCharPtr = (char*)kernel;
				float* gRow = (float*)(gCharPtr + i*gstep);

				float spatial = gRow[j];
				// Calcualte the range gaussian values for L, a, b values
				//	uses difference between center color and current window location
				gLab.x = gaussian1d_gpu(cLab.x - pLab.x, r);
				gLab.y = gaussian1d_gpu(cLab.y - pLab.y, r);
				gLab.z = gaussian1d_gpu(cLab.z - pLab.z, r);

				// The combined spatial and range gaussian
				wLab.x = gLab.x * spatial;
				wLab.y = gLab.y * spatial;
				wLab.z = gLab.z * spatial;

				// Add this part of the window's weight to the total normalization
				nLab.x = nLab.x + w;
				nLab.y = nLab.y + w;
				nLab.z = nLab.z + w;

				// Calcuate response
				rLab.x = rLab.x + (pLab.x * wLab.x);
				rLab.y = rLab.y + (pLab.y * wLab.y);
				rLab.z = rLab.z + (pLab.z * wLab.z);
			}

		// Normalize the response through use of the normalization value found in the loop
		rLab.x /= nLab.x;
		rLab.y /= nLab.y;
		rLab.z /= nLab.z;

		char* outCharPtr = (char*)input;
		float* outRow = (float*)(outCharPtr + yIdx*step);
		outRow[xIdx] = rLab.x;
		outRow[xIdx+1] = rLab.y;
		outRow[xIdx+2] = rLab.z;
	}
}

// Kernel code for bilateral filtering of gray image
__global__ 
void gray_bilateral_filter(const float* input,
							const float* kernel,
							const float r,
							const int w,
							const int width,
							const int height,
							const int step, 
							const int gstep,
							float* output)
{
	// 2D Index of current thread
	const int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads can perform memory I/O
	if ((xIdx < width) && (yIdx < height))
	{

	}
}

Mat BilateralFilter::ApplyFilterCUDA(Mat img)
{
#ifdef _DEBUG
	std::cout << "Data type of image:" << type2str(img.type()) << endl;
	std::cout << "Data type of G Mat:" << type2str(G.type()) << endl;
	cv::Point3_<float>* p = img.ptr<cv::Point3_<float>>(1023, 1023);
	std::cout << "image at (0,0):" << endl <<
		"\t" << p->x << endl <<
		"\t" << p->y << endl <<
		"\t" << p->z << endl;
#endif

	Mat out;

	cv::cvtColor(img, img, CV_BGR2Lab);

	const size_t bytes = img.step * img.rows;

	const size_t Gbytes = G.step * G.rows;

	float* d_input, *d_output;

	// Allocation of device memory
	CHECK_CUDA_ERROR(cudaMalloc<float>(&d_input, bytes), "CUDA Malloc Failed");
	CHECK_CUDA_ERROR(cudaMalloc<float>(&d_output, bytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	CHECK_CUDA_ERROR(cudaMemcpy(d_input, img.ptr<float>(), bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host to Device Failed");

	// Allocate kernel on device memory and copy G to it if it has not already
	if (!d_kernel)
	{
		CHECK_CUDA_ERROR(cudaMalloc<float>(&d_kernel, Gbytes), "CUDA Malloc Failed");
		// Copy data from OpenCV Mat G to device memory
		CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, G.ptr<float>(), Gbytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host to Device Failed");
	}

	// Define block size
	const dim3 block(16, 16);

	// Grid size in order to cover whole image
	const dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

	if (img.channels() > 1)
	{
		out = Mat::zeros(img.rows, img.cols, CV_32FC3);
		// Launch bilateral filter kernel for color image
		color_bilateral_filter << <grid, block >> >(d_input, d_kernel, r, w, img.cols, img.rows, img.step, G.step, d_output);
	}
	else
	{
		out = Mat::zeros(img.rows, img.cols, CV_32FC1);
		// Launch bilateral filter kernel for grayscale image
		gray_bilateral_filter << <grid, block >> >(d_input, d_kernel, r, w, img.cols, img.rows, img.step, G.step, d_output);
	}

	// Synchronize to check for kernel launch errors
	CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from the destination device memory to the OpenCV output image
	CHECK_CUDA_ERROR(cudaMemcpy(out.ptr<float>(), d_output, bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	CHECK_CUDA_ERROR(cudaFree(d_input), "CUDA Free Failed");
	CHECK_CUDA_ERROR(cudaFree(d_output), "CUDA Free Failed");

	cv::cvtColor(out, out, CV_Lab2BGR);

	return out;
}

BilateralFilter::~BilateralFilter()
{
	// Free kernel memory
	G.deallocate();

	// If the kernel had its info copied to device, free it
	if(d_kernel)
		CHECK_CUDA_ERROR(cudaFree(d_kernel), "CUDA Free Failed");
}