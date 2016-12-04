#include "BilateralFilter.h"

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

// Kernel code for bilateral filtering of color image
__global__ 
void color_bilateral_filter(const float* input,
							const float* kernel,
							const float r,
							const int w,
							const int width, 
							const int height,
							const int step,
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
		float L = srcRow[xIdx];
		float a = srcRow[xIdx + 1];
		float b = srcRow[xIdx + 2];

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


			}
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
		color_bilateral_filter << <grid, block >> >(d_input, d_kernel, r, w, img.cols, img.rows, img.step, d_output);
	}
	else
	{
		out = Mat::zeros(img.rows, img.cols, CV_32FC1);
		// Launch bilateral filter kernel for grayscale image
		gray_bilateral_filter << <grid, block >> >(d_input, d_kernel, r, w, img.cols, img.rows, img.step, d_output);
	}

	// Synchronize to check for kernel launch errors
	CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from the destination device memory to the OpenCV output image
	CHECK_CUDA_ERROR(cudaMemcpy(out.ptr<float>(), d_output, bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	CHECK_CUDA_ERROR(cudaFree(d_input), "CUDA Free Failed");
	CHECK_CUDA_ERROR(cudaFree(d_output), "CUDA Free Failed");

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