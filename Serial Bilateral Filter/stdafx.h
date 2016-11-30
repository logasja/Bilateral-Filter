#pragma once

#define NOMINMAX			// Use standard library min/max

#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

// My headers
#include "BilateralFilter.h"
#include "../const.h"

// opencv requires debug libraries when running in debug mode
#if _DEBUG
#pragma comment(lib, "opencv_core310d")
#pragma comment(lib, "opencv_imgproc310d")
#pragma comment(lib, "opencv_highgui310d")
#pragma comment(lib, "opencv_ml310d")
#pragma comment(lib, "opencv_video310d")
#pragma comment(lib, "opencv_features2d310d")
#pragma comment(lib, "opencv_calib3d310d")
#pragma comment(lib, "opencv_objdetect310d")
#pragma comment(lib, "opencv_flann310d")
#pragma comment(lib, "opencv_photo310d")
#pragma comment(lib, "opencv_shape310d")
#pragma comment(lib, "opencv_calib3d310d")
#pragma comment(lib, "opencv_imgcodecs310d")
#else
#pragma comment(lib, "opencv_core310")
#pragma comment(lib, "opencv_imgproc310")
#pragma comment(lib, "opencv_highgui310")
#pragma comment(lib, "opencv_ml310")
#pragma comment(lib, "opencv_video310")
#pragma comment(lib, "opencv_features2d310")
#pragma comment(lib, "opencv_calib3d310")
#pragma comment(lib, "opencv_objdetect310")
#pragma comment(lib, "opencv_flann310")
#pragma comment(lib, "opencv_photo310")
#pragma comment(lib, "opencv_shape310")
#pragma comment(lib, "opencv_calib3d310")
#pragma comment(lib, "opencv_imgcodecs310")
#endif