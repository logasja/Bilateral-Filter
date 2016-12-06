# Bilateral-Filter-Rocket

## Required
- [CUDA 8.0 Libraries](https://developer.nvidia.com/cuda-downloads)
- [OpenCV 3.1 Libraries](http://opencv.org/downloads.html)
- [Visual Studio 2015](https://www.visualstudio.com/vs/)

## Instructions
- General Build
  - Either build the OpenCV package in C:\DEV\opencv or be aware that the library and include locations need to be changed
  - If change is necessary follow these steps:
    1. Under the solution explorer right-click the Bilateral Filter project and select "Properties".
    2. Now put the path to the OpenCV include folder in *Configuration Properties -> C/C++ -> Additional Include Directories*
    3. Similarly, put the path to the OpenCV library files in *Configuration Properties -> Linker -> Additional Library Directories*
    4. Finally, add the OpenCV bin folder to your path.
- Debug Build
  - The debug build executes must more slowly however for the serial implementation, each row is shown as it is computed.
- Release Build
  - This provides much faster run times as some functions were optimized and there are no debug messages.