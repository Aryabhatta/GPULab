/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: convolution
* file:    convolution_cpu.cpp
*
* 
\******* PLEASE ENTER YOUR CORRECT STUDENT LOGIN, NAME AND ID BELOW *********/
const char* cpu_studentLogin = "p110";
const char* cpu_studentName  = "Shrikant Vinchurkar";
const int   cpu_studentID    = 03636145;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* makeGaussianKernel
* cpu_convolutionGrayImage
*
\****************************************************************************/


#include "convolution_cpu.h"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <iostream>


const char* cpu_getStudentLogin() { return cpu_studentLogin; };
const char* cpu_getStudentName()  { return cpu_studentName; };
int         cpu_getStudentID()    { return cpu_studentID; };
bool cpu_checkStudentData() { return strcmp(cpu_studentLogin, "p010") != 0 && strcmp(cpu_studentName, "John Doe") != 0 && cpu_studentID != 1234567; };



float *makeGaussianKernel(int kRadiusX, int kRadiusY, float sigmaX, float sigmaY)
{
  const int kWidth  = (kRadiusX << 1) + 1;
  const int kHeight = (kRadiusY << 1) + 1;
  const int kernelSize = kWidth*kHeight;
  float *kernel = new float[kernelSize];

  // Calculating the mu Values
  float muX = kWidth / 2 ; // kWidth & kHeight are odd
  float muY = kHeight / 2;

  int i = 0, j = 0;
  float xExpo = 0.0, yExpo = 0.0;
  float sumKernel = 0.0f, temp = 0.0f;

  // Calculating sum for normalization

  // ### build a normalized gaussian kernel ###
  for( i = 0; i < kWidth; i++ )
  {
	  for( j = 0; j < kHeight; j++ )
	  {
		  xExpo = 0.0;
	  	  xExpo = -0.5 * (( i - muX )/ sigmaX ) * (( i - muX )/ sigmaX );

	  	  yExpo = 0.0;
		  yExpo = -0.5 * (( j - muY )/ sigmaY ) * (( j - muY )/ sigmaY );

		  temp = exp( xExpo ) * exp( yExpo );
		  kernel[ i* kWidth + j ] = temp;
		  sumKernel = sumKernel + temp;
	  }
  }

  // normalize
  for( i = 0; i< kWidth * kHeight; i++)
  {
	  kernel[ i ]  = kernel[ i ] / sumKernel;
  }

  return kernel;
}


// the following kernel normalization is only for displaying purposes with openCV
float *cpu_normalizeGaussianKernel_cv(const float *kernel, float *kernelImgdata, int kRadiusX, int kRadiusY, int step)
{
  int i,j;
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  // the kernel is assumed to be decreasing when going outwards from the center and rotational symmetric
  // for normalization we subtract the minimum and divide by the maximum
  const float minimum = kernel[0];                                      // top left pixel is assumed to contain the minimum kernel value
  const float maximum = kernel[2*kRadiusX*kRadiusY+kRadiusX+kRadiusY] - minimum;  // center pixel is assumed to contain the maximum value

  for (i=0;i<kHeight;i++) 
	  for (j=0;j<kWidth;j++) 
		  kernelImgdata[i*step+j] = (kernel[i*kWidth+j]-minimum) / maximum;

  return kernelImgdata;
}


// mode 0: standard cpu implementation of a convolution
void cpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage, int iWidth, int iHeight, int kRadiusX, int kRadiusY)
{
  int i = 0, j = 0, k = 0, l = 0;

  const int kWidth  = (kRadiusX << 1) + 1;
  const int kHeight = (kRadiusY << 1) + 1;

  // initializing output image
  for( i = 0; i < iWidth ; i++)
  {
	  for( j = 0; j< iHeight; j++)
	  {
		  outputImage[ j * iWidth + i] = 0.0;
	  }
  }

  int IndX = 0, IndY = 0;
  int shiftX = 0, shiftY =0;

  // ### implement a convolution ### 
  for( i = 0; i < iWidth ; i++)
  {
	for( j = 0; j < iHeight; j++)
	{
		// calculations independent of variables in inner loop DONE HERE
		shiftX = i - kRadiusX;
		shiftY = j - kRadiusY;

      for( k = 0; k < kWidth; k++ )
	  {
		  for( l = 0; l < kHeight; l++)
		  {
			  IndX = shiftX + k;
			  IndY = shiftY + l;

			// handling boundary conditions
			if(IndX < 0 ){		IndX = 0;			}
			if(IndY < 0 ){		IndY = 0;			}

			if( IndX > iWidth -1) {			IndX = iWidth - 1;			}
			if( IndY > iHeight -1){			IndY = iHeight- 1;			}

			outputImage[ j * iWidth + i] += inputImage[ IndY * iWidth + IndX ] * \
					                        kernel[ l * kWidth + k];
		  }
	    }
	  }
	}
 }




void cpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage, int iWidth, int iHeight, int kRadiusX, int kRadiusY)
{
  // for separated red, green and blue channels a convolution is straightforward by using the gray value convolution for each color channel
  const int imgSize = iWidth*iHeight;
  cpu_convolutionGrayImage(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY);
  cpu_convolutionGrayImage(inputImage+imgSize, kernel, outputImage+imgSize, iWidth, iHeight, kRadiusX, kRadiusY);
  cpu_convolutionGrayImage(inputImage+(imgSize<<1), kernel, outputImage+(imgSize<<1), iWidth, iHeight, kRadiusX, kRadiusY);
}



void cpu_convolutionBenchmark(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                              int numKernelTestCalls)
{
  clock_t startTime, endTime;
  float fps;

  startTime = clock();

  for(int c=0;c<numKernelTestCalls;c++)
    cpu_convolutionRGB(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY);

  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - cpu version\n";
}
