/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: convolution
* file:    convolution_gpu.cu
*
* 
\******* PLEASE ENTER YOUR CORRECT STUDENT LOGIN, NAME AND ID BELOW *********/
const char* gpu_studentLogin = "p110";
const char* gpu_studentName  = "Shrikant Vinchurkar";
const int   gpu_studentID    = 03636145;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* gpu_convolutionGrayImage_gm_d
* gpu_convolutionGrayImage_gm_cm_d
* gpu_convolutionGrayImage_sm_d
* gpu_convolutionGrayImage_sm_cm_d
* gpu_convolutionGrayImage_dsm_cm_d
* gpu_convolutionInterleavedRGB_dsm_cm_d
* gpu_convolutionInterleavedRGB_tex_cm_d
*
\****************************************************************************/


#include "convolution_gpu.cuh"

#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <iostream>


#define TEXTURE_OFFSET      0.5f  // offset for indexing textures

#define BW                  16    // block width
#define BH                  16    // block height
#define MAXKERNELRADIUS     20    // maximum allowed kernel radius

#define MAXKERNELSIZE       ( 1+2*MAXKERNELRADIUS)*( 1+2*MAXKERNELRADIUS)
#define MAXSHAREDMEMSIZE    (BW+2*MAXKERNELRADIUS)*(BH+2*MAXKERNELRADIUS)

#if (MAXSHAREDMEMSIZE > 4000)   // Note: MAXSHAREDMEMSIZE <= 4000 should hold for most graphic cards to work
  #error "This program will most likely not run properly because of insufficient shared memory, please reduce BW/BH/MAXKERNELRADIUS!"
#endif



// constant memory block on device
__constant__ float constKernel[MAXKERNELSIZE];

// texture memory and descriptor
cudaChannelFormatDesc tex_Image_desc = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_Image;

cudaChannelFormatDesc tex_Image_descF4 = cudaCreateChannelDesc<float4>();
texture<float4, 2, cudaReadModeElementType> tex_ImageF4;


const char* gpu_getStudentLogin() { return gpu_studentLogin; };
const char* gpu_getStudentName()  { return gpu_studentName; };
int         gpu_getStudentID()    { return gpu_studentID; };
bool gpu_checkStudentData() { return strcmp(gpu_studentLogin, "p010") != 0 && strcmp(gpu_studentName, "John Doe") != 0 && gpu_studentID != 1234567; };
bool gpu_checkStudentNameAndID() { return strcmp(gpu_studentName, "John Doe") != 0 && gpu_studentID != 1234567; };



//----------------------------------------------------------------------------
// Gray Image Functions
//----------------------------------------------------------------------------


// mode 1 (gray): using global memory only
__global__ void gpu_convolutionGrayImage_gm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch)
{ 
  // ### implement me ###
  int k = 0, l = 0;
  
  const int kWidth  = (kRadiusX << 1) + 1;
  const int kHeight = (kRadiusY << 1) + 1;
  
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
  // initialising output image
  if( ix < iWidth && iy < iHeight )
    outputImage[ iy * iPitch + ix] = 0.0;
  
  int IndX = 0, IndY = 0;
  int shiftX = 0, shiftY = 0;
  
  shiftX = ix - kRadiusX;
  shiftY = iy - kRadiusY;

  // ### implement a convolution ### 
  for( k = 0; k < kWidth; k++ )
  {
    for( l = 0; l < kHeight; l++)
  	{	
  		IndX = shiftX + k;
  		IndY = shiftY + l;

  		// handling boundary conditions
  		if( IndX < 0 ){			IndX = 0;  		}
  		if( IndY < 0){ 			IndY = 0;  		}
  		
  		if( IndX > iWidth -1){		IndX = iWidth - 1;  		}
  		if( IndY > iHeight -1){		IndY = iHeight- 1;	 		}

  		if( ix < iWidth && iy < iHeight ) // guards
  		  outputImage[ iy * iPitch + ix ] += inputImage[ IndY * iPitch + IndX ] * \
  			   		                         kernel[ l * kPitch + k];
  	  }
  	} //replacing kWidth by kPitch gives correct indexing in kernel
  	  // same goes as iPitch
}



// mode 2 (gray): using global memory and constant memory for kernel
__global__ void gpu_convolutionGrayImage_gm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{  
	  // ### implement me ###
	  int k = 0, l = 0;
	  
	  const int kWidth  = (kRadiusX << 1) + 1;
	  const int kHeight = (kRadiusY << 1) + 1;
	  
	  int ix = threadIdx.x + blockIdx.x * blockDim.x;
	  int iy = threadIdx.y + blockIdx.y * blockDim.y;
	    
	  // initialising output image
	  if( ix < iWidth && iy < iHeight )
	    outputImage[ iy * iPitch + ix] = 0.0;
	  
	  int IndX = 0, IndY = 0;
	  int shiftX = 0, shiftY = 0;
	    
	  shiftX = ix - kRadiusX;
	  shiftY = iy - kRadiusY;

	  // ### implement a convolution ### 
	  for( k = 0; k < kWidth; k++ )
	  {
	    for( l = 0; l < kHeight; l++)
	  	{	
	  		IndX = shiftX + k;
	  		IndY = shiftY + l;

	  		// handling boundary conditions
	  		if( IndX < 0 ){			IndX = 0;	  		}
	  		if( IndY < 0) {			IndY = 0;	  		}
	  		
	  		if( IndX > iWidth -1){		IndX = iWidth - 1;	  		}
	  		if( IndY > iHeight -1){		IndY = iHeight- 1;	  		}

	  		if( ix < iWidth && iy < iHeight ) // guards
	  		  outputImage[ iy * iPitch + ix ] += inputImage[ IndY * iPitch + IndX ] * \
	  					                         constKernel[ l * kWidth + k];
	  	  }
	  	} // need nt replace kWidth by kPitch here   
}


// mode 3 (gray): using shared memory for image and globel memory for kernel access
__global__ void gpu_convolutionGrayImage_sm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch)
{
  // make use of the constant MAXSHAREDMEMSIZE in order to define the shared memory size

  // ### implement me ###	
  __shared__ float input[ MAXSHAREDMEMSIZE ];
  
  const int kWidth  = (kRadiusX << 1) + 1;
  const int kHeight = (kRadiusY << 1) + 1;
    
  // Copy input array equal to blocksize + 2 * kRadius  
  // size of block to be copied = (BW + 2 * kRadiusX)  * (BH + 2* kRadiusY)
  int sWidth  = blockDim.x + 2* kRadiusX;
  int sHeight = blockDim.y + 2* kRadiusY;
    
  int InputSize = sWidth * sHeight;
  int blockSize = blockDim.x * blockDim.y;

  int NoLoops = 0;
  if( InputSize % blockSize == 0)
  {
    NoLoops = InputSize / blockSize;
  }
  else
  {
    NoLoops = (InputSize/blockSize) + 1;
  }
  
  int ix = threadIdx.x + blockIdx.x * blockDim.x; // global indexes for img (input & output) 
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  
  // for example 16*16 blocksize & 16*16 no of blocks,ix & iy value from 0...255
  // here,though blockSize=16*16, #blocks_inX = iWidth/16,#blocks_inY=iHeight/16
   
  // shifting global indices to align it to required start point
  int start_X = (blockIdx.x * blockDim.x - kRadiusX);
  int start_Y = (blockIdx.y * blockDim.y - kRadiusY);
  
  int IndX = 0;
  int IndY = 0;

  // aligning locations equal to blockSize in input (without offset)
  int inpStart = threadIdx.x + threadIdx.y * blockDim.x;
  int inputIdx = 0;
  int inputOffset = 0; // initial offset

  for( int i = 0; i < NoLoops ; i++ )
  {    
    if( inputIdx < InputSize ) // guard
    {
    	inputOffset = i * blockSize; // 0 offset in first loop, +blockSize in next each
   	    inputIdx = inpStart + inputOffset; // final index for input

   	    // aligning global index as per read sequence in Input
        IndX = start_X  + inputIdx % sWidth;
        IndY = start_Y  + inputIdx / sHeight;

	    // guards
        if ( IndX < 0 ) { IndX  = 0; }
        if ( IndY < 0 ) { IndY  = 0; }

        if ( IndX > iWidth -1 ) { IndX  = iWidth -1; }
        if ( IndY > iHeight-1 ) { IndY  = iHeight-1; }

        input[ inputIdx ] = inputImage[ IndY * iPitch + IndX ];
    }
  }
  
  // syncthreads
  __syncthreads();

  // initialising output image
  if( ix < iWidth && iy < iHeight) //guards
    outputImage[ iy * iPitch + ix ] = 0.0; // indexing globally

  int tx = threadIdx.x;
  int ty = threadIdx.y;
   
  int index = 0, k = 0, l = 0;
    
 // ### implement a convolution ###
  for( k = 0; k < kWidth; k++ )
  {
	for( l = 0; l < kHeight; l++)
	{
		// index for input array
	   index = (ty+l) * sWidth + (tx+k);
	   
	   if( ix < iWidth &&  iy < iHeight) //guards
	     if( index < InputSize )
	       outputImage[ iy * iPitch + ix ] += input[ index ] * \
		   								      kernel[ l * kPitch + k];
	}
  }  
}


// mode 4 (gray): using shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_sm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{
  // make use of the constant MAXSHAREDMEMSIZE in order to define the shared memory size

  // ### implement me ###
	__shared__ float input[ MAXSHAREDMEMSIZE ];
	  
	  const int kWidth  = (kRadiusX << 1) + 1;
	  const int kHeight = (kRadiusY << 1) + 1;
	    
	  // Copy input array equal to blocksize + 2 * kRadius  
	  // size of block to be copied = (BW + 2 * kRadiusX)  * (BH + 2* kRadiusY)
	  int sWidth  = blockDim.x + 2* kRadiusX;
	  int sHeight = blockDim.y + 2* kRadiusY;
	    
	  int InputSize = sWidth * sHeight;
	  int blockSize = blockDim.x * blockDim.y;

	  int NoLoops = 0;
	  if( InputSize % blockSize == 0)
	  {
	    NoLoops = InputSize / blockSize;
	  }
	  else
	  {
	    NoLoops = (InputSize/blockSize) + 1;
	  }
	  
	  int ix = threadIdx.x + blockIdx.x * blockDim.x; // global indexes for img (input & output) 
	  int iy = threadIdx.y + blockIdx.y * blockDim.y;
	  
	  // for example 16*16 blocksize & 16*16 no of blocks,ix & iy value from 0...255
	  // here,though blockSize=16*16, #blocks_inX = iWidth/16,#blocks_inY=iHeight/16
	   
	  // shifting global indices to align it to requires start point
	  int start_X = (blockIdx.x * blockDim.x - kRadiusX);
	  int start_Y = (blockIdx.y * blockDim.y - kRadiusY);
	  
	  int IndX = 0;
	  int IndY = 0;

	  // aligning locations equal to blockSize in input (without offset)
	  int inpStart = threadIdx.x + threadIdx.y * blockDim.x;
	  int inputIdx = 0;
	  int inputOffset = 0; // initial offset

	  for( int i = 0; i < NoLoops ; i++ )
	  {    
	    if( inputIdx < InputSize ) // guard
	    {
	    	inputOffset = i * blockSize; // 0 in first loop, +blockSize in next each
	   	    inputIdx = inpStart + inputOffset; // final index for input

	   	    // aligning global index as per read sequence in Input
	        IndX = start_X  + inputIdx % sWidth;
	        IndY = start_Y  + inputIdx / sHeight;

		    // guards
	        if ( IndX < 0 ) { IndX  = 0; }
	        if ( IndY < 0 ) { IndY  = 0; }

	        if ( IndX > iWidth -1 ) { IndX  = iWidth -1; }
	        if ( IndY > iHeight-1 ) { IndY  = iHeight-1; }

	        input[ inputIdx ] = inputImage[ IndY * iPitch + IndX ];
	    }
	  }
	  
	  // syncthreads
	  __syncthreads();

	  // initialising output image
	  if( ix < iWidth && iy < iHeight) //guards
	    outputImage[ iy * iPitch + ix ] = 0.0; // indexing globally

	  // anchoring points in array input
	  int tx = threadIdx.x;
	  int ty = threadIdx.y;
	   
	  int index = 0, k = 0, l = 0;
	    
	 // ### implement a convolution ###
	  for( k = 0; k < kWidth; k++ )
	  {
		for( l = 0; l < kHeight; l++)
		{
			// index for input array
		   index = (ty+l) * sWidth + (tx+k);
		   
		   if( ix < iWidth &&  iy < iHeight) //guards
		     if( index < InputSize )
		       outputImage[ iy * iPitch + ix ] += input[ index ] * \
			   								      constKernel[ l * kWidth + k];
		}
	  }  
} 


// mode 5 (gray): using dynamically allocated shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_dsm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{

  // ### implement me ###  
	extern __shared__ float input[];
	  
	const int kWidth  = (kRadiusX << 1) + 1;
	const int kHeight = (kRadiusY << 1) + 1;
	    
	  // Copy input array equal to blocksize + 2 * kRadius  
	  // size of block to be copied = (BW + 2 * kRadiusX)  * (BH + 2* kRadiusY)
	  int sWidth  = blockDim.x + 2* kRadiusX;
	  int sHeight = blockDim.y + 2* kRadiusY;
	    
	  int InputSize = sWidth * sHeight;
	  int blockSize = blockDim.x * blockDim.y;

	  int NoLoops = 0;
	  if( InputSize % blockSize == 0)
	  {
	    NoLoops = InputSize / blockSize;
	  }
	  else
	  {
	    NoLoops = (InputSize/blockSize) + 1;
	  }
	  
	  int ix = threadIdx.x + blockIdx.x * blockDim.x; // global indexes for img (input & output) 
	  int iy = threadIdx.y + blockIdx.y * blockDim.y;
	  
	  // for example 16*16 blocksize & 16*16 no of blocks,ix & iy value from 0...255
	  // here,though blockSize=16*16, #blocks_inX = iWidth/16,#blocks_inY=iHeight/16
	   
	  // shifting global indices to align it to requires start point
	  int start_X = (blockIdx.x * blockDim.x - kRadiusX);
	  int start_Y = (blockIdx.y * blockDim.y - kRadiusY);
	  
	  int IndX = 0;
	  int IndY = 0;

	  // aligning locations equal to blockSize in input (without offset)
	  int inpStart = threadIdx.x + threadIdx.y * blockDim.x;
	  int inputIdx = 0;
	  int inputOffset = 0; // initial offset

	  for( int i = 0; i < NoLoops ; i++ )
	  {    
	    if( inputIdx < InputSize ) // guard
	    {
	    	inputOffset = i * blockSize; // 0 in first loop, +blockSize in next each
	   	    inputIdx = inpStart + inputOffset; // final index for input

	   	    // aligning global index as per read sequence in Input
	        IndX = start_X  + inputIdx % sWidth;
	        IndY = start_Y  + inputIdx / sHeight;

		    // guards
	        if ( IndX < 0 ) { IndX  = 0; }
	        if ( IndY < 0 ) { IndY  = 0; }

	        if ( IndX > iWidth -1 ) { IndX  = iWidth -1; }
	        if ( IndY > iHeight-1 ) { IndY  = iHeight-1; }

	        input[ inputIdx ] = inputImage[ IndY * iPitch + IndX ];
	    }
	  }
	  
	  // syncthreads
	  __syncthreads();

	  // initialising output image
	  if( ix < iWidth && iy < iHeight) //guards
	    outputImage[ iy * iPitch + ix ] = 0.0f; // indexing globally

	  // anchoring points in array input
	  int tx = threadIdx.x;
	  int ty = threadIdx.y;
	   
	  int index = 0, k = 0, l = 0;
	    
	 // ### implement a convolution ###
	  for( k = 0; k < kWidth; k++ )
	  {
		for( l = 0; l < kHeight; l++)
		{
			// index for input array
		   index = (ty+l) * sWidth + (tx+k);
		   
		   if( ix < iWidth &&  iy < iHeight) //guards
		     if( index < InputSize )
		       outputImage[ iy * iPitch + ix ] += input[ index ] * \
			   								      constKernel[ l * kWidth + k];
		}
	  }
} 




// mode 6 (gray): using texture memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_tex_cm_d(float *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY,
    size_t iPitch)
{
	// updated code from Martin's last mail
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iWidth || y >= iHeight) return;

	const float xx = (float)(x) + TEXTURE_OFFSET;
	const float yy = (float)(y) + TEXTURE_OFFSET;
	const int kWidth  = (kRadiusX<<1) + 1;

	float value = 0.0f;

	for (int yk = -kRadiusY; yk <= kRadiusY; yk++)
	  for (int xk = -kRadiusX; xk <= kRadiusX; xk++)
	    value += tex2D(tex_Image, xx-xk, yy-yk) * constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX];

	outputImage[y*iPitch + x] = value;
}



void gpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage, 
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  size_t iPitchBytes, kPitchBytes;
  size_t iPitch, kPitch;
  float *d_inputImage;
  float *d_kernel;
  float *d_outputImage;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);
  
  // allocate device memory
  cutilSafeCall( cudaMallocPitch( (void**)&d_inputImage, &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&d_outputImage, &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&d_kernel, &kPitchBytes, kWidth*sizeof(float), kHeight ) );  
  
  iPitch = iPitchBytes/sizeof(float);
  kPitch = kPitchBytes/sizeof(float);
  
  //std::cout << "iPitchBytes=" << iPitchBytes << " iPitch=" << iPitch << " kPitchBytes=" << kPitchBytes << " kPitch=" << kPitch << std::endl;  
  
  cutilSafeCall( cudaMemcpy2D(d_inputImage, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy2D(d_kernel, kPitchBytes, kernel, kWidth*sizeof(float), kWidth*sizeof(float), kHeight, cudaMemcpyHostToDevice) );

  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemory(d_inputImage, iWidth, iHeight, iPitchBytes);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  /*std::cout << "ImageWidth=" << iWidth << " ImageHeight=" << iHeight << std::endl;
  std::cout << "blockWidth=" << BW << " blockHeight=" << BH << std::endl;
  std::cout << "gridSize.x=" << gridSize.x << " gridSize.y=" << gridSize.y << std::endl;*/ 

  // invoke the kernel of your choice here
  const int smSize =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float);  

  switch(mode) {
    case 1:
      gpu_convolutionGrayImage_gm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
      break;
    case 2:
      gpu_convolutionGrayImage_gm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 3:
      gpu_convolutionGrayImage_sm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
      break;
    case 4:
      gpu_convolutionGrayImage_sm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 5:
      gpu_convolutionGrayImage_dsm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 6:
      gpu_convolutionGrayImage_tex_cm_d<<<gridSize,blockSize>>>(d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    default:
      std::cout << "gpu_convolutionGrayImage() Warning: mode " << mode << " is not supported." << std::endl;
  }

  cutilSafeCall( cudaThreadSynchronize() );
  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), d_outputImage, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );
  

  // free memory
  gpu_unbindTextureMemory();
  cutilSafeCall( cudaFree(d_inputImage) );
  cutilSafeCall( cudaFree(d_outputImage) );
  cutilSafeCall( cudaFree(d_kernel) );
}




//----------------------------------------------------------------------------
// RGB Image Functions (for separated color channels)
//----------------------------------------------------------------------------



void gpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage, 
                        int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  const int imgSize = iWidth*iHeight;
  gpu_convolutionGrayImage(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY, mode);
  gpu_convolutionGrayImage(inputImage+imgSize, kernel, outputImage+imgSize, iWidth, iHeight, kRadiusX, kRadiusY, mode);
  gpu_convolutionGrayImage(inputImage+(imgSize<<1), kernel, outputImage+(imgSize<<1), iWidth, iHeight, kRadiusX, kRadiusY, mode);
}


//----------------------------------------------------------------------------
// RGB Image Functions (for interleaved color channels)
//----------------------------------------------------------------------------


// mode 5 (interleaved): using dynamically allocated shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionInterleavedRGB_dsm_cm_d(const float3 *inputImage, float3 *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitchBytes)
{
  extern __shared__ float3 inputF3[];
	
  const int x = blockIdx.x * blockDim.x + threadIdx.x;// global indexes for img (input & output)
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  float3 value = make_float3(0.0f, 0.0f, 0.0f);

  // ### implement me ###	  
  const int kWidth  = (kRadiusX << 1) + 1;
  const int kHeight = (kRadiusY << 1) + 1;
	    
  // Copy input array equal to blocksize + 2 * kRadius  
  // size of block to be copied = (BW + 2 * kRadiusX)  * (BH + 2* kRadiusY)
  int sWidth  = blockDim.x + 2 * kRadiusX;
  int sHeight = blockDim.y + 2 * kRadiusY;
	    
  int InputSize = sWidth * sHeight;
  int blockSize = blockDim.x * blockDim.y;

  int NoLoops = 0;
  if( InputSize % blockSize == 0)
  {
    NoLoops = InputSize / blockSize;
  }
  else
  {
	NoLoops = (InputSize/blockSize) + 1;
  }
	   
	// shifting global indices to align it to requires start point
	int start_X = (blockIdx.x * blockDim.x - kRadiusX);
	int start_Y = (blockIdx.y * blockDim.y - kRadiusY);
	  
	int IndX = 0;
	int IndY = 0;

	// aligning locations equal to blockSize in input (without offset)
	int inpStart = threadIdx.x + threadIdx.y * blockDim.x;
	int inputIdx = 0;
	int inputOffset = 0; // initial offset

	for( int i = 0; i < NoLoops ; i++ )
	{    
		if( inputIdx < InputSize ) // guard
	    {
	    	inputOffset = i * blockSize; // 0 in first loop, +blockSize in next each
	   	    inputIdx = inpStart + inputOffset; // final index for input

	   	    // aligning global index as per read sequence in Input
	        IndX = start_X  + inputIdx % sWidth;
	        IndY = start_Y  + inputIdx / sHeight;

		    // guards
	        if ( IndX < 0 ) { IndX  = 0; }
	        if ( IndY < 0 ) { IndY  = 0; }

	        if ( IndX > iWidth -1 ) { IndX  = iWidth -1; }
	        if ( IndY > iHeight-1 ) { IndY  = iHeight-1; }

	        inputF3[ inputIdx ] = *((float3*)(((char*)inputImage) + IndY*iPitchBytes) + IndX); 
	    }
	}
	  
	  // syncthreads
	  __syncthreads();

	  // initialising output image
	  if( x < iWidth && y < iHeight) //guards
		  *((float3*)(((char*)outputImage) + y*iPitchBytes) + x) = value; // indexing globally

	  // anchoring points in array input
	  int tx = threadIdx.x;
	  int ty = threadIdx.y;
	   
	  int index = 0, k = 0, l = 0;
	    
	 // ### implement a convolution ###
	  for( k = 0; k < kWidth; k++ )
	  {
		for( l = 0; l < kHeight; l++)
		{
			// index for input array
		   index = (ty+l) * sWidth + (tx+k);
		   	   
		     if( index < InputSize )
		     {
		    	 value.x += inputF3[ index ].x * constKernel[ l * kWidth + k];
		    	 value.y += inputF3[ index ].y * constKernel[ l * kWidth + k];
		    	 value.z += inputF3[ index ].z * constKernel[ l * kWidth + k];
		     }
		}
	  }
	  if( x < iWidth &&  y < iHeight) //guards
	  {
		  *((float3*)(((char*)outputImage) + y*iPitchBytes) + x) = value;
	  }
} 





__global__ void gpu_ImageFloat3ToFloat4_d(const float3 *inputImage, float4 *outputImage, int iWidth, int iHeight, size_t iPitchBytes, size_t oPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= iWidth || y >= iHeight) return;

  float3 rgb = *((float3*)((char*)inputImage + y*iPitchBytes)+x);
  *((float4*)(((char*)outputImage) + y*oPitchBytes)+ x) = make_float4(rgb.x, rgb.y, rgb.z, 0.0f);
}




// mode 6 (interleaved): using texture memory for image and constant memory for kernel access
__global__ void gpu_convolutionInterleavedRGB_tex_cm_d(float3 *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY, size_t oPitchBytes)
{

  // ### implement me ### 
	// updated code from Martin's last mail
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iWidth || y >= iHeight) return;

	const float xx = (float)(x) + TEXTURE_OFFSET;
	const float yy = (float)(y) + TEXTURE_OFFSET;
	const int kWidth  = (kRadiusX<<1) + 1;

	float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float3 imgValue = make_float3(0.0f, 0.0f, 0.0f);
	
	for (int yk = -kRadiusY; yk <= kRadiusY; yk++)
	  for (int xk = -kRadiusX; xk <= kRadiusX; xk++)
	  {
		value = tex2D(tex_ImageF4, xx-xk, yy-yk);
		imgValue.x += value.x * constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX];
	    imgValue.y += value.y * constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX];
	    imgValue.z += value.z * constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX];
	  }
	
	*((float3*)(((char*)outputImage) + y*oPitchBytes) + x) = imgValue;
}



void gpu_convolutionInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                   int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  size_t iPitchBytesF3, iPitchBytesF4;
  float3 *d_inputImageF3, *d_outputImageF3;
  float4 *d_inputImageF4;
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF4), &iPitchBytesF4, iWidth*sizeof(float4), iHeight ) );

  cutilSafeCall( cudaMemcpy2D(d_inputImageF3, iPitchBytesF3, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );

  dim3 blockSize(BW,BH);
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSizeF3 =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float3);
  
  // convert image from float3* to float4*
  gpu_ImageFloat3ToFloat4_d<<<gridSize, blockSize>>>(d_inputImageF3, d_inputImageF4, iWidth, iHeight, iPitchBytesF3, iPitchBytesF4);
  
  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemoryF4(d_inputImageF4, iWidth, iHeight, iPitchBytesF4);


  switch(mode) {
    case 1:
    case 2:
    case 3:
    case 4:
      std::cout << "gpu_convolutionInterleavedRGB() Warning: mode " << mode << " is not supported." << std::endl;
      break;
    case 5:
      gpu_convolutionInterleavedRGB_dsm_cm_d<<<gridSize,blockSize,smSizeF3>>>(d_inputImageF3, d_outputImageF3,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
      break;
    case 6:
      gpu_convolutionInterleavedRGB_tex_cm_d<<<gridSize,blockSize>>>(d_outputImageF3,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
      break;
    default:
      std::cout << "gpu_convolutionInterleavedRGB() Warning: mode " << mode << " is not supported." << std::endl;
  }

  cutilSafeCall( cudaThreadSynchronize() );
  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), d_outputImageF3, iPitchBytesF3, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );


  // free memory
  gpu_unbindTextureMemoryF4();
  cutilSafeCall( cudaFree(d_inputImageF4) );
  cutilSafeCall( cudaFree(d_inputImageF3) );
  cutilSafeCall( cudaFree(d_outputImageF3) );
}












//----------------------------------------------------------------------------
// Benchmark Functions
//----------------------------------------------------------------------------




void gpu_convolutionKernelBenchmarkGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                                             int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                             int numKernelTestCalls)
{
  size_t iPitchBytes, kPitchBytes;
  size_t iPitch, kPitch;
  clock_t startTime, endTime;
  float *d_inputImage, *d_kernel, *d_outputImage;
  float fps;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSize =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float);

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImage), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImage), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_kernel), &kPitchBytes, kWidth*sizeof(float), kHeight ) );   
  iPitch = iPitchBytes/sizeof(float);
  kPitch = kPitchBytes/sizeof(float);
  
  cutilSafeCall( cudaMemcpy2D(d_inputImage, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy2D(d_kernel, kPitchBytes, kernel, kWidth*sizeof(float), kWidth*sizeof(float), kHeight, cudaMemcpyHostToDevice) );

  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemory(d_inputImage, iWidth, iHeight, iPitchBytes);

  // --- global memory only ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_gm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - global memory only\n";
  
  
  // --- global memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_gm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - global memory for image & constant memory for kernel access\n";


  // --- shared memory for image and global memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_sm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - shared memory for image & global memory for kernel access\n";

  
  // --- shared memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_sm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - shared memory for image & constant memory for kernel access\n";


   // --- shared memory for image and constant memory for kernel access ---  
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_dsm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - dyn. shared memory for image & const memory for kernel access\n";



  // --- texture memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_tex_cm_d<<<gridSize,blockSize>>>(d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  std::cout << fps << " fps - texture memory for image & const memory for kernel access\n";


  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), d_outputImage, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );

  // free memory
  gpu_unbindTextureMemory();
  cutilSafeCall( cudaFree(d_inputImage) );
  cutilSafeCall( cudaFree(d_outputImage) );
  cutilSafeCall( cudaFree(d_kernel) );
}



void gpu_convolutionKernelBenchmarkInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                                  int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                                  int numKernelTestCalls)
{
  size_t iPitchBytesF3, iPitchBytesF4;
  clock_t startTime, endTime;
  float3 *d_inputImageF3, *d_outputImageF3;
  float4 *d_inputImageF4;
  float fps;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSizeF3 =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float3);

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF4), &iPitchBytesF4, iWidth*sizeof(float4), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );   
  cutilSafeCall( cudaMemcpy2D(d_inputImageF3, iPitchBytesF3, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );

  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  
  
  // --- shared memory for interleaved image and constant memory for kernel access ---  
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionInterleavedRGB_dsm_cm_d<<<gridSize,blockSize,smSizeF3>>>(d_inputImageF3, d_outputImageF3, iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC * 3;
  std::cout << fps << " fps - dyn. shared mem for interleaved img & const mem for kernel\n";


  // --- texture memory for interleaved image and constant memory for kernel access ---
  gpu_ImageFloat3ToFloat4_d<<<gridSize, blockSize>>>(d_inputImageF3, d_inputImageF4, iWidth, iHeight, iPitchBytesF3, iPitchBytesF4);
  gpu_bindTextureMemoryF4(d_inputImageF4, iWidth, iHeight, iPitchBytesF4);

  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionInterleavedRGB_tex_cm_d<<<gridSize,blockSize>>>(d_outputImageF3, iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC * 3;
  std::cout << fps << " fps - texture mem for interleaved img & const mem for kernel access\n";


  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), d_outputImageF3, iPitchBytesF3, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );
  

  // free memory
  gpu_unbindTextureMemoryF4();
  cutilSafeCall( cudaFree(d_inputImageF3) );
  cutilSafeCall( cudaFree(d_outputImageF3) );
  cutilSafeCall( cudaFree(d_inputImageF4) );
}





void gpu_bindConstantMemory(const float *kernel, int size) 
{
  cutilSafeCall( cudaMemcpyToSymbol(constKernel, kernel, size*sizeof(float)) );
}



void gpu_bindTextureMemory(float *d_inputImage, int iWidth, int iHeight, size_t iPitchBytes)
{
  // >>>> prepare usage of texture memory
  tex_Image.addressMode[0] = cudaAddressModeClamp;
  tex_Image.addressMode[1] = cudaAddressModeClamp;
  tex_Image.filterMode = cudaFilterModeLinear;
  tex_Image.normalized = false;
  // <<<< prepare usage of texture memory

  cutilSafeCall( cudaBindTexture2D(0, &tex_Image, d_inputImage, &tex_Image_desc, iWidth, iHeight, iPitchBytes) );
}


void gpu_unbindTextureMemory()
{
  cutilSafeCall( cudaUnbindTexture(tex_Image) );
}



void gpu_bindTextureMemoryF4(float4 *d_inputImageF4, int iWidth, int iHeight, size_t iPitchBytesF4)
{
  // >>>> prepare usage of texture memory
  tex_ImageF4.addressMode[0] = cudaAddressModeClamp;
  tex_ImageF4.addressMode[1] = cudaAddressModeClamp;
  tex_ImageF4.filterMode = cudaFilterModeLinear;
  tex_ImageF4.normalized = false;
  // <<<< prepare usage of texture memory

  cutilSafeCall( cudaBindTexture2D(0, &tex_ImageF4, d_inputImageF4, &tex_Image_descF4, iWidth, iHeight, iPitchBytesF4) );
}


void gpu_unbindTextureMemoryF4()
{
  cutilSafeCall( cudaUnbindTexture(tex_ImageF4) );
}

