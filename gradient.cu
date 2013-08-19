/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: gradient
* file:    gradient.cu
*
* 
\******* PLEASE ENTER YOUR CORRECT STUDENT LOGIN, NAME AND ID BELOW *********/
const char* studentLogin = "p110";
const char* studentName  = "Shrikant Vinchurkar";
const int   studentID    = 03636145;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* derivativeY_sm_d(const float *inputImage, ... )
* derivativeY_sm_d(const float3 *inputImage, ... )
* gradient_magnitude_d(const float *inputImage, ... )
* gradient_magnitude_d(const float3 *inputImage, ... )
*
\****************************************************************************/


#include "gradient.cuh"



#define BW 16
#define BH 16



const char* getStudentLogin() { return studentLogin; };
const char* getStudentName()  { return studentName; };
int         getStudentID()    { return studentID; };
bool checkStudentData() { return strcmp(studentLogin, "p010") != 0 && strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };
bool checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };




__global__ void derivativeX_sm_d(const float *inputImage, float *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float u[BW+2][BH];


  if (x < iWidth && y < iHeight) {
    u[threadIdx.x+1][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x);

    if (x == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x-1);

    if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x+1);
  }

  __syncthreads();

  if (x < iWidth && y < iHeight)
    *((float*)(((char*)outputImage) + y*iPitchBytes)+ x) = 0.5f*(u[threadIdx.x+2][threadIdx.y]-u[threadIdx.x][threadIdx.y]);
}




__global__ void derivativeX_sm_d(const float3 *inputImage, float3 *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 imgValue;
  __shared__ float3 u[BW+2][BH];

  if (x < iWidth && y < iHeight) {
    u[threadIdx.x+1][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);

    if (x == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x-1);
    
    if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x+1);
  }

  __syncthreads();

  
  if (x < iWidth && y < iHeight) {
    imgValue.x = 0.5f*(u[threadIdx.x+2][threadIdx.y].x - u[threadIdx.x][threadIdx.y].x);
    imgValue.y = 0.5f*(u[threadIdx.x+2][threadIdx.y].y - u[threadIdx.x][threadIdx.y].y);
    imgValue.z = 0.5f*(u[threadIdx.x+2][threadIdx.y].z - u[threadIdx.x][threadIdx.y].z);
    *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = imgValue;
  }
}



__global__ void derivativeY_sm_d(const float *inputImage, float *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
  
  // ### implement me ###
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float u[BW][BH+2];
  
  if (x < iWidth && y < iHeight) {
      u[threadIdx.x][threadIdx.y + 1] = *((float*)((char*)inputImage + y*iPitchBytes)+x);

      if (y == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
      else if (threadIdx.y == 0) u[threadIdx.x][threadIdx.y] = *((float*)((char*)inputImage + (y-1)*iPitchBytes)+x);

      if (y == (iHeight-1)) u[threadIdx.x][threadIdx.y+2] = u[threadIdx.x][threadIdx.y+1];
      else if (threadIdx.y == blockDim.y-1) u[threadIdx.x][threadIdx.y+2] = *((float*)((char*)inputImage + (y+1)*iPitchBytes)+x);
    }

    __syncthreads();

    if (x < iWidth && y < iHeight)//guards
      *((float*)(((char*)outputImage) + y*iPitchBytes)+ x) = 0.5f*(u[threadIdx.x][threadIdx.y+2]-u[threadIdx.x][threadIdx.y]);
}



__global__ void derivativeY_sm_d(const float3 *inputImage, float3 *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
 
  // ### implement me ###
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 imgValue;
  __shared__ float3 u[BW][BH+2];

  if (x < iWidth && y < iHeight) {
      u[threadIdx.x][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);

      if (y == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
      else if (threadIdx.y == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + (y-1)*iPitchBytes)+x);
      
      if (y == (iHeight-1)) u[threadIdx.x][threadIdx.y+2] = u[threadIdx.x][threadIdx.y+1];
      else if (threadIdx.y == blockDim.y-1) u[threadIdx.x][threadIdx.y+2] = *((float3*)((char*)inputImage + (y+1)*iPitchBytes)+x);
    }

    __syncthreads();

    
    if (x < iWidth && y < iHeight) {
      imgValue.x = 0.5f*(u[threadIdx.x][threadIdx.y+2].x - u[threadIdx.x][threadIdx.y].x);
      imgValue.y = 0.5f*(u[threadIdx.x][threadIdx.y+2].y - u[threadIdx.x][threadIdx.y].y);
      imgValue.z = 0.5f*(u[threadIdx.x][threadIdx.y+2].z - u[threadIdx.x][threadIdx.y].z);
      *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = imgValue;
    }

}


__global__ void gradient_magnitude_d(const float *inputImage, float *outputImage,
                                     int iWidth, int iHeight, size_t iPitchBytes)
{
  // ### implement me ###
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  __shared__ float u[BW+2][BH+2];

  if (x < iWidth && y < iHeight) {
	  u[threadIdx.x+1][threadIdx.y+1] = *((float*)((char*)inputImage + y*iPitchBytes)+x);
	  
	  // may have conflicts at four extreme corners of image but of no concern as these 
	  // values are never used for calculating gradient of any pixel

	  if (y == 0) u[threadIdx.x+1][threadIdx.y] = u[threadIdx.x+1][threadIdx.y+1];
	  else if (threadIdx.y == 0) u[threadIdx.x+1][threadIdx.y] = *((float*)((char*)inputImage + (y-1)*iPitchBytes)+x);

	  if (y == (iHeight-1)) u[threadIdx.x+1][threadIdx.y+2] = u[threadIdx.x+1][threadIdx.y+1];
	  else if (threadIdx.y == blockDim.y-1) u[threadIdx.x+1][threadIdx.y+2] = *((float*)((char*)inputImage + (y+1)*iPitchBytes)+x);
	  
	  if (x == 0) u[threadIdx.x][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y+1];
      else if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y+1] = *((float*)((char*)inputImage + y*iPitchBytes)+x-1);

      if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y+1];
      else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y+1] = *((float*)((char*)inputImage + y*iPitchBytes)+x+1);
   }

  __syncthreads();
  
  float derX = 0.5f*(u[threadIdx.x+2][threadIdx.y+1]-u[threadIdx.x][threadIdx.y+1]);
  float derY = 0.5f*(u[threadIdx.x+1][threadIdx.y+2]-u[threadIdx.x+1][threadIdx.y]);
  
  // calculate image gradient
  if (x < iWidth && y < iHeight)
        *((float*)(((char*)outputImage) + y*iPitchBytes)+ x) = sqrt( derX*derX + derY*derY );   
}


__global__ void gradient_magnitude_d(const float3 *inputImage, float3 *outputImage,
                                     int iWidth, int iHeight, size_t iPitchBytes)
{

  // ### implement me ###
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  float3 imgValue;
    
  __shared__ float3 u[BW+2][BH+2];

	if (x < iWidth && y < iHeight) {
		  u[threadIdx.x+1][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);		  
		  
		  // may have conflicts at four extreme corners of image (in shared block) but of no  
		  // concern as these values are never used for calculating gradient of any pixel

		  if (y == 0) u[threadIdx.x+1][threadIdx.y] = u[threadIdx.x+1][threadIdx.y+1];
		  else if (threadIdx.y == 0) u[threadIdx.x+1][threadIdx.y] = *((float3*)((char*)inputImage + (y-1)*iPitchBytes)+x);
		        
		  if (y == (iHeight-1)) u[threadIdx.x+1][threadIdx.y+2] = u[threadIdx.x+1][threadIdx.y+1];
		  else if (threadIdx.y == blockDim.y-1) u[threadIdx.x+1][threadIdx.y+2] = *((float3*)((char*)inputImage + (y+1)*iPitchBytes)+x);
  
		  if (x == 0) u[threadIdx.x][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y+1];
		  else if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x-1);
		      
		  if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y+1];
		  else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x+1);		  
		 	  
	   }

	  __syncthreads();
	  
	  // calculate image gradient
	  if (x < iWidth && y < iHeight)
	  {
		float derX_x = 0.5f*(u[threadIdx.x+2][threadIdx.y+1].x-u[threadIdx.x][threadIdx.y+1].x);
		float derY_x = 0.5f*(u[threadIdx.x+1][threadIdx.y+2].x-u[threadIdx.x+1][threadIdx.y].x);
	
	  imgValue.x = sqrt( derX_x * derX_x + derY_x * derY_x );
	  
	  float derX_y = 0.5f*(u[threadIdx.x+2][threadIdx.y+1].y-u[threadIdx.x][threadIdx.y+1].y); 
	  float derY_y = 0.5f*(u[threadIdx.x+1][threadIdx.y+2].y-u[threadIdx.x+1][threadIdx.y].y); 

	  imgValue.y = sqrt( derX_y * derX_y + derY_y * derY_y );			  
	  
	  float derX_z = 0.5f*(u[threadIdx.x+2][threadIdx.y+1].z-u[threadIdx.x][threadIdx.y+1].z);
	  float derY_z = 0.5f*(u[threadIdx.x+1][threadIdx.y+2].z-u[threadIdx.x+1][threadIdx.y].z);
	  
	  imgValue.z = sqrt( derX_z * derX_z + derY_z * derY_z );
			  
	  }
	  
	  if( x < iWidth && y < iHeight) // not putting this guard creates error at cudaThreadSynchronize
	  *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = imgValue;
}


void gpu_derivative_sm_d(const float *inputImage, float *outputImage,
                         int iWidth, int iHeight, int iSpectrum, int mode)
{
  size_t iPitchBytes;
  float *inputImage_d = 0, *outputImage_d = 0;

  dim3 blockSize(BW, BH);  
  dim3 gridSize( (int)ceil(iWidth/(float)BW), (int)ceil(iHeight/(float)BH) );
  //dim3 smSize(BW+2,BH);

  if(iSpectrum == 1) {
    cutilSafeCall( cudaMallocPitch( (void**)&(inputImage_d), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(outputImage_d), &iPitchBytes, iWidth*sizeof(float), iHeight ) );

    cutilSafeCall( cudaMemcpy2D(inputImage_d, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );

    if (mode == 0)
      derivativeX_sm_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 1)
      derivativeY_sm_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 2)
      gradient_magnitude_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);

    cutilSafeCall( cudaThreadSynchronize() );
    cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), outputImage_d, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );
  }
  else if(iSpectrum == 3) {
    cutilSafeCall( cudaMallocPitch( (void**)&(inputImage_d), &iPitchBytes, iWidth*sizeof(float3), iHeight ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(outputImage_d), &iPitchBytes, iWidth*sizeof(float3), iHeight ) );

    cutilSafeCall( cudaMemcpy2D(inputImage_d, iPitchBytes, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );

    if (mode == 0)
      derivativeX_sm_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 1)
      derivativeY_sm_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 2)
      gradient_magnitude_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);

    cutilSafeCall( cudaThreadSynchronize() );
    cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), outputImage_d, iPitchBytes, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );
  }

  cutilSafeCall( cudaFree(inputImage_d) );
  cutilSafeCall( cudaFree(outputImage_d) );
}
