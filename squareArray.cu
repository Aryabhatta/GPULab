/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: squareArray
* file:    squareArray.cu
*
*
* In this file the following methods have to be edited or completed:
*
* square_array_kernel
* square_array_gpu
* 
\****************************************************************************/

#include <stdio.h>
#include <cuda.h>


void square_array_cpu(float *a, unsigned int numElements)
{
  for (unsigned int i=0; i<numElements; i++)
    a[i] = a[i]*a[i];
}


// Kernel that executes on the CUDA device
__global__ void square_array_kernel(float *a, unsigned int numElements)
{
	// kernel code
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Check so as not to extend the array bound
	if( i < numElements )
	{
		a[i] = a[i] * a[i];
	}	
}



// function that invokes the gpu kernel
__host__ void square_array_gpu(float *a_host, unsigned int numElements)
{
  float *a_device;
  size_t size = numElements*sizeof(float);

  // allocate memory on the device
  cudaMalloc( &a_device, size );
  
  // copy array from host to device memory
  cudaMemcpy( a_device, a_host, size, cudaMemcpyHostToDevice );
  
  // do calculation on device
  int block_size = 4;
  int grid_size = numElements/block_size + (numElements%block_size ? 1:0);
  
  square_array_kernel <<< grid_size, block_size >>> ( a_device, numElements );
  
  // Retrieve result from device and store it in host array
  cudaMemcpy( a_host, a_device, size, cudaMemcpyDeviceToHost );

  // free device memory
  cudaFree( a_device );
}



// main routine that executes on the host
int main(void)
{
  float *a_host;                            // pointer to array in host memory
  const unsigned int numElements = 10;      // number of elements in the array
  size_t size = numElements * sizeof(float);
  a_host = (float *)malloc(size);           // allocate array on host
  
  // initialize host array with some data
  for (int i=0; i<numElements; i++) a_host[i] = (float)i;
  printf("\nCPU-version:\n");

  square_array_cpu(a_host, numElements);
  
  // print results
  for (int i=0; i<numElements; i++) printf("%d %f\n", i, a_host[i]);  

  // re-initialize host array to do the same on the gpu again
  for (int i=0; i<numElements; i++) a_host[i] = (float)i;
  printf("\nGPU-version:\n");

  square_array_gpu(a_host, numElements);

  // print results
  for (int i=0; i<numElements; i++) printf("%d %f\n", i, a_host[i]);
  
  // cleanup
  free(a_host);  

  printf("\nPress ENTER to exit...\n");
  getchar();
}
