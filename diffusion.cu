/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: diffusion
* file:    diffusion.cu
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
* diffuse_linear_isotrop_shared(const float  *d_input, ... )
* diffuse_linear_isotrop_shared(const float3 *d_input, ... )
* diffuse_nonlinear_isotrop_shared(const float  *d_input, ... )
* diffuse_nonlinear_isotrop_shared(const float3 *d_input, ... )
* compute_tv_diffusivity_shared
* compute_tv_diffusivity_joined_shared
* compute_tv_diffusivity_separate_shared
* jacobi_shared(float  *d_output, ... )
* jacobi_shared(float3 *d_output, ... )
* sor_shared(float  *d_output, ... )
* sor_shared(float3 *d_output, ... )
*
\****************************************************************************/


#define DIFF_BW 16
#define DIFF_BH 16

#define TV_EPSILON 0.1f


#include "diffusion.cuh"



const char* getStudentLogin() { return studentLogin; };
const char* getStudentName()  { return studentName; };
int         getStudentID()    { return studentID; };
bool checkStudentData() { return strcmp(studentLogin, "p010") != 0 && strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };
bool checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };


//----------------------------------------------------------------------------
// Linear Diffusion
//----------------------------------------------------------------------------


// mode 0 gray: linear diffusion
__global__ void diffuse_linear_isotrop_shared(
  const float *d_input,
  float *d_output,
  float timeStep, 
  int nx, int ny,
  size_t pitch)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = d_input[idx];

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
  }

  __syncthreads();

  // ### implement me ###
  // calculating linear isotropic diffusion
  if( x < nx && y < ny )//guards
  {  
      float isoD = u[tx+1][ty] + u[tx-1][ty] + u[tx][ty+1] + u[tx][ty-1] - 4*u[tx][ty];
      
	  d_output[idx] = u[tx][ty] + timeStep * isoD;
  }
}


// mode 0 interleaved: linear diffusion
__global__ void diffuse_linear_isotrop_shared
(
 const float3 *d_input,
 float3 *d_output,
 float timeStep,
 int nx, int ny,
 size_t pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* imgO = (char*)d_output + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  float3 imgValue;

  // load data into shared memory
  if (x < nx && y < ny) {

    imgValue = *( (float3*)imgP );
    u[tx][ty] = imgValue;

    if (x == 0)  u[0][ty] = imgValue;
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = imgValue;
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = imgValue;
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = imgValue;
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();

 
  // ### implement me ###
  // calculating linear isotropic diffusion
   if( x < nx && y < ny )//guards
   {  
      imgValue.x = u[tx][ty].x + timeStep * (u[tx+1][ty].x + u[tx-1][ty].x + u[tx][ty+1].x + u[tx][ty-1].x - 4*u[tx][ty].x);
      imgValue.y = u[tx][ty].y + timeStep * (u[tx+1][ty].y + u[tx-1][ty].y + u[tx][ty+1].y + u[tx][ty-1].y - 4*u[tx][ty].y);
      imgValue.z = u[tx][ty].z + timeStep * (u[tx+1][ty].z + u[tx-1][ty].z + u[tx][ty+1].z + u[tx][ty-1].z - 4*u[tx][ty].z);
      
  	   *( (float3 *)imgO ) = imgValue;
    }
}


//----------------------------------------------------------------------------
// Non-linear Diffusion - explicit scheme
//----------------------------------------------------------------------------


// mode 1 gray: nonlinear diffusion
__global__ void diffuse_nonlinear_isotrop_shared
(
 const float *d_input,
 const float *d_diffusivity,
 float *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0) {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
      
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }


    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
      
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    } 
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();

  
  // ### implement me ###
  if( x < nx && y < ny )//guards
  {
	  float phiR = 0.5f *( g[tx+1][ty] + g[tx][ty] );
	  float phiL = 0.5f *( g[tx-1][ty] + g[tx][ty] );
	  float phiU = 0.5f *( g[tx][ty-1] + g[tx][ty] );
	  float phiD = 0.5f *( g[tx][ty+1] + g[tx][ty] );
	  
	  float sum = phiR * u[tx+1][ty] + phiL * u[tx-1][ty] + phiU * u[tx][ty-1] + phiD * u[tx][ty+1] -
			  	  ( phiR + phiL + phiU + phiD ) * u[tx][ty];
	  
	  d_output[idx] = u[tx][ty] + timeStep * sum;
  }
}



// mode 1 interleaved: nonlinear diffusion
__global__ void diffuse_nonlinear_isotrop_shared
(
 const float3 *d_input,
 const float3 *d_diffusivity,
 float3 *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);
  const char* imgO = (char*)d_output + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];
  float3 value;


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0) {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    } 
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    } 
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###
  if( x < nx && y < ny )//guards
  {
	  // calculate value.x
	  float phiR_x = 0.5f *( g[tx+1][ty].x + g[tx][ty].x );
  	  float phiL_x = 0.5f *( g[tx-1][ty].x + g[tx][ty].x );
  	  float phiU_x = 0.5f *( g[tx][ty-1].x + g[tx][ty].x );
  	  float phiD_x = 0.5f *( g[tx][ty+1].x + g[tx][ty].x );
  	  
  	  float sum_x = phiR_x * u[tx+1][ty].x + phiL_x * u[tx-1][ty].x + 
  			        phiU_x * u[tx][ty-1].x + phiD_x * u[tx][ty+1].x - 
  			        ( phiR_x + phiL_x + phiU_x + phiD_x ) * u[tx][ty].x;
  	  
  	  value.x = u[tx][ty].x + timeStep * sum_x;
  	  
  	// calculate value.y
  	float phiR_y = 0.5f *( g[tx+1][ty].y + g[tx][ty].y );
  	float phiL_y = 0.5f *( g[tx-1][ty].y + g[tx][ty].y );
  	float phiU_y = 0.5f *( g[tx][ty-1].y + g[tx][ty].y );
  	float phiD_y = 0.5f *( g[tx][ty+1].y + g[tx][ty].y );
  	
  	float sum_y = phiR_y * u[tx+1][ty].y + phiL_y * u[tx-1][ty].y + 
  	              phiU_y * u[tx][ty-1].y + phiD_y * u[tx][ty+1].y - 
  	  		    ( phiR_y + phiL_y + phiU_y + phiD_y ) * u[tx][ty].y;
  	  	  
  	value.y = u[tx][ty].y + timeStep * sum_y;
  	
  	// calculate value.z
  	float phiR_z = 0.5f *( g[tx+1][ty].z + g[tx][ty].z );
  	float phiL_z = 0.5f *( g[tx-1][ty].z + g[tx][ty].z );
  	float phiU_z = 0.5f *( g[tx][ty-1].z + g[tx][ty].z );
  	float phiD_z = 0.5f *( g[tx][ty+1].z + g[tx][ty].z );
  	
  	float sum_z = phiR_z * u[tx+1][ty].z + phiL_z * u[tx-1][ty].z + 
  	              phiU_z * u[tx][ty-1].z + phiD_z * u[tx][ty+1].z - 
  	   		    ( phiR_z + phiL_z + phiU_z + phiD_z ) * u[tx][ty].z;
  	  	  	  
  	value.z = u[tx][ty].z + timeStep * sum_z;
  	  
  	*( (float3 *) imgO) = value;
  }
}

// diffusivity computation for modes 1-3 gray
__global__ void compute_tv_diffusivity_shared
(
 const float *d_input,
 float *d_output,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = d_input[idx];

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];      
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
  }

  __syncthreads();

 
  // make use of the constant TV_EPSILON

  // ### implement me ###
  if( x < nx && y < ny )//guards
  {
	  // calculate gradient magnitude
	  float derX = 0.5f * (u[tx+1][ty] - u[tx-1][ty]);
	  float derY = 0.5f * (u[tx][ty+1] - u[tx][ty-1]);
	  float temp = sqrt( derX * derX + derY * derY  + TV_EPSILON );
	  
	  d_output[idx] = 1 / temp;
  }
}


/*! Computes a joined diffusivity for an RGB Image:
 *  (g_R,g_G,g_B)(R,G,B) := 
 *  (g((R+G+B)/3),g((R+G+B)/3),g((R+G+B)/3))
 * */
__global__ void compute_tv_diffusivity_joined_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* imgO = (char*)d_output + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  float3 imgValue;

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();
  
  
  // make use of the constant TV_EPSILON

  // ### implement me ###
  /*  (g_R,g_G,g_B)(R,G,B) := 
   *  (g((R+G+B)/3),g((R+G+B)/3),g((R+G+B)/3))*/
  if( x < nx && y < ny )//guards
  {
	  // calculate g((R+G+B)/3)
	  float derX = 0.5f * ((u[tx+1][ty].x + u[tx+1][ty].y + u[tx+1][ty].z)/3 - 
			  	  	  	   (u[tx-1][ty].x + u[tx-1][ty].y + u[tx-1][ty].z)/3 );
	  
	  float derY = 0.5f * ((u[tx][ty+1].x + u[tx][ty+1].y + u[tx][ty+1].z)/3 - 
			  	  	  	   (u[tx][ty-1].x + u[tx][ty-1].y + u[tx][ty-1].z)/3);
	  
	  float temp = 1 / (sqrt( derX * derX + derY * derY  + TV_EPSILON ));
	  
	  imgValue.x = temp;
	  imgValue.y = temp;
	  imgValue.z = temp;
    	  
      *( (float3 *)imgO ) = imgValue;
  }
}


/*! Computes a separate diffusivity for an RGB Image:
 *  (g_R,g_G,g_B)(R,G,B) := 
 *  (g(R),g(G),g(B))
 * */
__global__ void compute_tv_diffusivity_separate_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* imgO = (char*)d_output + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  float3 imgValue;

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[threadIdx.x][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();

  
  // make use of the constant TV_EPSILON

  // ### implement me ###
  if( x < nx && y < ny )//guards
  {
  	  // calculate gradient magnitude
	  
	  // calculate g(R)
  	  float derX_x = 0.5f * (u[tx+1][ty].x - u[tx-1][ty].x);
  	  float derY_x = 0.5f * (u[tx][ty+1].x - u[tx][ty-1].x);
  	  float temp_x = sqrt( derX_x * derX_x + derY_x * derY_x  + TV_EPSILON );
  	  imgValue.x = 1 / temp_x;
  	  
  	  // calculate g(G)
  	  float derX_y = 0.5f * (u[tx+1][ty].y - u[tx-1][ty].y);
  	  float derY_y = 0.5f * (u[tx][ty+1].y - u[tx][ty-1].y);
  	  float temp_y = sqrt( derX_y * derX_y + derY_y * derY_y  + TV_EPSILON );
  	  imgValue.y = 1 / temp_y;
  	  
  	  // calculate g(B)
  	  float derX_z = 0.5f * (u[tx+1][ty].z - u[tx-1][ty].z);
  	  float derY_z = 0.5f * (u[tx][ty+1].z - u[tx][ty-1].z);
  	  float temp_z = sqrt( derX_z * derX_z + derY_z * derY_z  + TV_EPSILON );
  	  imgValue.z = 1 / temp_z;
  	  
  	  *( (float3 *)imgO ) = imgValue;
  }

}

//----------------------------------------------------------------------------
// Non-linear Diffusion - Jacobi scheme
//----------------------------------------------------------------------------

// mode 2 gray: Jacobi solver
__global__ void jacobi_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();

  
  // ### implement me ###
  
  // dnt calculate A - waste of time & space
  if (x < nx && y < ny) //guards
  {
	  // setting boundary values to 0 to avoid corruption of Jacobi scheme
	  float phiR = ( x == nx-1) ? 0.0f : 0.5f *( g[tx+1][ty] + g[tx][ty] );
	  float phiL = ( x == 0 )   ? 0.0f : 0.5f *( g[tx-1][ty] + g[tx][ty] );
	  float phiU = ( y == 0 )   ? 0.0f : 0.5f *( g[tx][ty-1] + g[tx][ty] );
	  float phiD = ( y == ny-1) ? 0.0f : 0.5f *( g[tx][ty+1] + g[tx][ty] );
	  	  
	  float Aii = ( 1 + (phiR + phiL + phiU + phiD ) * weight) ;
	    
	  float sumN = weight* ( phiR * u[tx+1][ty] + phiL * u[tx-1][ty] + 
	                         phiU * u[tx][ty-1] + phiD * u[tx][ty+1]);
	    
	  d_output[idx] = (d_original[idx] + sumN) / Aii;   
  }
}



// mode 2 interleaved: Jacobi solver
__global__ void jacobi_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);
  
  const char* imgOut = (char*)d_output + y*pitchBytes + x*sizeof(float3);
  const char* imgOrig = (char*)d_original + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];
  
  float3 imgValue;


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###
  if (x < nx && y < ny) //guards
  {
	  // setting boundary values to 0 to avoid corruption of Jacobi scheme
  	  float phiR_x =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].x + g[tx][ty].x );
  	  float phiL_x =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].x + g[tx][ty].x );
  	  float phiU_x =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].x + g[tx][ty].x );
  	  float phiD_x =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].x + g[tx][ty].x );
  	  
  	  float Aii_x = ( 1 + (phiR_x + phiL_x + phiU_x + phiD_x ) * weight) ;  	    
  	  float sumN_x = weight* ( phiR_x * u[tx+1][ty].x + phiL_x * u[tx-1][ty].x + 
  	                           phiU_x * u[tx][ty-1].x + phiD_x * u[tx][ty+1].x);
  	  imgValue.x = (*((float*)imgOrig) + sumN_x)/Aii_x;
  	  
  	  float phiR_y =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].y + g[tx][ty].y );
  	  float phiL_y =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].y + g[tx][ty].y );
  	  float phiU_y =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].y + g[tx][ty].y );
  	  float phiD_y =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].y + g[tx][ty].y );
  	  
  	  float Aii_y = ( 1 + (phiR_y + phiL_y + phiU_y + phiD_y ) * weight) ;  	    
  	  float sumN_y = weight* ( phiR_y * u[tx+1][ty].y + phiL_y * u[tx-1][ty].y + 
  			  	  	  	  	   phiU_y * u[tx][ty-1].y + phiD_y * u[tx][ty+1].y);
  	  imgValue.y = (*((float*)imgOrig+1) + sumN_y)/Aii_y;
  	  	  
  	  float phiR_z =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].z + g[tx][ty].z );
  	  float phiL_z =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].z + g[tx][ty].z );
  	  float phiU_z =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].z + g[tx][ty].z );
  	  float phiD_z =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].z + g[tx][ty].z );
  	  
  	  float Aii_z = ( 1 + (phiR_z + phiL_z + phiU_z + phiD_z ) * weight) ;  	    
  	  float sumN_z = weight* ( phiR_z * u[tx+1][ty].z + phiL_z * u[tx-1][ty].z + 
  	                           phiU_z * u[tx][ty-1].z + phiD_z * u[tx][ty+1].z);
  	  imgValue.z = (*((float*)imgOrig+2) + sumN_z)/Aii_z;
  	    
  	  *((float3*)imgOut) =imgValue;   
    }
}

//----------------------------------------------------------------------------
// Non-linear Diffusion - Successive Over-Relaxation (SOR)
//----------------------------------------------------------------------------

// mode 3 gray: SOR solver
__global__ void sor_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitch,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;
  
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();


  // ### implement me ###
  // dnt calculate A - waste of time & space
  if (x < nx && y < ny)//guards 
  { 
	   	  	  	  	  	  	 // implementing the checkerboard pattern
	 if( (tx+ty) % 2 == red) // allow only threads with even sum if red==0 & odd sum if red==1 
	 {		  
		 float phiR =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty] + g[tx][ty] );
		 float phiL =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty] + g[tx][ty] );
		 float phiU =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1] + g[tx][ty] );
		 float phiD =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1] + g[tx][ty] );
				  
		 float Aii = ( 1 + (phiR + phiL + phiU + phiD ) * weight) ;
				
		 float sumNU = weight * ( phiR * u[tx+1][ty] + phiU * u[tx][ty-1] );
		 float sumNL = weight * ( phiL * u[tx-1][ty] + phiD * u[tx][ty+1] );  	  
			 
		 d_output[idx] = (1-overrelaxation) * u[tx][ty] + 
		                 (d_original[idx] + sumNU + sumNL) * overrelaxation / Aii;
	  }
  }
}


// mode 3 interleaved: SOR solver
__global__ void sor_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitchBytes,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);
  const char* imgOut = (char*)d_output + y*pitchBytes + x*sizeof(float3);
  const char* imgOrig = (char*)d_original + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];

  float3 imgValue;


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###
  if (x < nx && y < ny) //guards
  { 
	                		  // implementing the checkerboard pattern
	if( (tx+ty) % 2 == red) // allow only threads with even sum if red==0 & odd sum if red==1 
	{		  
		// calculate imgValue.x
		float phiR_x =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].x + g[tx][ty].x );
		float phiL_x =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].x + g[tx][ty].x );
		float phiU_x =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].x + g[tx][ty].x );
		float phiD_x =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].x + g[tx][ty].x );
			  
		float Aii_x = ( 1 + (phiR_x + phiL_x + phiU_x + phiD_x ) * weight) ;
				
		float sumNU_x = weight * ( phiR_x * u[tx+1][ty].x + phiU_x * u[tx][ty-1].x );
		float sumNL_x = weight * ( phiL_x * u[tx-1][ty].x + phiD_x * u[tx][ty+1].x );  	  
			 
		imgValue.x = (1-overrelaxation) * u[tx][ty].x + 
					 ( *((float *) imgOrig)+ sumNU_x + sumNL_x) * overrelaxation / Aii_x;
			  
		// calculate imgValue.y
		float phiR_y =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].y + g[tx][ty].y );
		float phiL_y =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].y + g[tx][ty].y );
		float phiU_y =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].y + g[tx][ty].y );
		float phiD_y =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].y + g[tx][ty].y );
			  			  
		float Aii_y = ( 1 + (phiR_y + phiL_y + phiU_y + phiD_y ) * weight) ;
			  				
		float sumNU_y = weight * ( phiR_y * u[tx+1][ty].y + phiU_y * u[tx][ty-1].y );
		float sumNL_y = weight * ( phiL_y * u[tx-1][ty].y + phiD_y * u[tx][ty+1].y );  	  
			  			 
		imgValue.y = (1-overrelaxation) * u[tx][ty].y + 
					 ( *((float *) imgOrig + 1)+ sumNU_y + sumNL_y) * overrelaxation / Aii_y;
			  
		// calculate imgValue.z
		float phiR_z =( x == nx-1 ) ? 0.0f : 0.5f *( g[tx+1][ty].z + g[tx][ty].z );
		float phiL_z =( x == 0 )    ? 0.0f : 0.5f *( g[tx-1][ty].z + g[tx][ty].z );
		float phiU_z =( y == 0 )    ? 0.0f : 0.5f *( g[tx][ty-1].z + g[tx][ty].z );
		float phiD_z =( y == ny-1 ) ? 0.0f : 0.5f *( g[tx][ty+1].z + g[tx][ty].z );
			  			  
		float Aii_z = ( 1 + (phiR_z + phiL_z + phiU_z + phiD_z ) * weight) ;
			  				
		float sumNU_z = weight * ( phiR_z * u[tx+1][ty].z + phiU_z * u[tx][ty-1].z );
		float sumNL_z = weight * ( phiL_z * u[tx-1][ty].z + phiD_z * u[tx][ty+1].z );  	  
			  			 
		imgValue.z = (1-overrelaxation) * u[tx][ty].z + 
					 ( *((float *) imgOrig + 2)+ sumNU_z + sumNL_z) * overrelaxation / Aii_z;
			  
		*((float3*) imgOut ) = imgValue;
	}
  }
}




//----------------------------------------------------------------------------
// Host function
//----------------------------------------------------------------------------



void gpu_diffusion
(
 const float *input,
 float *output,
 int nx, int ny, int nc, 
 float timeStep,
 int iterations,
 float weight,
 int lagged_iterations,
 float overrelaxation,
 int mode,
 bool jointDiffusivity
 )
{
  int i,j;
  size_t pitchF1, pitchBytesF1, pitchBytesF3;
  float *d_input = 0;
  float *d_output = 0;
  float *d_diffusivity = 0;
  float *d_original = 0;
  float *temp = 0;

  dim3 dimGrid((int)ceil((float)nx/DIFF_BW), (int)ceil((float)ny/DIFF_BH));
  dim3 dimBlock(DIFF_BW,DIFF_BH);

  // Allocation of GPU Memory
  if (nc == 1) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF1, nx*sizeof(float), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF1, nx*sizeof(float), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF1, input, nx*sizeof(float), nx*sizeof(float), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF1, d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToDevice) );

    pitchF1 = pitchBytesF1/sizeof(float);

  } else if (nc == 3) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF3, nx*sizeof(float3), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF3, nx*sizeof(float3), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF3, input, nx*sizeof(float3), nx*sizeof(float3), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF3, d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToDevice) );

  }


  // Execution of the Diffusion Kernel

  if (mode == 0) {   // linear isotropic diffision
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input, d_output, timeStep, nx, ny, pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 1) {  // nonlinear isotropic diffusion
    if (nc == 1) {

      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,d_output,timeStep,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);


        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>
          ((float3*)d_input,(float3*)d_diffusivity,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 2) {    // Jacobi-method
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>> (d_output,d_input,d_original,
            d_diffusivity,weight,nx,ny,pitchF1);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_output,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,nx,ny,pitchBytesF3);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }    
  }
  else if (mode == 3) {    // Successive Over Relaxation (Gauss-Seidel with extrapolation)
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for(j=0;j<lagged_iterations;j++) {					
          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
    if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
  }


  if (nc == 1) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_output, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
  } else if (nc == 3) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_output, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
  }


  // clean up
  if (d_original) cutilSafeCall( cudaFree(d_original) );
  if (d_diffusivity) cutilSafeCall( cudaFree(d_diffusivity) );
  if (d_output) cutilSafeCall( cudaFree(d_output) );
  if (d_input)  cutilSafeCall( cudaFree(d_input) );
}