#include <math.h>
#include <float.h>
#include <stdio.h>

//#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_interp.h"
#include "../kernels/cuda/tcf/device_vars.h"
#include "../tree/struct_tree.h"
FLOAT *d_xC;
FLOAT *d_yC;
FLOAT *d_zC;

// RL - initialize/free device memories
extern "C"
void CUDA_Setup_Interp(
    int totalNumberInterpolationPoints,
    FLOAT *xC, FLOAT *yC, FLOAT *zC)
{
    cudaErr = cudaMalloc(&d_xC, sizeof(FLOAT)*totalNumberInterpolationPoints);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_yC, sizeof(FLOAT)*totalNumberInterpolationPoints);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_zC, sizeof(FLOAT)*totalNumberInterpolationPoints);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_xC, xC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_yC, yC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_zC, zC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    return;
}

extern "C"
void CUDA_Free_Interp(int totalNumberInterpolationPoints,FLOAT *xC, FLOAT *yC, FLOAT *zC)
{
    cudaErr = cudaMemcpy(xC, d_xC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(yC, d_yC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(zC, d_zC, sizeof(FLOAT)*totalNumberInterpolationPoints, cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaFree(d_xC);
    cudaFree(d_yC);
    cudaFree(d_zC);

    return;
}




__global__ 
static void CUDA_COMP_INTERP(
    int interp_order_lim,int cluster_ind_start, FLOAT x0, FLOAT x1, FLOAT y0, FLOAT y1,
    FLOAT z0, FLOAT z1, FLOAT *xC, FLOAT *yC, FLOAT *zC)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < interp_order_lim) { // loop over interpolation points, set (cx,cy,cz) for this point
        
      FLOAT tt = cos(i * M_PI / (interp_order_lim - 1));
       xC[cluster_ind_start + i] = x0 + (tt + 1.0)/2.0 * (x1 - x0); 
       yC[cluster_ind_start + i] = y0 + (tt + 1.0)/2.0 * (y1 - y0); 
       zC[cluster_ind_start + i] = z0 + (tt + 1.0)/2.0 * (z1 - z0); 
    }

    return;
}


__host__
void K_CUDA_COMP_INTERP(
    const struct Tree *tree, int idx, int interpolationOrder,
    int stream_id)
{
    int interp_order_lim       = interpolationOrder + 1;
    
    FLOAT x0 = tree->x_min[idx];
    FLOAT x1 = tree->x_max[idx];
    FLOAT y0 = tree->y_min[idx];
    FLOAT y1 = tree->y_max[idx];
    FLOAT z0 = tree->z_min[idx];
    FLOAT z1 = tree->z_max[idx];
    


    int cluster_ind_start   = idx * interp_order_lim;

    int nthreads = 256;
    int nblocks = (interp_order_lim-1)/nthreads + 1;
    CUDA_COMP_INTERP<<<nblocks,nthreads,0,stream[stream_id]>>>( interp_order_lim,
            cluster_ind_start,x0,x1,y0,y1,z0,z1,
            d_xC, d_yC, d_zC);
    cudaError_t kernelErr = cudaGetLastError();
    if ( kernelErr != cudaSuccess )
        printf("Kernel failed with error \"%s\".\n", cudaGetErrorString(kernelErr));
    
    return;
}


