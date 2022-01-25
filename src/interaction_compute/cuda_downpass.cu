#include <math.h>
#include <float.h>
#include <stdio.h>

#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_downpass.h"
#include "../kernels/cuda/tcf/device_vars.h"

FLOAT *d_coeff_x;
FLOAT *d_coeff_y;
FLOAT *d_coeff_z;

// RL - initialize/free device memories
extern "C"
void CUDA_Setup2(
    int sizeof_coeff_x, int sizeof_coeff_y, int sizeof_coeff_z,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z)
{
    cudaErr = cudaMalloc(&d_coeff_x, sizeof(FLOAT)*sizeof_coeff_x);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_coeff_y, sizeof(FLOAT)*sizeof_coeff_y);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_coeff_z, sizeof(FLOAT)*sizeof_coeff_z);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_coeff_x, coeff_x, sizeof(FLOAT)*sizeof_coeff_x, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_coeff_y, coeff_y, sizeof(FLOAT)*sizeof_coeff_y, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_coeff_z, coeff_z, sizeof(FLOAT)*sizeof_coeff_z, cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
        printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    return;
}

extern "C"
void CUDA_Free2()
{
    cudaFree(d_coeff_x);
    cudaFree(d_coeff_y);
    cudaFree(d_coeff_z);

    return;
}


extern "C"
void CUDA_Wrapup(int target_xyz_dim, double *potential)
{
    cudaErr = cudaMemcpy(potential, d_potential,
                         target_xyz_dim * sizeof(double), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaFree(d_cluster_q);
    cudaFree(d_potential);

    return;
}


__global__ 
static void CUDA_CP_COMP_DOWNPASS(
    int interp_pts_per_cluster, int interp_order_lim,
    int coeff_start_ind, int cluster_charge_start, int child_cluster_charge_start,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z,
    double *cluster_q)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < interp_pts_per_cluster) { // loop over interpolation points, set (cx,cy,cz) for this point
        int child_k3 = i%interp_order_lim;
        int child_kk = (i-child_k3)/interp_order_lim;
        int child_k2 = child_kk%interp_order_lim;
        child_kk = child_kk - child_k2;
        int child_k1 = child_kk / interp_order_lim;

        int coeff_x_start = coeff_start_ind + child_k1 * interp_order_lim;
        int coeff_y_start = coeff_start_ind + child_k2 * interp_order_lim;
        int coeff_z_start = coeff_start_ind + child_k3 * interp_order_lim;
        
        FLOAT temp = 0.0;

        for (int j = 0; j < interp_pts_per_cluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point
            int k3 = j%interp_order_lim;
            int kk = (j-k3)/interp_order_lim;
            int k2 = kk%interp_order_lim;
            kk = kk - k2;
            int k1 = kk / interp_order_lim;

            FLOAT cq = cluster_q[cluster_charge_start + j];

            temp += coeff_x[coeff_x_start + k1] *
                    coeff_y[coeff_y_start + k2] *
                    coeff_z[coeff_z_start + k3] * cq;
        }

        atomicAdd(cluster_q + child_cluster_charge_start + i, (double)temp);
    }

    return;
}


__host__
void K_CUDA_CP_COMP_DOWNPASS(
    int idx, int child_idx, int interp_order, int coeff_start,
    int stream_id)
{
    int interp_order_lim       = interp_order + 1;
    int interp_pts_per_cluster = interp_order_lim * interp_order_lim * interp_order_lim;
    
    int cluster_charge_start          = idx * interp_pts_per_cluster;
    int child_cluster_charge_start    = child_idx * interp_pts_per_cluster;

    int coeff_start_ind = interp_order_lim * interp_order_lim * coeff_start;

    int nthreads = 256;
    int nblocks = (interp_pts_per_cluster-1)/nthreads + 1;
    CUDA_CP_COMP_DOWNPASS<<<nblocks,nthreads,0,stream[stream_id]>>>(interp_pts_per_cluster, interp_order_lim,
            coeff_start_ind, cluster_charge_start, child_cluster_charge_start,
            d_coeff_x, d_coeff_y, d_coeff_z, d_cluster_q);
    cudaError_t kernelErr = cudaGetLastError();
    if ( kernelErr != cudaSuccess )
        printf("Kernel failed with error \"%s\".\n", cudaGetErrorString(kernelErr));
    
    return;
}


__global__ 
static void CUDA_CP_COMP_POT(
    int idx, int interp_order,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_yz_dim, int target_z_dim_glob,
    int coeff_x_start, int coeff_y_start, int coeff_z_start,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z,
    double *cluster_q, double *potential)
{
    int interp_order_lim = interp_order + 1;
    int orderlim3 = interp_order_lim*interp_order_lim*interp_order_lim;
    int cluster_charge_start =idx * orderlim3;
    // ix/iy/iz always start from 0
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix < target_x_high_ind - target_x_low_ind + 1 &&
        iy < target_y_high_ind - target_y_low_ind + 1 &&
        iz < target_z_high_ind - target_z_low_ind + 1 ) {
        
        int ii = ((ix + target_x_low_ind) * target_yz_dim) +
                 ((iy + target_y_low_ind) * target_z_dim_glob ) +
                  (iz + target_z_low_ind);

        int iix = coeff_x_start + ix *interp_order_lim;
        int iiy = coeff_y_start + iy *interp_order_lim;
        int iiz = coeff_z_start + iz *interp_order_lim;

        FLOAT temporary_potential = 0.0;

        for (int j=0; j < orderlim3; j++){

            int k3 = j%interp_order_lim; 
            int kk = (j-k3)/interp_order_lim;
            int k2 = kk%interp_order_lim;
            kk = kk - k2;
            int k1 = kk/interp_order_lim;
            
            FLOAT cq = cluster_q[cluster_charge_start +j];

            temporary_potential += coeff_x[iix + k1] *
                                   coeff_y[iiy + k2] *
                                   coeff_z[iiz + k3] * cq;
        }                           

        atomicAdd(potential+ii, (double)temporary_potential);
    }

    return;
}

__host__
void K_CUDA_CP_COMP_POT(
    int idx, int interp_order,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int coeff_x_start,
    int coeff_y_start,
    int coeff_z_start,
    int stream_id)
{
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    int target_x_dim = target_x_high_ind - target_x_low_ind + 1;
    int target_y_dim = target_y_high_ind - target_y_low_ind + 1;
    int target_z_dim = target_z_high_ind - target_z_low_ind + 1;

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((target_x_dim-1)/threadsperblock + 1,
                 (target_y_dim-1)/threadsperblock + 1,
                 (target_z_dim-1)/threadsperblock + 1);

    CUDA_CP_COMP_POT<<<nblocks,nthreads,0,stream[stream_id]>>>(idx, interp_order,
            target_x_low_ind, target_y_low_ind, target_z_low_ind,
            target_x_high_ind, target_y_high_ind, target_z_high_ind,
            target_yz_dim, target_z_dim_glob,coeff_x_start, coeff_y_start, 
            coeff_z_start, d_coeff_x, d_coeff_y, d_coeff_z, d_cluster_q, d_potential);
    cudaError_t kernelErr = cudaGetLastError();
    if ( kernelErr != cudaSuccess )
        printf("Kernel failed with error \"%s\".\n", cudaGetErrorString(kernelErr));

    return;
}
