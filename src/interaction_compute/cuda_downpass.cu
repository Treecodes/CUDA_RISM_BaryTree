#include <math.h>
#include <float.h>
#include <stdio.h>

//#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_downpass.h"
#include "device_vars2.h"

cudaError_t cudaErr2;
cudaStream_t stream2[16];
FLOAT *d_coeff_x;
FLOAT *d_coeff_y;
FLOAT *d_coeff_z;
FLOAT *d_potential2;
FLOAT *d_cluster_q2;

// RQ - initialize streams
extern "C"
void initStream2()
{
    for (int i = 0; i < 16; ++i) {
        cudaErr2 = cudaStreamCreate(&stream2[i]);
        if ( cudaErr2 != cudaSuccess )
            printf("Stream creation failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }
}

extern "C"
void delStream2()
{
    for (int i = 0; i < 8; ++i) {
        cudaErr2 = cudaStreamDestroy(stream2[i]);
        if ( cudaErr2 != cudaSuccess )
            printf("Stream destruction failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }
}

// RL - initialize/free device memories
extern "C"
void CUDA_Setup2(int call_type,
    int sizeof_coeff_x, int sizeof_coeff_y, int sizeof_coeff_z, int num_charge, int target_xyz_dim,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z,
    double *cluster_q, double *potential)
{
    if ( call_type == 1 || call_type == 3 ) {
        cudaErr2 = cudaMalloc(&d_coeff_x, sizeof(FLOAT)*sizeof_coeff_x);
        if ( cudaErr2 != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr2 = cudaMalloc(&d_coeff_y, sizeof(FLOAT)*sizeof_coeff_y);
        if ( cudaErr2 != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_coeff_z, sizeof(FLOAT)*sizeof_coeff_z);
        if ( cudaErr2 != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr2 = cudaMalloc(&d_cluster_q2, sizeof(FLOAT)*num_charge);
        if ( cudaErr2 != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr2 = cudaMalloc(&d_potential2, sizeof(FLOAT)*target_xyz_dim);
        if ( cudaErr2 != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }


    if ( call_type == 1 || call_type == 3 ) {
        cudaErr2 = cudaMemcpy(d_coeff_x, coeff_x, sizeof(FLOAT)*sizeof_coeff_x, cudaMemcpyHostToDevice);
        if ( cudaErr2 != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr2 = cudaMemcpy(d_coeff_y, coeff_y, sizeof(FLOAT)*sizeof_coeff_y, cudaMemcpyHostToDevice);
        if ( cudaErr2 != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr2 = cudaMemcpy(d_coeff_z, coeff_z, sizeof(FLOAT)*sizeof_coeff_z, cudaMemcpyHostToDevice);
        if ( cudaErr2 != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr2 = cudaMemcpy(d_cluster_q2, cluster_q, sizeof(FLOAT)*num_charge, cudaMemcpyHostToDevice);
        if ( cudaErr2 != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr2 = cudaMemcpy(d_potential2, potential, sizeof(FLOAT)*target_xyz_dim, cudaMemcpyHostToDevice);
        if ( cudaErr2 != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        //printf("CUDA copied data into device %d %d\n", num_source, target_xyz_dim);
    }

    return;
}

extern "C"
void CUDA_Free2(int call_type,
    int target_xyz_dim, double *potential)
{
    if ( call_type == 1 || call_type == 3 ) {
        cudaErr2 = cudaMemcpy(potential, d_potential2,
                             target_xyz_dim * sizeof(FLOAT), cudaMemcpyDeviceToHost);
        if ( cudaErr2 != cudaSuccess )
            printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaFree(d_coeff_x);
        cudaFree(d_coeff_y);
        cudaFree(d_coeff_z);
        cudaFree(d_cluster_q2);
        cudaFree(d_potential2);
    }


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
    FLOAT *cluster_q,
    FLOAT *potential)
{
     int interp_order_lim = interp_order + 1;
     int orderlim3 = interp_order_lim*interp_order_lim*interp_order_lim;
     int cluster_charge_start =idx * orderlim3;
    // ix/iy/iz always start from 0
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix < target_x_high_ind - target_x_low_ind + 1  &&
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
            
            double cq = cluster_q[cluster_charge_start +j];
             temporary_potential += coeff_x[iix + k1] * coeff_y[iiy + k2]
                          * coeff_z[iiz + k3] * cq;
          }                           

        
        potential[ii] += temporary_potential;

    }

    return;
}

__host__
void K_CUDA_CP_COMP_POT(
    int call_type, int idx, double *potential, int interp_order,
        int target_x_low_ind,  int target_x_high_ind,
        int target_y_low_ind,  int target_y_high_ind,
        int target_z_low_ind,  int target_z_high_ind,
        int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
        double *cluster_q,
        int coeff_x_start, double *coeff_x,
        int coeff_y_start, double *coeff_y,
        int coeff_z_start, double *coeff_z, 
        struct RunParams *run_params, int stream_id)
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
    // RQ - test without stream
    //CUDA_TCF_PP<<<nblocks,nthreads>>>(eta, kap, kap_eta_2,
    CUDA_CP_COMP_POT<<<nblocks,nthreads,0,stream2[stream_id]>>>(idx, interp_order,
                  target_x_low_ind, target_y_low_ind, target_z_low_ind,
                  target_x_high_ind, target_y_high_ind, target_z_high_ind,
                  target_yz_dim, target_z_dim_glob,coeff_x_start, coeff_y_start, 
                  coeff_z_start, d_coeff_x, d_coeff_y, d_coeff_z, d_cluster_q2, d_potential2);
   cudaError_t kernelErr = cudaGetLastError();
 if ( kernelErr != cudaSuccess )
            printf("Kernel failed with error \"%s\".\n", cudaGetErrorString(kernelErr));

    // RQ
    //cudaStreamSynchronize(stream[stream_id]);
    //cudaDeviceSynchronize();

    return;

}
