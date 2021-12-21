#include <math.h>
#include <float.h>
#include <stdio.h>

//#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_tcf_pp.h"
#include "device_vars.h"

cudaError_t cudaErr;
cudaStream_t stream[16];
FLOAT *d_source_x;
FLOAT *d_source_y;
FLOAT *d_source_z;
FLOAT *d_source_q;
FLOAT *d_potential;
FLOAT *d_cluster_x;
FLOAT *d_cluster_y;
FLOAT *d_cluster_z;
FLOAT *d_cluster_q;

// RQ - initialize streams
extern "C"
void initStream()
{
    for (int i = 0; i < 16; ++i) {
        cudaErr = cudaStreamCreate(&stream[i]);
        if ( cudaErr != cudaSuccess )
            printf("Stream creation failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }
}

extern "C"
void delStream()
{
    for (int i = 0; i < 8; ++i) {
        cudaErr = cudaStreamDestroy(stream[i]);
        if ( cudaErr != cudaSuccess )
            printf("Stream destruction failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }
}

// RL - initialize/free device memories
extern "C"
void CUDA_Setup(int call_type,
    int num_source, int num_cluster, int num_charge, int target_xyz_dim,
    FLOAT *source_x, FLOAT *source_y,  FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z,
    double *cluster_q, double *potential)
{
    if ( call_type == 1 || call_type == 3 ) {
        cudaErr = cudaMalloc(&d_source_x, sizeof(FLOAT)*num_source);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_source_y, sizeof(FLOAT)*num_source);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_source_z, sizeof(FLOAT)*num_source);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_source_q, sizeof(FLOAT)*num_source);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr = cudaMalloc(&d_potential, sizeof(FLOAT)*target_xyz_dim);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }

    if ( call_type == 1 ) {
        cudaErr = cudaMalloc(&d_cluster_x, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_cluster_y, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_cluster_z, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr = cudaMalloc(&d_cluster_q, sizeof(FLOAT)*num_charge);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }


    if ( call_type == 1 || call_type == 3 ) {
        cudaErr = cudaMemcpy(d_source_x, source_x, sizeof(FLOAT)*num_source, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_source_y, source_y, sizeof(FLOAT)*num_source, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_source_z, source_z, sizeof(FLOAT)*num_source, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_source_q, source_q, sizeof(FLOAT)*num_source, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr = cudaMemcpy(d_potential, potential, sizeof(FLOAT)*target_xyz_dim, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        //printf("CUDA copied data into device %d %d\n", num_source, target_xyz_dim);
    }

    if ( call_type == 1 ) {
        cudaErr = cudaMemcpy(d_cluster_x, cluster_x, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_cluster_y, cluster_y, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_cluster_z, cluster_z, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_cluster_q, cluster_q, sizeof(FLOAT)*num_charge, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        //printf("CUDA copied data into device %d %d\n", num_cluster, num_charge);
    }
    return;
}

extern "C"
void CUDA_Free(int call_type,
    int num_charge, int target_xyz_dim,
    double *cluster_q, double *potential)
{
    if ( call_type == 1 || call_type == 3 ) {
        cudaErr = cudaMemcpy(potential, d_potential,
                             target_xyz_dim * sizeof(FLOAT), cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess )
            printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaFree(d_source_x);
        cudaFree(d_source_y);
        cudaFree(d_source_z);
        cudaFree(d_source_q);
        cudaFree(d_potential);
    }

    if ( call_type == 1 ) {
        cudaErr = cudaMemcpy(cluster_q, d_cluster_q, sizeof(FLOAT)*num_charge, cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess )
            printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaFree(d_cluster_x);
        cudaFree(d_cluster_y);
        cudaFree(d_cluster_z);
        cudaFree(d_cluster_q);
    }

    return;
}


__global__ 
static void CUDA_TCF_PP(
    FLOAT eta, FLOAT kap, FLOAT kap_eta_2,
    int cluster_num_sources, int cluster_idx_start,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_yz_dim, int target_z_dim,
    FLOAT target_xmin, FLOAT target_ymin, FLOAT target_zmin,
    FLOAT target_xdd, FLOAT target_ydd, FLOAT target_zdd,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *potential)
{
    // ix/iy/iz always start from 0
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix < target_x_high_ind - target_x_low_ind + 1 &&
        iy < target_y_high_ind - target_y_low_ind + 1 &&
        iz < target_z_high_ind - target_z_low_ind + 1) {
        
        int ii = ((ix + target_x_low_ind) * target_yz_dim) +
                 ((iy + target_y_low_ind) * target_z_dim ) +
                  (iz + target_z_low_ind);

        FLOAT temporary_potential = 0.0;

        FLOAT tx = target_xmin + ix * target_xdd;
        FLOAT ty = target_ymin + iy * target_ydd;
        FLOAT tz = target_zmin + iz * target_zdd;

        for (int j=0; j < cluster_num_sources; j++){

            int jj = cluster_idx_start + j;
            FLOAT dx = tx - source_x[jj];
            FLOAT dy = ty - source_y[jj];
            FLOAT dz = tz - source_z[jj];
            FLOAT r  = sqrt(dx*dx + dy*dy + dz*dz);

            if (r > DBL_MIN) {
                FLOAT kap_r = kap * r;
                FLOAT r_eta = r / eta;
                temporary_potential += source_q[jj] / r 
                                     *(exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                                     - exp( kap_r) * erfc(kap_eta_2 + r_eta));
            }

        }
        potential[ii] += temporary_potential;

    }

    return;
}

__host__
void K_CUDA_TCF_PP(
    int call_type,         int num_source,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    FLOAT target_xmin,    FLOAT target_ymin,    FLOAT target_zmin,
    FLOAT target_xdd,     FLOAT target_ydd,     FLOAT target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    struct RunParams *run_params, double *potential, int stream_id)
{
    int target_yz_dim_glob = target_y_dim_glob * target_z_dim_glob;
    FLOAT kap = (FLOAT)run_params->kernel_params[0];
    FLOAT eta = (FLOAT)run_params->kernel_params[1];
    FLOAT kap_eta_2 = kap * eta / 2.0;

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
    CUDA_TCF_PP<<<nblocks,nthreads,0,stream[stream_id]>>>(eta, kap, kap_eta_2,
                    cluster_num_sources, cluster_idx_start,
                    target_x_low_ind, target_y_low_ind, target_z_low_ind,
                    target_x_high_ind, target_y_high_ind, target_z_high_ind,
                    target_yz_dim_glob, target_z_dim_glob,
                    target_xmin, target_ymin, target_zmin,
                    target_xdd, target_ydd, target_zdd,
                    d_source_x, d_source_y, d_source_z, d_source_q, d_potential);

    // RQ
    //cudaStreamSynchronize(stream[stream_id]);
    //cudaDeviceSynchronize();

    return;

}
