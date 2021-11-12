#include <math.h>
#include <float.h>
#include <stdio.h>

#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_coulomb_pp.h"

__global__ 
static void CUDA_Coulomb_PP(
    int cluster_num_sources, int cluster_idx_start,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_yz_dim, int target_z_dim,
    FLOAT target_xmin, FLOAT target_ymin, FLOAT target_zmin,
    FLOAT target_xdd, FLOAT target_ydd, FLOAT target_zdd,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *d_potential)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix >= target_x_low_ind && ix <= target_x_high_ind &&
        iy >= target_y_low_ind && iy <= target_y_high_ind &&
        iz >= target_z_low_ind && iz <= target_z_high_ind){

        FLOAT tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
        FLOAT ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
        FLOAT tz = target_zmin + (iz - target_z_low_ind) * target_zdd;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim) + iz;
        FLOAT temporary_potential = 0.0;
        for (int j=0;j < cluster_num_sources;j++){

            int jj = cluster_idx_start + j;
            FLOAT dx = tx - source_x[jj];
            FLOAT dy = ty - source_y[jj];
            FLOAT dz = tz - source_z[jj];
            FLOAT r  = sqrt(dx*dx + dy*dy + dz*dz);
            if (r > DBL_MIN) {
                temporary_potential += source_q[jj] / r;
            }
        }
        d_potential[ii]+= temporary_potential;
    }

    return;
}


__host__
void K_CUDA_Coulomb_PP(
    int call_type,         int num_source,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    FLOAT target_xmin,    FLOAT target_ymin,    FLOAT target_zmin,
    FLOAT target_xdd,     FLOAT target_ydd,     FLOAT target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    struct RunParams *run_params, double *potential )
{
    FLOAT *d_source_x;
    FLOAT *d_source_y;
    FLOAT *d_source_z;
    FLOAT *d_source_q;

    printf("CUDA received call_type: %d\n", call_type);
    cudaError_t cudaErr;
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
        printf("CUDA copied data into device %d\n", num_source);
    }

    int target_x_dim = target_x_high_ind - target_x_low_ind + 1;
    int target_y_dim = target_y_high_ind - target_y_low_ind + 1;
    int target_z_dim = target_z_high_ind - target_z_low_ind + 1;
    int target_yz_dim = target_y_dim * target_z_dim;
    int target_xyz_dim = target_x_dim * target_yz_dim;

    FLOAT *h_potential;
    FLOAT *d_potential;
    cudaErr = cudaMallocHost(&h_potential, sizeof(FLOAT)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_potential, sizeof(FLOAT)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((target_x_dim-1)/threadsperblock + 1,
                 (target_y_dim-1)/threadsperblock + 1,
                 (target_z_dim-1)/threadsperblock + 1); 
    CUDA_Coulomb_PP<<<nblocks,nthreads>>>(cluster_num_sources, cluster_idx_start,
                                    0, 0, 0,
                                    target_x_dim-1, target_y_dim-1, target_z_dim-1,
                                    target_yz_dim, target_z_dim,
                                    target_xmin, target_ymin, target_zmin,
                                    target_xdd, target_ydd, target_zdd,
                                    d_source_x, d_source_y, d_source_z, d_source_q, d_potential);
    cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(h_potential, d_potential,
                         target_xyz_dim * sizeof(FLOAT), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    int target_yz_dim_glob = target_y_dim_glob * target_z_dim_glob;
    //printf("grid block x low/high %d %d\n", target_x_low_ind, target_x_high_ind);
    //printf("grid block y low/high %d %d\n", target_y_low_ind, target_y_high_ind);
    //printf("grid block z low/high %d %d\n", target_z_low_ind, target_z_high_ind);
    for (int ix_glob = target_x_low_ind; ix_glob <= target_x_high_ind; ix_glob++) {
    for (int iy_glob = target_y_low_ind; iy_glob <= target_y_high_ind; iy_glob++) {
    for (int iz_glob = target_z_low_ind; iz_glob <= target_z_high_ind; iz_glob++) {
        int ii_glob = (ix_glob * target_yz_dim_glob) + (iy_glob * target_z_dim_glob) + iz_glob;
        int ix = ix_glob - target_x_low_ind;
        int iy = iy_glob - target_y_low_ind;
        int iz = iz_glob - target_z_low_ind;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim ) + iz;
        potential[ii_glob] += (double)h_potential[ii];
        //printf("direct potential, %d %15.6e\n", ii_glob, h_potential[ii]);
    }
    }
    }

    cudaFree(h_potential);
    cudaFree(d_potential);
    if ( call_type == 1 || call_type == 3 ) {
        cudaFree(d_source_x);
        cudaFree(d_source_y);
        cudaFree(d_source_z);
        cudaFree(d_source_q);
    }

    return;
}

