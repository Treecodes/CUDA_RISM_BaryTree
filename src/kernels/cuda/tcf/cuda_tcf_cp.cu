#include <math.h>
#include <float.h>
#include <stdio.h>

//#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_tcf_cp.h"


__global__ 
void  CUDA_TCF_CP_Lagrange(
    FLOAT eta, FLOAT kap, FLOAT kap_eta_2,
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z, FLOAT *d_potential)
{
    int k1 = threadIdx.x + blockDim.x * blockIdx.x;
    int k2 = threadIdx.y + blockDim.y * blockIdx.y;
    int k3 = threadIdx.z + blockDim.z * blockIdx.z;

    if ( k1 >=0 && k1 < interp_order_lim &&
         k2 >=0 && k2 < interp_order_lim &&
         k3 >=0 && k3 < interp_order_lim ){

        FLOAT temporary_potential = 0.0;

        FLOAT cx = cluster_x[cluster_pts_start + k1];
        FLOAT cy = cluster_y[cluster_pts_start + k2];
        FLOAT cz = cluster_z[cluster_pts_start + k3];

        int ii = cluster_q_start + k1 * interp_order_lim*interp_order_lim + k2 * interp_order_lim + k3;

        for (int j = 0; j < batch_num_sources; j++) {

            int jj = batch_idx_start + j;
            FLOAT dx = cx - source_x[jj];
            FLOAT dy = cy - source_y[jj];
            FLOAT dz = cz - source_z[jj];
            FLOAT r = sqrt(dx*dx + dy*dy + dz*dz);

            if (r > DBL_MIN) {
                FLOAT kap_r = kap * r;
                FLOAT r_eta = r / eta;
                temporary_potential += source_q[jj] / r
                                     * (exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                                     -  exp( kap_r) * erfc(kap_eta_2 + r_eta));
            }
        } // end loop over interpolation points
        d_potential[ii] += temporary_potential;
    }
    return;
}

__host__
void CUDA_Setup_CP(
    int num_source, int num_cluster,
    FLOAT *source_x, FLOAT *source_y,  FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z,
    FLOAT *d_source_x, FLOAT *d_source_y, FLOAT *d_source_z, FLOAT *d_source_q,
    FLOAT *d_cluster_x, FLOAT *d_cluster_y, FLOAT *d_cluster_z)
{

    return;
}


__host__
void CUDA_Free_CP(
    FLOAT *d_source_x, FLOAT *d_source_y, FLOAT *d_source_z, FLOAT *d_source_q,
    FLOAT *d_cluster_x, FLOAT *d_cluster_y, FLOAT *d_cluster_z)
{

    return;
}


__host__
void K_CUDA_TCF_CP_Lagrange(
    int call_type, int num_source, int num_cluster, int num_charge,
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z, double *cluster_q,
    struct RunParams *run_params)
{
    cudaError_t cudaErr;
    FLOAT *d_source_x;
    FLOAT *d_source_y; 
    FLOAT *d_source_z;
    FLOAT *d_source_q;
    FLOAT *d_cluster_x;
    FLOAT *d_cluster_y;
    FLOAT *d_cluster_z;
    FLOAT *d_potential;

    printf("TCF_CP received call_type: %d\n", call_type);
    if ( call_type == 1 ) {
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

        cudaErr = cudaMalloc(&d_cluster_x, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_cluster_y, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMalloc(&d_cluster_z, sizeof(FLOAT)*num_cluster);
        if ( cudaErr != cudaSuccess )
            printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaErr = cudaMalloc(&d_potential, sizeof(FLOAT)*num_charge);
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
        cudaErr = cudaMemcpy(d_cluster_x, cluster_x, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_cluster_y, cluster_y, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_cluster_z, cluster_z, sizeof(FLOAT)*num_cluster, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
        cudaErr = cudaMemcpy(d_potential, cluster_q, sizeof(FLOAT)*num_charge, cudaMemcpyHostToDevice);
        if ( cudaErr != cudaSuccess )
            printf("Host to Device MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    }

    FLOAT kap = (FLOAT)run_params->kernel_params[0];
    FLOAT eta = (FLOAT)run_params->kernel_params[1];
    FLOAT kap_eta_2 = kap * eta / 2.0;

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((interp_order_lim-1)/threadsperblock + 1,
                 (interp_order_lim-1)/threadsperblock + 1,
                 (interp_order_lim-1)/threadsperblock + 1);
    CUDA_TCF_CP_Lagrange<<<nblocks,nthreads>>>(eta, kap, kap_eta_2,
                    batch_num_sources, batch_idx_start,
                    cluster_q_start, cluster_pts_start, interp_order_lim,
                    d_source_x,  d_source_y,  d_source_z,  d_source_q,
                    d_cluster_x, d_cluster_y, d_cluster_z, d_potential);
    //cudaErr = cudaDeviceSynchronize();
    //if ( cudaErr != cudaSuccess )
    //    printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    if ( call_type == 2 ) {
        cudaErr = cudaMemcpy(cluster_q, d_potential, sizeof(FLOAT)*num_charge, cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess )
            printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

        cudaFree(d_source_x);
        cudaFree(d_source_y);
        cudaFree(d_source_z);
        cudaFree(d_source_q);
        cudaFree(d_cluster_x);
        cudaFree(d_cluster_y);
        cudaFree(d_cluster_z);
        cudaFree(d_potential);
    }

    return;
}

