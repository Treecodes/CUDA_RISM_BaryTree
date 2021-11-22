#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cuda.h>

//#define SINGLE

#ifdef SINGLE
#define FLOAT float
#else
#define FLOAT double
#endif

#include "cuda_coulomb_cp.h"


__global__ 
void CUDA_Coulomb_CP_Lagrange(int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z,
    FLOAT *temporary_potential )
{
    int fid=threadIdx.x + blockDim.x * blockIdx.x;
    int cid_lim2 = interp_order_lim*interp_order_lim;
    int cid_lim3 = interp_order_lim*cid_lim2;
    if (fid < batch_num_sources * cid_lim3){
        int cid = fid/batch_num_sources;
        int j = fid-cid*batch_num_sources;
        int k1 = cid/cid_lim2; int tmp = cid - k1*cid_lim2;
        int k2 = tmp/interp_order_lim;
        int k3 = tmp%interp_order_lim;
        FLOAT cx = cluster_x[cluster_pts_start + k1];
        FLOAT cy = cluster_y[cluster_pts_start + k2];
        FLOAT cz = cluster_z[cluster_pts_start + k3];

        int jj = batch_idx_start + j;
        FLOAT dx = cx - source_x[jj];
        FLOAT dy = cy - source_y[jj];
        FLOAT dz = cz - source_z[jj];
        FLOAT r = sqrt(dx*dx + dy*dy + dz*dz);
        temporary_potential[j+batch_num_sources*cid] = source_q[jj] / r;
    }
}

__host__
void K_CUDA_Coulomb_CP_Lagrange(
    int call_type, int num_source, int num_cluster,
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z, double *cluster_q,
    struct RunParams *run_params)
{
    FLOAT *d_source_x;
    FLOAT *d_source_y; 
    FLOAT *d_source_z;
    FLOAT *d_source_q;
    FLOAT *d_cluster_x;
    FLOAT *d_cluster_y;
    FLOAT *d_cluster_z;

    //printf("CUDA received call_type: %d\n", call_type);
    cudaError_t cudaErr;
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
        //printf("CUDA copied data into device %d %d\n", num_source, num_cluster);
    }

    FLOAT *h_temporary_potential, *d_temporary_potential;
    int lim3 = interp_order_lim*interp_order_lim*interp_order_lim;
    cudaErr = cudaMallocHost(&h_temporary_potential, sizeof(FLOAT)*(batch_num_sources*lim3));
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_temporary_potential, sizeof(FLOAT)*(batch_num_sources*lim3));
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    int nthreads = 256;
    int nblocks = (batch_num_sources*lim3 - 1)/ nthreads + 1;
    CUDA_Coulomb_CP_Lagrange<<<nblocks,nthreads>>>(batch_num_sources, batch_idx_start,
                    cluster_q_start, cluster_pts_start, interp_order_lim,
                    d_source_x,  d_source_y,  d_source_z,  d_source_q,
                    d_cluster_x, d_cluster_y, d_cluster_z, d_temporary_potential);
    cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(h_temporary_potential, d_temporary_potential,
                    sizeof(FLOAT)*(batch_num_sources*lim3), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    for (int cid = 0; cid < lim3; cid++) {
        int ii = cluster_q_start + cid;
        for (int j = 0; j < batch_num_sources; j++) {
            cluster_q[ii] += (double)h_temporary_potential[j+batch_num_sources*cid];
        }
    }

    cudaFree(h_temporary_potential);
    cudaFree(d_temporary_potential);
    if ( call_type == 2 ) {
        cudaFree(d_source_x);
        cudaFree(d_source_y);
        cudaFree(d_source_z);
        cudaFree(d_source_q);
        cudaFree(d_cluster_x);
        cudaFree(d_cluster_y);
        cudaFree(d_cluster_z);
    }

    return;
}
