#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_tcf_cp.h"


__global__ 
void  CUDA_TCF_CP_Lagrange(double kap, int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *temporary_potential)
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
        double cx = cluster_x[cluster_pts_start + k1];
        double cy = cluster_y[cluster_pts_start + k2];
        double cz = cluster_z[cluster_pts_start + k3];

        int jj = batch_idx_start + j;
        double dx = cx - source_x[jj];
        double dy = cy - source_y[jj];
        double dz = cz - source_z[jj];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        temporary_potential[j+batch_num_sources*cid] = source_q[jj] / r * 2. * exp(-kap * r);
    }
}

__host__
void K_CUDA_TCF_CP_Lagrange(
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    double kap = run_params->kernel_params[0];
    cudaError_t cudaErr;

    double *h_temporary_potential, *d_temporary_potential;
    int lim3 = interp_order_lim*interp_order_lim*interp_order_lim;
    cudaErr = cudaMallocHost(&h_temporary_potential, sizeof(double)*(batch_num_sources*lim3));
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_temporary_potential, sizeof(double)*(batch_num_sources*lim3));
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    int nthreads = 256;
    int nblocks = (batch_num_sources * lim3 - 1) / nthreads + 1;
    CUDA_TCF_CP_Lagrange<<<nblocks,nthreads>>>(kap, batch_num_sources, batch_idx_start,
                    cluster_q_start, cluster_pts_start, interp_order_lim,
                    source_x,  source_y,  source_z,  source_q,
                    cluster_x, cluster_y, cluster_z, d_temporary_potential);
    cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(h_temporary_potential, d_temporary_potential,
                    sizeof(double)*(batch_num_sources*lim3), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    for (int cid = 0; cid < lim3; cid++) {
        int ii = cluster_q_start + cid;
        for (int j = 0; j < batch_num_sources; j++) {
            cluster_q[ii] += h_temporary_potential[j+batch_num_sources*cid];
        }
        //printf("new %i %15.6e\n", cid, cluster_q[ii]);
    }

    cudaFree(h_temporary_potential);
    cudaFree(d_temporary_potential);

    return;
}
