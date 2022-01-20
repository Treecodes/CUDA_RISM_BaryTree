#include <math.h>
#include <float.h>
#include <stdio.h>

#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_tcf_cp.h"
#include "device_vars.h"

__global__ 
void  CUDA_TCF_CP_Lagrange(
    FLOAT eta, FLOAT kap, FLOAT kap_eta_2,
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z,
    double *potential)
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

            //if (r > DBL_MIN) {
            FLOAT kap_r = kap * r;
            FLOAT r_eta = r / eta;
            temporary_potential += source_q[jj] / r
                                 * (exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                                 -  exp( kap_r) * erfc(kap_eta_2 + r_eta));
            //}

        } // end loop over interpolation points

        atomicAdd(potential+ii, (double)temporary_potential);

    }
    return;
}

__host__
void K_CUDA_TCF_CP_Lagrange(
    int num_source, int num_cluster, int num_charge,
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    struct RunParams *run_params, int stream_id)
{
    FLOAT kap = (FLOAT)run_params->kernel_params[0];
    FLOAT eta = (FLOAT)run_params->kernel_params[1];
    FLOAT kap_eta_2 = kap * eta / 2.0;

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((interp_order_lim-1)/threadsperblock + 1,
                 (interp_order_lim-1)/threadsperblock + 1,
                 (interp_order_lim-1)/threadsperblock + 1);

    CUDA_TCF_CP_Lagrange<<<nblocks,nthreads,0,stream[stream_id]>>>(eta, kap, kap_eta_2,
                    batch_num_sources, batch_idx_start,
                    cluster_q_start, cluster_pts_start, interp_order_lim,
                    d_source_x,  d_source_y,  d_source_z,  d_source_q,
                    d_cluster_x, d_cluster_y, d_cluster_z, d_cluster_q);

    return;
}

