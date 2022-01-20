#include <math.h>
#include <float.h>
#include <stdio.h>

//#define SINGLE

#ifdef SINGLE
    #define FLOAT float
#else
    #define FLOAT double
#endif

#include "cuda_coulomb_pp.h"
#include "../tcf/device_vars.h"

__global__ 
static void CUDA_Coulomb_PP(
    int cluster_num_sources, int cluster_idx_start,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_yz_dim, int target_z_dim,
    FLOAT target_xmin, FLOAT target_ymin, FLOAT target_zmin,
    FLOAT target_xdd, FLOAT target_ydd, FLOAT target_zdd,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,
    FLOAT *potential)
{
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
        atomicAdd(potential+ii, temporary_potential);
    }

    return;
}


__host__
void K_CUDA_Coulomb_PP(
    int num_source,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    FLOAT target_xmin,    FLOAT target_ymin,    FLOAT target_zmin,
    FLOAT target_xdd,     FLOAT target_ydd,     FLOAT target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    struct RunParams *run_params, int stream_id )
{
    int target_x_dim = target_x_high_ind - target_x_low_ind + 1;
    int target_y_dim = target_y_high_ind - target_y_low_ind + 1;
    int target_z_dim = target_z_high_ind - target_z_low_ind + 1;
    int target_yz_dim_glob = target_y_dim_glob * target_z_dim_glob;

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((target_x_dim-1)/threadsperblock + 1,
                 (target_y_dim-1)/threadsperblock + 1,
                 (target_z_dim-1)/threadsperblock + 1); 

    CUDA_Coulomb_PP<<<nblocks,nthreads,0,stream[stream_id]>>>(cluster_num_sources, cluster_idx_start,
                    target_x_low_ind, target_y_low_ind, target_z_low_ind,
                    target_x_high_ind, target_y_high_ind, target_z_high_ind,
                    target_yz_dim_glob, target_z_dim_glob,
                    target_xmin, target_ymin, target_zmin,
                    target_xdd, target_ydd, target_zdd,
                    d_source_x, d_source_y, d_source_z, d_source_q, d_potential);

    return;
}

