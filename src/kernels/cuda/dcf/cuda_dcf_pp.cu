#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_dcf_pp.h"

__managed__ double *d_potential;

__global__ 
static void CUDA_DCF_PP(
    double eta, int cluster_num_sources, int cluster_idx_start,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    double target_xmin, double target_ymin, double target_zmin,
    double target_xdd, double target_ydd, double target_zdd,
    double *source_x, double *source_y, double *source_z, double *source_q)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

    if (ix >= target_x_low_ind && ix <= target_x_high_ind &&
        iy >= target_y_low_ind && iy <= target_y_high_ind &&
        iz >= target_z_low_ind && iz <= target_z_high_ind){

        double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
        double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
        double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
        double temporary_potential = 0.0;
        for (int j=0;j < cluster_num_sources;j++){

            int jj = cluster_idx_start + j;
            double dx = tx - source_x[jj];
            double dy = ty - source_y[jj];
            double dz = tz - source_z[jj];
            double r  = sqrt(dx*dx + dy*dy + dz*dz);
            if (r > DBL_MIN) {
                temporary_potential += source_q[jj] * erf(r / eta)  / r;
            }
        }
        d_potential[ii]+= temporary_potential;

    }

    return;
}


__host__
void K_CUDA_DCF_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    double eta = run_params->kernel_params[1];

    cudaError_t cudaErr;

    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((target_x_high_ind-target_x_low_ind)/threadsperblock + 1,
                 (target_y_high_ind-target_y_low_ind)/threadsperblock + 1,
                 (target_z_high_ind-target_z_low_ind)/threadsperblock + 1);
    CUDA_DCF_PP<<<nblocks,nthreads>>>(eta,cluster_num_sources, cluster_idx_start,
                                    target_x_low_ind,target_y_low_ind,target_z_low_ind,
                                    target_x_high_ind,target_y_high_ind,target_z_high_ind,
                                    target_x_dim_glob,target_y_dim_glob,target_z_dim_glob,
                                    target_xmin,target_ymin,target_zmin,
                                    target_xdd,target_ydd,target_zdd,
                                    source_x, source_y, source_z, source_q);
    cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

#ifdef TESTDIRECT
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    int target_xyz_dim = target_x_dim_glob * target_yz_dim;
    double *h_potential;
    cudaErr = cudaMallocHost(&h_potential, sizeof(double)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(h_potential, potential,
                         target_xyz_dim * sizeof(double), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
    for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
    for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {
        int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
        printf("direct potential, %d %15.6e\n", ii, h_potential[ii]);
    }
    }
    }
    cudaFree(h_potential);
    exit(0);
#endif

    return;
}
