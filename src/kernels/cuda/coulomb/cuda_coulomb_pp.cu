#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_coulomb_pp.h"

__global__ 
static void CUDA_Coulomb_PP(
    int cluster_num_sources, int cluster_idx_start,
    int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
    int target_x_high_ind, int target_y_high_ind, int target_z_high_ind,
    int target_yz_dim, int target_z_dim,
    double target_xmin, double target_ymin, double target_zmin,
    double target_xdd, double target_ydd, double target_zdd,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *d_potential)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix >= target_x_low_ind && ix <= target_x_high_ind &&
        iy >= target_y_low_ind && iy <= target_y_high_ind &&
        iz >= target_z_low_ind && iz <= target_z_high_ind){

        double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
        double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
        double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim) + iz;
        double temporary_potential = 0.0;
        for (int j=0;j < cluster_num_sources;j++){

            int jj = cluster_idx_start + j;
            double dx = tx - source_x[jj];
            double dy = ty - source_y[jj];
            double dz = tz - source_z[jj];
            double r  = sqrt(dx*dx + dy*dy + dz*dz);
            printf("r, %15.6e\n",r);
            printf("source_q, %15.6e\n",source_q[jj]);
            if (r > DBL_MIN) {
                temporary_potential += source_q[jj] / r;
               /// printf("temporary potential, %15.6e\n",temporary_potential);
            }
        }
        d_potential[ii]+= temporary_potential;
      ///   printf("device potential, %15.6e\n",d_potential[ii]);
    }

    return;
}


__host__
void K_CUDA_Coulomb_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,
    struct RunParams *run_params, double *potential, int gpu_async_stream_id )
{
    for( int i = 0; i<= cluster_num_sources; i++){
         printf("source_x, source_y, source_z, source_q, %15.6e, %15.6e, %15.6e, %15.6e\n", source_x[i], source_y[i], 
              source_z[i], source_q[i]);}
    cudaError_t cudaErr;

<<<<<<< HEAD
///    double *d_source_x, *d_source_y, *d_source_z, *d_source_q;
    cudaMallocManaged( &source_x, cluster_num_sources * sizeof(double));
    cudaMallocManaged( &source_y, cluster_num_sources * sizeof(double));
    cudaMallocManaged( &source_z, cluster_num_sources * sizeof(double));
   cudaErr =  cudaMallocManaged( &source_q, cluster_num_sources * sizeof(double));
if ( cudaErr != cudaSuccess )
        printf("MallocManaged failed \"%s\".\n", cudaGetErrorString(cudaErr));
///    cudaErr = cudaMemcpy(source_x, d_source_x, cluster_num_sources * sizeof(double), cudaMemcpyHostToDevice);
///if ( cudaErr != cudaSuccess )
///        printf("Memcpy failed \"%s\".\n", cudaGetErrorString(cudaErr));
///    cudaMemcpy(source_y, d_source_y, cluster_num_sources * sizeof(double), cudaMemcpyHostToDevice);
///    cudaMemcpy(source_z, d_source_z, cluster_num_sources * sizeof(double), cudaMemcpyHostToDevice);
///    cudaMemcpy(source_q, d_source_q, cluster_num_sources * sizeof(double), cudaMemcpyHostToDevice);
 
=======
    int target_x_dim = target_x_high_ind - target_x_low_ind + 1;
    int target_y_dim = target_y_high_ind - target_y_low_ind + 1;
    int target_z_dim = target_z_high_ind - target_z_low_ind + 1;
    int target_yz_dim = target_y_dim * target_z_dim;
    int target_xyz_dim = target_x_dim * target_yz_dim;

    double *h_potential;
    double *d_potential;
    cudaErr = cudaMallocHost(&h_potential, sizeof(double)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_potential, sizeof(double)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

>>>>>>> 2e87ce7deba0232897c9cd74fe8b4cbca8e46b05
    int threadsperblock = 8;
    dim3 nthreads(threadsperblock, threadsperblock, threadsperblock);
    dim3 nblocks((target_x_dim-1)/threadsperblock + 1,
                 (target_y_dim-1)/threadsperblock + 1,
                 (target_z_dim-1)/threadsperblock + 1); 
    CUDA_Coulomb_PP<<<nblocks,nthreads>>>(cluster_num_sources, cluster_idx_start,
                                    target_x_low_ind, target_y_low_ind, target_z_low_ind,
                                    target_x_high_ind, target_y_high_ind, target_z_high_ind,
                                    target_yz_dim, target_z_dim,
                                    target_xmin, target_ymin, target_zmin,
                                    target_xdd, target_ydd, target_zdd,
                                    source_x, source_y, source_z, source_q, d_potential);
    cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

<<<<<<< HEAD
///#ifdef TESTDIRECT
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    int target_xyz_dim = target_x_dim_glob * target_yz_dim;
    double *h_potential;
    cudaErr = cudaMallocHost(&h_potential, sizeof(double)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
=======
>>>>>>> 2e87ce7deba0232897c9cd74fe8b4cbca8e46b05
    cudaErr = cudaMemcpy(h_potential, d_potential,
                         target_xyz_dim * sizeof(double), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    int target_yz_dim_glob = target_y_dim_glob * target_z_dim_glob;
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
    for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
    for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {
        int ii = (ix * target_yz_dim) + (iy * target_z_dim) + iz;
        int ix_glob = ix + target_x_low_ind;
        int iy_glob = iy + target_y_low_ind;
        int iz_glob = iz + target_z_low_ind;
        int ii_glob = (ix_glob * target_yz_dim_glob) + (iy_glob * target_z_dim_glob) + iz_glob;
        potential[ii_glob] = h_potential[ii];
        //printf("direct potential, %d %15.6e\n", ii, h_potential[ii]);
    }
    }
    }
    cudaFree(h_potential);
<<<<<<< HEAD
    exit(0);
///#endif

    return;
}

__host__
void CUDA_Setup_PP( int target_xyz_dim )
{
    cudaError_t cudaErr = cudaMallocManaged(&d_potential, sizeof(double)*target_xyz_dim);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
=======
    cudaFree(d_potential);
>>>>>>> 2e87ce7deba0232897c9cd74fe8b4cbca8e46b05

    return;
}

