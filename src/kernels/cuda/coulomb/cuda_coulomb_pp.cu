#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_coulomb_pp.h"


__global__ 
static void CUDA_Coulomb_PP_Lagrange(int fid_high_ind, int cluster_num_sources, int cluster_idx_start,
                                     int target_x_low_ind, int target_y_low_ind, int target_z_low_ind,
                                     int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
                                     double target_xmin, double target_ymin, double target_zmin,
                                     double target_xdd, double target_ydd, double target_zdd,
                                     double *source_x, double *source_y, double *source_z, double *source_q,
                                     double *temporary_potential )
{
    int fid=threadIdx.x + blockDim.x * blockIdx.x;
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

    if(fid < fid_high_ind*cluster_num_sources){
        int tid = fid/cluster_num_sources;

        int j = fid-tid*cluster_num_sources;
        int ix = tid/target_yz_dim; int tmp = tid - ix*target_yz_dim;
        int iy = tmp/target_z_dim_glob;
        int iz = tmp%target_z_dim_glob;

        double tx = target_xmin + ix * target_xdd;
        double ty = target_ymin + iy * target_ydd;
        double tz = target_zmin + iz * target_zdd;

        int jj = cluster_idx_start + j;
        double dx = tx - source_x[jj];
        double dy = ty - source_y[jj];
        double dz = tz - source_z[jj];
        double r  = sqrt(dx*dx + dy*dy + dz*dz);
        if (r > DBL_MIN) {
            temporary_potential[j+cluster_num_sources*fid_high_ind] = source_q[jj] / r;
        }
    }
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
    struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{
    printf("Before Kernel call :::: \n");
    printf("1. Allocating Host and Device working memory ...\n");
    cudaError_t cudaErr;
    double *h_temp_pot,*d_temporary_potential,*potential2;

    int fid_high_ind = (target_x_high_ind - target_x_low_ind)*
                       (target_y_high_ind - target_y_low_ind)*
                       (target_z_high_ind - target_z_low_ind);
    cudaErr = cudaMallocHost(&h_temp_pot, sizeof(double)*cluster_num_sources*fid_high_ind);
    if ( cudaErr != cudaSuccess )
        printf("Host malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_temporary_potential, sizeof(double)*cluster_num_sources*fid_high_ind);
    if ( cudaErr != cudaSuccess )
        printf("Device malloc failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    printf("2. Sending Device memories to Kernel ...\n");

    int nthreads = 512;
    int nblocks = ((cluster_num_sources * fid_high_ind))/ nthreads + 1;

    CUDA_Coulomb_PP_Lagrange<<<nblocks,nthreads>>>(fid_high_ind, cluster_num_sources, cluster_idx_start,
                                     target_x_low_ind, target_y_low_ind, target_z_low_ind,
                                     target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
                                     target_xmin, target_ymin, target_zmin,
                                     target_xdd, target_ydd, target_zdd,
                                     source_x, source_y, source_z,source_q,
                                     d_temporary_potential );
        cudaErr = cudaDeviceSynchronize();
    if ( cudaErr != cudaSuccess )
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    printf("3. Downloading Device memory to Host ...\n");
    cudaErr = cudaMemcpy(h_temp_pot,d_temporary_potential, (cluster_num_sources) * fid_high_ind *sizeof(double),cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess )
        printf("Device to Host MemCpy failed with error \"%s\".\n", cudaGetErrorString(cudaErr));

    printf("4. Collecting cluster potentials on Host ...\n");
//       potential2 = (double*) malloc(11*cluster_num_sources * fid_high_ind *sizeof(double));
        int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
        for (int tid = 0; tid <= fid_high_ind; tid++) {
        int ix = tid/target_yz_dim; int tmp = tid - ix*target_yz_dim;
        int iy = tmp/target_z_dim_glob;
        int iz = tmp%target_z_dim_glob;
        ix = ix + target_x_low_ind;
        iy = iy + target_y_low_ind;
        iz = iz + target_z_low_ind;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
        for (int j = 0; j < cluster_num_sources; j++) {
            potential[ii] += h_temp_pot[j+cluster_num_sources*fid_high_ind];
            printf("new pp %i %15.6e\n",ii, potential[ii]);
        }
       }
    printf("5. Cleaning up working memory ...\n");
    cudaFree(h_temp_pot);
    cudaFree(d_temporary_potential);

    printf("Exiting :::: \n");

    return;
}

