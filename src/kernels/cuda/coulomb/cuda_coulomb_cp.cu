#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cuda.h>

#include "cuda_coulomb_cp.h"


__global__ 
static void CUDA_Coulomb_CP_Lagrange(int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q)
{
    printf("Hello Coulomb_CP thread %d, block %d\n", threadIdx.x, blockIdx.x);
          int i=threadIdx.x + blockDim.x * blockIdx.x;
          int j=blockDim.x * gridDim.x;
          int p=0;
          double dx, dy,dz,r2,tp,potential;
      for( int k=i; k<interp_order_lim; k +=j){
                     p=i-((floorf(i/(batch_num_sources)))*(batch_num_sources));
                     dx= cluster_x[k]-source_x[p];
                     dy= cluster_y[k]-source_y[p];
                     dz= cluster_z[k]-source_z[p];

                      r2=(dx*dx)+(dy*dy)+(dz*dz);

}


}


__host__
void K_CUDA_Coulomb_CP_Lagrange(
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    CUDA_Coulomb_CP_Lagrange<<<1,32>>>(batch_num_sources, batch_idx_start,
                      cluster_q_start, cluster_pts_start, interp_order_lim,
                      source_x,  source_y,  source_z,  source_q,
                      cluster_x, cluster_y, cluster_z, cluster_q);
    cudaDeviceSynchronize();

    return;
}
