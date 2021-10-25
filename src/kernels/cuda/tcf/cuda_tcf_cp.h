/* Interaction Kernels */
#ifndef H_K_CUDA_TCF_CP_H
#define H_K_CUDA_TCF_CP_H
 
#include "../../../run_params/struct_run_params.h"


#ifdef __CUDACC__
extern "C" {
#endif
void K_CUDA_TCF_CP_Lagrange(
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    //double *h_temporary_potential, double *d_temporary_potential,
    struct RunParams *run_params, int gpu_async_stream_id);

#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_TCF_CP_H */
