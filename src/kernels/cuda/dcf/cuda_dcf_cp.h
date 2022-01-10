/* Interaction Kernels */
#ifndef H_K_CUDA_DCF_CP_H
#define H_K_CUDA_DCF_CP_H
 
#include "../../../run_params/struct_run_params.h"


#ifdef __CUDACC__
extern "C" {
#endif
void K_CUDA_DCF_CP_Lagrange(
    int call_type, int num_source, int num_cluster,
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    struct RunParams *run_params, int stream_id);

#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_DCF_CP_H */
