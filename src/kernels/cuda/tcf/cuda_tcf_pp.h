/* Interaction Kernels */
#ifndef H_K_CUDA_TCF_PP_H
#define H_K_CUDA_TCF_PP_H
 
#include "../../../run_params/struct_run_params.h"

#ifdef __CUDACC__


extern "C" {
#endif
// RQ
void initStream();
void delStream();
// RL
void CUDA_Setup(int call_type,
    int num_source, int num_cluster, int num_charge, int target_xyz_dim,
    FLOAT *source_x, FLOAT *source_y,  FLOAT *source_z, FLOAT *source_q,
    FLOAT *cluster_x, FLOAT *cluster_y, FLOAT *cluster_z,
    double *cluster_q, double *potential);
void CUDA_Free(int call_type,
    int num_charge, int target_xyz_dim,
    double *cluster_q, double *potential);

void K_CUDA_TCF_PP(
    int num_source,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,

    FLOAT target_xmin,    FLOAT target_ymin,    FLOAT target_zmin,
    FLOAT target_xdd,     FLOAT target_ydd,     FLOAT target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int cluster_num_sources, int cluster_idx_start,
    struct RunParams *run_params, int stream_id);
#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_TCF_PP_H */
