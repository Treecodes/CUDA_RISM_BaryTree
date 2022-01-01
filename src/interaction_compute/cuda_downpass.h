/* Interaction Kernels */
#ifndef H_K_CUDA_DOWNPASS_H
#define H_K_CUDA_DOWNPASS_H
 
#include "../run_params/struct_run_params.h"

#ifdef __CUDACC__


extern "C" {
#endif
// RQ
void initStream2();
void delStream2();
// RL
void CUDA_Setup2(int call_type,
    int sizeof_coeff_x, int sizeof_coeff_y, int sizeof_coeff_z, int num_charge, int target_xyz_dim,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z,
    double *cluster_q, double *potential);
void CUDA_Free2(int call_type,
    int target_xyz_dim, double *potential);

void K_CUDA_CP_COMP_POT(
    int call_type, int idx, double *potential, int interp_order,
        int target_x_low_ind,  int target_x_high_ind,
        int target_y_low_ind,  int target_y_high_ind,
        int target_z_low_ind,  int target_z_high_ind,
        int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
        double *cluster_q,
        int coeff_x_start, double *coeff_x,
        int coeff_y_start, double *coeff_y,
        int coeff_z_start, double *coeff_z,
        struct RunParams *run_params, int stream_id);
#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_TCF_PP_H */
