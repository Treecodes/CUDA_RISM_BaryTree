/* Interaction Kernels */
#ifndef H_K_CUDA_DOWNPASS_H
#define H_K_CUDA_DOWNPASS_H
 
#include "../run_params/struct_run_params.h"

#ifdef __CUDACC__
extern "C" {
#endif
// RL
void CUDA_Setup2(
    int sizeof_coeff_x, int sizeof_coeff_y, int sizeof_coeff_z,
    FLOAT *coeff_x, FLOAT *coeff_y, FLOAT *coeff_z);
void CUDA_Free2(void);
void CUDA_Wrapup(
    int target_xyz_dim, double *potential);

void K_CUDA_CP_COMP_DOWNPASS(
    int idx, int child_idx, int interp_order, int coeff_start,
    int stream_id);
void K_CUDA_CP_COMP_POT(
    int idx, int interp_order,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int coeff_x_start,
    int coeff_y_start,
    int coeff_z_start,
    int stream_id);
#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_TCF_PP_H */
