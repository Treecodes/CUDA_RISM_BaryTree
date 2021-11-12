/* Interaction Kernels */
#ifndef H_K_CUDA_COULOMB_PP_H
#define H_K_CUDA_COULOMB_PP_H
 
#include "../../../run_params/struct_run_params.h"

#ifdef __CUDACC__
extern "C" {
#endif
void K_CUDA_Coulomb_PP(
    int call_type,         int num_source,
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    
    FLOAT target_xmin,    FLOAT target_ymin,    FLOAT target_zmin,
    FLOAT target_xdd,     FLOAT target_ydd,     FLOAT target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int number_of_source_points_in_cluster, int starting_index_of_source,
    FLOAT *source_x, FLOAT *source_y, FLOAT *source_z, FLOAT *source_q,

    struct RunParams *run_params, double *potential);
#ifdef __CUDACC__
}
#endif

#endif /* H_K_CUDA_COULOMB_PP_H */
