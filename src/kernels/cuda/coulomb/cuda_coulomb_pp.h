/* Interaction Kernels */
#ifndef H_K_CUDA_COULOMB_PP_H
#define H_K_CUDA_COULOMB_PP_H
 
#include "../../../run_params/struct_run_params.h"

#ifdef __CUDACC__
extern "C" {
#endif
void K_CUDA_Coulomb_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
    
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int number_of_source_points_in_cluster, int starting_index_of_source,
    double *source_x, double *source_y, double *source_z, double *source_q,

    struct RunParams *run_params, int gpu_async_stream_id);
#ifdef __CUDACC__
}
#endif

#ifdef __CUDACC__
extern "C" {
#endif
void CUDA_Setup_PP(int target_xyz_dim);
#ifdef __CUDACC__
}
#endif

#ifdef __CUDACC__
extern "C" {
#endif
void CUDA_Cleanup_PP(int target_xyz_dim, double *potential);
#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_COULOMB_PP_H */
