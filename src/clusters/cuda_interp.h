/* Interaction Kernels */
#ifndef H_K_CUDA_INTERP_H
#define H_K_CUDA_INTERP_H
 
#include "../run_params/struct_run_params.h"

#ifdef __CUDACC__
extern "C" {
#endif
// EA
void CUDA_Setup_Interp(
    int totalNumberInterpolationPoints,
    FLOAT *xC, FLOAT *yC, FLOAT *zC);
void CUDA_Free_Interp(int totalNumberInterpolationPoints,
    FLOAT *xC, FLOAT *yC, FLOAT *zC);

void K_CUDA_COMP_INTERP( 
    const struct Tree *tree, int idx, int interpolationOrder,
    int stream_id);
#ifdef __CUDACC__
}
#endif


#endif /* H_K_CUDA_TCF_PP_H */
