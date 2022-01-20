#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../utilities/array.h"

#include "../particles/struct_particles.h"
#include "../run_params/struct_run_params.h"

#include "../kernels/coulomb/coulomb.h"
#include "../kernels/tcf/tcf.h"
#include "../kernels/dcf/dcf.h"

#ifdef CUDA_ENABLED
    //#define SINGLE
    #ifdef SINGLE
        #define FLOAT float
    #else
        #define FLOAT double
    #endif
    #include "../kernels/cuda/coulomb/cuda_coulomb.h"
    #include "../kernels/cuda/tcf/cuda_tcf.h"
    #include "../kernels/cuda/dcf/cuda_dcf.h"
#endif

#include "interaction_compute.h"


void InteractionCompute_Direct(double *potential,
                               struct Particles *sources, struct Particles *targets,
                               struct RunParams *run_params)
{

    int num_sources   = sources->num;
    double *source_x  = sources->x;
    double *source_y  = sources->y;
    double *source_z  = sources->z;
    double *source_q  = sources->q;

    double target_xdd = targets->xdd;
    double target_ydd = targets->ydd;
    double target_zdd = targets->zdd;

    double target_xmin = targets->xmin;
    double target_ymin = targets->ymin;
    double target_zmin = targets->zmin;

    int target_xdim = targets->xdim;
    int target_ydim = targets->ydim;
    int target_zdim = targets->zdim;

#ifdef SINGLE
    float *s_source_x  = (float*)malloc(sizeof(float)*num_sources);
    float *s_source_y  = (float*)malloc(sizeof(float)*num_sources);
    float *s_source_z  = (float*)malloc(sizeof(float)*num_sources);
    float *s_source_q  = (float*)malloc(sizeof(float)*num_sources);

    for (int i = 0; i < num_sources-1; i++) {
        s_source_x[i] = source_x[i];
        s_source_y[i] = source_y[i];
        s_source_z[i] = source_z[i];
        s_source_q[i] = source_q[i];
    }
#endif

#ifdef CUDA_ENABLED
    int call_type = 3;
    int target_xyz_dim = target_xdim*target_ydim*target_zdim;
    int num_clusters = 1;
    int num_charges = 1;
    double *cluster_x = NULL;
    double *cluster_y = NULL;
    double *cluster_z = NULL;
    double *cluster_q = NULL;
    CUDA_Setup(call_type, num_sources, num_clusters, num_charges, target_xyz_dim,
               source_x, source_y, source_z, source_q, cluster_x, cluster_y, cluster_z,
               cluster_q, potential);
#endif

/* * ********************************************************/
/* * ************** COMPLETE DIRECT SUM *********************/
/* * ********************************************************/


    /* * *************************************/
    /* * ******* Coulomb *********************/
    /* * *************************************/

    if (run_params->kernel == COULOMB) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
        K_CUDA_Coulomb_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,

            (float)target_xmin, (float)target_ymin, (float)target_zmin,
            (float)target_xdd,  (float)target_ydd,  (float)target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #else
        K_CUDA_Coulomb_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            
            target_xmin, target_ymin, target_zmin,
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #endif
#else
        K_Coulomb_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif


    /* * *************************************/
    /* * ******* TCF *************************/
    /* * *************************************/

    } else if (run_params->kernel == TCF) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
        K_CUDA_TCF_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,

            (float)target_xmin, (float)target_ymin, (float)target_zmin,
            (float)target_xdd,  (float)target_ydd,  (float)target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #else
        K_CUDA_TCF_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            
            target_xmin, target_ymin, target_zmin,
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #endif
#else
        K_TCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,

            target_xmin, target_ymin, target_zmin,
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif

                        
    /* * *************************************/
    /* * ******* DCF *************************/
    /* * *************************************/

    } else if (run_params->kernel == DCF) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
        K_CUDA_DCF_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,

            (float)target_xmin, (float)target_ymin, (float)target_zmin,
            (float)target_xdd,  (float)target_ydd,  (float)target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #else
        K_CUDA_DCF_PP(
            num_sources,

            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            
            target_xmin, target_ymin, target_zmin,
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,

            run_params, 0);
    #endif
#else
        K_DCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif
                        
    }

#ifdef CUDA_ENABLED
    CUDA_Free(call_type, num_charges, target_xyz_dim, cluster_q, potential);
#endif
#ifdef SINGLE
    free(s_source_x );
    free(s_source_y );
    free(s_source_z );
    free(s_source_q );
#endif
    return;
}
