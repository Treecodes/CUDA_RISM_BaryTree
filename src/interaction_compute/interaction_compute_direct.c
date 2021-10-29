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

    int num_targets   = targets->num;

    double target_xdd = targets->xdd;
    double target_ydd = targets->ydd;
    double target_zdd = targets->zdd;

    double target_xmin = targets->xmin;
    double target_ymin = targets->ymin;
    double target_zmin = targets->zmin;

    int target_xdim = targets->xdim;
    int target_ydim = targets->ydim;
    int target_zdim = targets->zdim;

///#ifdef OPENACC_ENABLED
///    #pragma acc data copyin(source_x[0:num_sources], source_y[0:num_sources], \
                            source_z[0:num_sources], source_q[0:num_sources]) 
///    {
///#endif

/* * ********************************************************/
/* * ************** COMPLETE DIRECT SUM *********************/
/* * ********************************************************/


#ifdef CUDA_ENABLED
///    #pragma acc host_data use_device( \
  ///              source_x, source_y, source_z, source_q)
    {
    //int target_xyz_dim = target_xdim * target_ydim * target_zdim;
    //CUDA_Setup_PP(target_xyz_dim);
#endif
    /* * *************************************/
    /* * ******* Coulomb *********************/
    /* * *************************************/

    if (run_params->kernel == COULOMB) {

#ifdef CUDA_ENABLED
        K_CUDA_Coulomb_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
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
        K_CUDA_TCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
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
        K_CUDA_DCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
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
    //CUDA_Cleanup_PP(target_xyz_dim, potential);
    }
#endif

///#ifdef OPENACC_ENABLED
///    #pragma acc wait
///    } // end acc data region
///#endif

    //int target_yzdim = target_ydim*target_zdim;
    //for (int ix = 0; ix <= target_xdim-1; ix++) {
    //for (int iy = 0; iy <= target_ydim-1; iy++) {
    //for (int iz = 0; iz <= target_zdim-1; iz++) {
    //    int ii = (ix * target_yzdim) + (iy * target_zdim) + iz;
    //    printf("direct sum pot, %d %15.6e\n", ii, potential[ii]);
    //}
    //}
    //}

    return;
}
