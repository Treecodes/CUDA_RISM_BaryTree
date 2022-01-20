#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../utilities/array.h"

#include "../tree/struct_tree.h"
#include "../particles/struct_particles.h"
#include "../run_params/struct_run_params.h"
#include "../interaction_lists/struct_interaction_lists.h"

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


void InteractionCompute_CP(double *potential, struct Tree *tree, struct Tree *batches,
                           struct InteractionLists *interaction_list,
                           struct Particles *sources, struct Particles *targets,
                           struct Clusters *clusters, struct RunParams *run_params)
{
    int cluster_num_interp_pts = run_params->interp_pts_per_cluster;
    int interp_order_lim = run_params->interp_order+1;

    int num_source = sources->num;
    double *source_x  = sources->x;
    double *source_y  = sources->y;
    double *source_z  = sources->z;
    double *source_q  = sources->q;

    int num_cluster = clusters->num;
    double *cluster_x = clusters->x;
    double *cluster_y = clusters->y;
    double *cluster_z = clusters->z;

    int num_charge = clusters->num_charges;
    double *cluster_q = clusters->q;

    int **approx_inter_list = interaction_list->approx_interactions;
    int **direct_inter_list = interaction_list->direct_interactions;
    
    int *num_approx = interaction_list->num_approx;
    int *num_direct = interaction_list->num_direct;
    
    
    int *cluster_ind = tree->cluster_ind;
    
    int *target_tree_x_low_ind = tree->x_low_ind;
    int *target_tree_y_low_ind = tree->y_low_ind;
    int *target_tree_z_low_ind = tree->z_low_ind;
    
    int *target_tree_x_high_ind = tree->x_high_ind;
    int *target_tree_y_high_ind = tree->y_high_ind;
    int *target_tree_z_high_ind = tree->z_high_ind;
    
    double *target_tree_x_min = tree->x_min;
    double *target_tree_y_min = tree->y_min;
    double *target_tree_z_min = tree->z_min;
    
    
    int target_x_dim_glob = targets->xdim;
    int target_y_dim_glob = targets->ydim;
    int target_z_dim_glob = targets->zdim;
    
    double target_xdd = targets->xdd;
    double target_ydd = targets->ydd;
    double target_zdd = targets->zdd;

#ifdef SINGLE
    float *s_source_x  = (float*)malloc(sizeof(float)*num_source);
    float *s_source_y  = (float*)malloc(sizeof(float)*num_source);
    float *s_source_z  = (float*)malloc(sizeof(float)*num_source);
    float *s_source_q  = (float*)malloc(sizeof(float)*num_source);
    float *s_cluster_x = (float*)malloc(sizeof(float)*num_cluster);
    float *s_cluster_y = (float*)malloc(sizeof(float)*num_cluster);
    float *s_cluster_z = (float*)malloc(sizeof(float)*num_cluster);

    for (int i = 0; i < num_source-1; i++) {
        s_source_x[i] = source_x[i];
        s_source_y[i] = source_y[i];
        s_source_z[i] = source_z[i];
        s_source_q[i] = source_q[i];
    }
    for (int i = 0; i < num_cluster-1; i++) {
        s_cluster_x[i] = cluster_x[i];
        s_cluster_y[i] = cluster_y[i];
        s_cluster_z[i] = cluster_z[i];
    }
#endif

#ifdef CUDA_ENABLED
    // RQ: Initialize the streams
    int call_type = 1;
    int target_xyz_dim = target_x_dim_glob*target_y_dim_glob*target_z_dim_glob;
    CUDA_Setup(call_type, num_source, num_cluster, num_charge, target_xyz_dim,
               source_x, source_y, source_z, source_q, cluster_x, cluster_y, cluster_z,
               cluster_q, potential);
    initStream();
#endif


/* * ********************************************************/
/* * ************ POTENTIAL FROM APPROX *********************/
/* * ********************************************************/

    for (int i = 0; i < batches->numnodes; i++) {
    
        int batch_ibeg = batches->ibeg[i];
        int batch_iend = batches->iend[i];
        
        int num_approx_in_batch = num_approx[i];
        int num_direct_in_batch = num_direct[i];

        int batch_num_sources = batch_iend - batch_ibeg + 1;
        int batch_idx_start =  batch_ibeg - 1;

        for (int j = 0; j < num_approx_in_batch; j++) {

            int node_index = approx_inter_list[i][j];

            int cluster_q_start = cluster_num_interp_pts*cluster_ind[node_index];
            int cluster_pts_start = interp_order_lim*cluster_ind[node_index];

            int stream_id = (i*num_approx_in_batch+j)%512;

    /* * *********************************************/
    /* * *************** Coulomb *********************/
    /* * *********************************************/

            if (run_params->kernel == COULOMB) {

                if (run_params->approximation == LAGRANGE) {

#ifdef CUDA_ENABLED
                    K_CUDA_Coulomb_CP_Lagrange(
                        num_source, num_cluster,
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        run_params, stream_id);
#else
                    K_Coulomb_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
#endif

                } else if (run_params->approximation == HERMITE) {
                    // NOT IMPLEMENTED
                }


    /* * *************************************/
    /* * ******* TCF *************************/
    /* * *************************************/

            } else if (run_params->kernel == TCF) {

                if (run_params->approximation == LAGRANGE) {

#ifdef CUDA_ENABLED
                    K_CUDA_TCF_CP_Lagrange(
                        num_source, num_cluster, num_charge,
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        run_params, stream_id);
#else
                    K_TCF_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
#endif

                } else if (run_params->approximation == HERMITE) {
                    // NOT IMPLEMENTED
                }


    /* * *************************************/
    /* * ******* DCF *************************/
    /* * *************************************/

            } else if (run_params->kernel == DCF) {

                if (run_params->approximation == LAGRANGE) {

#ifdef CUDA_ENABLED
                    K_CUDA_DCF_CP_Lagrange(
                        num_source, num_cluster,
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        run_params, stream_id);
#else
                    K_DCF_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
#endif

                } else if (run_params->approximation == HERMITE) {
                    // NOT IMPLEMENTED
                }

            }

        } // end loop over cluster approximations

    } // end loop over target batches

//
/* * ********************************************************/
/* * ************ POTENTIAL FROM DIRECT *********************/
/* * ********************************************************/
//

    for (int i = 0; i < batches->numnodes; i++) {
    
        int batch_ibeg = batches->ibeg[i];
        int batch_iend = batches->iend[i];
        
        int num_approx_in_batch = num_approx[i];
        int num_direct_in_batch = num_direct[i];

        int batch_num_sources = batch_iend - batch_ibeg + 1;
        int batch_idx_start =  batch_ibeg - 1;

        for (int j = 0; j < num_direct_in_batch; j++) {

            int node_index = direct_inter_list[i][j];
    
            int target_x_low_ind = target_tree_x_low_ind[node_index];
            int target_y_low_ind = target_tree_y_low_ind[node_index];
            int target_z_low_ind = target_tree_z_low_ind[node_index];
    
            int target_x_high_ind = target_tree_x_high_ind[node_index];
            int target_y_high_ind = target_tree_y_high_ind[node_index];
            int target_z_high_ind = target_tree_z_high_ind[node_index];
    
            double target_x_min = target_tree_x_min[node_index];
            double target_y_min = target_tree_y_min[node_index];
            double target_z_min = target_tree_z_min[node_index];

            int stream_id = (i*num_approx_in_batch+j)%128;

    /* * *********************************************/
    /* * *************** Coulomb *********************/
    /* * *********************************************/

            if (run_params->kernel == COULOMB) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
                K_CUDA_Coulomb_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    (float)target_x_min, (float)target_y_min, (float)target_z_min,
                    (float)target_xdd,   (float)target_ydd,   (float)target_zdd,
                    target_x_dim_glob,   target_y_dim_glob,   target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #else
                K_CUDA_Coulomb_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #endif
#else
                K_Coulomb_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
#endif


    /* * *********************************************/
    /* * ********* TCF *******************************/
    /* * *********************************************/

            } else if (run_params->kernel == TCF) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
                K_CUDA_TCF_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    (float)target_x_min, (float)target_y_min, (float)target_z_min,
                    (float)target_xdd,   (float)target_ydd,   (float)target_zdd,
                    target_x_dim_glob,   target_y_dim_glob,   target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #else
                K_CUDA_TCF_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #endif
#else
                K_TCF_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
#endif


    /* * *********************************************/
    /* * ********* DCF *******************************/
    /* * *********************************************/

            } else if (run_params->kernel == DCF) {

#ifdef CUDA_ENABLED
    #ifdef SINGLE
                K_CUDA_DCF_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    (float)target_x_min, (float)target_y_min, (float)target_z_min,
                    (float)target_xdd,   (float)target_ydd,   (float)target_zdd,
                    target_x_dim_glob,   target_y_dim_glob,   target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #else
                K_CUDA_DCF_PP(
                    num_source,

                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,

                    run_params, stream_id);
    #endif

#else
                K_DCF_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
#endif

            }

        } // end loop over number of direct interactions

    } // end loop over target batches

#ifdef CUDA_ENABLED
    // RQ: Destroy the streams
    // RL: Release device memories
    delStream();
    CUDA_Free(call_type, num_charge, target_xyz_dim, cluster_q, potential);
#endif

    // debugging cluster potentials
//    for (int i = 0; i < batches->numnodes; i++) {
//        int num_approx_in_batch = num_approx[i];
//        for (int j = 0; j < num_approx_in_batch; j++) {
//            int node_index = approx_inter_list[i][j];
//            int cluster_q_start = cluster_num_interp_pts*cluster_ind[node_index];
//            for (int ii = cluster_q_start;
//                ii < cluster_q_start + interp_order_lim*interp_order_lim*interp_order_lim; ii++) {
//                printf("returned cluster_q %d %15.6e\n", ii, cluster_q[ii]);
//             }
//        }
//    }

    // debugging direct potentials
//    int target_yzdim = target_y_dim_glob*target_z_dim_glob;
//    for (int ix = 0; ix <= target_x_dim_glob-1; ix++) {
//        for (int iy = 0; iy <= target_y_dim_glob-1; iy++) {
//            for (int iz = 0; iz <= target_z_dim_glob-1; iz++) {
//                int ii = (ix * target_yzdim) + (iy * target_z_dim_glob) + iz;
//                printf("returned potential, %d %15.6e\n", ii, potential[ii]);
//            }
//        }
//    }

#ifdef SINGLE
    free(s_source_x );
    free(s_source_y );
    free(s_source_z );
    free(s_source_q );
    free(s_cluster_x);
    free(s_cluster_y);
    free(s_cluster_z);
#endif

    return;

} /* END of function pc_treecode */
