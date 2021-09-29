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

    double *source_x  = sources->x;
    double *source_y  = sources->y;
    double *source_z  = sources->z;
    double *source_q  = sources->q;

    double *cluster_x = clusters->x;
    double *cluster_y = clusters->y;
    double *cluster_z = clusters->z;
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


    for (int i = 0; i < batches->numnodes; i++) {
    
        int batch_ibeg = batches->ibeg[i];
        int batch_iend = batches->iend[i];
        
        int num_approx_in_batch = num_approx[i];
        int num_direct_in_batch = num_direct[i];

        int batch_num_sources = batch_iend - batch_ibeg + 1;
        int batch_idx_start =  batch_ibeg - 1;


/* * ********************************************************/
/* * ************ POTENTIAL FROM APPROX *********************/
/* * ********************************************************/

        for (int j = 0; j < num_approx_in_batch; j++) {

            int node_index = approx_inter_list[i][j];

            int cluster_q_start = cluster_num_interp_pts*cluster_ind[node_index];
            int cluster_pts_start = interp_order_lim*cluster_ind[node_index];
            int stream_id = j%3;


    /* * *********************************************/
    /* * *************** Coulomb *********************/
    /* * *********************************************/

            if (run_params->kernel == COULOMB) {

                if (run_params->approximation == LAGRANGE) {

#ifdef CUDA_ENABLED
                    #pragma acc host_data use_device( \
                            source_x, source_y, source_z, source_q, \
                            cluster_x, cluster_y, cluster_z, cluster_q)
                    {
                    //printf("CUDA Kernel call");
                    K_CUDA_Coulomb_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
                    }
#else
                    K_Coulomb_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
                    //test flattened loop
     //               test_flat(
     //                   batch_num_sources, batch_idx_start,
     //                   cluster_q_start, cluster_pts_start,
     //                   interp_order_lim,
     //                   source_x, source_y, source_z, source_q,
     //                   cluster_x, cluster_y, cluster_z, cluster_q,
     //                   run_params, stream_id);
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
                    #pragma acc host_data use_device( \
                            source_x, source_y, source_z, source_q, \
                            cluster_x, cluster_y, cluster_z, cluster_q)
                    {
                    K_CUDA_TCF_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
                    }
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
                    #pragma acc host_data use_device( \
                            source_x, source_y, source_z, source_q, \
                            cluster_x, cluster_y, cluster_z) //cluster_q removed 
                    {
                    K_CUDA_DCF_CP_Lagrange(
                        batch_num_sources, batch_idx_start,
                        cluster_q_start, cluster_pts_start,
                        interp_order_lim,
                        source_x, source_y, source_z, source_q,
                        cluster_x, cluster_y, cluster_z, cluster_q,
                        run_params, stream_id);
                    }
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


/* * ********************************************************/
/* * ************ POTENTIAL FROM DIRECT *********************/
/* * ********************************************************/

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


            int stream_id = j%3;


    /* * *********************************************/
    /* * *************** Coulomb *********************/
    /* * *********************************************/

            if (run_params->kernel == COULOMB) {

#ifdef CUDA_ENABLED
                #pragma acc host_data use_device(potential, \
                        source_x, source_y, source_z, source_q)
                {
                K_CUDA_Coulomb_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
                }
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
    //            test_flat_PP(
    //                target_x_low_ind, target_x_high_ind,
    //                target_y_low_ind, target_y_high_ind,
    //                target_z_low_ind, target_z_high_ind,

    //                target_x_min,      target_y_min,      target_z_min,
    //                target_xdd,        target_ydd,        target_zdd,
    //                target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

    //                batch_num_sources, batch_idx_start,
    //                source_x, source_y, source_z, source_q,

    //                run_params, potential, stream_id);

#endif


    /* * *********************************************/
    /* * ********* TCF *******************************/
    /* * *********************************************/

            } else if (run_params->kernel == TCF) {

#ifdef CUDA_ENABLED
                #pragma acc host_data use_device(potential, \
                        source_x, source_y, source_z, source_q)
                {
                K_CUDA_TCF_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
                }
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
                #pragma acc host_data use_device(potential, \
                        source_x, source_y, source_z, source_q)
                {
                K_CUDA_DCF_PP(
                    target_x_low_ind, target_x_high_ind,
                    target_y_low_ind, target_y_high_ind,
                    target_z_low_ind, target_z_high_ind,

                    target_x_min,      target_y_min,      target_z_min,
                    target_xdd,        target_ydd,        target_zdd,
                    target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

                    batch_num_sources, batch_idx_start,
                    source_x, source_y, source_z, source_q,

                    run_params, potential, stream_id);
                }
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

#ifdef OPENACC_ENABLED
        #pragma acc wait
#endif

    return;

} /* END of function pc_treecode */
