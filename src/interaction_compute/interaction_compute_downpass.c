#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../utilities/array.h"
#include "../tree/struct_tree.h"
#include "../particles/struct_particles.h"
#include "../run_params/struct_run_params.h"

#include "interaction_compute.h"

#ifdef CUDA_ENABLED
    #define SINGLE
    #ifdef SINGLE
        #define FLOAT float
    #else
        #define FLOAT double
    #endif
#include "cuda_downpass.h"
#include "../kernels/cuda/tcf/cuda_tcf.h"
#endif

static void cp_comp_downpass_coeffs(int idx, int child_idx, int interp_order,
        double *cluster_x, double *cluster_y, double *cluster_z,
        int coeff_start, double *coeff_x, double *coeff_y, double *coeff_z, double *weights);

static void cp_comp_downpass(int idx, int child_idx, int interp_order,
        int coeff_start, double *coeff_x, double *coeff_y, double *coeff_z, double *cluster_q);

static void cp_comp_pot_coeffs(int idx, int interp_order,
        int target_x_low_ind, int target_x_high_ind, double target_xmin,
        double target_xdd, double *cluster_x,
        int coeff_x_start, double *coeff_x, double *weights);

static void cp_comp_pot(int idx, double *potential, int interp_order,
        int target_x_low_ind,  int target_x_high_ind,
        int target_y_low_ind,  int target_y_high_ind,
        int target_z_low_ind,  int target_z_high_ind,
        int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
        double *cluster_q,
        int coeff_x_start, double *coeff_x,
        int coeff_y_start, double *coeff_y,
        int coeff_z_start, double *coeff_z);

void InteractionCompute_Downpass(double *potential, struct Tree *tree,
                                 struct Particles *targets, struct Clusters *clusters,
                                 struct RunParams *run_params)
{
    double *cluster_x = clusters->x;
    double *cluster_y = clusters->y;
    double *cluster_z = clusters->z;
    double *cluster_q = clusters->q;

    int tree_numnodes = tree->numnodes;
    int interp_order = run_params->interp_order;

    int *target_tree_x_low_ind = tree->x_low_ind;
    int *target_tree_y_low_ind = tree->y_low_ind;
    int *target_tree_z_low_ind = tree->z_low_ind;
    
    int *target_tree_x_high_ind = tree->x_high_ind;
    int *target_tree_y_high_ind = tree->y_high_ind;
    int *target_tree_z_high_ind = tree->z_high_ind;
    
    double *target_tree_x_min = tree->x_min;
    double *target_tree_y_min = tree->y_min;
    double *target_tree_z_min = tree->z_min;
    
    double *target_tree_x_max = tree->x_max;
    double *target_tree_y_max = tree->y_max;
    double *target_tree_z_max = tree->z_max;
    
    
    int target_x_dim_glob = targets->xdim;
    int target_y_dim_glob = targets->ydim;
    int target_z_dim_glob = targets->zdim;
    
    double target_xdd = targets->xdd;
    double target_ydd = targets->ydd;
    double target_zdd = targets->zdd;


    if (run_params->approximation == LAGRANGE) {

        double *weights = NULL;
        make_vector(weights, interp_order+1);
        for (int i = 0; i < interp_order + 1; i++) {
            weights[i] = ((i % 2 == 0)? 1 : -1);
            if (i == 0 || i == interp_order) weights[i] = ((i % 2 == 0)? 1 : -1) * 0.5;
        }

#ifdef CUDA_ENABLED
        initStream();
#endif

        //First go over each level
        for (int i = 0; i < tree->max_depth; ++i) {

            if (tree->levels_list_num[i] > 0) {

                int downpass_num = 0;
                for (int j = 0; j < tree->levels_list_num[i]; ++j)
                    downpass_num += tree->num_children[tree->levels_list[i][j]];

                int sizeof_coeffs = (interp_order + 1) * (interp_order + 1) * downpass_num;

                double *coeff_x = NULL; 
                double *coeff_y = NULL;
                double *coeff_z = NULL;
                make_vector(coeff_x, sizeof_coeffs);
                make_vector(coeff_y, sizeof_coeffs);
                make_vector(coeff_z, sizeof_coeffs);

                //Go over each cluster at that level
                int coeff_start = 0;
                for (int j = 0; j < tree->levels_list_num[i]; ++j) {
                    int idx = tree->levels_list[i][j];

                    //Interpolate down coeffs to each child of the cluster
                    for (int k = 0; k < tree->num_children[idx]; ++k) {
                        int child_idx = tree->children[8*idx + k];
                        cp_comp_downpass_coeffs(idx, child_idx, interp_order,
                                                cluster_x, cluster_y, cluster_z,
                                                coeff_start, coeff_x, coeff_y, coeff_z, weights);
                        coeff_start++;
                    }
                }
                
#ifdef CUDA_ENABLED
    #ifdef SINGLE
                float *s_coeff_x  = (float*)malloc(sizeof(float)*sizeof_coeffs);
                float *s_coeff_y  = (float*)malloc(sizeof(float)*sizeof_coeffs);
                float *s_coeff_z  = (float*)malloc(sizeof(float)*sizeof_coeffs);
                for (int k = 0; k < sizeof_coeffs; k++) {
                    s_coeff_x[k] = coeff_x[k];
                    s_coeff_y[k] = coeff_y[k];
                    s_coeff_z[k] = coeff_z[k];
                }
                CUDA_Setup2(sizeof_coeffs, sizeof_coeffs, sizeof_coeffs,
                        s_coeff_x, s_coeff_y, s_coeff_z);
    #else
                CUDA_Setup2(sizeof_coeffs, sizeof_coeffs, sizeof_coeffs,
                        coeff_x, coeff_y, coeff_z);
    #endif
#endif

                //Go over each cluster at that level
                coeff_start = 0;
                for (int j = 0; j < tree->levels_list_num[i]; ++j) {
                    int idx = tree->levels_list[i][j];

                    //Interpolate down to each child of the cluster
                    for (int k = 0; k < tree->num_children[idx]; ++k) {
                        int child_idx = tree->children[8*idx + k];
#ifdef CUDA_ENABLED
                        int stream_id = (j*(tree->num_children[idx])+k)%16;
                        K_CUDA_CP_COMP_DOWNPASS(idx, child_idx, interp_order,
                                         coeff_start, stream_id);
#else
                        cp_comp_downpass(idx, child_idx, interp_order,
                                         coeff_start, coeff_x, coeff_y, coeff_z, cluster_q);
#endif
                        coeff_start++;
                    }
                }
                free_vector(coeff_x);
                free_vector(coeff_y);
                free_vector(coeff_z);

#ifdef CUDA_ENABLED
    #ifdef SINGLE
                free(s_coeff_x);
                free(s_coeff_y);
                free(s_coeff_z);
    #endif
                CUDA_Free2();
#endif
            }
        }

#ifdef CUDA_ENABLED
        delStream();
#endif

        //Then go over the leaves to the targets

        if (tree->leaves_list_num > 0) {
            int sizeof_coeff_x = 0, sizeof_coeff_y = 0, sizeof_coeff_z = 0;
            for (int i = 0; i < tree->leaves_list_num; ++i) {
                int idx = tree->leaves_list[i];
                sizeof_coeff_x += tree->x_dim[idx];
                sizeof_coeff_y += tree->y_dim[idx];
                sizeof_coeff_z += tree->z_dim[idx];
            }

            sizeof_coeff_x *= (interp_order + 1);
            sizeof_coeff_y *= (interp_order + 1);
            sizeof_coeff_z *= (interp_order + 1);
        
            double *coeff_x = NULL; 
            double *coeff_y = NULL;
            double *coeff_z = NULL;
            make_vector(coeff_x, sizeof_coeff_x);
            make_vector(coeff_y, sizeof_coeff_y);
            make_vector(coeff_z, sizeof_coeff_z);

            int coeff_x_start=0, coeff_y_start=0, coeff_z_start=0;
            for (int i = 0; i < tree->leaves_list_num; ++i) {
                int idx = tree->leaves_list[i];

                int target_x_low_ind = target_tree_x_low_ind[idx];
                int target_y_low_ind = target_tree_y_low_ind[idx];
                int target_z_low_ind = target_tree_z_low_ind[idx];

                int target_x_high_ind = target_tree_x_high_ind[idx];
                int target_y_high_ind = target_tree_y_high_ind[idx];
                int target_z_high_ind = target_tree_z_high_ind[idx];

                double target_x_min = target_tree_x_min[idx];
                double target_y_min = target_tree_y_min[idx];
                double target_z_min = target_tree_z_min[idx];
                
                cp_comp_pot_coeffs(idx, interp_order,
                               target_x_low_ind, target_x_high_ind, target_x_min, 
                               target_xdd,
                               cluster_x, coeff_x_start, coeff_x, weights);
    
                cp_comp_pot_coeffs(idx, interp_order,
                               target_y_low_ind, target_y_high_ind, target_y_min, 
                               target_ydd,
                               cluster_y, coeff_y_start, coeff_y, weights);
    
                cp_comp_pot_coeffs(idx, interp_order,
                               target_z_low_ind, target_z_high_ind, target_z_min, 
                               target_zdd,
                               cluster_z, coeff_z_start, coeff_z, weights);

                coeff_x_start += tree->x_dim[idx] * (interp_order + 1);
                coeff_y_start += tree->y_dim[idx] * (interp_order + 1);
                coeff_z_start += tree->z_dim[idx] * (interp_order + 1);
            }

#ifdef CUDA_ENABLED
            initStream();
    #ifdef SINGLE
            float *s_coeff_x  = (float*)malloc(sizeof(float)*sizeof_coeff_x);
            float *s_coeff_y  = (float*)malloc(sizeof(float)*sizeof_coeff_y);
            float *s_coeff_z  = (float*)malloc(sizeof(float)*sizeof_coeff_z);
            for (int k = 0; k < sizeof_coeff_x; k++)
                s_coeff_x[k] = coeff_x[k];
            for (int k = 0; k < sizeof_coeff_y; k++)
                s_coeff_y[k] = coeff_y[k];
            for (int k = 0; k < sizeof_coeff_z; k++)
                s_coeff_z[k] = coeff_z[k];
            CUDA_Setup2(sizeof_coeff_x, sizeof_coeff_y, sizeof_coeff_z,
                    s_coeff_x, s_coeff_y, s_coeff_z);
    #else
            CUDA_Setup2(sizeof_coeff_x, sizeof_coeff_y, sizeof_coeff_z,
                    coeff_x, coeff_y, coeff_z);
    #endif
#endif

            coeff_x_start = 0; coeff_y_start = 0; coeff_z_start = 0;
            for (int i = 0; i < tree->leaves_list_num; ++i) {
                int idx = tree->leaves_list[i];

                int target_x_low_ind = target_tree_x_low_ind[idx];
                int target_y_low_ind = target_tree_y_low_ind[idx];
                int target_z_low_ind = target_tree_z_low_ind[idx];
    
                int target_x_high_ind = target_tree_x_high_ind[idx];
                int target_y_high_ind = target_tree_y_high_ind[idx];
                int target_z_high_ind = target_tree_z_high_ind[idx];

#ifdef CUDA_ENABLED    
                 int stream_id = i%64;
                 K_CUDA_CP_COMP_POT(idx, interp_order,
                            target_x_low_ind, target_x_high_ind,
                            target_y_low_ind, target_y_high_ind,
                            target_z_low_ind, target_z_high_ind,
                            target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
                            coeff_x_start,
                            coeff_y_start,
                            coeff_z_start,
                            stream_id);
#else
                  cp_comp_pot(idx, potential, interp_order,
                            target_x_low_ind, target_x_high_ind,
                            target_y_low_ind, target_y_high_ind,
                            target_z_low_ind, target_z_high_ind,
                            target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
                            cluster_q,
                            coeff_x_start, coeff_x,
                            coeff_y_start, coeff_y,
                            coeff_z_start, coeff_z);
#endif

                coeff_x_start += tree->x_dim[idx] * (interp_order + 1);
                coeff_y_start += tree->y_dim[idx] * (interp_order + 1);
                coeff_z_start += tree->z_dim[idx] * (interp_order + 1);
            }

            free_vector(coeff_x);
            free_vector(coeff_y);
            free_vector(coeff_z);

#ifdef CUDA_ENABLED
    #ifdef SINGLE
            free(s_coeff_x);
            free(s_coeff_y);
            free(s_coeff_z);
    #endif
            CUDA_Free2();
            delStream();
#endif

        }

        free_vector(weights);

#ifdef CUDA_ENABLED
        int target_xyz_dim = target_x_dim_glob*target_y_dim_glob*target_z_dim_glob;
        CUDA_Wrapup(target_xyz_dim, potential);
//      debugging direct potentials
//       int target_yzdim = target_y_dim_glob*target_z_dim_glob;
//       for (int ix = 0; ix <= target_x_dim_glob-1; ix++) {
//           for (int iy = 0; iy <= target_y_dim_glob-1; iy++) {
//               for (int iz = 0; iz <= target_z_dim_glob-1; iz++) {
//                   int ii = (ix * target_yzdim) + (iy * target_z_dim_glob) + iz;
//                   if (potential[ii] < 0.0)
//                       printf("returned potential, %d %15.6e\n", ii, potential[ii]);
//               }
//           }
//        }
#endif

    } else {
        exit(1);
    }
        
    return;
}




/************************************/
/***** LOCAL FUNCTIONS **************/
/************************************/

void cp_comp_downpass_coeffs(int idx, int child_idx, int interp_order,
        double *cluster_x, double *cluster_y, double *cluster_z,
        int coeff_start, double *coeff_x, double *coeff_y, double *coeff_z, double *weights)
{
    int interp_order_lim       = interp_order + 1;
    
    int cluster_pts_start             = idx * interp_order_lim;
    int child_cluster_pts_start       = child_idx * interp_order_lim;

    int coeff_start_ind = interp_order_lim * interp_order_lim * coeff_start;
    
    //  Fill in arrays of unique x, y, and z coordinates for the interpolation points.

    for (int i = 0; i < interp_order_lim; i++) {
        double tx = cluster_x[child_cluster_pts_start + i];
        double ty = cluster_y[child_cluster_pts_start + i];
        double tz = cluster_z[child_cluster_pts_start + i];

        double denominatorx = 0.0;
        double denominatory = 0.0;
        double denominatorz = 0.0;

        int eix = -1;
        int eiy = -1;
        int eiz = -1;

        for (int j = 0; j < interp_order_lim; j++) {  // loop through the degree
            double cx = tx - cluster_x[cluster_pts_start + j];
            double cy = ty - cluster_y[cluster_pts_start + j];
            double cz = tz - cluster_z[cluster_pts_start + j];

            if (fabs(cx)<DBL_MIN) eix = j;
            if (fabs(cy)<DBL_MIN) eiy = j;
            if (fabs(cz)<DBL_MIN) eiz = j;

            denominatorx += weights[j] / cx;
            denominatory += weights[j] / cy;
            denominatorz += weights[j] / cz;
        }

        if (eix!=-1) denominatorx = 1;
        if (eiy!=-1) denominatory = 1;
        if (eiz!=-1) denominatorz = 1;

        for (int j = 0; j < interp_order_lim; j++) {  // loop through the degree
            double numeratorx = 1.0;
            double numeratory = 1.0;
            double numeratorz = 1.0;

            if (eix == -1) {
                numeratorx *= weights[j] / (tx - cluster_x[cluster_pts_start + j]);
            } else {
                if (eix != j) numeratorx *= 0;
            }

            if (eiy == -1) {
                numeratory *= weights[j] / (ty - cluster_y[cluster_pts_start + j]);
            } else {
                if (eiy != j) numeratory *= 0;
            }

            if (eiz == -1) {
                numeratorz *= weights[j] / (tz - cluster_z[cluster_pts_start + j]);
            } else {
                if (eiz != j) numeratorz *= 0;
            }

            coeff_x[coeff_start_ind + i * interp_order_lim + j] = numeratorx / denominatorx;
            coeff_y[coeff_start_ind + i * interp_order_lim + j] = numeratory / denominatory;
            coeff_z[coeff_start_ind + i * interp_order_lim + j] = numeratorz / denominatorz;

        }
    }

    return;
}




void cp_comp_downpass(int idx, int child_idx, int interp_order,
        int coeff_start, double *coeff_x, double *coeff_y, double *coeff_z, double *cluster_q)
{
    int interp_order_lim       = interp_order + 1;
    int interp_pts_per_cluster = interp_order_lim * interp_order_lim * interp_order_lim;
    
    int cluster_charge_start          = idx * interp_pts_per_cluster;
    int child_cluster_charge_start    = child_idx * interp_pts_per_cluster;


    int coeff_start_ind = interp_order_lim * interp_order_lim * coeff_start;

    
    for (int i = 0; i < interp_pts_per_cluster; i++) { // loop over interpolation points, set (cx,cy,cz) for this point
        int child_k3 = i%interp_order_lim;
        int child_kk = (i-child_k3)/interp_order_lim;
        int child_k2 = child_kk%interp_order_lim;
        child_kk = child_kk - child_k2;
        int child_k1 = child_kk / interp_order_lim;

        int coeff_x_start = coeff_start_ind + child_k1 * interp_order_lim;
        int coeff_y_start = coeff_start_ind + child_k2 * interp_order_lim;
        int coeff_z_start = coeff_start_ind + child_k3 * interp_order_lim;
        
        double temp = 0.0;

        for (int j = 0; j < interp_pts_per_cluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point
            int k3 = j%interp_order_lim;
            int kk = (j-k3)/interp_order_lim;
            int k2 = kk%interp_order_lim;
            kk = kk - k2;
            int k1 = kk / interp_order_lim;

            temp += coeff_x[coeff_x_start + k1] * coeff_y[coeff_y_start + k2] * coeff_z[coeff_z_start + k3] 
                  * cluster_q[cluster_charge_start + j];

        }

        cluster_q[child_cluster_charge_start + i] += temp;
    }

    
    return;
}




void cp_comp_pot_coeffs(int idx, int interp_order,
        int target_x_low_ind, int target_x_high_ind, double target_xmin,
        double target_xdd, double *cluster_x,
        int coeff_x_start, double *coeff_x, double *weights)
{

    int interp_order_lim = interp_order + 1;
    int cluster_pts_start = idx * interp_order_lim;

    //  Fill in arrays of unique x, y, and z coordinates for the interpolation points.

    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
        double denominator = 0.0;
        int eix = -1;

        for (int j = 0; j < interp_order_lim; j++) {  // loop through the degree
            double cx = tx - cluster_x[cluster_pts_start+j];
            if (fabs(cx)<DBL_MIN) eix = j;
            denominator += weights[j] / cx;
        }

        if (eix!=-1) denominator = 1;

        for (int j = 0; j < interp_order_lim; j++) {  // loop through the degree
            double numerator = 1.0;
            if (eix == -1) {
                numerator *= weights[j] / (tx - cluster_x[cluster_pts_start+j]);
            } else {
                if (eix != j) numerator *= 0;
            }

            coeff_x[coeff_x_start + (ix-target_x_low_ind) * interp_order_lim + j] = numerator / denominator;
        }
    }

    return;
}




void cp_comp_pot(int idx, double *potential, int interp_order,
        int target_x_low_ind,  int target_x_high_ind,
        int target_y_low_ind,  int target_y_high_ind,
        int target_z_low_ind,  int target_z_high_ind,
        int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
        double *cluster_q,
        int coeff_x_start, double *coeff_x,
        int coeff_y_start, double *coeff_y,
        int coeff_z_start, double *coeff_z)
{

    int interp_order_lim       = interp_order + 1;
    int interp_pts_per_cluster = interp_order_lim * interp_order_lim * interp_order_lim;
    int target_yz_dim          = target_y_dim_glob * target_z_dim_glob;
    int cluster_charge_start   = idx * interp_pts_per_cluster;
//    printf("target_x_low_ind, target_x_high_ind %12d %12d \n",target_x_low_ind, target_x_high_ind); 
//    printf("target_y_low_ind, target_y_high_ind %12d %12d \n",target_y_low_ind, target_y_high_ind); 
//    printf("target_z_low_ind, target_z_high_ind %12d %12d \n",target_z_low_ind, target_z_high_ind); 
//    printf("target_y_dim_glob, target_z_dim_glob %12d %12d \n",target_y_dim_glob, target_z_dim_glob); 
//    printf("interp_order_lim, idx %12d %12d \n",interp_order_lim, idx); 


    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
            for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {

                int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
                int iix = coeff_x_start + (ix - target_x_low_ind) * interp_order_lim;
                int iiy = coeff_y_start + (iy - target_y_low_ind) * interp_order_lim;
                int iiz = coeff_z_start + (iz - target_z_low_ind) * interp_order_lim;
              //  printf("ix,iy,iz %12d %12d %12d  \n",ix,iy,iz);
              //  printf("ii, iix, iiy,iiz %12d %12d %12d %12d \n",ii,iix,iiy,iiz);
                double temp = 0.0;

                for (int j = 0; j < interp_pts_per_cluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point
                    int k3 = j%interp_order_lim;
                    int kk = (j-k3)/interp_order_lim;
                    int k2 = kk%interp_order_lim;
                    kk = kk - k2;
                    int k1 = kk / interp_order_lim;

                    double cq = cluster_q[cluster_charge_start + j];
                    //printf("cq %15.6f \n",cq);
                    temp += coeff_x[iix + k1] * coeff_y[iiy + k2] 
                          * coeff_z[iiz + k3] * cq;
                }

                potential[ii] += temp;
            }
        }
    }

    return;
}
