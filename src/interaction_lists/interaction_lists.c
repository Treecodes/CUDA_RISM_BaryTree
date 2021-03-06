#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../utilities/tools.h"
#include "../utilities/array.h"

#include "../tree/struct_tree.h"
#include "../run_params/struct_run_params.h"

#include "struct_interaction_lists.h"
#include "interaction_lists.h"
                

void pc_compute_interaction_list(int tree_node, const int *tree_numpar, const double *tree_radius,
                const double *tree_x_mid, const double *tree_y_mid, const double *tree_z_mid,
                const int *tree_num_children, const int *tree_children, int *tree_used,

                double batch_radius, double batch_x_mid, double batch_y_mid, double batch_z_mid,

                int **batch_approx_list, int **batch_direct_list,
                int *sizeof_approx_list, int *sizeof_direct_list,
                int *approx_index_counter, int *direct_index_counter,
                const struct RunParams *run_params);
                

void cc_compute_interaction_list(
                int source_tree_node, const int *source_tree_numpar, const double *source_tree_radius,
                const double *source_tree_x_mid, const double *source_tree_y_mid, const double *source_tree_z_mid,
                const int *source_tree_num_children, const int *source_tree_children,

                int target_tree_node, const int *target_tree_numpar, const double *target_tree_radius,
                const double *target_tree_x_mid, const double *target_tree_y_mid, const double *target_tree_z_mid,
                const int *target_tree_num_children, const int *target_tree_children,

                int **target_approx_list, int **target_direct_list,
                int *sizeof_approx_list, int *sizeof_direct_list,
                int *approx_index_counter, int *direct_index_counter,
                
                int **cc_source_approx_list, int **cc_target_approx_list,
                int *sizeof_source_approx_list, int *sizeof_target_approx_list,
                int *cc_source_approx_index_counter, int *cc_target_approx_index_counter,
                    
                const struct RunParams *run_params);
                
                
void InteractionLists_Make(struct InteractionLists **interaction_list_addr,
                          const struct Tree *source_tree,
                          const struct Tree *target_tree,
                          const struct RunParams *run_params)
{

    *interaction_list_addr = malloc(sizeof(struct InteractionLists));
    struct InteractionLists *interaction_list = *interaction_list_addr;
    
    
    /* Nullify unallocated arrays in interaction_list struct */
    
    interaction_list->approx_interactions = NULL;
    interaction_list->direct_interactions = NULL;
    
    interaction_list->num_approx = NULL;
    interaction_list->num_direct = NULL;
    
    interaction_list->cc_source_approx_interactions = NULL;
    interaction_list->cc_target_approx_interactions = NULL;
    
    interaction_list->num_cc_source_approx = NULL;
    interaction_list->num_cc_target_approx = NULL;
    
    
    /* Set addresses for interaction lists common to PC, CP, and CC */

    int ***approx_inter_list_addr = &(interaction_list->approx_interactions);
    int ***direct_inter_list_addr = &(interaction_list->direct_interactions);
    
    int **num_approx_addr = &(interaction_list->num_approx);
    int **num_direct_addr = &(interaction_list->num_direct);
    
    
    /* Set addresses for variables pointing to source and target tree struct members */
    
    int source_tree_numnodes = source_tree->numnodes;
    const int *source_tree_numpar = source_tree->numpar;
    const double *source_tree_radius = source_tree->radius;
    const double *source_tree_x_mid = source_tree->x_mid;
    const double *source_tree_y_mid = source_tree->y_mid;
    const double *source_tree_z_mid = source_tree->z_mid;

    const int *source_tree_num_children = source_tree->num_children;
    const int *source_tree_children = source_tree->children;
   
   
    int target_tree_numnodes = target_tree->numnodes;
    const int *target_tree_numpar = target_tree->numpar;
    const double *target_tree_radius = target_tree->radius;
    const double *target_tree_x_mid = target_tree->x_mid;
    const double *target_tree_y_mid = target_tree->y_mid;
    const double *target_tree_z_mid = target_tree->z_mid;
   
    const int *target_tree_num_children = target_tree->num_children;
    const int *target_tree_children = target_tree->children;


   /* Allocate and initialize interaction lists common to PC, CP, and CC */

    make_matrix(*approx_inter_list_addr, target_tree_numnodes, 50);
    make_matrix(*direct_inter_list_addr, target_tree_numnodes, 50);
    int **approx_inter_list = *approx_inter_list_addr;
    int **direct_inter_list = *direct_inter_list_addr;
   
    make_vector(*num_approx_addr, target_tree_numnodes);
    make_vector(*num_direct_addr, target_tree_numnodes);
    int *num_approx_inter = *num_approx_addr;
    int *num_direct_inter = *num_direct_addr;
   

    int *sizeof_approx_inter_list, *sizeof_direct_inter_list;
    make_vector(sizeof_approx_inter_list, target_tree_numnodes);
    make_vector(sizeof_direct_inter_list, target_tree_numnodes);
   

    for (int i = 0; i < target_tree_numnodes; i++) sizeof_approx_inter_list[i] = 50;
    for (int i = 0; i < target_tree_numnodes; i++) sizeof_direct_inter_list[i] = 50;
   
    for (int i = 0; i < target_tree_numnodes; i++)
        for (int j = 0; j < 50; j++)
            approx_inter_list[i][j] = -1;

    for (int i = 0; i < target_tree_numnodes; i++)
        for (int j = 0; j < 50; j++)
            direct_inter_list[i][j] = -1;
           
    for (int i = 0; i < target_tree_numnodes; i++) num_approx_inter[i] = 0;
    for (int i = 0; i < target_tree_numnodes; i++) num_direct_inter[i] = 0;
    
    
    if (run_params->compute_type == CLUSTER_PARTICLE) {

        int *source_tree_used = source_tree->used;
    
        /* Build PC and CP interaction lists */
    
        for (int i = 0; i < target_tree_numnodes; i++) {
            pc_compute_interaction_list(
                    0, source_tree_numpar, source_tree_radius,
                    source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                    source_tree_num_children, source_tree_children, source_tree_used,

                    target_tree_radius[i], target_tree_x_mid[i], target_tree_y_mid[i], target_tree_z_mid[i],

                    &(approx_inter_list[i]), &(direct_inter_list[i]),
                    &(sizeof_approx_inter_list[i]), &(sizeof_direct_inter_list[i]),
                    &(num_approx_inter[i]), &(num_direct_inter[i]),
                    run_params);
        }
    
    } else if (run_params->compute_type == CLUSTER_CLUSTER) {

        int *target_tree_used = target_tree->used;
    
        /* Allocate interaction lists exclusive to CC */
        
        int ***cc_source_approx_inter_list_addr = &(interaction_list->cc_source_approx_interactions);
        int ***cc_target_approx_inter_list_addr = &(interaction_list->cc_target_approx_interactions);
    
        int **num_cc_source_approx_addr = &(interaction_list->num_cc_source_approx);
        int **num_cc_target_approx_addr = &(interaction_list->num_cc_target_approx);
        
        make_matrix(*cc_source_approx_inter_list_addr, target_tree_numnodes, 50);
        make_matrix(*cc_target_approx_inter_list_addr, target_tree_numnodes, 50);
        int **cc_source_approx_inter_list = *cc_source_approx_inter_list_addr;
        int **cc_target_approx_inter_list = *cc_target_approx_inter_list_addr;
   
        make_vector(*num_cc_source_approx_addr, target_tree_numnodes);
        make_vector(*num_cc_target_approx_addr, target_tree_numnodes);
        int *num_cc_source_approx_inter = *num_cc_source_approx_addr;
        int *num_cc_target_approx_inter = *num_cc_target_approx_addr;
   
        int *sizeof_cc_source_approx_inter_list, *sizeof_cc_target_approx_inter_list;
        make_vector(sizeof_cc_source_approx_inter_list, target_tree_numnodes);
        make_vector(sizeof_cc_target_approx_inter_list, target_tree_numnodes);
        
        for (int i = 0; i < target_tree_numnodes; i++) sizeof_cc_source_approx_inter_list[i] = 50;
        for (int i = 0; i < target_tree_numnodes; i++) sizeof_cc_target_approx_inter_list[i] = 50;
       
        for (int i = 0; i < target_tree_numnodes; i++)
            for (int j = 0; j < 50; j++)
                cc_source_approx_inter_list[i][j] = -1;

        for (int i = 0; i < target_tree_numnodes; i++)
            for (int j = 0; j < 50; j++)
                cc_target_approx_inter_list[i][j] = -1;
               
        for (int i = 0; i < target_tree_numnodes; i++) num_cc_source_approx_inter[i] = 0;
        for (int i = 0; i < target_tree_numnodes; i++) num_cc_target_approx_inter[i] = 0;
    
        /* Build CC interaction lists */
        
        cc_compute_interaction_list(
                    0, source_tree_numpar, source_tree_radius,
                    source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                    source_tree_num_children, source_tree_children,

                    0, target_tree_numpar, target_tree_radius,
                    target_tree_x_mid, target_tree_y_mid, target_tree_z_mid,
                    target_tree_num_children, target_tree_children,

                    approx_inter_list, direct_inter_list,
                    sizeof_approx_inter_list, sizeof_direct_inter_list,
                    num_approx_inter, num_direct_inter,
                    
                    cc_source_approx_inter_list, cc_target_approx_inter_list,
                    sizeof_cc_source_approx_inter_list, sizeof_cc_target_approx_inter_list,
                    num_cc_source_approx_inter, num_cc_target_approx_inter,
                    
                    run_params);

        for (int i = 0; i < target_tree_numnodes; i++) {
            if (num_approx_inter[i] != 0 || num_cc_target_approx_inter[i] != 0) {
                target_tree_used[i] = 1;
            }
        }
                    
        free_vector(sizeof_cc_source_approx_inter_list);
        free_vector(sizeof_cc_target_approx_inter_list);
    }
                    
    free_vector(sizeof_approx_inter_list);
    free_vector(sizeof_direct_inter_list);
    
    return;

} /* END of function Interaction_MakeList */



void InteractionLists_Free(struct InteractionLists **interaction_list_addr)
{
    struct InteractionLists *interaction_list = *interaction_list_addr;

    free_matrix(interaction_list->approx_interactions);
    free_matrix(interaction_list->direct_interactions);
    
    free_vector(interaction_list->num_approx);
    free_vector(interaction_list->num_direct);
    
    free_matrix(interaction_list->cc_source_approx_interactions);
    free_matrix(interaction_list->cc_target_approx_interactions);
    
    free_vector(interaction_list->num_cc_source_approx);
    free_vector(interaction_list->num_cc_target_approx);
    
    free(interaction_list);
    
    interaction_list = NULL;

    return;
}



/**********************************************/
/************ LOCAL FUNCTIONS *****************/
/**********************************************/

void pc_compute_interaction_list(
                int tree_node, const int *tree_numpar, const double *tree_radius,
                const double *tree_x_mid, const double *tree_y_mid, const double *tree_z_mid,
                const int *tree_num_children, const int *tree_children, int *tree_used,

                double batch_radius, double batch_x_mid, double batch_y_mid, double batch_z_mid,

                int **batch_approx_list, int **batch_direct_list,
                int *sizeof_approx_list, int *sizeof_direct_list,
                int *approx_index_counter, int *direct_index_counter,
                const struct RunParams *run_params)
{

    /* determine DIST for MAC test */
    double tx = batch_x_mid - tree_x_mid[tree_node];
    double ty = batch_y_mid - tree_y_mid[tree_node];
    double tz = batch_z_mid - tree_z_mid[tree_node];
    double dist = sqrt(tx*tx + ty*ty + tz*tz);


    if ((tree_radius[tree_node] + batch_radius) < dist * run_params->theta) {

        if ((tree_radius[tree_node] == 0.00) ||
           (run_params->size_check_factor * run_params->interp_pts_per_cluster > tree_numpar[tree_node])) {

            if (*direct_index_counter >= *sizeof_direct_list) {
                (*sizeof_direct_list) *= 1.5;
                *batch_direct_list = realloc_vector(*batch_direct_list, *sizeof_direct_list);
            }

            (*batch_direct_list)[*direct_index_counter] = tree_node; 
            (*direct_index_counter)++;

        } else {
    /*
     * If MAC is accepted use the expansion for the approximation.
     */
            if (*approx_index_counter >= *sizeof_approx_list) {
                (*sizeof_approx_list) *= 1.5;
                (*batch_approx_list) = realloc_vector(*batch_approx_list, *sizeof_approx_list);
            }

            (*batch_approx_list)[*approx_index_counter] = tree_node;
            (*approx_index_counter)++;
            tree_used[tree_node] = 1;
        }

    } else {
    /*
     * If MAC fails check to see if there are children. If not, perform direct
     * calculation. If there are children, call routine recursively for each.
     */
        if (tree_num_children[tree_node] == 0) {

            if (*direct_index_counter >= *sizeof_direct_list) {
                (*sizeof_direct_list) *= 1.5;
                *batch_direct_list = realloc_vector(*batch_direct_list, *sizeof_direct_list);
            }

            (*batch_direct_list)[*direct_index_counter] = tree_node; 
            (*direct_index_counter)++;

        } else {
            for (int i = 0; i < tree_num_children[tree_node]; i++) {
                pc_compute_interaction_list(tree_children[8*tree_node + i],
                           tree_numpar, tree_radius,
                           tree_x_mid, tree_y_mid, tree_z_mid,
                           tree_num_children, tree_children, tree_used,

                           batch_radius, batch_x_mid, batch_y_mid, batch_z_mid,

                           batch_approx_list, batch_direct_list,
                           sizeof_approx_list, sizeof_direct_list,
                           approx_index_counter, direct_index_counter,
                           run_params);
            }
        }
    }

    return;

} 



void cc_compute_interaction_list(
                int source_tree_node, const int *source_tree_numpar, const double *source_tree_radius,
                const double *source_tree_x_mid, const double *source_tree_y_mid, const double *source_tree_z_mid,
                const int *source_tree_num_children, const int *source_tree_children,

                int target_tree_node, const int *target_tree_numpar, const double *target_tree_radius,
                const double *target_tree_x_mid, const double *target_tree_y_mid, const double *target_tree_z_mid,
                const int *target_tree_num_children, const int *target_tree_children,

                int **approx_list, int **direct_list,
                int *sizeof_approx_list, int *sizeof_direct_list,
                int *approx_index_counter, int *direct_index_counter,
                
                int **source_approx_list, int **target_approx_list,
                int *sizeof_source_approx_list, int *sizeof_target_approx_list,
                int *source_approx_index_counter, int *target_approx_index_counter,
                
                const struct RunParams *run_params)
{
    
    int size_check = run_params->size_check_factor * run_params->interp_pts_per_cluster;

    /* determine DIST for MAC test */
    double tx = target_tree_x_mid[target_tree_node] - source_tree_x_mid[source_tree_node];
    double ty = target_tree_y_mid[target_tree_node] - source_tree_y_mid[source_tree_node];
    double tz = target_tree_z_mid[target_tree_node] - source_tree_z_mid[source_tree_node];
    double dist = sqrt(tx*tx + ty*ty + tz*tz);

    if ((source_tree_radius[source_tree_node] + target_tree_radius[target_tree_node])
         < dist * run_params->theta) {
    
        if ((source_tree_numpar[source_tree_node] <= size_check) &&
            (target_tree_numpar[target_tree_node] <= size_check)) {
            
            /* add to direct list */
            
            if (direct_index_counter[target_tree_node] >= sizeof_direct_list[target_tree_node]) {
                sizeof_direct_list[target_tree_node] *= 1.5;
                direct_list[target_tree_node] = realloc_vector(direct_list[target_tree_node],
                                                        sizeof_direct_list[target_tree_node]);
            }
            direct_list[target_tree_node][direct_index_counter[target_tree_node]] = source_tree_node;
            direct_index_counter[target_tree_node]++;
            
        } else if (source_tree_numpar[source_tree_node] <= size_check) {
        
            /* add to CP approx list */
            
            if (target_approx_index_counter[target_tree_node] >= sizeof_target_approx_list[target_tree_node]) {
                sizeof_target_approx_list[target_tree_node] *= 1.5;
                target_approx_list[target_tree_node] = realloc_vector(target_approx_list[target_tree_node],
                                                               sizeof_target_approx_list[target_tree_node]);
            }
            target_approx_list[target_tree_node][target_approx_index_counter[target_tree_node]] = source_tree_node;
            target_approx_index_counter[target_tree_node]++;
            
        } else if (target_tree_numpar[target_tree_node] <= size_check) {
        
            /* add to PC approx list */
            
            if (source_approx_index_counter[target_tree_node] >= sizeof_source_approx_list[target_tree_node]) {
                sizeof_source_approx_list[target_tree_node] *= 1.5;
                source_approx_list[target_tree_node] = realloc_vector(source_approx_list[target_tree_node],
                                                               sizeof_source_approx_list[target_tree_node]);
            }
            source_approx_list[target_tree_node][source_approx_index_counter[target_tree_node]] = source_tree_node;
            source_approx_index_counter[target_tree_node]++;
          
        } else {
        
            /* add to CC approx list */
            
            if (approx_index_counter[target_tree_node] >= sizeof_approx_list[target_tree_node]) {
                sizeof_approx_list[target_tree_node] *= 1.5;
                approx_list[target_tree_node] = realloc_vector(approx_list[target_tree_node],
                                                        sizeof_approx_list[target_tree_node]);
            }
            approx_list[target_tree_node][approx_index_counter[target_tree_node]] = source_tree_node;
            approx_index_counter[target_tree_node]++;
            
        }
    

    } else {
   /*
    * If MAC fails check to see if there are children. If not, perform direct
    * calculation. If there are children, call routine recursively for each.
    */
        if ((target_tree_num_children[target_tree_node] == 0) &&
            (source_tree_num_children[source_tree_node] == 0)) {
            
            /* add to direct list */

            if (direct_index_counter[target_tree_node] >= sizeof_direct_list[target_tree_node]) {
                sizeof_direct_list[target_tree_node] *= 1.5;
                direct_list[target_tree_node] = realloc_vector(direct_list[target_tree_node],
                                                        sizeof_direct_list[target_tree_node]);
            }

            direct_list[target_tree_node][direct_index_counter[target_tree_node]] = source_tree_node;
            direct_index_counter[target_tree_node]++;
            
        } else if (source_tree_num_children[source_tree_node] == 0) {
        
            /* traverse target tree */
            
            for (int i = 0; i < target_tree_num_children[target_tree_node]; i++) {
                cc_compute_interaction_list(
                           source_tree_node, source_tree_numpar, source_tree_radius,
                           source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                           source_tree_num_children, source_tree_children,
                           
                           target_tree_children[8*target_tree_node + i],
                           target_tree_numpar, target_tree_radius,
                           target_tree_x_mid, target_tree_y_mid, target_tree_z_mid,
                           target_tree_num_children, target_tree_children,

                           approx_list, direct_list,
                           sizeof_approx_list, sizeof_direct_list,
                           approx_index_counter, direct_index_counter,
                           
                           source_approx_list, target_approx_list,
                           sizeof_source_approx_list, sizeof_target_approx_list,
                           source_approx_index_counter, target_approx_index_counter,
                
                           run_params);
            }
        
        } else if (target_tree_num_children[target_tree_node] == 0) {
        
            /* traverse source tree */
        
            for (int i = 0; i < source_tree_num_children[source_tree_node]; i++) {
                cc_compute_interaction_list(
                           source_tree_children[8*source_tree_node + i],
                           source_tree_numpar, source_tree_radius,
                           source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                           source_tree_num_children, source_tree_children,
                           
                           target_tree_node, target_tree_numpar, target_tree_radius,
                           target_tree_x_mid, target_tree_y_mid, target_tree_z_mid,
                           target_tree_num_children, target_tree_children,

                           approx_list, direct_list,
                           sizeof_approx_list, sizeof_direct_list,
                           approx_index_counter, direct_index_counter,
                           
                           source_approx_list, target_approx_list,
                           sizeof_source_approx_list, sizeof_target_approx_list,
                           source_approx_index_counter, target_approx_index_counter,
                           
                           run_params);
            }

        } else if (source_tree_numpar[source_tree_node] <
                   target_tree_numpar[target_tree_node]) {
                   
            /* traverse target tree */
                   
            for (int i = 0; i < target_tree_num_children[target_tree_node]; i++) {
                cc_compute_interaction_list(
                           source_tree_node, source_tree_numpar, source_tree_radius,
                           source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                           source_tree_num_children, source_tree_children,
                           
                           target_tree_children[8*target_tree_node + i],
                           target_tree_numpar, target_tree_radius,
                           target_tree_x_mid, target_tree_y_mid, target_tree_z_mid,
                           target_tree_num_children, target_tree_children,

                           approx_list, direct_list,
                           sizeof_approx_list, sizeof_direct_list,
                           approx_index_counter, direct_index_counter,
                           
                           source_approx_list, target_approx_list,
                           sizeof_source_approx_list, sizeof_target_approx_list,
                           source_approx_index_counter, target_approx_index_counter,
                
                           run_params);
            }
            
        } else {
        
            /* traverse source tree */
            
            for (int i = 0; i < source_tree_num_children[source_tree_node]; i++) {
                cc_compute_interaction_list(
                           source_tree_children[8*source_tree_node + i],
                           source_tree_numpar, source_tree_radius,
                           source_tree_x_mid, source_tree_y_mid, source_tree_z_mid,
                           source_tree_num_children, source_tree_children,
                           
                           target_tree_node, target_tree_numpar, target_tree_radius,
                           target_tree_x_mid, target_tree_y_mid, target_tree_z_mid,
                           target_tree_num_children, target_tree_children,

                           approx_list, direct_list,
                           sizeof_approx_list, sizeof_direct_list,
                           approx_index_counter, direct_index_counter,
                           
                           source_approx_list, target_approx_list,
                           sizeof_source_approx_list, sizeof_target_approx_list,
                           source_approx_index_counter, target_approx_index_counter,
                           
                           run_params);
            }
        }
    }

    return;

}
