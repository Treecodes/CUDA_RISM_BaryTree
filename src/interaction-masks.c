#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#include "array.h"
#include "treedriver.h"
#include "tools.h"
#include "particles.h"
#include "sort.h"
#include "tnode.h"
#include "batch.h"


void remote_interaction_lists(const struct tnode_array *tree_array, struct batch *batches,
                              int *approx_list, int *direct_list, int numnodes)
{
    /* local variables */
    int i, j;

    int **batches_ind;
    double **batches_center;
    double *batches_radius;

    int tree_numnodes;
    const int *tree_numpar, *tree_level;
    const double *tree_x_mid, *tree_y_mid, *tree_z_mid, *tree_radius;

    batches_ind = batches->index;
    batches_center = batches->center;
    batches_radius = batches->radius;

    tree_numnodes = tree_array->numnodes;
    tree_numpar = tree_array->numpar;
    tree_level = tree_array->level;
    tree_radius = tree_array->radius;
    tree_x_mid = tree_array->x_mid;
    tree_y_mid = tree_array->y_mid;
    tree_z_mid = tree_array->z_mid;

    for (i = 0; i < numnodes; i++){
    	approx_list[i] = -1;
    	direct_list[i] = -1;
    }


    // Make interaction lists and set to -1
    double *temp_tree_inter_list, *temp_direct_inter_list;
    make_vector(temp_tree_inter_list, batches->num * numnodes);
	make_vector(temp_direct_inter_list, batches->num * numnodes);

	printf("Allocated temp_tree_inter_list and temp_direct_inter_list.\n");

	for (i = 0; i < batches->num * numnodes; i++)
		temp_tree_inter_list[i] = -1;

	for (i = 0; i < batches->num * numnodes; i++)
		temp_direct_inter_list[i] = -1;


	// Fill interaction lists
    for (i = 0; i < batches->num; i++){
    	// fill interaction lists
//    	printf("Filling lists for batch %i\n", i);
        pc_compute_interaction_list(tree_numnodes, tree_level, tree_numpar,
                tree_radius, tree_x_mid, tree_y_mid, tree_z_mid,
                batches_ind[i], batches_center[i], batches_radius[i],
                &(temp_tree_inter_list[i*numnodes]), &(temp_direct_inter_list[i*numnodes]));
    }

    // Update masks using interaction lists (overkill, but okay for now)
    int approx_counter=0, direct_counter=0;
    for (i=0; i<numnodes; i++){
    	for (j=0;j<batches->num;j++){
    		if (temp_tree_inter_list[j*numnodes]!=-1){ // then at least one target batch accepted the MAC for the ith node
    			approx_list[approx_counter]=i;
				approx_counter++;
    		}
    		if (temp_direct_inter_list[j*numnodes]!=-1){ // then at least one target batch interacts directly with the ith node
    			direct_list[direct_counter]=i;
    			direct_counter++;
			}
    	}

    }

    free_vector(temp_tree_inter_list);
    free_vector(temp_direct_inter_list);


    // Example:
    // At end of this function, approx_list and direct_list look like [c0, c3, c5, -1, -1, -1 ... ]
    // indicating the 0th, 3rd, and 5th clusters in remote tree array are needed.

    printf("Exiting remote_interaction_lists\n");
    return;

} /* END of function pc_treecode */

