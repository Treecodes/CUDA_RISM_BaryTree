#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../utilities/array.h"
#include "../utilities/enums.h"

#include "struct_run_params.h"
#include "run_params.h"


void RunParams_Setup(struct RunParams **run_params_addr,
                     KERNEL kernel, int num_kernel_params, double *kernel_params,
                     APPROXIMATION approximation,
                     COMPUTE_TYPE compute_type,
                     double theta, double size_check_factor, int interp_order,
                     int max_per_source_leaf, int max_per_target_leaf,
                     int verbosity)
{
    RunParams_Free(run_params_addr);
    *run_params_addr = malloc(sizeof (struct RunParams));  
    struct RunParams *run_params = *run_params_addr;

	run_params->kernel = kernel;
    run_params->num_kernel_params = num_kernel_params;
	if (run_params->num_kernel_params > 0) make_vector(run_params->kernel_params, num_kernel_params);

    for (int i = 0; i < num_kernel_params; ++i)
        run_params->kernel_params[i] = kernel_params[i];

    run_params->approximation = approximation;
    run_params->compute_type = compute_type;

    run_params->theta = theta;
    run_params->size_check_factor = size_check_factor;

    run_params->interp_order = interp_order;
    run_params->interp_pts_per_cluster = (interp_order+1) * (interp_order+1) * (interp_order+1);
    run_params->interp_charges_per_cluster = run_params->interp_pts_per_cluster;

    run_params->max_per_source_leaf = max_per_source_leaf;
    run_params->max_per_target_leaf = max_per_target_leaf;

    run_params->verbosity = verbosity;

    return;
}



void RunParams_Validate(struct RunParams *run_params)
{
    int interp_order_lim = run_params->interp_order + 1;
    run_params->interp_pts_per_cluster = interp_order_lim * interp_order_lim * interp_order_lim;
    run_params->interp_charges_per_cluster = run_params->interp_pts_per_cluster;

    return;
}



void RunParams_Free(struct RunParams **run_params_addr)
{
    struct RunParams *run_params = *run_params_addr;

    if (run_params != NULL) {
	    if (run_params->num_kernel_params > 0) free_vector(run_params->kernel_params);
        free(run_params);
    }
    
    run_params = NULL;

    return;
}



void RunParams_Print(struct RunParams *run_params)
{
    printf("[BaryTree]\n");
    printf("[BaryTree] RunParams struct has been set to the following:\n");
    printf("[BaryTree]\n");
    printf("[BaryTree]                     kernel = %d\n", run_params->kernel);
    printf("[BaryTree]          num_kernel_params = %d\n", run_params->num_kernel_params);
    printf("[BaryTree]              kernel_params = ");
    for (int i = 0; i < run_params->num_kernel_params; ++i) {
        printf("%f, ", run_params->kernel_params[i]);
    }
    printf("\n");
    printf("[BaryTree]              approximation = %d\n", run_params->approximation);
    printf("[BaryTree]               compute_type = %d\n", run_params->compute_type);
    printf("[BaryTree]                      theta = %f\n", run_params->theta);
    printf("[BaryTree]          size_check_factor = %f\n", run_params->size_check_factor);
    printf("[BaryTree]               interp_order = %d\n", run_params->interp_order);
    printf("[BaryTree]     interp_pts_per_cluster = %d\n", run_params->interp_pts_per_cluster);
    printf("[BaryTree] interp_charges_per_cluster = %d\n", run_params->interp_charges_per_cluster);
    printf("[BaryTree]        max_per_source_leaf = %d\n", run_params->max_per_source_leaf);
    printf("[BaryTree]        max_per_target_leaf = %d\n", run_params->max_per_target_leaf);
    printf("[BaryTree]                  verbosity = %d\n", run_params->verbosity);
    printf("[BaryTree]\n");
}
