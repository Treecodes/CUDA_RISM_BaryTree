#ifndef H_RUN_PARAMS_H
#define H_RUN_PARAMS_H

#include "../utilities/enums.h"


struct RunParams
{
    KERNEL kernel;
    int num_kernel_params;
    double *kernel_params;

    APPROXIMATION approximation;
    COMPUTE_TYPE compute_type;

    double theta;
    double size_check_factor;

    int interp_order;
    int interp_pts_per_cluster;
    int interp_charges_per_cluster;

    int max_per_source_leaf;
    int max_per_target_leaf;

    int verbosity;
};


#endif /* H_RUN_PARAMS_H */
