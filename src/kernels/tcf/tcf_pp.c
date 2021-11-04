#include <math.h>
#include <float.h>
#include <stdio.h>

#include "tcf_pp.h"

void K_TCF_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
        
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,

    struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{
    double kap = run_params->kernel_params[0];
    double eta = run_params->kernel_params[1];
    double kap_eta_2 = kap * eta / 2.0;
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

    printf("grid block x low/high %d %d\n", target_x_low_ind, target_x_high_ind);
    printf("grid block y low/high %d %d\n", target_y_low_ind, target_y_high_ind);
    printf("grid block z low/high %d %d\n", target_z_low_ind, target_z_high_ind);
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
            for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {

                int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
                double temporary_potential = 0.0;

                double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
                double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
                double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

                for (int j = 0; j < cluster_num_sources; j++) {

                    int jj = cluster_idx_start + j;
                    double dx = tx - source_x[jj];
                    double dy = ty - source_y[jj];
                    double dz = tz - source_z[jj];
                    double r  = sqrt(dx*dx + dy*dy + dz*dz);

                    if (r > DBL_MIN) {
                        double kap_r = kap * r;
                        double r_eta = r / eta;
                        temporary_potential += source_q[jj] / r
                                             * (exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                                             -  exp( kap_r) * erfc(kap_eta_2 + r_eta));
                    }
                } // end loop over interpolation points
                potential[ii] += temporary_potential;
                printf("direct potential, %d %15.6e\n", ii, potential[ii]);
            }
        }
    }
    return;
}
