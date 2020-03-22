#ifdef OPENACC_ENABLED
    #include <accelmath.h>
    #define M_PI 3.14159265358979323846264338327950288
#else
    #include <math.h>
#endif
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "atan_pc.h"


void K_Atan_PC_Lagrange(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster,
        double *target_x, double *target_y, double *target_z,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double domainLength = run_params->kernel_params[0];
    double delta = run_params->kernel_params[1];
    double wadj = 1. / (1. - delta / sqrt(1. + delta * delta));

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, potential)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        int ii = starting_index_of_target + i;
        double temporary_potential = 0.0;
        double tz = target_z[ii];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            int jj = starting_index_of_cluster + j;
            double dz = (tz - cluster_z[jj]) / domainLength;

            if (fabs(dz - 0.5) > DBL_MIN) {
                if (fabs(dz + 0.5) > DBL_MIN) {

                    double modz = fmod(0.5 - dz, 1.0);
                    modz += (0.5 - dz < 0 ? 1.0 : 0);

                    temporary_potential += cluster_charge[jj]
                                         * (1. / M_PI * atan(sqrt(1. + 1. / (delta * delta)) * tan(M_PI * dz))
                                         + modz - 0.5);
                }
            }


        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[ii] += wadj * temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}




void K_Atan_PC_Hermite(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster, int total_number_interpolation_points,
        double *target_x, double *target_y, double *target_z,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    printf("#######################################\n");
    printf("ATAN KERNEL NOT IMPLEMENTED FOR HERMITE\n");
    printf("#######################################\n");

    // total_number_interpolation_points is the stride, separating clustersQ, clustersQx, clustersQy, etc.
    double *cluster_charge_          = &cluster_charge[8*starting_index_of_cluster + 0*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_x   = &cluster_charge[8*starting_index_of_cluster + 1*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_y   = &cluster_charge[8*starting_index_of_cluster + 2*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_z   = &cluster_charge[8*starting_index_of_cluster + 3*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xy  = &cluster_charge[8*starting_index_of_cluster + 4*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_yz  = &cluster_charge[8*starting_index_of_cluster + 5*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xz  = &cluster_charge[8*starting_index_of_cluster + 6*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xyz = &cluster_charge[8*starting_index_of_cluster + 7*number_of_interpolation_points_in_cluster];



#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, potential, \
                        cluster_charge_, cluster_charge_delta_x, cluster_charge_delta_y, cluster_charge_delta_z, \
                        cluster_charge_delta_xy, cluster_charge_delta_yz, cluster_charge_delta_xz, \
                        cluster_charge_delta_xyz)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        int ii = starting_index_of_target + i;
        double temporary_potential = 0.0;

        double tx = target_x[ii];
        double ty = target_y[ii];
        double tz = target_z[ii];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            int jj = starting_index_of_cluster + j;
            double dx = tx - cluster_x[jj];
            double dy = ty - cluster_y[jj];
            double dz = tz - cluster_z[jj];
            double r  = sqrt(dx*dx + dy*dy + dz*dz);

            double rinv  = 1 / r;
            double r3inv = rinv*rinv*rinv;
            double r5inv = r3inv*rinv*rinv;
            double r7inv = r5inv*rinv*rinv;

            if (r > DBL_MIN) {

                temporary_potential +=  rinv  * (cluster_charge_[j])
                                 +      r3inv * (cluster_charge_delta_x[j]*dx + cluster_charge_delta_y[j]*dy
                                               + cluster_charge_delta_z[j]*dz)
                                 +  3 * r5inv * (cluster_charge_delta_xy[j]*dx*dy + cluster_charge_delta_yz[j]*dy*dz
                                               + cluster_charge_delta_xz[j]*dx*dz)
                                 + 15 * r7inv *  cluster_charge_delta_xyz[j]*dx*dy*dz;

            }
        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[starting_index_of_target + i] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
