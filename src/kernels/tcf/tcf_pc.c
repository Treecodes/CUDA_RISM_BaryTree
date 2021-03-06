#include <math.h>
#include <float.h>
#include <stdio.h>

#include "tcf_pc.h"

/*
void K_TCF_PC_Lagrange(int target_x_low_ind,  int target_x_high_ind,
                       int target_y_low_ind,  int target_y_high_ind,
                       int target_z_low_ind,  int target_z_high_ind,
                       double target_xmin,    double target_ymin,    double target_zmin,
                       
                       double target_xdd,     double target_ydd,     double target_zdd,
                       int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

                       int cluster_num_interp_pts, int cluster_idx_start,
                       double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,

                       struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{
    double kap = run_params->kernel_params[0];
    double eta = run_params->kernel_params[1];
    double kap_eta_2 = kap * eta / 2.;
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(cluster_x, cluster_y, cluster_z, cluster_q, potential)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop gang collapse(3) independent
#endif	
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
            for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {

                int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;

                double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
                double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
                double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

                double temporary_potential = 0.0;

#ifdef OPENACC_ENABLED
                #pragma acc loop vector independent reduction(+:temporary_potential)
#endif
                for (int j = 0; j < cluster_num_interp_pts; j++) {
#ifdef OPENACC_ENABLED
                #pragma acc cache(cluster_x[cluster_idx_start : cluster_idx_start+cluster_num_interp_pts], \
                                  cluster_y[cluster_idx_start : cluster_idx_start+cluster_num_interp_pts], \
                                  cluster_z[cluster_idx_start : cluster_idx_start+cluster_num_interp_pts], \
                                  cluster_q[cluster_idx_start : cluster_idx_start+cluster_num_interp_pts])
#endif

                    int jj = cluster_idx_start + j;
                    double dx = tx - cluster_x[jj];
                    double dy = ty - cluster_y[jj];
                    double dz = tz - cluster_z[jj];
                    double r  = sqrt(dx*dx + dy*dy + dz*dz);

            //if (r > DBL_MIN) {
                    double kap_r = kap * r;
                    double r_eta = r / eta;
                //temporary_potential += cluster_q[jj] / r
                //                     * (exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                //                     -  exp( kap_r) * erfc(kap_eta_2 + r_eta));
                    temporary_potential += cluster_q[jj] / r * 2. * exp(-kap * r);
            //}
                } // end loop over interpolation points
#ifdef OPENACC_ENABLED
                #pragma acc atomic
#endif
                potential[ii] += temporary_potential;
            }
        }
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}




void K_TCF_PC_Hermite(int target_x_low_ind,  int target_x_high_ind,
                      int target_y_low_ind,  int target_y_high_ind,
                      int target_z_low_ind,  int target_z_high_ind,
                      double target_xdd,     double target_ydd,     double target_zdd,
                      double target_xmin,    double target_ymin,    double target_zmin,
                      int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

                      int cluster_num_interp_pts, int cluster_idx_start,
                      double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,

                      struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double kappa  = run_params->kernel_params[0];
    double kappa2 = kappa * kappa;
    double kappa3 = kappa * kappa2;
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

    // total_number_interpolation_points is the stride, separating clustersQ, clustersQx, clustersQy, etc.
    double *cluster_q_     = &cluster_q[8*cluster_idx_start + 0*cluster_num_interp_pts];
    double *cluster_q_dx   = &cluster_q[8*cluster_idx_start + 1*cluster_num_interp_pts];
    double *cluster_q_dy   = &cluster_q[8*cluster_idx_start + 2*cluster_num_interp_pts];
    double *cluster_q_dz   = &cluster_q[8*cluster_idx_start + 3*cluster_num_interp_pts];
    double *cluster_q_dxy  = &cluster_q[8*cluster_idx_start + 4*cluster_num_interp_pts];
    double *cluster_q_dyz  = &cluster_q[8*cluster_idx_start + 5*cluster_num_interp_pts];
    double *cluster_q_dxz  = &cluster_q[8*cluster_idx_start + 6*cluster_num_interp_pts];
    double *cluster_q_dxyz = &cluster_q[8*cluster_idx_start + 7*cluster_num_interp_pts];



#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present( \
                        cluster_x, cluster_y, cluster_z, cluster_q, potential, \
                        cluster_q_, cluster_q_dx, cluster_q_dy, cluster_q_dz, \
                        cluster_q_dxy, cluster_q_dyz, cluster_q_dxz, \
                        cluster_q_dxyz)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop collapse(3) independent
#endif	
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
            for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {

                int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;

                double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
                double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
                double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

                double temporary_potential = 0.0;

#ifdef OPENACC_ENABLED
                #pragma acc loop independent reduction(+:temporary_potential)
#endif
                for (int j = 0; j < cluster_num_interp_pts; j++) {
        
                    int jj = cluster_idx_start + j;
                    double dx = tx - cluster_x[jj];
                    double dy = ty - cluster_y[jj];
                    double dz = tz - cluster_z[jj];
                    double r = sqrt(dx*dx + dy*dy + dz*dz);
        
                    double r2 = r  * r;
                    double r3 = r2 * r;
                    double rinv = 1. / r;
                    double r3inv = rinv  * rinv * rinv;
                    double r5inv = r3inv * rinv * rinv;
                    double r7inv = r5inv * rinv * rinv;
        
                    temporary_potential += -2. * exp(-kappa * r)
                        * (rinv * (cluster_q_[j])
                        + r3inv * (1. + kappa * r)
                                * (cluster_q_dx[j]*dx + cluster_q_dy[j]*dy
                                 + cluster_q_dz[j]*dz)
                        + r5inv * (3. + 3. * kappa * r + kappa2 * r2)
                                * (cluster_q_dxy[j]*dx*dy + cluster_q_dyz[j]*dy*dz
                                 + cluster_q_dxz[j]*dx*dz)
                        + r7inv * (15. + 15. * kappa * r + 6. * kappa2 * r2 + kappa3 * r3)
                                * cluster_q_dxyz[j]*dx*dy*dz);
                } // end loop over interpolation points
#ifdef OPENACC_ENABLED
                #pragma acc atomic
#endif
                potential[ii] += temporary_potential;
            }
        }
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
*/
