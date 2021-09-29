#include <math.h>
#include <float.h>
#include <stdio.h>

#include "coulomb_pp.h"

void K_Coulomb_PP(
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
    double eta = run_params->kernel_params[0];
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(source_x, source_y, source_z, source_q, potential)
    {
    #pragma acc loop gang collapse(3) independent
#endif
    for (int ix = target_x_low_ind; ix <= target_x_high_ind; ix++) {
        for (int iy = target_y_low_ind; iy <= target_y_high_ind; iy++) {
            for (int iz = target_z_low_ind; iz <= target_z_high_ind; iz++) {

                int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
                double temporary_potential = 0.0;

                double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
                double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
                double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

#ifdef OPENACC_ENABLED
                #pragma acc loop vector independent reduction(+:temporary_potential)
#endif
                for (int j = 0; j < cluster_num_sources; j++) {
#ifdef OPENACC_ENABLED
                #pragma acc cache(source_x[cluster_idx_start : cluster_idx_start+cluster_num_sources], \
                                  source_y[cluster_idx_start : cluster_idx_start+cluster_num_sources], \
                                  source_z[cluster_idx_start : cluster_idx_start+cluster_num_sources], \
                                  source_q[cluster_idx_start : cluster_idx_start+cluster_num_sources])
#endif

                    int jj = cluster_idx_start + j;
                    printf("old PP J,JJ,CNS,FHI,II %i,%i,%i,%i,%i,%i\n", j,jj,ix,iy,iz,ii);
                    double dx = tx - source_x[jj];
                    double dy = ty - source_y[jj];
                    double dz = tz - source_z[jj];
                    double r  = sqrt(dx*dx + dy*dy + dz*dz);

                    if (r > DBL_MIN) {
                        temporary_potential += source_q[jj]  / r;
                    }
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

void test_flat_PP(
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
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    int target_xyz_dim = target_x_dim_glob*target_yz_dim;

    double *temporary_potential;
    temporary_potential = (double*) malloc(sizeof(double)*target_xyz_dim*cluster_num_sources);

    int fid_high_ind = (target_x_high_ind - target_x_low_ind)*
                       (target_y_high_ind - target_y_low_ind)*
                       (target_z_high_ind - target_z_low_ind);
    for (int fid = 0; fid <= fid_high_ind*cluster_num_sources; fid++) {
        int tid = fid/cluster_num_sources;

        int j = fid-tid*cluster_num_sources;
        int ix = tid/target_yz_dim; int tmp = tid - ix*target_yz_dim;
        int iy = tmp/target_z_dim_glob;
        int iz = tmp%target_z_dim_glob;

        double tx = target_xmin + ix * target_xdd;
        double ty = target_ymin + iy * target_ydd;
        double tz = target_zmin + iz * target_zdd;

        int jj = cluster_idx_start + j;
        printf("new PP J,JJ,CNS,FHI,II %i,%i,%i,%i,%i\n", j,jj,cluster_num_sources,fid_high_ind,(cluster_idx_start+tid));
        //double dx = tx - source_x[jj];
       // double dy = ty - source_y[jj];
       // double dz = tz - source_z[jj];
      //  double r  = sqrt(dx*dx + dy*dy + dz*dz);

       // temporary_potential[j+cluster_num_sources*fid_high_ind] = 0.0;
       // if (r > DBL_MIN) {
        //    temporary_potential[j+cluster_num_sources*fid_high_ind] = source_q[jj] / r;
      //  }
    } // end of flattened loop

    for (int tid = 0; tid <= fid_high_ind; tid++) {
        int ix = tid/target_yz_dim; int tmp = tid - ix*target_yz_dim;
        int iy = tmp/target_z_dim_glob;
        int iz = tmp%target_z_dim_glob;
        ix = ix + target_x_low_ind;
        iy = iy + target_y_low_ind;
        iz = iz + target_z_low_ind;
        int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
        for (int j = 0; j < cluster_num_sources; j++) {
          //  potential[ii] += temporary_potential[j+cluster_num_sources*fid_high_ind];
        }
    } // end of collect loop
    return;
}

