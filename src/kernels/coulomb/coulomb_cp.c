#include <math.h>
#include <float.h>
#include <stdio.h>

#include "coulomb_cp.h"

void K_Coulomb_CP_Lagrange(
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start,
    int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    double kap = run_params->kernel_params[0];
    double eta = run_params->kernel_params[1];
    double kap_eta_2 = kap * eta / 2.0;

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(source_x, source_y, source_z, source_q, \
                        cluster_x, cluster_y, cluster_z, cluster_q)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop gang collapse(3) independent
#endif	
    for (int k1 = 0; k1 < interp_order_lim; k1++) {
    for (int k2 = 0; k2 < interp_order_lim; k2++) {
    for (int k3 = 0; k3 < interp_order_lim; k3++) {

        double temporary_potential = 0.0;
            
    //    printf("K1: %i\n", k1);
    //    printf("K2: %i\n", k2);
    //    printf("K3: %i\n", k3);

        double cx = cluster_x[cluster_pts_start + k1];
        double cy = cluster_y[cluster_pts_start + k2];
        double cz = cluster_z[cluster_pts_start + k3];

        int ii = cluster_q_start + k1 * interp_order_lim*interp_order_lim + k2 * interp_order_lim + k3;
     //   printf("II: %i\n", ii);

#ifdef OPENACC_ENABLED
        #pragma acc loop vector independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < batch_num_sources; j++) {
#ifdef OPENACC_ENABLED
            #pragma acc cache(source_x[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_y[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_z[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_q[batch_idx_start : batch_idx_start+batch_num_sources])
#endif

            int jj = batch_idx_start + j;
           // printf("old J,JJ,K1,K2,K3,II %i,%i,%i,%i,%i,%i\n", j,jj,k1,k2,k3,ii);
           // printf("cx: %lf\n", cx);
           // printf("source_x: %lf\n", source_x[jj]);
            double dx = cx - source_x[jj];
            double dy = cy - source_y[jj];
            double dz = cz - source_z[jj];
            double r = sqrt(dx*dx + dy*dy + dz*dz);

            temporary_potential += source_q[jj] / r;
        }
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        cluster_q[ii] += temporary_potential;
        printf("old %15.6e\n", cluster_q[ii]);
    }
    }
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}


void test_flat(
    int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start,
    int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    double kap = run_params->kernel_params[0];
    double eta = run_params->kernel_params[1];
    double kap_eta_2 = kap * eta / 2.0;

    int cid_lim2 = interp_order_lim*interp_order_lim;
    int cid_lim3 = interp_order_lim*cid_lim2;
    double *temporary_potential;
    temporary_potential = (double*) malloc(sizeof(double)*batch_num_sources * cid_lim3);
       printf("new batch_num_sources: %i\n", batch_num_sources);
       printf("new cid_lim3: %i\n", cid_lim3);
     //    return;
   for (int fid = 0; fid < batch_num_sources * cid_lim3; fid++) {
       //printf("new fid: %i\n", fid);
        int cid = fid/batch_num_sources;
        int j = fid-cid*batch_num_sources;
        int k1 = cid/cid_lim2; int tmp = cid - k1*cid_lim2;
        int k2 = tmp/interp_order_lim;
        int k3 = tmp%interp_order_lim;
        double cx = cluster_x[cluster_pts_start + k1];
        double cy = cluster_y[cluster_pts_start + k2];
        double cz = cluster_z[cluster_pts_start + k3];

        int jj = batch_idx_start + j;
        printf("new source_q: %d\n", source_q[jj]);
        printf("new temp: %d\n", temporary_potential[j+batch_num_sources*cid]);
       // printf("new J,JJ,K1,K2,K3,II %i,%i,%i,%i,%i,%i\n", j,jj,k1,k2,k3,(cluster_q_start+cid));
        //fflush(stdout);
        double dx = cx - source_x[jj];
        double dy = cy - source_y[jj];
        double dz = cz - source_z[jj];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        temporary_potential[j+batch_num_sources*cid] = source_q[jj] / r;
      }
    for (int cid = 0; cid < cid_lim3; cid++) {
        //printf(" new cid: %i\n", cid);
       // fflush(stdout);
        int ii = cluster_q_start + cid;
        for (int j = 0; j < batch_num_sources; j++) {
        //    cluster_q[ii] += temporary_potential[j+batch_num_sources*cid];
        } 
        //printf("new %15.6e\n", cluster_q[ii]);
        //fflush(stdout);
    } // end collect
      free(temporary_potential);
        //printf("end");

    return;
}

/*
void K_Coulomb_CP_Hermite(
        int batch_num_sources, int cluster_num_interp_pts,
        int batch_idx_start, int cluster_idx_start,
        double *source_x, double *source_y, double *source_z, double *source_q,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
        struct RunParams *run_params, int gpu_async_stream_id)
{
    double kappa  = run_params->kernel_params[0];
    double kappa2 = kappa * kappa;
    double kappa3 = kappa * kappa2;

    double *cluster_q_     = &cluster_q[8*cluster_idx_start + 0*cluster_num_interp_pts];
    double *cluster_q_dx   = &cluster_q[8*cluster_idx_start + 1*cluster_num_interp_pts];
    double *cluster_q_dy   = &cluster_q[8*cluster_idx_start + 2*cluster_num_interp_pts];
    double *cluster_q_dz   = &cluster_q[8*cluster_idx_start + 3*cluster_num_interp_pts];
    double *cluster_q_dxy  = &cluster_q[8*cluster_idx_start + 4*cluster_num_interp_pts];
    double *cluster_q_dyz  = &cluster_q[8*cluster_idx_start + 5*cluster_num_interp_pts];
    double *cluster_q_dxz  = &cluster_q[8*cluster_idx_start + 6*cluster_num_interp_pts];
    double *cluster_q_dxyz = &cluster_q[8*cluster_idx_start + 7*cluster_num_interp_pts];


#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(source_x, source_y, source_z, source_q, \
                        cluster_x, cluster_y, cluster_z, \
                        cluster_q_, cluster_q_dx, cluster_q_dy, cluster_q_dz, \
                        cluster_q_dxy, cluster_q_dyz, cluster_q_dxz, \
                        cluster_q_dxyz)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < cluster_num_interp_pts; i++) {

        double temp_pot_     = 0.0;
        double temp_pot_dx   = 0.0;
        double temp_pot_dy   = 0.0;
        double temp_pot_dz   = 0.0;
        double temp_pot_dxy  = 0.0;
        double temp_pot_dyz  = 0.0;
        double temp_pot_dxz  = 0.0;
        double temp_pot_dxyz = 0.0;
        
        int ii = cluster_idx_start + i;
        double cx = cluster_x[ii];
        double cy = cluster_y[ii];
        double cz = cluster_z[ii];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temp_pot_dx)  reduction(+:temp_pot_dy)  reduction(+:temp_pot_dz)  \
                                     reduction(+:temp_pot_dxy) reduction(+:temp_pot_dyz) reduction(+:temp_pot_dxz) \
                                     reduction(+:temp_pot_)    reduction(+:temp_pot_dxyz)
                                               
#endif
        for (int j = 0; j < batch_num_sources; j++) {
#ifdef OPENACC_ENABLED
            #pragma acc cache(source_x[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_y[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_z[batch_idx_start : batch_idx_start+batch_num_sources], \
                              source_q[batch_idx_start : batch_idx_start+batch_num_sources])
#endif

            int jj = batch_idx_start + j;
            double dx = source_x[jj] - cx;
            double dy = source_y[jj] - cy;
            double dz = source_z[jj] - cz;
            double r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 > DBL_MIN) {
                double r  = sqrt(r2);
                double r3 = r2 * r;

                double r2inv = 1. / r2;
                double rinvq = 2. * source_q[jj] / r * exp(-kappa * r);
                double r3inv = rinvq * r2inv;
                double r5inv = r3inv * r2inv;
                double r7inv = r5inv * r2inv;
                
                double term_d1 = r3inv * (1. + kappa * r);
                double term_d2 = r5inv * (3. + 3. * kappa * r + kappa2 * r2);
                double term_d3 = r7inv * (15. + 15. * kappa * r + 6. * kappa2 * r2 + kappa3 * r3);

                temp_pot_     += rinvq;
                temp_pot_dx   += term_d1 * dx;
                temp_pot_dy   += term_d1 * dy;
                temp_pot_dz   += term_d1 * dz;
                temp_pot_dxy  += term_d2 * dx * dy;
                temp_pot_dyz  += term_d2 * dy * dz;
                temp_pot_dxz  += term_d2 * dx * dz;
                temp_pot_dxyz += term_d3 * dx * dy * dz;
            }

        } // end loop over interpolation points
        
#ifdef OPENACC_ENABLED
        #pragma acc atomic
        cluster_q_[i]     += temp_pot_;
        #pragma acc atomic
        cluster_q_dx[i]   += temp_pot_dx;
        #pragma acc atomic
        cluster_q_dy[i]   += temp_pot_dy;
        #pragma acc atomic
        cluster_q_dz[i]   += temp_pot_dz;
        #pragma acc atomic
        cluster_q_dxy[i]  += temp_pot_dxy;
        #pragma acc atomic
        cluster_q_dyz[i]  += temp_pot_dyz;
        #pragma acc atomic
        cluster_q_dxz[i]  += temp_pot_dxz;
        #pragma acc atomic
        cluster_q_dxyz[i] += temp_pot_dxyz;
#else
        cluster_q_[i]     += temp_pot_;
        cluster_q_dx[i]   += temp_pot_dx;
        cluster_q_dy[i]   += temp_pot_dy;
        cluster_q_dz[i]   += temp_pot_dz;
        cluster_q_dxy[i]  += temp_pot_dxy;
        cluster_q_dyz[i]  += temp_pot_dyz;
        cluster_q_dxz[i]  += temp_pot_dxz;
        cluster_q_dxyz[i] += temp_pot_dxyz;
#endif

    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
*/
