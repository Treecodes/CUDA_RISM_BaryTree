#include <stdio.h>
#include <mpi.h>

#include "array.h"
#include "globvars.h"
#include "tnode.h"
#include "tools.h"
#include "tree.h"

#include "treedriver.h"


/* definition of primary treecode driver */

void treecode(double *xS, double *yS, double *zS, double *qS, 
              double *xT, double *yT, double *zT,
              int numparsS, int numparsT, double *tEn, double *tpeng, 
              int order, double theta, int shrink, int maxparnode,
              double *timetree, int treelevel, int iflag,
              int pot_type, double kappa, int tree_type)
{

    /* local variables */
    struct tnode *troot = NULL;
    int level;
    double xyzminmax[6];

    /* date and time */
    double time1, time2;

    
    time1 = MPI_Wtime();

    /* call setup to allocate arrays for Taylor expansions and setup global vars */
    if (tree_type == 0) {
        if (pot_type == 0)
            setup(xT, yT, zT, numparsT, order, theta, xyzminmax);
        else if (pot_type == 1)
            setup_yuk(xT, yT, zT, numparsT, order, theta, xyzminmax);
    } else if (tree_type == 1) {
        if (pot_type == 0)
            setup(xS, yS, zS, numparsS, order, theta, xyzminmax);
        else if (pot_type == 1)
            setup_yuk(xS, yS, zS, numparsS, order, theta, xyzminmax);
    }
    

    /* set global variables to track tree levels during construction */
    level = 0;
    minlevel = 50000;

    printf("Creating tree... \n\n");

    if (tree_type == 0) {
        if (iflag == 0) {
            maxlevel = 0;
            cp_create_tree_n0(&troot, 1, numparsT, xT, yT, zT,
                              shrink, maxparnode, xyzminmax, level);
        } else if (iflag == 1) {
            maxlevel = treelevel;
            cp_create_tree_lv(&troot, 1, numparsT, xT, yT, zT,
                              shrink, treelevel, xyzminmax, level);
        }
    } else if (tree_type == 1) {
        if (iflag == 0) {
            maxlevel = 0;
            pc_create_tree_n0(&troot, 1, numparsS, xS, yS, zS, qS,
                              shrink, maxparnode, xyzminmax, level);
        } else if (iflag == 1) {
            maxlevel = treelevel;
            pc_create_tree_lv(&troot, 1, numparsS, xS, yS, zS, qS,
                              shrink, treelevel, xyzminmax, level);
        }
    }


    time2 = MPI_Wtime();
    timetree[0] = time2-time1;


    printf("Tree created.\n\n");
    printf("Tree information: \n\n");

    printf("       numpar: %d\n", troot->numpar);
    printf("        x_mid: %e\n", troot->x_mid);
    printf("        y_mid: %e\n", troot->y_mid);
    printf("        z_mid: %e\n\n", troot->z_mid);
    printf("       radius: %f\n\n", troot->radius);
    printf("        x_len: %e\n", troot->x_max - troot->x_min);
    printf("        y_len: %e\n", troot->y_max - troot->y_min);
    printf("        z_len: %e\n\n", troot->z_max - troot->z_min);
    printf("       torder: %d\n", torder);
    printf("        theta: %f\n", theta);
    printf("       shrink: %d\n", shrink);
    printf("   maxparnode: %d\n", maxparnode);
    printf("        iflag: %d\n", iflag);
    printf("tree maxlevel: %d\n\n", treelevel);


    time1 = MPI_Wtime();

    if (tree_type == 0) {
        if (pot_type == 0) {
            cp_treecode(troot, xS, yS, zS, qS, xT, yT, zT,
                        tpeng, tEn, numparsS, numparsT, &timetree[1]);
        } else if (pot_type == 1) {
            cp_treecode_yuk(troot, xS, yS, zS, qS, xT, yT, zT,
                            tpeng, tEn, numparsS, numparsT, kappa, &timetree[1]);
        }
    } else if (tree_type == 1) {
        if (pot_type == 0) {
            pc_treecode(troot, xS, yS, zS, qS, xT, yT, zT,
                        tpeng, tEn, numparsS, numparsT);
        } else if (pot_type == 1) {
            pc_treecode_yuk(troot, xS, yS, zS, qS, xT, yT, zT,
                            tpeng, tEn, numparsS, numparsT, kappa);
        }
    }

    time2 = MPI_Wtime();
    timetree[3] = time2-time1 + timetree[0];

    //printf("       Tree building time (s): %f\n", *timetree - totaltime);
    //printf("    Tree computation time (s): %f\n\n", totaltime);
    printf("Deallocating tree structure... \n\n");

    cleanup(troot);

    return;

} /* END function treecode */

