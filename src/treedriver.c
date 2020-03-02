#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>

#include "array.h"
#include "tools.h"
#include "globvars.h"
#include "const.h"

#include "struct_nodes.h"
#include "struct_particles.h"
#include "struct_clusters.h"
#include "struct_run_params.h"

#include "interaction_lists.h"
#include "interaction_compute.h"
#include "tree.h"
#include "batches.h"
#include "clusters.h"
#include "particles.h"

#include "treedriver.h"


void treedriver(struct particles *sources, struct particles *targets, struct RunParams *run_params,
                double *potential_array, double *time_tree)
{
    int rank, numProcs, ierr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    RunParams_Validate(run_params);
    if (run_params->verbosity > 0) printf("Set rank %i and numProcs %i.\n", rank, numProcs);

    double time1;
    

    int totalNumberDirect = 0;
    int totalNumberApprox = 0;
    int totalNumberInteractions = 0;
    int cumulativeNumberInteractions = 0;
    int maxNumberInteractions = 0;
    int minNumberInteractions = 0;



    //--------------------------------------
    //--------------------------------------
    //CLUSTER PARTICLE
    //--------------------------------------
    //--------------------------------------
    


    /* call setup to allocate arrays for Taylor expansions and setup global vars */
    if (run_params->compute_type == CLUSTER_PARTICLE) {
    
        struct tnode *troot = NULL;
        struct tnode_array *tree_array = NULL;
        double xyzminmax[6];
        int numnodes = 0, numleaves = 0;

        struct tnode_array *batches = NULL;
        double batch_lim[6];

        struct clusters *clusters = NULL;
        

        time1 = MPI_Wtime();

        Tree_Setup(sources, targets, run_params->interp_order, xyzminmax);
        Tree_CP_Create(&troot, targets, 1, targets->num,
                       run_params->max_per_target_leaf, xyzminmax, 0, &numnodes, &numleaves);
        Tree_SetIndex(troot, 0);
        Tree_AllocArray(&tree_array, numnodes);
        Tree_CreateArray(troot, tree_array);


        time_tree[0] = MPI_Wtime() - time1; //time_maketreearray
        


        time1 = MPI_Wtime();

        Batches_Alloc(&batches, batch_lim, sources, run_params->max_per_source_leaf);
        Batches_CreateSourceBatches(batches, sources, 1, sources->num, run_params->max_per_source_leaf, batch_lim);


        time_tree[1] = MPI_Wtime() - time1; //time_createbatch
        

        time1 = MPI_Wtime();

        Clusters_CP_Setup(&clusters, run_params->interp_order, tree_array,
                          run_params->approximation, run_params->singularity);


        time_tree[2] = MPI_Wtime() - time1; //time_fillclusters
        
        
        //-------------------
        //BEGIN COMPUTE PHASE
        //-------------------
        
        
        MPI_Barrier(MPI_COMM_WORLD);

        time1 = MPI_Wtime();

        int numBatchesOnProc[numProcs];
        MPI_Allgather(&(batches->numnodes), 1, MPI_INT, numBatchesOnProc, 1, MPI_INT, MPI_COMM_WORLD);

        int pointsPerCluster = run_params->interp_pts_per_cluster;
        int chargesPerCluster = pointsPerCluster;

        if (run_params->approximation == HERMITE)
            chargesPerCluster *= 8;

        struct tnode_array *let_batches_array = NULL;
        int let_batches_array_length = 0;

        struct particles *let_sources = NULL;
        let_sources = malloc(sizeof(struct particles));  // let_sources will hold all source nodes needed for batches
        int let_sources_length = 0;  // previously let_sources included local.  Now it should not



        MPI_Win win_x_mid, win_y_mid, win_z_mid, win_radius, win_numpar, win_ibeg, win_iend;
        MPI_Win win_clusters_x, win_clusters_y, win_clusters_z, win_clusters_q, win_clusters_w;
        MPI_Win win_sources_x, win_sources_y, win_sources_z, win_sources_q, win_sources_w;
        MPI_Win win_children, win_num_children;
        
        MPI_Win_create(batches->x_mid,  batches->numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_x_mid);
        MPI_Win_create(batches->y_mid,  batches->numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_y_mid);
        MPI_Win_create(batches->z_mid,  batches->numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_z_mid);
        MPI_Win_create(batches->radius, batches->numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_radius);
        MPI_Win_create(batches->numpar, batches->numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_numpar);
        MPI_Win_create(batches->ibeg,   batches->numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_ibeg);
        MPI_Win_create(batches->iend,   batches->numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_iend);

        MPI_Win_create(clusters->x, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_x);
        MPI_Win_create(clusters->y, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_y);
        MPI_Win_create(clusters->z, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_z);
        MPI_Win_create(clusters->q, clusters->num_charges*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_q);

        MPI_Win_create(sources->x, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_x);
        MPI_Win_create(sources->y, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_y);
        MPI_Win_create(sources->z, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_z);
        MPI_Win_create(sources->q, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_q);
        MPI_Win_create(sources->w, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_w);

        // Perform MPI round robin, filling LET with remote data
        int new_sources_length_array[numProcs];
        int previous_let_sources_length_array[numProcs];
        MPI_Datatype direct_type[numProcs];



        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            // Allocate remote_tree_array
            struct tnode_array *remote_batches_array = NULL;
            Batches_AllocArray(&remote_batches_array, numBatchesOnProc[getFrom]);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_x_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_y_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_z_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_radius);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_numpar);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_ibeg);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_iend);

            MPI_Get(remote_batches_array->x_mid, numBatchesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_DOUBLE, win_x_mid);
            MPI_Get(remote_batches_array->y_mid, numBatchesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_DOUBLE, win_y_mid);
            MPI_Get(remote_batches_array->z_mid, numBatchesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_DOUBLE, win_z_mid);
            MPI_Get(remote_batches_array->radius, numBatchesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_DOUBLE, win_radius);
            MPI_Get(remote_batches_array->numpar, numBatchesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_INT, win_numpar);
            MPI_Get(remote_batches_array->ibeg, numBatchesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_INT, win_ibeg);
            MPI_Get(remote_batches_array->iend, numBatchesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numBatchesOnProc[getFrom], MPI_INT, win_iend);

            MPI_Win_unlock(getFrom, win_x_mid);
            MPI_Win_unlock(getFrom, win_y_mid);
            MPI_Win_unlock(getFrom, win_z_mid);
            MPI_Win_unlock(getFrom, win_radius);
            MPI_Win_unlock(getFrom, win_numpar);
            MPI_Win_unlock(getFrom, win_ibeg);
            MPI_Win_unlock(getFrom, win_iend);
            
            int *direct_list, *direct_ibeg_list, *direct_length_list;
            make_vector(direct_list, numBatchesOnProc[getFrom]);
            make_vector(direct_ibeg_list, numBatchesOnProc[getFrom]);
            make_vector(direct_length_list, numBatchesOnProc[getFrom]);

            InteractionList_CP_MakeRemote(tree_array, remote_batches_array,
                                          direct_list, run_params);


            // Count number of unique clusters adding to LET
            int previousBatchesArrayLength = let_batches_array_length;
            for (int i = 0; i < numBatchesOnProc[getFrom]; ++i) {
                if (direct_list[i] != -1) let_batches_array_length++;
            }
            

            if (procID == 1) {
                Batches_AllocArray(&let_batches_array, let_batches_array_length);
            } else {
                Batches_ReallocArray(let_batches_array, let_batches_array_length);
            }
            
            int previous_let_sources_length = let_sources_length;

            // Fill in LET tree array from Remote tree array.
            int appendCounter = 0;
            for (int i = 0; i < numBatchesOnProc[getFrom]; ++i) {
                if (direct_list[i] != -1) {
                    let_batches_array->x_mid[previousBatchesArrayLength + appendCounter] = remote_batches_array->x_mid[i];
                    let_batches_array->y_mid[previousBatchesArrayLength + appendCounter] = remote_batches_array->y_mid[i];
                    let_batches_array->z_mid[previousBatchesArrayLength + appendCounter] = remote_batches_array->z_mid[i];
                    let_batches_array->radius[previousBatchesArrayLength + appendCounter] = remote_batches_array->radius[i];
                    let_batches_array->numpar[previousBatchesArrayLength + appendCounter] = remote_batches_array->numpar[i];
                    
                    let_batches_array->numApprox[previousBatchesArrayLength + appendCounter] = remote_batches_array->numApprox[i];
                    let_batches_array->numDirect[previousBatchesArrayLength + appendCounter] = remote_batches_array->numDirect[i];
                        
                    // Set the beginning and ending particle indices for the associated nodes in the local sources list
                    let_batches_array->ibeg[previousBatchesArrayLength + appendCounter] = let_sources_length + 1;  // These are one-index based!!!
                    let_batches_array->iend[previousBatchesArrayLength + appendCounter] = let_sources_length + remote_batches_array->numpar[i];
                    let_sources_length += remote_batches_array->numpar[i];
                        
                    // Determine displacements and lengths for getting prticles from remote sources list
                    direct_ibeg_list[appendCounter] = remote_batches_array->ibeg[i] - 1; // These are zero-index based!!!
                    direct_length_list[appendCounter] = remote_batches_array->numpar[i];
                    appendCounter++;

                }
            }

            new_sources_length_array[getFrom] = let_sources_length - previous_let_sources_length;
            previous_let_sources_length_array[getFrom] = previous_let_sources_length;


            MPI_Type_indexed(appendCounter, direct_length_list, direct_ibeg_list,
                                          MPI_DOUBLE, &direct_type[getFrom]);
            MPI_Type_commit(&direct_type[getFrom]);

            free_vector(direct_list);
            free_vector(direct_ibeg_list);
            free_vector(direct_length_list);
            Batches_Free(remote_batches_array);
        } //end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win_x_mid);
        MPI_Win_free(&win_y_mid);
        MPI_Win_free(&win_z_mid);
        MPI_Win_free(&win_radius);
        MPI_Win_free(&win_numpar);
        MPI_Win_free(&win_ibeg);
        MPI_Win_free(&win_iend);

        if (let_sources_length > 0) Particles_AllocSources(let_sources, let_sources_length);


        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_q);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_w);

            MPI_Get(&(let_sources->x[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_x);
            MPI_Get(&(let_sources->y[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_y);
            MPI_Get(&(let_sources->z[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_z);
            MPI_Get(&(let_sources->q[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_q);
            MPI_Get(&(let_sources->w[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_w);

            MPI_Win_unlock(getFrom, win_sources_y);
            MPI_Win_unlock(getFrom, win_sources_z);
            MPI_Win_unlock(getFrom, win_sources_q);
            MPI_Win_unlock(getFrom, win_sources_w);

        } // end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Win_free(&win_sources_x);
        MPI_Win_free(&win_sources_y);
        MPI_Win_free(&win_sources_z);
        MPI_Win_free(&win_sources_q);
        MPI_Win_free(&win_sources_w);

        time_tree[3] = MPI_Wtime() - time1;


        // Beginning local computation
        
        time1 = MPI_Wtime();

        int **local_approx_inter_list, **local_direct_inter_list;

        InteractionList_Make(tree_array, batches, &local_approx_inter_list, &local_direct_inter_list,
                             run_params);

        time_tree[4] = MPI_Wtime() - time1; //time_constructlet

        time1 = MPI_Wtime();



        InteractionCompute_CP_1(tree_array, batches,
                        local_approx_inter_list, local_direct_inter_list,
                        sources->x, sources->y, sources->z, sources->q, sources->w,
                        targets->x, targets->y, targets->z, targets->q,
                        clusters->x, clusters->y, clusters->z, clusters->q, clusters->w,
                        potential_array, sources->num, targets->num, clusters->num,
                        run_params);
                        
        free_matrix(local_approx_inter_list);
        free_matrix(local_direct_inter_list);

        time_tree[5] = MPI_Wtime() - time1; //time_constructlet
        

        if (numProcs > 1) {
            time1 = MPI_Wtime();

            int **let_approx_inter_list, **let_direct_inter_list;
            

            InteractionList_Make(tree_array, let_batches_array, &let_approx_inter_list, &let_direct_inter_list,
                                 run_params);

            time_tree[6] = MPI_Wtime() - time1; //time_makeglobintlist

            // After filling LET, call interaction_list_treecode
            time1 = MPI_Wtime(); // start timer for tree evaluation


            InteractionCompute_CP_1(tree_array, let_batches_array,
                                    let_approx_inter_list, let_direct_inter_list,
                                    let_sources->x, let_sources->y, let_sources->z, let_sources->q, let_sources->w,
                                    targets->x, targets->y, targets->z, targets->q,
                                    clusters->x, clusters->y, clusters->z, clusters->q, clusters->w,
                                    potential_array, let_sources->num, targets->num, clusters->num,
                                    run_params);
            
            free_matrix(let_approx_inter_list);
            free_matrix(let_direct_inter_list);

            Particles_FreeSources(let_sources);
            Batches_Free(let_batches_array);
            
            time_tree[7] = MPI_Wtime() - time1;
        }


        time1 = MPI_Wtime();


        
        InteractionCompute_CP_2(tree_array,
                                targets->x, targets->y, targets->z, targets->q,
                                clusters->x, clusters->y, clusters->z, clusters->q, clusters->w,
                                potential_array,
                                targets->num, clusters->num_charges, clusters->num_weights,
                                run_params);

        time_tree[8] = MPI_Wtime() - time1;

        time1 = MPI_Wtime();


        
        InteractionCompute_SubtractionPotentialCorrection(potential_array, targets->q, targets->num, run_params);
        Particles_ReorderTargetsAndPotential(targets, potential_array);

        time_tree[9] = MPI_Wtime() - time1;
        
        
        //-------------------
        //CLEANUP
        //-------------------
        

        time1 = MPI_Wtime();
        
        free_vector(targets->order); // free particle order arrays
        free_vector(sources->order); // free particle order arrays
        
        Tree_Free(troot);
        Tree_FreeArray(tree_array);
        
        //I'm still not sure about deallocating the clusters and batches here.
        //Clusters_Free_Win(clusters);
        //Batches_Free(batches);
        
        time_tree[10] = MPI_Wtime() - time1; //time_cleanup
        time_tree[11] = time_tree[0] + time_tree[1] + time_tree[3] + time_tree[4] + time_tree[6]; //total setup time
        time_tree[12] = time_tree[5] + time_tree[7] + time_tree[8]; // total compute time
    
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "Done cleaning up.\n");

        return;
        
        
        
        //--------------------------------------
        //--------------------------------------
        //PARTICLE CLUSTER
        //--------------------------------------
        //--------------------------------------


        
    } else if (run_params->compute_type == PARTICLE_CLUSTER) {
    
        struct tnode *troot = NULL;
        struct tnode_array *tree_array = NULL;
        double xyzminmax[6];
        int numnodes = 0, numleaves = 0;

        struct tnode_array *batches = NULL;
        double batch_lim[6];

        struct clusters *clusters = NULL;

        time1 = MPI_Wtime();

        Tree_Setup(sources, targets, run_params->interp_order, xyzminmax);
        Tree_PC_Create(&troot, sources, 1, sources->num,
                       run_params->max_per_source_leaf, xyzminmax, 0, &numnodes, &numleaves);
        Tree_SetIndex(troot, 0);
        Tree_AllocArray(&tree_array, numnodes);
        Tree_CreateArray(troot, tree_array);

        time_tree[0] = MPI_Wtime() - time1; //time_maketreearray
        

        time1 = MPI_Wtime();

        Batches_Alloc(&batches, batch_lim, targets, run_params->max_per_target_leaf);
        Batches_CreateTargetBatches(batches, targets, 1, targets->num, run_params->max_per_target_leaf, batch_lim);

        time_tree[1] = MPI_Wtime() - time1; //time_createbatch
        

        time1 = MPI_Wtime();

        Clusters_PC_Setup(&clusters, sources, run_params->interp_order, tree_array,
                          run_params->approximation, run_params->singularity);

        time_tree[2] = MPI_Wtime() - time1; //time_fillclusters


        //-------------------
        //BEGIN COMPUTE PHASE
        //-------------------


        MPI_Barrier(MPI_COMM_WORLD);
  
        time1 = MPI_Wtime();

        int numNodesOnProc[numProcs];
        MPI_Allgather(&numnodes, 1, MPI_INT, numNodesOnProc, 1, MPI_INT, MPI_COMM_WORLD);

        int pointsPerCluster = run_params->interp_pts_per_cluster;
        int chargesPerCluster = pointsPerCluster;
        int weightsPerCluster = pointsPerCluster;

        if (run_params->approximation == HERMITE)
            chargesPerCluster *= 8;

        if (run_params->approximation == HERMITE && run_params->singularity == SUBTRACTION)
            weightsPerCluster *= 8;

        struct tnode_array *let_tree_array = NULL;
        int let_tree_array_length = 0;

        struct clusters *let_clusters = NULL;
        let_clusters = malloc(sizeof(struct clusters));
        int let_clusters_length = 0; // previously let_clusters included the local.  Now it should not

        struct particles *let_sources = NULL;
        let_sources = malloc(sizeof(struct particles));  // let_sources will hold all source nodes needed for direct interactions
        int let_sources_length = 0;  // previously let_sources included local.  Now it should not


        MPI_Win win_x_mid, win_y_mid, win_z_mid, win_radius, win_numpar, win_ibeg, win_iend, win_level;
        MPI_Win win_clusters_x, win_clusters_y, win_clusters_z, win_clusters_q, win_clusters_w;
        MPI_Win win_sources_x, win_sources_y, win_sources_z, win_sources_q, win_sources_w;
        MPI_Win win_children, win_num_children;
        
        MPI_Win_create(tree_array->x_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_x_mid);
        MPI_Win_create(tree_array->y_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_y_mid);
        MPI_Win_create(tree_array->z_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_z_mid);
        MPI_Win_create(tree_array->radius, numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_radius);
        MPI_Win_create(tree_array->numpar, numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_numpar);
        MPI_Win_create(tree_array->ibeg,   numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_ibeg);
        MPI_Win_create(tree_array->iend,   numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_iend);
        MPI_Win_create(tree_array->level,  numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_level);
        MPI_Win_create(tree_array->num_children,  numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_num_children);
        MPI_Win_create(tree_array->children,    8*numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_children);

        MPI_Win_create(clusters->x, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_x);
        MPI_Win_create(clusters->y, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_y);
        MPI_Win_create(clusters->z, clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_z);
        MPI_Win_create(clusters->q, clusters->num_charges*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_q);
        MPI_Win_create(clusters->w, clusters->num_weights*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_w);

        MPI_Win_create(sources->x, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_x);
        MPI_Win_create(sources->y, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_y);
        MPI_Win_create(sources->z, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_z);
        MPI_Win_create(sources->q, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_q);
        MPI_Win_create(sources->w, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_w);


        // Perform MPI round robin, filling LET with remote data
        int num_remote_approx_array[numProcs], new_sources_length_array[numProcs];
        int previous_let_clusters_length_array[numProcs], previous_let_sources_length_array[numProcs];
        MPI_Datatype approx_type[numProcs], approx_charges_type[numProcs], approx_weights_type[numProcs];
        MPI_Datatype direct_type[numProcs];
        int let_clusters_num = 0;


        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            // Allocate remote_tree_array
            struct tnode_array *remote_tree_array = NULL;
            Tree_AllocArray(&remote_tree_array, numNodesOnProc[getFrom]);

            // Get remote_tree_array
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_x_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_y_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_z_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_radius);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_numpar);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_ibeg);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_iend);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_level);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_children);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_num_children);

            
            MPI_Get(remote_tree_array->x_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_x_mid);
            MPI_Get(remote_tree_array->y_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_y_mid);
            MPI_Get(remote_tree_array->z_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_z_mid);
            MPI_Get(remote_tree_array->radius, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_radius);
            MPI_Get(remote_tree_array->numpar, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_numpar);
            MPI_Get(remote_tree_array->ibeg, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_ibeg);
            MPI_Get(remote_tree_array->iend, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_iend);
            MPI_Get(remote_tree_array->level, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_level);

            MPI_Get(remote_tree_array->children, 8*numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, 8*numNodesOnProc[getFrom], MPI_INT, win_children);
            MPI_Get(remote_tree_array->num_children, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_num_children);

            MPI_Win_unlock(getFrom, win_x_mid);
            MPI_Win_unlock(getFrom, win_y_mid);
            MPI_Win_unlock(getFrom, win_z_mid);
            MPI_Win_unlock(getFrom, win_radius);
            MPI_Win_unlock(getFrom, win_numpar);
            MPI_Win_unlock(getFrom, win_ibeg);
            MPI_Win_unlock(getFrom, win_iend);
            MPI_Win_unlock(getFrom, win_level);
            MPI_Win_unlock(getFrom, win_children);
            MPI_Win_unlock(getFrom, win_num_children);


            // Construct masks
            int *approx_list_packed, *approx_list_unpacked, *direct_list, *direct_ibeg_list, *direct_length_list;
            make_vector(approx_list_packed, numNodesOnProc[getFrom]);
            make_vector(approx_list_unpacked, numNodesOnProc[getFrom]);
            make_vector(direct_list, numNodesOnProc[getFrom]);
            make_vector(direct_ibeg_list, numNodesOnProc[getFrom]);
            make_vector(direct_length_list, numNodesOnProc[getFrom]);

            InteractionList_PC_MakeRemote(remote_tree_array, batches, approx_list_unpacked, approx_list_packed,
                                          direct_list, run_params);


            // Count number of unique clusters adding to LET
            int previousTreeArrayLength = let_tree_array_length;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {
                let_tree_array_length++;
            }
            

            if (procID == 1) {
                Tree_AllocArray(&let_tree_array, let_tree_array_length);
            } else {
                Tree_ReallocArray(let_tree_array, let_tree_array_length);
            }

            int numberOfRemoteApprox = 0;
            int previous_let_clusters_length = let_clusters_length;

            int numberOfRemoteDirect = 0;
            int previous_let_sources_length = let_sources_length;



            // Fill in LET tree array from Remote tree array.
            int appendCounter = 0;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {

                let_tree_array->x_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->x_mid[i];
                let_tree_array->y_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->y_mid[i];
                let_tree_array->z_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->z_mid[i];
                let_tree_array->radius[previousTreeArrayLength + appendCounter] = remote_tree_array->radius[i];
                let_tree_array->numpar[previousTreeArrayLength + appendCounter] = remote_tree_array->numpar[i];
                let_tree_array->level[previousTreeArrayLength + appendCounter] = remote_tree_array->level[i];
                let_tree_array->num_children[previousTreeArrayLength + appendCounter] = remote_tree_array->num_children[i];
                
                for (int j = 0; j < let_tree_array->num_children[previousTreeArrayLength + appendCounter]; ++j)
                    let_tree_array->children[8 * (previousTreeArrayLength + appendCounter) + j]
                        = remote_tree_array->children[8*i + j];
                    
                if (approx_list_unpacked[i] != -1) {
                    let_tree_array->cluster_ind[previousTreeArrayLength + appendCounter] = let_clusters_num;
                    let_clusters_length += pointsPerCluster;
                    let_clusters_num++;
                    numberOfRemoteApprox++;
                }
                    
                if (direct_list[i] != -1) {
                        
                    // Set the beginning and ending particle indices for the associated nodes in the local sources list
                    let_tree_array->ibeg[previousTreeArrayLength + appendCounter] = let_sources_length + 1;  // These are one-index based!!!
                    let_tree_array->iend[previousTreeArrayLength + appendCounter] = let_sources_length + remote_tree_array->numpar[i];
                    let_sources_length += remote_tree_array->numpar[i];
                        
                    // Determine displacements and lengths for getting prticles from remote sources list
                    direct_ibeg_list[numberOfRemoteDirect] = remote_tree_array->ibeg[i] - 1; // These are zero-index based!!!
                    direct_length_list[numberOfRemoteDirect] = remote_tree_array->numpar[i];
                    numberOfRemoteDirect++;
                }
                
                appendCounter++;
            }

            
            num_remote_approx_array[getFrom] = numberOfRemoteApprox;
            new_sources_length_array[getFrom] = let_sources_length - previous_let_sources_length;
            previous_let_clusters_length_array[getFrom] = previous_let_clusters_length;
            previous_let_sources_length_array[getFrom] = previous_let_sources_length;
            
            int *approx_list_displacements, *approx_charges_list_displacements, *approx_weights_list_displacements;
            make_vector(approx_list_displacements, numNodesOnProc[getFrom]);
            make_vector(approx_charges_list_displacements, numNodesOnProc[getFrom]);
            make_vector(approx_weights_list_displacements, numNodesOnProc[getFrom]);

            // Use masks to get remote data
            for (int ii = 0; ii < numberOfRemoteApprox; ++ii) {
                approx_list_displacements[ii] = approx_list_packed[ii] * pointsPerCluster;
                approx_charges_list_displacements[ii] = approx_list_packed[ii] * chargesPerCluster;
                approx_weights_list_displacements[ii] = approx_list_packed[ii] * weightsPerCluster;
            }
            
            MPI_Type_create_indexed_block(numberOfRemoteApprox, pointsPerCluster, approx_list_displacements,
                                          MPI_DOUBLE, &approx_type[getFrom]);
            MPI_Type_commit(&approx_type[getFrom]);

            MPI_Type_create_indexed_block(numberOfRemoteApprox, chargesPerCluster, approx_charges_list_displacements,
                                          MPI_DOUBLE, &approx_charges_type[getFrom]);
            MPI_Type_commit(&approx_charges_type[getFrom]);

            MPI_Type_create_indexed_block(numberOfRemoteApprox, weightsPerCluster, approx_weights_list_displacements,
                                          MPI_DOUBLE, &approx_weights_type[getFrom]);
            MPI_Type_commit(&approx_weights_type[getFrom]);

            MPI_Type_indexed(numberOfRemoteDirect, direct_length_list, direct_ibeg_list,
                                          MPI_DOUBLE, &direct_type[getFrom]);
            MPI_Type_commit(&direct_type[getFrom]);

            free_vector(approx_list_packed);
            free_vector(approx_list_unpacked);
            free_vector(approx_list_displacements);
            free_vector(approx_charges_list_displacements);
            free_vector(approx_weights_list_displacements);
            free_vector(direct_list);
            free_vector(direct_ibeg_list);
            free_vector(direct_length_list);
            Tree_FreeArray(remote_tree_array);
        } //end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win_x_mid);
        MPI_Win_free(&win_y_mid);
        MPI_Win_free(&win_z_mid);
        MPI_Win_free(&win_radius);
        MPI_Win_free(&win_numpar);
        MPI_Win_free(&win_ibeg);
        MPI_Win_free(&win_iend);
        MPI_Win_free(&win_level);
        MPI_Win_free(&win_children);
        MPI_Win_free(&win_num_children);


        if (let_sources_length > 0) Particles_AllocSources(let_sources, let_sources_length);
        if (let_clusters_length > 0) Clusters_Alloc(let_clusters, let_clusters_length,
                                                    run_params->approximation, run_params->singularity);
    
        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_w);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_q);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_q);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_w);


            MPI_Get(&(let_clusters->x[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_x);
            MPI_Get(&(let_clusters->y[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_y);
            MPI_Get(&(let_clusters->z[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_z);

            MPI_Get(&(let_clusters->q[(chargesPerCluster/pointsPerCluster) * previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * chargesPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_charges_type[getFrom], win_clusters_q);
            MPI_Get(&(let_clusters->w[(weightsPerCluster/pointsPerCluster) * previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * weightsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_weights_type[getFrom], win_clusters_w);

            MPI_Get(&(let_sources->x[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_x);
            MPI_Get(&(let_sources->y[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_y);
            MPI_Get(&(let_sources->z[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_z);
            MPI_Get(&(let_sources->q[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_q);
            MPI_Get(&(let_sources->w[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_w);

            
            MPI_Win_unlock(getFrom, win_clusters_x);
            MPI_Win_unlock(getFrom, win_clusters_y);
            MPI_Win_unlock(getFrom, win_clusters_z);
            MPI_Win_unlock(getFrom, win_clusters_q);
            MPI_Win_unlock(getFrom, win_clusters_w);
            
            MPI_Win_unlock(getFrom, win_sources_x);
            MPI_Win_unlock(getFrom, win_sources_y);
            MPI_Win_unlock(getFrom, win_sources_z);
            MPI_Win_unlock(getFrom, win_sources_q);
            MPI_Win_unlock(getFrom, win_sources_w);

        } // end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win_clusters_x);
        MPI_Win_free(&win_clusters_y);
        MPI_Win_free(&win_clusters_z);
        MPI_Win_free(&win_clusters_q);
        MPI_Win_free(&win_clusters_w);

        MPI_Win_free(&win_sources_x);
        MPI_Win_free(&win_sources_y);
        MPI_Win_free(&win_sources_z);
        MPI_Win_free(&win_sources_q);
        MPI_Win_free(&win_sources_w);

        time_tree[3] = MPI_Wtime() - time1;


        // Beginning local computation
        
        time1 = MPI_Wtime();

        int **local_approx_inter_list, **local_direct_inter_list;
        
        InteractionList_Make(tree_array, batches, &local_approx_inter_list, &local_direct_inter_list,
                             run_params);

        time_tree[4] = MPI_Wtime() - time1; //time_constructlet


        if (run_params->verbosity > 0) {
            for (int j = 0; j < batches->numnodes; j++){
                totalNumberApprox += batches->numApprox[j];
                totalNumberDirect += batches->numDirect[j];
            }
        }
        time1 = MPI_Wtime();



        InteractionCompute_PC(tree_array, batches,
                        local_approx_inter_list, local_direct_inter_list,
                        sources->x, sources->y, sources->z, sources->q, sources->w,
                        targets->x, targets->y, targets->z, targets->q,
                        clusters->x, clusters->y, clusters->z, clusters->q, clusters->w,
                        potential_array,
                        sources->num, targets->num, clusters->num,
                        run_params);
                        
        free_matrix(local_approx_inter_list);
        free_matrix(local_direct_inter_list);

        time_tree[5] = MPI_Wtime() - time1; //time_constructlet


        // Compute interaction lists based on LET
        if (numProcs > 1) {
            time1 = MPI_Wtime();

            int **let_approx_inter_list, **let_direct_inter_list;

            InteractionList_Make(let_tree_array, batches, &let_approx_inter_list, &let_direct_inter_list,
                                 run_params);

            // Count number of interactions

            if (run_params->verbosity > 0) {
                for (int j = 0; j < batches->numnodes; j++){
                    totalNumberApprox += batches->numApprox[j];
                    totalNumberDirect += batches->numDirect[j];
                }
            }
            time_tree[6] = MPI_Wtime() - time1; //time_makeglobintlist

            // After filling LET, call interaction_list_treecode
            time1 = MPI_Wtime(); // start timer for tree evaluation

            InteractionCompute_PC(let_tree_array, batches,
                                   let_approx_inter_list, let_direct_inter_list,
                                   let_sources->x, let_sources->y, let_sources->z, let_sources->q, let_sources->w,
                                   targets->x, targets->y, targets->z, targets->q,
                                   let_clusters->x, let_clusters->y, let_clusters->z, let_clusters->q, let_clusters->w,
                                   potential_array,
                                   let_sources->num, targets->num, let_clusters->num,
                                   run_params);
            
            free_matrix(let_approx_inter_list);
            free_matrix(let_direct_inter_list);
            
            Clusters_Free(let_clusters);
            Particles_FreeSources(let_sources);
            Tree_FreeArray(let_tree_array);
            
            time_tree[7] = MPI_Wtime() - time1;
        }


        time1 = MPI_Wtime();
        
        InteractionCompute_SubtractionPotentialCorrection(potential_array, targets->q, targets->num,
                                  run_params);

        Particles_ReorderTargetsAndPotential(targets, potential_array);

        time_tree[8] = 0.0;
        time_tree[9] = MPI_Wtime() - time1;
        
        
        //-------------------
        //CLEANUP
        //-------------------
        
        
        time1 = MPI_Wtime();
        
        free_vector(targets->order); // free particle order arrays
        free_vector(sources->order); // free particle order arrays
        
        Tree_Free(troot);
        Tree_FreeArray(tree_array);
        Clusters_Free_Win(clusters);
        Batches_Free(batches);
        
        time_tree[10] = MPI_Wtime() - time1; //time_cleanup
        time_tree[11] = time_tree[0] + time_tree[1] + time_tree[3] + time_tree[4] + time_tree[6]; //total setup time
        time_tree[12] = time_tree[5] + time_tree[7] + time_tree[8]; // total compute time
    
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "Done cleaning up.\n");

        return;



        //--------------------------------------
        //--------------------------------------
        //CLUSTER CLUSTER
        //--------------------------------------
        //--------------------------------------



    } else if (run_params->compute_type == CLUSTER_CLUSTER) {
    
        struct tnode *source_tree_root = NULL;
        struct tnode_array *source_tree_array = NULL;

        struct tnode *target_tree_root = NULL;
        struct tnode_array *target_tree_array = NULL;

        struct clusters *source_clusters = NULL;
        struct clusters *target_clusters = NULL;

        double source_xyzminmax[6], target_xyzminmax[6];
        int source_numnodes = 0, source_numleaves = 0;
        int target_numnodes = 0, target_numleaves = 0;

        time1 = MPI_Wtime();

        Tree_CC_Setup(sources, targets, run_params->interp_order,
                      source_xyzminmax, target_xyzminmax);

        Tree_PC_Create(&source_tree_root, sources, 1, sources->num,
                       run_params->max_per_source_leaf, source_xyzminmax, 0,
                       &source_numnodes, &source_numleaves);
        Tree_SetIndex(source_tree_root, 0);

        Tree_AllocArray(&source_tree_array, source_numnodes);
        Tree_CreateArray(source_tree_root, source_tree_array);

        time_tree[0] = MPI_Wtime() - time1;


        time1 = MPI_Wtime();

        Tree_CP_Create(&target_tree_root, targets, 1, targets->num,
                       run_params->max_per_target_leaf, target_xyzminmax, 0,
                       &target_numnodes, &target_numleaves);
        Tree_SetIndex(target_tree_root, 0);

        Tree_AllocArray(&target_tree_array, target_numnodes);
        Tree_CreateArray(target_tree_root, target_tree_array);
        make_vector(target_tree_array->numApprox, target_tree_array->numnodes);
        make_vector(target_tree_array->numDirect, target_tree_array->numnodes);

        time_tree[1] = MPI_Wtime() - time1; //time_maketreearray
         

        time1 = MPI_Wtime();

        Clusters_PC_Setup(&source_clusters, sources, run_params->interp_order, source_tree_array,
                          run_params->approximation, run_params->singularity);

        Clusters_CP_Setup(&target_clusters, run_params->interp_order, target_tree_array,
                          run_params->approximation, run_params->singularity);

        time_tree[2] = MPI_Wtime() - time1; //time_fillclusters
        
        
        //-------------------
        //COMPUTE PHASE
        //-------------------
        
        
        MPI_Barrier(MPI_COMM_WORLD);
  
        time1 = MPI_Wtime();

        int numNodesOnProc[numProcs];
        MPI_Allgather(&source_numnodes, 1, MPI_INT, numNodesOnProc, 1, MPI_INT, MPI_COMM_WORLD);

        int pointsPerCluster = run_params->interp_pts_per_cluster;
        int chargesPerCluster = pointsPerCluster;
        int weightsPerCluster = pointsPerCluster;

        if (run_params->approximation == HERMITE)
            chargesPerCluster *= 8;

        if (run_params->approximation == HERMITE && run_params->singularity == SUBTRACTION)
            weightsPerCluster *= 8;

        struct tnode_array *let_tree_array = NULL;
        int let_tree_array_length = 0;

        struct clusters *let_clusters = malloc(sizeof(struct clusters));
        int let_clusters_length = 0;

        struct particles *let_sources = malloc(sizeof(struct particles));
        int let_sources_length = 0;


        MPI_Win win_x_mid, win_y_mid, win_z_mid, win_radius, win_numpar, win_ibeg, win_iend, win_level;
        MPI_Win win_clusters_x, win_clusters_y, win_clusters_z, win_clusters_q, win_clusters_w;
        MPI_Win win_sources_x, win_sources_y, win_sources_z, win_sources_q, win_sources_w;
        MPI_Win win_children, win_num_children;
        
        MPI_Win_create(source_tree_array->x_mid,  source_numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_x_mid);
        MPI_Win_create(source_tree_array->y_mid,  source_numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_y_mid);
        MPI_Win_create(source_tree_array->z_mid,  source_numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_z_mid);
        MPI_Win_create(source_tree_array->radius, source_numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_radius);
        MPI_Win_create(source_tree_array->numpar, source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_numpar);
        MPI_Win_create(source_tree_array->ibeg,   source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_ibeg);
        MPI_Win_create(source_tree_array->iend,   source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_iend);
        MPI_Win_create(source_tree_array->level,  source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_level);
        MPI_Win_create(source_tree_array->num_children,  source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_num_children);
        MPI_Win_create(source_tree_array->children,    8*source_numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_children);

        MPI_Win_create(source_clusters->x, source_clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_x);
        MPI_Win_create(source_clusters->y, source_clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_y);
        MPI_Win_create(source_clusters->z, source_clusters->num*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_z);
        MPI_Win_create(source_clusters->q, source_clusters->num_charges*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_q);
        MPI_Win_create(source_clusters->w, source_clusters->num_weights*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_w);

        MPI_Win_create(sources->x, source_tree_root->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_x);
        MPI_Win_create(sources->y, source_tree_root->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_y);
        MPI_Win_create(sources->z, source_tree_root->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_z);
        MPI_Win_create(sources->q, source_tree_root->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_q);
        MPI_Win_create(sources->w, source_tree_root->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_w);


        // Perform MPI round robin, filling LET with remote data
        int num_remote_approx_array[numProcs], new_sources_length_array[numProcs];
        int previous_let_clusters_length_array[numProcs], previous_let_sources_length_array[numProcs];
        MPI_Datatype approx_type[numProcs], approx_charges_type[numProcs], approx_weights_type[numProcs];
        MPI_Datatype direct_type[numProcs];
        int let_clusters_num = 0;
         

        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            struct tnode_array *remote_tree_array = NULL;
            Tree_AllocArray(&remote_tree_array, numNodesOnProc[getFrom]);

            // Get remote_tree_array
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_x_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_y_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_z_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_radius);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_numpar);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_ibeg);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_iend);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_level);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_children);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_num_children);

            
            MPI_Get(remote_tree_array->x_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_x_mid);
            MPI_Get(remote_tree_array->y_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_y_mid);
            MPI_Get(remote_tree_array->z_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_z_mid);
            MPI_Get(remote_tree_array->radius, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_radius);
            MPI_Get(remote_tree_array->numpar, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_numpar);
            MPI_Get(remote_tree_array->ibeg, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_ibeg);
            MPI_Get(remote_tree_array->iend, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_iend);
            MPI_Get(remote_tree_array->level, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_level);

            MPI_Get(remote_tree_array->children, 8*numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, 8*numNodesOnProc[getFrom], MPI_INT, win_children);
            MPI_Get(remote_tree_array->num_children, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_num_children);
            
            MPI_Win_unlock(getFrom, win_x_mid);
            MPI_Win_unlock(getFrom, win_y_mid);
            MPI_Win_unlock(getFrom, win_z_mid);
            MPI_Win_unlock(getFrom, win_radius);
            MPI_Win_unlock(getFrom, win_numpar);
            MPI_Win_unlock(getFrom, win_ibeg);
            MPI_Win_unlock(getFrom, win_iend);
            MPI_Win_unlock(getFrom, win_level);
            MPI_Win_unlock(getFrom, win_children);
            MPI_Win_unlock(getFrom, win_num_children);


            // Construct lists
            int *approx_list_packed, *approx_list_unpacked, *direct_list, *direct_ibeg_list, *direct_length_list;
            make_vector(approx_list_packed, numNodesOnProc[getFrom]);
            make_vector(approx_list_unpacked, numNodesOnProc[getFrom]);
            make_vector(direct_list, numNodesOnProc[getFrom]);
            make_vector(direct_ibeg_list, numNodesOnProc[getFrom]);
            make_vector(direct_length_list, numNodesOnProc[getFrom]);

            InteractionList_CC_MakeRemote(remote_tree_array, target_tree_array,
                                          approx_list_unpacked, approx_list_packed, direct_list,
                                          run_params);


            // Count number of unique clusters adding to LET
            int previousTreeArrayLength = let_tree_array_length;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {
                let_tree_array_length++;
            }
            

            if (procID == 1) {
                Tree_AllocArray(&let_tree_array, let_tree_array_length);
            } else {
                Tree_ReallocArray(let_tree_array, let_tree_array_length);
            }

            int numberOfRemoteApprox = 0;
            int previous_let_clusters_length = let_clusters_length;

            int numberOfRemoteDirect = 0;
            int previous_let_sources_length = let_sources_length;


            // Fill in LET tree array from Remote tree array.
            int appendCounter = 0;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {

                let_tree_array->x_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->x_mid[i];
                let_tree_array->y_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->y_mid[i];
                let_tree_array->z_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->z_mid[i];
                let_tree_array->radius[previousTreeArrayLength + appendCounter] = remote_tree_array->radius[i];
                let_tree_array->numpar[previousTreeArrayLength + appendCounter] = remote_tree_array->numpar[i];
                let_tree_array->level[previousTreeArrayLength + appendCounter] = remote_tree_array->level[i];
                let_tree_array->num_children[previousTreeArrayLength + appendCounter] = remote_tree_array->num_children[i];
                
                for (int j = 0; j < let_tree_array->num_children[previousTreeArrayLength + appendCounter]; ++j)
                    let_tree_array->children[8 * (previousTreeArrayLength + appendCounter) + j]
                        = remote_tree_array->children[8*i + j];
                    
                if (approx_list_unpacked[i] != -1) {
                    let_tree_array->cluster_ind[previousTreeArrayLength + appendCounter] = let_clusters_num;
                    let_clusters_length += pointsPerCluster;
                    let_clusters_num++;
                    numberOfRemoteApprox++;
                }
                    
                if (direct_list[i] != -1) {
                        
                    // Set the beginning and ending particle indices for the associated nodes in the local sources list
                    let_tree_array->ibeg[previousTreeArrayLength + appendCounter] = let_sources_length + 1;  // These are one-index based!!!
                    let_tree_array->iend[previousTreeArrayLength + appendCounter] = let_sources_length + remote_tree_array->numpar[i];
                    let_sources_length += remote_tree_array->numpar[i];
                        
                    // Determine displacements and lengths for getting prticles from remote sources list
                    direct_ibeg_list[numberOfRemoteDirect] = remote_tree_array->ibeg[i] - 1; // These are zero-index based!!!
                    direct_length_list[numberOfRemoteDirect] = remote_tree_array->numpar[i];
                    numberOfRemoteDirect++;
                }
                
                appendCounter++;
            }

            
            num_remote_approx_array[getFrom] = numberOfRemoteApprox;
            new_sources_length_array[getFrom] = let_sources_length - previous_let_sources_length;
            previous_let_clusters_length_array[getFrom] = previous_let_clusters_length;
            previous_let_sources_length_array[getFrom] = previous_let_sources_length;
            
            
            int *approx_list_displacements, *approx_charges_list_displacements, *approx_weights_list_displacements;
            make_vector(approx_list_displacements, numNodesOnProc[getFrom]);
            make_vector(approx_charges_list_displacements, numNodesOnProc[getFrom]);
            make_vector(approx_weights_list_displacements, numNodesOnProc[getFrom]);

            // Use masks to get remote data
            for (int ii = 0; ii < numberOfRemoteApprox; ++ii) {
                approx_list_displacements[ii] = approx_list_packed[ii] * pointsPerCluster;
                approx_charges_list_displacements[ii] = approx_list_packed[ii] * chargesPerCluster;
                approx_weights_list_displacements[ii] = approx_list_packed[ii] * weightsPerCluster;
            }
            
            MPI_Type_create_indexed_block(numberOfRemoteApprox, pointsPerCluster, approx_list_displacements,
                                          MPI_DOUBLE, &approx_type[getFrom]);
            MPI_Type_commit(&approx_type[getFrom]);

            MPI_Type_create_indexed_block(numberOfRemoteApprox, chargesPerCluster, approx_charges_list_displacements,
                                          MPI_DOUBLE, &approx_charges_type[getFrom]);
            MPI_Type_commit(&approx_charges_type[getFrom]);

            MPI_Type_create_indexed_block(numberOfRemoteApprox, weightsPerCluster, approx_weights_list_displacements,
                                          MPI_DOUBLE, &approx_weights_type[getFrom]);
            MPI_Type_commit(&approx_weights_type[getFrom]);

            MPI_Type_indexed(numberOfRemoteDirect, direct_length_list, direct_ibeg_list,
                                          MPI_DOUBLE, &direct_type[getFrom]);
            MPI_Type_commit(&direct_type[getFrom]);

            free_vector(approx_list_packed);
            free_vector(approx_list_unpacked);
            free_vector(approx_list_displacements);
            free_vector(approx_charges_list_displacements);
            free_vector(approx_weights_list_displacements);
            free_vector(direct_list);
            free_vector(direct_ibeg_list);
            free_vector(direct_length_list);
            Tree_FreeArray(remote_tree_array);
        } //end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win_x_mid);
        MPI_Win_free(&win_y_mid);
        MPI_Win_free(&win_z_mid);
        MPI_Win_free(&win_radius);
        MPI_Win_free(&win_numpar);
        MPI_Win_free(&win_ibeg);
        MPI_Win_free(&win_iend);
        MPI_Win_free(&win_level);
        MPI_Win_free(&win_children);
        MPI_Win_free(&win_num_children);


        if (let_sources_length > 0) Particles_AllocSources(let_sources, let_sources_length);
        if (let_clusters_length > 0) Clusters_Alloc(let_clusters, let_clusters_length,
                                                    run_params->approximation, run_params->singularity);
    
        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_w);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_q);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_q);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_w);


            MPI_Get(&(let_clusters->x[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_x);
            MPI_Get(&(let_clusters->y[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_y);
            MPI_Get(&(let_clusters->z[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_z);

            MPI_Get(&(let_clusters->q[(chargesPerCluster/pointsPerCluster) * previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * chargesPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_charges_type[getFrom], win_clusters_q);
            MPI_Get(&(let_clusters->w[(weightsPerCluster/pointsPerCluster) * previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * weightsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_weights_type[getFrom], win_clusters_w);

            MPI_Get(&(let_sources->x[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_x);
            MPI_Get(&(let_sources->y[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_y);
            MPI_Get(&(let_sources->z[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_z);
            MPI_Get(&(let_sources->q[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_q);
            MPI_Get(&(let_sources->w[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_w);

            
            MPI_Win_unlock(getFrom, win_clusters_x);
            MPI_Win_unlock(getFrom, win_clusters_y);
            MPI_Win_unlock(getFrom, win_clusters_z);
            MPI_Win_unlock(getFrom, win_clusters_q);
            MPI_Win_unlock(getFrom, win_clusters_w);
            
            MPI_Win_unlock(getFrom, win_sources_x);
            MPI_Win_unlock(getFrom, win_sources_y);
            MPI_Win_unlock(getFrom, win_sources_z);
            MPI_Win_unlock(getFrom, win_sources_q);
            MPI_Win_unlock(getFrom, win_sources_w);

        } // end loop over numProcs


        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win_clusters_x);
        MPI_Win_free(&win_clusters_y);
        MPI_Win_free(&win_clusters_z);
        MPI_Win_free(&win_clusters_q);
        MPI_Win_free(&win_clusters_w);

        MPI_Win_free(&win_sources_x);
        MPI_Win_free(&win_sources_y);
        MPI_Win_free(&win_sources_z);
        MPI_Win_free(&win_sources_q);
        MPI_Win_free(&win_sources_w);

        time_tree[3] = MPI_Wtime() - time1;

        
        time1 = MPI_Wtime();

        int **local_approx_inter_list, **local_direct_inter_list;
        InteractionList_CC_Make(source_tree_array, target_tree_array, &local_approx_inter_list, &local_direct_inter_list,
                                run_params);

        time_tree[4] = MPI_Wtime() - time1; //time_constructlet


        if (run_params->verbosity > 0) {
            for (int j = 0; j < target_tree_array->numnodes; j++){
                totalNumberApprox += target_tree_array->numApprox[j];
                totalNumberDirect += target_tree_array->numDirect[j];
            }
        }
        time1 = MPI_Wtime();

        InteractionCompute_CC_1(source_tree_array, target_tree_array,
                        local_approx_inter_list, local_direct_inter_list,
                        sources->x, sources->y, sources->z, sources->q, sources->w,
                        targets->x, targets->y, targets->z, targets->q,
                        source_clusters->x, source_clusters->y, source_clusters->z,
                        source_clusters->q, source_clusters->w,
                        target_clusters->x, target_clusters->y, target_clusters->z,
                        target_clusters->q, target_clusters->w,
                        potential_array,
                        sources->num, targets->num, source_clusters->num, target_clusters->num,
                        run_params);
                        
        free_matrix(local_approx_inter_list);
        free_matrix(local_direct_inter_list);

        time_tree[5] = MPI_Wtime() - time1; //time_constructlet


        // Compute interaction lists based on LET
        if (numProcs > 1) {
            time1 = MPI_Wtime();

            int **let_approx_inter_list, **let_direct_inter_list;
            
            InteractionList_CC_Make(let_tree_array, target_tree_array, &let_approx_inter_list, &let_direct_inter_list,
                                    run_params);

            // Count number of interactions

            if (run_params->verbosity > 0) {
                for (int j = 0; j < target_tree_array->numnodes; j++){
                    totalNumberApprox += target_tree_array->numApprox[j];
                    totalNumberDirect += target_tree_array->numDirect[j];
                }
            }
            time_tree[6] = MPI_Wtime() - time1; //time_makeglobintlist

            // After filling LET, call interaction_list_treecode
            time1 = MPI_Wtime(); // start timer for tree evaluation

            InteractionCompute_CC_1(let_tree_array, target_tree_array,
                                   let_approx_inter_list, let_direct_inter_list,
                                   let_sources->x, let_sources->y, let_sources->z,
                                   let_sources->q, let_sources->w,
                                   targets->x, targets->y, targets->z, targets->q,
                                   let_clusters->x, let_clusters->y, let_clusters->z,
                                   let_clusters->q, let_clusters->w,
                                   target_clusters->x, target_clusters->y, target_clusters->z,
                                   target_clusters->q, target_clusters->w,
                                   potential_array,
                                   let_sources->num, targets->num, let_clusters->num, target_clusters->num,
                                   run_params);

            free_matrix(let_approx_inter_list);
            free_matrix(let_direct_inter_list);
            
            Clusters_Free(let_clusters);
            Particles_FreeSources(let_sources);
            Tree_FreeArray(let_tree_array);
            
            time_tree[7] = MPI_Wtime() - time1;
        }
        
        time1 = MPI_Wtime();
        
        InteractionCompute_CP_2(target_tree_array,
                                targets->x, targets->y, targets->z, targets->q,
                                target_clusters->x, target_clusters->y, target_clusters->z,
                                target_clusters->q, target_clusters->w,
                                potential_array, 
                                targets->num, target_clusters->num_charges, target_clusters->num_weights,
                                run_params);

        time_tree[8] = MPI_Wtime() - time1;


        time1 = MPI_Wtime();
        
        InteractionCompute_SubtractionPotentialCorrection(potential_array, targets->q, targets->num,
                                  run_params);

        Particles_ReorderTargetsAndPotential(targets, potential_array);

        time_tree[9] = MPI_Wtime() - time1;
               
               
        //-------------------
        //CLEANUP
        //-------------------
        
        
        time1 = MPI_Wtime();
        
        free_vector(targets->order); // free particle order arrays
        free_vector(sources->order); // free particle order arrays

        free_vector(target_tree_array->numApprox);
        free_vector(target_tree_array->numDirect);
        
        Tree_Free(source_tree_root);
        Tree_Free(target_tree_root);
        Tree_FreeArray(source_tree_array);
        Tree_FreeArray(target_tree_array);
        
        Clusters_Free_Win(source_clusters);
        Clusters_Free(target_clusters);
        
        time_tree[10] = MPI_Wtime() - time1; //time_cleanup
        time_tree[11] = time_tree[0] + time_tree[1] + time_tree[3] + time_tree[4] + time_tree[6]; //total setup time
        time_tree[12] = time_tree[5] + time_tree[7] + time_tree[8]; // total compute time
        
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "Done cleaning up.\n");

        return;
        
    }
    
    
    

//    if (run_params->verbosity > 0) {
//        printf("Tree information: \n\n");
//
//        printf("                      numpar: %d\n", troot->numpar);
//        printf("                       x_mid: %e\n", troot->x_mid);
//        printf("                       y_mid: %e\n", troot->y_mid);
//        printf("                       z_mid: %e\n\n", troot->z_mid);
//        printf("                      radius: %f\n\n", troot->radius);
//        printf("                       x_len: %e\n", troot->x_max - troot->x_min);
//        printf("                       y_len: %e\n", troot->y_max - troot->y_min);
//        printf("                       z_len: %e\n\n", troot->z_max - troot->z_min);
//        printf("                      torder: %d\n", interpolationOrder);
//        printf("                       theta: %f\n", theta);
//        printf("                  maxparnode: %d\n", maxparnode);
//        printf("            number of leaves: %d\n", numleaves);
//        printf("             number of nodes: %d\n", numnodes);
//        printf("           target batch size: %d\n", batch_size);
//        printf("           number of batches: %d\n\n", batches->numnodes);
//    }




//    if (run_params->verbosity > 0) {
//        totalNumberInteractions=totalNumberDirect+totalNumberApprox;
//        printf("Interaction information: \n");
//        printf("rank %d: number of direct batch-cluster interactions: %d\n", rank, totalNumberApprox);
//        printf("rank %d: number of approx batch-cluster interactions: %d\n", rank, totalNumberDirect);
//        printf("rank %d:  total number of batch-cluster interactions: %d\n\n", rank, totalNumberInteractions);
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//
//    if (run_params -> verbosity > 0) {
//        MPI_Reduce(&totalNumberInteractions,&cumulativeNumberInteractions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//        MPI_Reduce(&totalNumberInteractions,&maxNumberInteractions, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
//        MPI_Reduce(&totalNumberInteractions,&minNumberInteractions, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
//        if (rank==0){
//            printf("Cumulative number of interactions across all ranks: %d\n", cumulativeNumberInteractions);
//            printf("   Maximum number of interactions across all ranks: %d\n", maxNumberInteractions);
//            printf("   Minimum number of interactions across all ranks: %d\n", minNumberInteractions);
//            printf("                                             Ratio: %f\n\n", (double)maxNumberInteractions/(double)minNumberInteractions );
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }

} /* END function treecode */
