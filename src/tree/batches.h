#ifndef H_BATCHFUNCTIONS_H
#define H_BATCHFUNCTIONS_H

#include "../particles/struct_particles.h"
#include "struct_nodes.h"

void Batches_Alloc(struct tnode_array **batches, double *batch_lim,
                   struct particles *particles, int batch_size);

void Batches_AllocArray(struct tnode_array **batches, int length);

void Batches_ReallocArray(struct tnode_array *batches, int length);

void Batches_Free(struct tnode_array *batches);

void Batches_Free_Win(struct tnode_array *batches);

void Batches_CreateTargetBatches(struct tnode_array *batches, struct particles *particles,
                   int ibeg, int iend, int maxparnode, double *xyzmm);

void Batches_CreateSourceBatches(struct tnode_array *batches, struct particles *particles,
                   int ibeg, int iend, int maxparnode, double *xyzmm);

#endif