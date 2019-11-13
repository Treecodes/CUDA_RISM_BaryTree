#ifndef H_CLUSTERS_H
#define H_CLUSTERS_H

#include "tnode.h"
#include "particles.h"

void Clusters_PC_Setup(struct particles *clusters, struct particles *sources, int order,
                       struct tnode_array *tree_array, char *approxName, char *singularityHandling);

#endif /* H_CLUSTERS_H */