#ifndef H_STRUCT_TREE_LINKED_LIST_NODE_H
#define H_STRUCT_TREE_LINKED_LIST_NODE_H

struct TreeLinkedListNode
{
    int numpar, ibeg, iend;
    
    double x_min, y_min, z_min;
    double x_max, y_max, z_max;
    double x_mid, y_mid, z_mid;
    
    double radius, aspect;
    
    int num_children;
    struct TreeLinkedListNode *child[8];
    struct TreeLinkedListNode *parent;

    int node_index;

    int x_dim, y_dim, z_dim;
    int x_low_ind, y_low_ind, z_low_ind;
    int x_high_ind, y_high_ind, z_high_ind;

    int level;
};

#endif /* H_STRUCT_TREE_LINKED_LIST_NODE_H */
