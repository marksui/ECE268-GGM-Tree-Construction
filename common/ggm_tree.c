/*
 * common/ggm_tree.c
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ggm_tree.h"

static inline size_t flat_index(int level, size_t i) {
    return ((size_t)1 << level) - 1 + i;
}

static inline size_t total_nodes(int depth) {
    return ((size_t)2 << depth) - 1;   /* 2^(depth+1) - 1 */
}

int ggm_tree_build(ggm_tree_t    *tree,
                   const prf_t   *prf,
                   const uint8_t *root_seed,
                   int            depth)
{
    if (!tree || !prf || !root_seed)        return -1;
    if (depth < 0 || depth > GGM_MAX_DEPTH) return -1;

    size_t n          = total_nodes(depth);
    size_t seed_bytes = prf->seed_bytes;

    tree->data = (uint8_t *)calloc(n, seed_bytes);
    if (!tree->data) return -1;

    tree->depth      = depth;
    tree->seed_bytes = seed_bytes;
    tree->prf        = prf;

    memcpy(tree->data, root_seed, seed_bytes);

    for (int l = 0; l < depth; l++) {
        size_t count = (size_t)1 << l;
        for (size_t i = 0; i < count; i++) {
            uint8_t *parent = tree->data + flat_index(l,   i    ) * seed_bytes;
            uint8_t *left   = tree->data + flat_index(l+1, 2*i  ) * seed_bytes;
            uint8_t *right  = tree->data + flat_index(l+1, 2*i+1) * seed_bytes;
            prf->expand(parent, seed_bytes, left, right, seed_bytes);
        }
    }
    return 0;
}

void ggm_tree_free(ggm_tree_t *tree) {
    if (!tree) return;
    free(tree->data);
    tree->data = NULL;
}

const uint8_t *ggm_tree_get_node(const ggm_tree_t *tree, int level, size_t index) {
    if (!tree || !tree->data)               return NULL;
    if (level < 0 || level > tree->depth)  return NULL;
    if (index >= ((size_t)1 << level))     return NULL;
    return tree->data + flat_index(level, index) * tree->seed_bytes;
}

const uint8_t *ggm_tree_get_leaves(const ggm_tree_t *tree) {
    return ggm_tree_get_node(tree, tree->depth, 0);
}

size_t ggm_tree_num_leaves(const ggm_tree_t *tree) {
    if (!tree) return 0;
    return (size_t)1 << tree->depth;
}

void ggm_tree_print(const ggm_tree_t *tree, int max_depth_to_print) {
    if (!tree) return;
    int limit = (max_depth_to_print < tree->depth) ? max_depth_to_print : tree->depth;
    printf("GGM Tree [prf=%s, depth=%d, seed_bytes=%llu]\n",
           tree->prf->name, tree->depth, (unsigned long long)tree->seed_bytes);
    for (int l = 0; l <= limit; l++) {
        size_t count = (size_t)1 << l;
        printf("  Level %d (%llu node%s):\n", l, (unsigned long long)count, count == 1 ? "" : "s");
        for (size_t i = 0; i < count; i++) {
            const uint8_t *node = ggm_tree_get_node(tree, l, i);
            printf("    [%llu] ", (unsigned long long)i);
            for (size_t b = 0; b < tree->seed_bytes; b++) printf("%02x", node[b]);
            printf("\n");
        }
    }
}
