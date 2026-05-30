#pragma once
/*
 * gpu/ggm_tree_gpu.cuh
 *
 * GPU GGM tree builder — flat BFS layout, level-by-level expansion.
 * Layout matches common/ggm_tree.c: node (level l, index i) is at
 * flat index (2^l - 1 + i).
 *
 * Two PRF backends are available:
 *   ggm_gpu_tree_build_spongent — uses Spongent-128 (16-byte seeds)
 *   ggm_gpu_tree_build_keccak  — uses Keccak-f1600  (32-byte seeds)
 */

#include <stdint.h>
#include <stddef.h>
#include "../common/ggm_tree.h"

typedef struct {
    uint8_t *d_data;    /* device memory, flat BFS tree */
    int      depth;
    size_t   seed_bytes;
} ggm_gpu_tree_t;

/* Total node count = 2^(depth+1) - 1 */
size_t ggm_gpu_tree_total_nodes(int depth);

/* Pointer to node on device (returns NULL on bad inputs) */
uint8_t *ggm_gpu_tree_get_node(const ggm_gpu_tree_t *tree, int level, size_t index);

/* Build tree using Spongent-128 kernel (16-byte seeds) */
int ggm_gpu_tree_build_spongent(ggm_gpu_tree_t *tree,
                                const uint8_t  *root_seed,
                                int             depth);

/* Build tree using Keccak-f1600 kernel (32-byte seeds) */
int ggm_gpu_tree_build_keccak(ggm_gpu_tree_t *tree,
                              const uint8_t  *root_seed,
                              int             depth);

/* Copy full device tree to host buffer (out_bytes must be >= total size) */
int ggm_gpu_tree_copy_to_host(const ggm_gpu_tree_t *tree,
                              uint8_t *out, size_t out_bytes);

/* Free device memory */
void ggm_gpu_tree_free(ggm_gpu_tree_t *tree);
