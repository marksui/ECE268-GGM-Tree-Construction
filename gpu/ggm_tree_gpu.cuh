#pragma once

#include <stdint.h>
#include <stddef.h>
#include "../cpu/ggm_tree_cpu.h"

typedef struct {
    uint8_t *d_data;
    int      depth;
    size_t   seed_bytes;
} ggm_gpu_tree_t;

size_t ggm_gpu_tree_total_nodes(int depth);

uint8_t *ggm_gpu_tree_get_node(const ggm_gpu_tree_t *tree, int level, size_t index);

int ggm_gpu_tree_build_spongent(ggm_gpu_tree_t *tree,
                                const uint8_t  *root_seed,
                                int             depth);

int ggm_gpu_tree_build_spongent_tpb(ggm_gpu_tree_t *tree,
                                    const uint8_t  *root_seed,
                                    int             depth,
                                    int             threads_per_block);

int ggm_gpu_tree_build_keccak(ggm_gpu_tree_t *tree,
                              const uint8_t  *root_seed,
                              int             depth);

int ggm_gpu_tree_copy_to_host(const ggm_gpu_tree_t *tree,
                              uint8_t *out, size_t out_bytes);

void ggm_gpu_tree_free(ggm_gpu_tree_t *tree);
