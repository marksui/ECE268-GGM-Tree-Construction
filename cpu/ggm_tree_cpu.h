#pragma once

#include <stdint.h>
#include <stddef.h>
#include "../common/prf_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGM_MAX_DEPTH 20

typedef struct {
    uint8_t *data;
    int depth;
    size_t seed_bytes;
    const prf_t *prf;
} ggm_tree_t;

int ggm_tree_build(ggm_tree_t *tree, const prf_t *prf, const uint8_t *root_seed, int depth);

void ggm_tree_free(ggm_tree_t *tree);

const uint8_t *ggm_tree_get_node(const ggm_tree_t *tree, int level, size_t index);

const uint8_t *ggm_tree_get_leaves(const ggm_tree_t *tree);

size_t ggm_tree_num_leaves(const ggm_tree_t *tree);

void ggm_tree_print(const ggm_tree_t *tree, int max_depth_to_print);

#ifdef __cplusplus
}
#endif
