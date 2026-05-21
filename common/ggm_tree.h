#pragma once
/*
 * common/ggm_tree.h
 *
 * GGM binary PRF tree — CPU implementation.
 * Member C will build a GPU counterpart (gpu/ggm_tree_gpu.cu)
 * that calls Member A/B's __global__ expand kernels.
 *
 * Memory layout (flat BFS):
 *   Node (level l, index i) lives at flat index (2^l - 1 + i).
 *   All nodes at level l are contiguous, left-to-right.
 *
 *              [0]            <- level 0 (root)
 *           [1]    [2]        <- level 1
 *        [3][4]  [5][6]       <- level 2  (leaves when depth=2)
 *
 * Max depth: 20  (~2M leaves, 32 MB at 128-bit seeds)
 */

#include <stdint.h>
#include <stddef.h>
#include "prf_interface.h"

#define GGM_MAX_DEPTH 20

typedef struct {
    uint8_t      *data;        /* flat BFS allocation                  */
    int           depth;       /* leaf level                           */
    size_t        seed_bytes;  /* bytes per node                       */
    const prf_t  *prf;         /* PRF used to build this tree          */
} ggm_tree_t;

/* Build a complete tree. Returns 0 on success, -1 on error.
 * Caller must call ggm_tree_free() when done. */
int ggm_tree_build(ggm_tree_t    *tree,
                   const prf_t   *prf,
                   const uint8_t *root_seed,
                   int            depth);

void           ggm_tree_free(ggm_tree_t *tree);

/* Pointer to node (level, index). Returns NULL on bad inputs. */
const uint8_t *ggm_tree_get_node(const ggm_tree_t *tree, int level, size_t index);

/* Pointer to first leaf (level == depth, index 0). Leaves are contiguous. */
const uint8_t *ggm_tree_get_leaves(const ggm_tree_t *tree);

/* Number of leaves = 2^depth. */
size_t         ggm_tree_num_leaves(const ggm_tree_t *tree);

/* Print tree to stdout (small depths only). */
void           ggm_tree_print(const ggm_tree_t *tree, int max_depth_to_print);
