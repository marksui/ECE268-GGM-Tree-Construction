/*
 * gpu/ggm_tree_gpu.cu
 *
 * Builds a complete GGM tree on the GPU, level by level.
 * Supports both Spongent-128 and Keccak-f1600 expand kernels.
 *
 * Memory layout matches common/ggm_tree.c (flat BFS):
 *   Node (level l, index i) is at flat index (2^l - 1 + i).
 */

#include <cuda_runtime.h>
#include "ggm_tree_gpu.cuh"
#include "../spongent/spongent.cuh"
#include "../spongent/spongent_prf.cuh"
#include "../keccak/keccak_f1600.cuh"
#include "../keccak/keccak_prf.cuh"

#define THREADS_PER_BLOCK 256

/* -----------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------- */
static inline size_t flat_index(int level, size_t i) {
    return ((size_t)1 << level) - 1 + i;
}

size_t ggm_gpu_tree_total_nodes(int depth) {
    if (depth < 0 || depth > GGM_MAX_DEPTH) return 0;
    return ((size_t)2 << depth) - 1;
}

uint8_t *ggm_gpu_tree_get_node(const ggm_gpu_tree_t *tree, int level, size_t index) {
    if (!tree || !tree->d_data) return NULL;
    if (level < 0 || level > tree->depth) return NULL;
    if (index >= ((size_t)1 << level)) return NULL;
    return tree->d_data + flat_index(level, index) * tree->seed_bytes;
}

static int cleanup_fail(ggm_gpu_tree_t *tree) {
    if (tree && tree->d_data) cudaFree(tree->d_data);
    if (tree) { tree->d_data = NULL; tree->depth = 0; tree->seed_bytes = 0; }
    return -1;
}

/* -----------------------------------------------------------------------
 * Shared internal builder — takes a kernel function pointer.
 * expand_kernel  : __global__ fn with signature (parents, children, N)
 * seed_bytes     : bytes per node (16 for Spongent, 32 for Keccak)
 * upload_tables  : host fn to push constant-memory tables; may be NULL
 * -------------------------------------------------------------------- */
typedef void (*expand_kernel_t)(const uint8_t *, uint8_t *, size_t);

static int ggm_gpu_tree_build_impl(ggm_gpu_tree_t  *tree,
                                   const uint8_t   *root_seed,
                                   int              depth,
                                   size_t           seed_bytes,
                                   expand_kernel_t  kernel,
                                   void           (*upload_tables)(void))
{
    if (!tree || !root_seed) return -1;
    if (depth < 0 || depth > GGM_MAX_DEPTH) return -1;

    tree->d_data    = NULL;
    tree->depth     = depth;
    tree->seed_bytes = seed_bytes;

    size_t nodes = ggm_gpu_tree_total_nodes(depth);
    size_t bytes = nodes * seed_bytes;

    cudaError_t err = cudaMalloc((void **)&tree->d_data, bytes);
    if (err != cudaSuccess) return cleanup_fail(tree);

    err = cudaMemcpy(tree->d_data, root_seed, seed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_fail(tree);

    if (upload_tables) {
        upload_tables();
        err = cudaGetLastError();
        if (err != cudaSuccess) return cleanup_fail(tree);
    }

    for (int level = 0; level < depth; level++) {
        size_t   N        = (size_t)1 << level;
        uint8_t *parents  = ggm_gpu_tree_get_node(tree, level,     0);
        uint8_t *children = ggm_gpu_tree_get_node(tree, level + 1, 0);
        int blocks = (int)((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        kernel<<<blocks, THREADS_PER_BLOCK>>>(parents, children, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) return cleanup_fail(tree);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return cleanup_fail(tree);
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * Public API — Spongent-128
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_build_spongent(ggm_gpu_tree_t *tree,
                                const uint8_t  *root_seed,
                                int             depth)
{
    return ggm_gpu_tree_build_impl(tree, root_seed, depth,
                                   SPONGENT128_HASH_BYTES,
                                   spongent_expand_level,
                                   spongent128_upload_tables);
}

/* -----------------------------------------------------------------------
 * Public API — Keccak-f1600
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_build_keccak(ggm_gpu_tree_t *tree,
                              const uint8_t  *root_seed,
                              int             depth)
{
    return ggm_gpu_tree_build_impl(tree, root_seed, depth,
                                   KECCAK1600_HASH_BYTES,
                                   keccak_expand_level,
                                   keccak_f1600_init_cuda);
}

/* -----------------------------------------------------------------------
 * Copy full tree from device to host buffer
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_copy_to_host(const ggm_gpu_tree_t *tree,
                              uint8_t *out, size_t out_bytes)
{
    if (!tree || !tree->d_data || !out) return -1;

    size_t bytes = ggm_gpu_tree_total_nodes(tree->depth) * tree->seed_bytes;
    if (out_bytes < bytes) return -1;

    cudaError_t err = cudaMemcpy(out, tree->d_data, bytes, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/* -----------------------------------------------------------------------
 * Free device memory
 * -------------------------------------------------------------------- */
void ggm_gpu_tree_free(ggm_gpu_tree_t *tree) {
    if (!tree) return;
    if (tree->d_data) cudaFree(tree->d_data);
    tree->d_data    = NULL;
    tree->depth     = 0;
    tree->seed_bytes = 0;
}
