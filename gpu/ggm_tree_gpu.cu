/*
 * gpu/ggm_tree_gpu.cu
 *
 * Builds a complete GGM tree on the GPU, level by level.
 * Supports both Spongent-128 and Keccak-f1600 expand kernels.
 *
 * Memory layout matches cpu/ggm_tree_cpu.c (flat BFS):
 *   Node (level l, index i) is at flat index (2^l - 1 + i).
 *
 * Optimisation (v2): single cudaDeviceSynchronize() after all levels.
 *   The original code called spongent_launch_expand_level() /
 *   keccak_launch_expand_level() which each ended with a
 *   cudaDeviceSynchronize(), meaning depth-1 unnecessary CPU–GPU
 *   round-trips blocked the host between levels.
 *
 *   CUDA guarantees that kernels submitted to the *same stream* execute
 *   in submission order, so level l+1's kernel cannot begin reading the
 *   children buffer until level l's kernel has finished writing it — no
 *   explicit per-level sync is needed.  We now:
 *     a) Launch all level kernels back-to-back without blocking.
 *     b) Call cudaDeviceSynchronize() exactly once after the loop.
 *
 *   For depth 12 this removes 11 redundant CPU–GPU round-trips; early
 *   levels (1–128 nodes) finish in microseconds, so the CPU was spending
 *   most of its inter-level time in sync overhead rather than doing useful
 *   work.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "ggm_tree_gpu.cuh"
#include "../cpu/spongent/spongent.cuh"
#include "../cpu/spongent/spongent_prf.cuh"
#include "../cpu/keccak/keccak_f1600.cuh"
#include "../cpu/keccak/keccak_prf.cuh"

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

static int cleanup_fail_msg(ggm_gpu_tree_t *tree, const char *where, cudaError_t err) {
    printf("CUDA error at %s: %s\n", where, cudaGetErrorString(err));
    return cleanup_fail(tree);
}

/* -----------------------------------------------------------------------
 * Shared internal setup: allocate device memory and copy root seed.
 * -------------------------------------------------------------------- */
static int setup_gpu_tree(ggm_gpu_tree_t *tree,
                          const uint8_t  *root_seed,
                          int             depth,
                          size_t          seed_bytes,
                          void          (*upload_tables)(void))
{
    if (!tree || !root_seed) return -1;
    if (depth < 0 || depth > GGM_MAX_DEPTH) return -1;

    tree->d_data     = NULL;
    tree->depth      = depth;
    tree->seed_bytes = seed_bytes;

    size_t nodes = ggm_gpu_tree_total_nodes(depth);
    size_t bytes = nodes * seed_bytes;

    cudaError_t err = cudaMalloc((void **)&tree->d_data, bytes);
    if (err != cudaSuccess) return cleanup_fail_msg(tree, "cudaMalloc", err);

    err = cudaMemcpy(tree->d_data, root_seed, seed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_fail_msg(tree, "copy root to device", err);

    if (upload_tables) {
        upload_tables();
        err = cudaGetLastError();
        if (err != cudaSuccess) return cleanup_fail_msg(tree, "upload constant tables", err);
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * Public API — Spongent-128
 *
 * Change from v1: kernels are launched without per-level sync.
 * A single cudaDeviceSynchronize() is issued after the complete level loop.
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_build_spongent(ggm_gpu_tree_t *tree,
                                const uint8_t  *root_seed,
                                int             depth)
{
    int rc = setup_gpu_tree(tree, root_seed, depth,
                            SPONGENT128_HASH_BYTES,
                            spongent128_upload_tables);
    if (rc != 0) return -1;

    for (int level = 0; level < depth; level++) {
        size_t   N        = (size_t)1 << level;
        uint8_t *parents  = ggm_gpu_tree_get_node(tree, level,     0);
        uint8_t *children = ggm_gpu_tree_get_node(tree, level + 1, 0);

        /*
         * Launch without blocking.  CUDA stream ordering guarantees
         * this kernel completes before the next level's kernel begins.
         */
        rc = spongent_launch_expand_level(parents, children, N,
                                          THREADS_PER_BLOCK);
        if (rc != 0) return cleanup_fail(tree);
    }

    /* Single synchronisation after all levels are dispatched. */
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return cleanup_fail_msg(tree, "spongent tree final sync", err);

    return 0;
}

/* -----------------------------------------------------------------------
 * Block-size variant — Spongent-128
 * Same as ggm_gpu_tree_build_spongent but accepts threads_per_block at
 * runtime. Used by the block-size sensitivity benchmark (Section 2).
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_build_spongent_tpb(ggm_gpu_tree_t *tree,
                                    const uint8_t  *root_seed,
                                    int             depth,
                                    int             threads_per_block)
{
    int rc = setup_gpu_tree(tree, root_seed, depth,
                            SPONGENT128_HASH_BYTES,
                            spongent128_upload_tables);
    if (rc != 0) return -1;

    for (int level = 0; level < depth; level++) {
        size_t   N        = (size_t)1 << level;
        uint8_t *parents  = ggm_gpu_tree_get_node(tree, level,     0);
        uint8_t *children = ggm_gpu_tree_get_node(tree, level + 1, 0);
        rc = spongent_launch_expand_level(parents, children, N,
                                         threads_per_block);
        if (rc != 0) return cleanup_fail(tree);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return cleanup_fail_msg(tree, "spongent tpb final sync", err);

    return 0;
}

/* -----------------------------------------------------------------------
 * Public API — Keccak-f1600
 * (same single-sync pattern applied for consistency)
 * -------------------------------------------------------------------- */
int ggm_gpu_tree_build_keccak(ggm_gpu_tree_t *tree,
                              const uint8_t  *root_seed,
                              int             depth)
{
    int rc = setup_gpu_tree(tree, root_seed, depth,
                            KECCAK1600_HASH_BYTES,
                            keccak_f1600_init_cuda);
    if (rc != 0) return -1;

    for (int level = 0; level < depth; level++) {
        size_t   N        = (size_t)1 << level;
        uint8_t *parents  = ggm_gpu_tree_get_node(tree, level,     0);
        uint8_t *children = ggm_gpu_tree_get_node(tree, level + 1, 0);

        rc = keccak_launch_expand_level(parents, children, N,
                                        THREADS_PER_BLOCK);
        if (rc != 0) return cleanup_fail(tree);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return cleanup_fail_msg(tree, "keccak tree final sync", err);

    return 0;
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
    if (err == cudaSuccess) return 0;
    printf("CUDA error at copy tree to host: %s\n", cudaGetErrorString(err));
    return -1;
}

/* -----------------------------------------------------------------------
 * Free device memory
 * -------------------------------------------------------------------- */
void ggm_gpu_tree_free(ggm_gpu_tree_t *tree) {
    if (!tree) return;
    if (tree->d_data) cudaFree(tree->d_data);
    tree->d_data     = NULL;
    tree->depth      = 0;
    tree->seed_bytes = 0;
}
