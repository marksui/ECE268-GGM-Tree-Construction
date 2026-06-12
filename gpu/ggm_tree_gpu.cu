#include <cuda_runtime.h>
#include <stdio.h>
#include "ggm_tree_gpu.cuh"
#include "../cpu/spongent/spongent.cuh"
#include "../cpu/spongent/spongent_prf.cuh"
#include "../cpu/keccak/keccak_f1600.cuh"
#include "../cpu/keccak/keccak_prf.cuh"

#define THREADS_PER_BLOCK 256

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

        rc = spongent_launch_expand_level(parents, children, N,
                                          THREADS_PER_BLOCK);
        if (rc != 0) return cleanup_fail(tree);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return cleanup_fail_msg(tree, "spongent tree final sync", err);

    return 0;
}

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

void ggm_gpu_tree_free(ggm_gpu_tree_t *tree) {
    if (!tree) return;
    if (tree->d_data) cudaFree(tree->d_data);
    tree->d_data     = NULL;
    tree->depth      = 0;
    tree->seed_bytes = 0;
}
