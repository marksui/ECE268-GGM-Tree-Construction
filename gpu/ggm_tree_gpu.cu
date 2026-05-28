/*
 * gpu/ggm_tree_gpu.cu
 *
 * Builds a complete GGM tree on the GPU, level by level.
 * For now this uses the Spongent expand kernel.
 */

#include <cuda_runtime.h>
#include "ggm_tree_gpu.cuh"
#include "../spongent/spongent.cuh"
#include "../spongent/spongent_prf.cuh"

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
    if (tree) {
        tree->d_data = NULL;
        tree->depth = 0;
        tree->seed_bytes = 0;
    }
    return -1;
}

int ggm_gpu_tree_build_spongent(ggm_gpu_tree_t *tree, const uint8_t *root_seed, int depth) {
    if (!tree || !root_seed) return -1;
    if (depth < 0 || depth > GGM_MAX_DEPTH) return -1;

    tree->d_data = NULL;
    tree->depth = depth;
    tree->seed_bytes = SPONGENT128_HASH_BYTES;

    size_t nodes = ggm_gpu_tree_total_nodes(depth);
    size_t bytes = nodes * tree->seed_bytes;

    cudaError_t err = cudaMalloc((void **)&tree->d_data, bytes);
    if (err != cudaSuccess) return cleanup_fail(tree);

    err = cudaMemcpy(tree->d_data, root_seed, tree->seed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_fail(tree);

    spongent128_upload_tables();
    err = cudaGetLastError();
    if (err != cudaSuccess) return cleanup_fail(tree);

    for (int level = 0; level < depth; level++) {
        size_t parents_count = (size_t)1 << level;
        uint8_t *parents = ggm_gpu_tree_get_node(tree, level, 0);
        uint8_t *children = ggm_gpu_tree_get_node(tree, level + 1, 0);
        int blocks = (int)((parents_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        spongent_expand_level<<<blocks, THREADS_PER_BLOCK>>>(parents, children, parents_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) return cleanup_fail(tree);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return cleanup_fail(tree);
    }

    return 0;
}

int ggm_gpu_tree_copy_to_host(const ggm_gpu_tree_t *tree, uint8_t *out, size_t out_bytes) {
    if (!tree || !tree->d_data || !out) return -1;

    size_t bytes = ggm_gpu_tree_total_nodes(tree->depth) * tree->seed_bytes;
    if (out_bytes < bytes) return -1;

    cudaError_t err = cudaMemcpy(out, tree->d_data, bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        return 0;
    } else {
        return -1;
    }
}

void ggm_gpu_tree_free(ggm_gpu_tree_t *tree) {
    if (!tree) return;
    if (tree->d_data) cudaFree(tree->d_data);
    tree->d_data = NULL;
    tree->depth = 0;
    tree->seed_bytes = 0;
}
