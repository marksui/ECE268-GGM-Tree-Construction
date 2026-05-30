#pragma once
/*
 * keccak/keccak_prf.cuh
 *
 * Keccak-f1600 length-doubling PRG for the GGM tree.
 *
 * CPU usage:
 *   #include "keccak/keccak_prf.cuh"
 *   ggm_tree_build(&tree, &KECCAK1600_PRF, root_seed, depth);
 *
 * GPU usage (Member C):
 *   keccak_f1600_init_cuda();           // upload tables once
 *   keccak_expand_level<<<g,b>>>(parents, children, N);
 */

#include <stdint.h>
#include <stddef.h>
#include "../common/prf_interface.h"

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__
#endif

/* -----------------------------------------------------------------------
 * keccak1600_expand  (__host__ __device__)
 *
 * Expand one 32-byte seed into two 32-byte children.
 * Callable from CPU code and GPU device functions.
 * -------------------------------------------------------------------- */
__host__ __device__
void keccak1600_expand(const uint8_t *seed, size_t seed_len,
                       uint8_t *out0, uint8_t *out1, size_t out_len);

/* -----------------------------------------------------------------------
 * keccak_expand_level  (__global__ kernel)
 *
 * One GPU thread per parent node.
 *   parents  : device pointer, N × 32 bytes (all nodes at level l)
 *   children : device pointer, 2N × 32 bytes (all nodes at level l+1)
 *   N        : number of parent nodes (= 2^l)
 *
 * Requires keccak_f1600_init_cuda() called once before first launch.
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
__global__
void keccak_expand_level(const uint8_t *parents,
                         uint8_t       *children,
                         size_t         N);
#endif

/* CPU-side prf_t vtable for ggm_tree_build(). */
extern const prf_t KECCAK1600_PRF;
