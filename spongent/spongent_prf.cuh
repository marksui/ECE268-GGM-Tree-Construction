#pragma once
/*
 * spongent/spongent_prf.cuh
 *
 * Wraps Spongent-128 into a length-doubling PRG for the GGM tree.
 *
 * Domain-separated expand:
 *   out0 = Spongent128_Hash( 0x00 || seed )   <- left  child
 *   out1 = Spongent128_Hash( 0x01 || seed )   <- right child
 *
 * Both spongent128_expand (CPU+GPU) and spongent_expand_level (GPU kernel)
 * live here. Member C calls spongent_expand_level from their GPU tree loop.
 *
 * Usage — CPU tree:
 *   #include "spongent/spongent_prf.cuh"
 *   ggm_tree_build(&tree, &SPONGENT128_PRF, root_seed, depth);
 *
 * Usage — GPU tree (Member C):
 *   spongent128_upload_tables();   // once, before any kernel launches
 *   spongent_expand_level<<<grid, block>>>(parents, children, N);
 */

#include <stdint.h>
#include <stddef.h>
#include "../common/prf_interface.h"

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__
#endif

/*
 * spongent128_expand  (__host__ __device__)
 *
 * Expand one 16-byte seed into two 16-byte children.
 * Callable from CPU code and from GPU device functions.
 */
__host__ __device__
void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len);

/*
 * spongent_expand_level  (__global__ kernel)
 *
 * One GPU thread per parent node.
 *   parents  : device pointer, N × 16 bytes  (all nodes at level l)
 *   children : device pointer, 2N × 16 bytes (all nodes at level l+1)
 *   N        : number of parent nodes (= 2^l)
 *
 * Launch: spongent_expand_level<<<(N+255)/256, 256>>>(parents, children, N)
 *
 * Called by Member C's gpu/ggm_tree_gpu.cu.
 */
#ifdef __CUDACC__
__global__
void spongent_expand_level(const uint8_t *parents,
                           uint8_t       *children,
                           size_t         N);
#endif

/*
 * Upload pLayer and S-box tables to GPU constant memory.
 * Call once from host before any kernel launch.
 */
#ifdef __CUDACC__
void spongent128_upload_tables(void);
#endif

/* CPU-side prf_t vtable for ggm_tree_build(). */
extern const prf_t SPONGENT128_PRF;
