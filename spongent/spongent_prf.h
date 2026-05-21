#pragma once
/*
 * spongent/spongent_prf.h
 *
 * Wraps Spongent-128 into a length-doubling PRG for use in the GGM tree.
 *
 * PRG construction:
 *   Given a 16-byte seed s, produce two 16-byte outputs (left, right):
 *
 *     out0 = Spongent128_Hash( 0x00 || s )   [left  child]
 *     out1 = Spongent128_Hash( 0x01 || s )   [right child]
 *
 *   Domain separation via the leading byte prevents related outputs.
 *
 * Usage with GGM tree:
 *   #include "spongent/spongent_prf.h"
 *   ggm_tree_build(&tree, &SPONGENT128_PRF, root_seed, depth);
 */

#include "../common/prf_interface.h"

/*
 * Low-level expand — can be called directly or via the vtable below.
 *
 *   seed     : 16-byte input seed
 *   seed_len : must be 16 (SPONGENT128_HASH_BYTES)
 *   out0     : 16-byte left  child output
 *   out1     : 16-byte right child output
 *   out_len  : must be 16
 */
void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len);

/*
 * SPONGENT128_PRF  —  ready-to-use prf_t for ggm_tree_build().
 *
 * Example:
 *   uint8_t root[16] = { ... };
 *   ggm_tree_t tree;
 *   ggm_tree_build(&tree, &SPONGENT128_PRF, root, 8);
 */
extern const prf_t SPONGENT128_PRF;
