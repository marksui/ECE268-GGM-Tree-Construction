#pragma once
/*
 * common/prf_interface.h
 *
 * Abstract PRF/PRG interface for the GGM tree construction.
 *
 * Both Keccak-f1600 (Member A) and Spongent (Member B) plug in here.
 * Member C's GPU tree loop also consumes this interface.
 *
 * GGM construction:
 *   G(seed) -> (left_child, right_child)
 *   where left  = PRF(0x00 || seed)
 *         right = PRF(0x01 || seed)
 *   Applied level-by-level to build a complete binary tree.
 */

#include <stdint.h>
#include <stddef.h>

/*
 * prf_expand_fn  —  CPU-side function pointer type.
 *
 * Expands one seed into two children of equal length.
 * Both out0 and out1 must be caller-allocated with out_len bytes.
 */
typedef void (*prf_expand_fn)(
    const uint8_t *seed,
    size_t         seed_len,
    uint8_t       *out0,
    uint8_t       *out1,
    size_t         out_len
);

/*
 * prf_t  —  vtable for a CPU PRF implementation.
 * Pass to ggm_tree_build() to select which PRF to use.
 */
typedef struct {
    const char    *name;        /* e.g. "Spongent-128"       */
    size_t         seed_bytes;  /* 16 for 128-bit security   */
    prf_expand_fn  expand;      /* CPU expand function       */
} prf_t;
