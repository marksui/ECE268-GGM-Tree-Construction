#pragma once
/*
 * spongent/spongent.cuh
 *
 * Spongent-128/128/8  —  CUDA-compatible header.
 *
 * Variant: Spongent-128/128/8
 *   State size  b  = 136 bits = 17 bytes
 *   Rate        r  =   8 bits =  1 byte   (state[0])
 *   Capacity    c  = 128 bits = 16 bytes
 *   Hash output    = 128 bits = 16 bytes
 *   Rounds         = 70
 *
 * All core functions are tagged __host__ __device__ so they compile
 * for both CPU (tests, CPU tree) and GPU (tree expansion kernel).
 *
 * References:
 *   Bogdanov et al., "SPONGENT: A Lightweight Hash Function", CHES 2011.
 *   https://link.springer.com/chapter/10.1007/978-3-642-23951-9_21
 */

#include <stdint.h>
#include <stddef.h>

/*
 * CUDA annotation shim.
 * When compiled by a plain C/C++ compiler (e.g. for unit tests without nvcc),
 * __host__ and __device__ expand to nothing so the same header works everywhere.
 */
#ifndef __CUDACC__
  #define __host__
  #define __device__
#endif

/* -----------------------------------------------------------------------
 * Parameters
 * -------------------------------------------------------------------- */
#define SPONGENT128_STATE_BYTES  17
#define SPONGENT128_RATE_BYTES    1
#define SPONGENT128_CAP_BYTES    16
#define SPONGENT128_HASH_BYTES   16
#define SPONGENT128_NR_ROUNDS    70
#define SPONGENT128_LFSR_INIT  0x7A   /* 0b1111010 — official Spongent-128 value */

/* -----------------------------------------------------------------------
 * Building blocks  (__host__ __device__ — run on CPU or GPU thread)
 * -------------------------------------------------------------------- */

/* Advance the 7-bit LFSR one step. Poly: x^7 + x + 1. */
__host__ __device__
uint8_t spongent128_lfsr_step(uint8_t state);

/* XOR round constant derived from LFSR value lc into state. */
__host__ __device__
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc);

/* Apply the 4-bit PRESENT S-box to all 34 nibbles. */
__host__ __device__
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]);

/* Apply the bit permutation p(i) = (34i) mod 135, p(135)=135. */
__host__ __device__
void spongent128_player(uint8_t state[SPONGENT128_STATE_BYTES]);

/* Full 70-round permutation. */
__host__ __device__
void spongent128_permute(uint8_t state[SPONGENT128_STATE_BYTES]);

/* -----------------------------------------------------------------------
 * Sponge hash  (__host__ __device__)
 *
 * Absorbs msg_len bytes, squeezes 16 bytes into digest.
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_hash(const uint8_t *msg, size_t msg_len,
                      uint8_t digest[SPONGENT128_HASH_BYTES]);
