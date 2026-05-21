#pragma once
/*
 * spongent/spongent.h
 *
 * Spongent-128 sponge-based hash / permutation.
 *
 * Variant: Spongent-128/128/8
 *   State size  b  = 136 bits = 17 bytes
 *   Rate        r  =   8 bits =  1 byte   (state[0])
 *   Capacity    c  = 128 bits = 16 bytes
 *   Hash output    = 128 bits = 16 bytes
 *   Rounds         = 70
 *
 * References:
 *   Bogdanov et al., "SPONGENT: A Lightweight Hash Function", CHES 2011.
 *   https://link.springer.com/chapter/10.1007/978-3-642-23951-9_21
 *
 * NOTE: verify output against official test vectors before use.
 *       Test vectors are in tests/test_spongent.c.
 */

#include <stdint.h>
#include <stddef.h>

/* -----------------------------------------------------------------------
 * Spongent-128 parameters
 * -------------------------------------------------------------------- */
#define SPONGENT128_STATE_BYTES  17   /* b = 136 bits    */
#define SPONGENT128_RATE_BYTES    1   /* r =   8 bits    */
#define SPONGENT128_CAP_BYTES    16   /* c = 128 bits    */
#define SPONGENT128_HASH_BYTES   16   /* output 128 bits */
#define SPONGENT128_NR_ROUNDS    70
#define SPONGENT128_LFSR_BITS     7   /* ceil(log2(70+1)) */
#define SPONGENT128_LFSR_INIT  0x7E  /* 0b1111110        */

/* -----------------------------------------------------------------------
 * Core permutation
 *
 * Applies 70 rounds of: AddRoundConstant → sBoxLayer → pLayer
 * in-place on the 17-byte state.
 * -------------------------------------------------------------------- */
void spongent128_permute(uint8_t state[SPONGENT128_STATE_BYTES]);

/* -----------------------------------------------------------------------
 * Sponge hash
 *
 * Absorbs `msg_len` bytes, then squeezes 16 bytes into `digest`.
 * Returns 0 on success.
 * -------------------------------------------------------------------- */
int spongent128_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[SPONGENT128_HASH_BYTES]);

/* -----------------------------------------------------------------------
 * Low-level building blocks (exposed for testing / GPU porting)
 * -------------------------------------------------------------------- */

/* Apply 4-bit PRESENT S-box to all 34 nibbles of state. */
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]);

/* Apply the bit-permutation pLayer to state. */
void spongent128_player(uint8_t state[SPONGENT128_STATE_BYTES]);

/*
 * XOR the round constant into state.
 *   lc  = current 7-bit LFSR counter value.
 * lc is XORed (MSB-first) into leftmost 7 state bits,
 * bitrev(lc) is XORed into rightmost 7 state bits.
 */
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc);

/* Advance the 7-bit LFSR one step. Polynomial: x^7 + x + 1. */
uint8_t spongent128_lfsr_step(uint8_t state);
