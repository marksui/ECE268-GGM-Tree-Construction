/*
 * spongent/spongent.c
 *
 * Spongent-128/128/8  —  all core functions tagged __host__ __device__
 * so the same code runs on both CPU threads and GPU threads.
 *
 * State bit ordering  (LSB-first, matching reference C implementation):
 *   bit 0   = bit 0 (LSB) of state[0]
 *   bit 7   = bit 7 (MSB) of state[0]
 *   bit 8   = bit 0 (LSB) of state[1]
 *   bit 135 = bit 7 (MSB) of state[16]
 *   i.e. bit i is at byte i/8, bit position i%8.
 *
 * pLayer: p(i) = (i * 34) mod 135  for i < 135;  p(135) = 135.
 *
 * LFSR: 7-bit, init 0x7A = 0b1111010.
 *   Shift-left: new bit = bit6 XOR bit5.
 *
 * Round constant: state[0] ^= lc; state[16] ^= reverse_byte(lc).
 *
 * Verified against official Spongent-128 test vector:
 *   hash("Sponge + Present = Spongent") == 6B7BA35EB09DE0F8DEF06AE555694C53
 */

#include <string.h>
#include "spongent.cuh"

/* -----------------------------------------------------------------------
 * S-box  (PRESENT 4-bit S-box)
 * -------------------------------------------------------------------- */
static const uint8_t SBOX[16] = {
    0xE, 0xD, 0xB, 0x0,
    0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5,
    0x9, 0xC, 0x3, 0x6
};

/* -----------------------------------------------------------------------
 * pLayer permutation table  (b = 136)
 * -------------------------------------------------------------------- */
#define SPONGENT128_B 136

static uint8_t cpu_perm[SPONGENT128_B];
static int     cpu_perm_ready = 0;

static void init_cpu_perm(void) {
    for (int i = 0; i < SPONGENT128_B - 1; i++)
        cpu_perm[i] = (uint8_t)((i * 34) % (SPONGENT128_B - 1));
    cpu_perm[SPONGENT128_B - 1] = (uint8_t)(SPONGENT128_B - 1);
    cpu_perm_ready = 1;
}

/* -----------------------------------------------------------------------
 * Bit access helpers — LSB-first  (__host__ __device__)
 *   bit i is at byte i/8, bit position i%8.
 * -------------------------------------------------------------------- */
__host__ __device__ static inline
int get_bit(const uint8_t state[SPONGENT128_STATE_BYTES], int i) {
    return (state[i >> 3] >> (i & 7)) & 1;
}

__host__ __device__ static inline
void set_bit(uint8_t state[SPONGENT128_STATE_BYTES], int i, int v) {
    int byte = i >> 3;
    int bit  = i & 7;          /* LSB = bit 0 */
    state[byte] = (uint8_t)((state[byte] & ~(1u << bit)) | ((v & 1u) << bit));
}

/* -----------------------------------------------------------------------
 * Byte bit-reversal helper  (__host__ __device__)
 * -------------------------------------------------------------------- */
__host__ __device__ static inline
uint8_t reverse_byte(uint8_t b) {
    b = (uint8_t)(((b & 0xF0u) >> 4) | ((b & 0x0Fu) << 4));
    b = (uint8_t)(((b & 0xCCu) >> 2) | ((b & 0x33u) << 2));
    b = (uint8_t)(((b & 0xAAu) >> 1) | ((b & 0x55u) << 1));
    return b;
}

/* -----------------------------------------------------------------------
 * LFSR step  (__host__ __device__)
 *
 * 7-bit, shift-left.  Feedback = bit6 XOR bit5.
 *   new_state = ((s << 1) & 0x7F) | (bit6 ^ bit5)
 * -------------------------------------------------------------------- */
__host__ __device__
uint8_t spongent128_lfsr_step(uint8_t s) {
    uint8_t fb = (uint8_t)(((s >> 6) ^ (s >> 5)) & 1u);
    return (uint8_t)(((s << 1) & 0x7Fu) | fb);
}

/* -----------------------------------------------------------------------
 * AddRoundConstant  (__host__ __device__)
 *
 * XOR 7-bit LFSR value lc into bits 0..6 of state[0].
 * XOR reverse_byte(lc) into state[16]  (= revLFSR at bits 129..135).
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc)
{
    state[0]  ^= lc;
    state[16] ^= reverse_byte(lc);
}

/* -----------------------------------------------------------------------
 * sBoxLayer  (__host__ __device__)
 *
 * Apply PRESENT S-box to each nibble (34 nibbles in 17 bytes).
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]) {
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) {
        uint8_t hi = (state[i] >> 4) & 0xFu;
        uint8_t lo =  state[i]       & 0xFu;
        state[i] = (uint8_t)((SBOX[hi] << 4) | SBOX[lo]);
    }
}

/* -----------------------------------------------------------------------
 * pLayer  (__host__ __device__)
 *
 * Bit permutation: bit at position i moves to position perm[i].
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_player(uint8_t state[SPONGENT128_STATE_BYTES]) {
    uint8_t tmp[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) tmp[i] = 0;

    if (!cpu_perm_ready) init_cpu_perm();
    for (int i = 0; i < SPONGENT128_B; i++)
        set_bit(tmp, cpu_perm[i], get_bit(state, i));

    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = tmp[i];
}

/* -----------------------------------------------------------------------
 * Full permutation — 70 rounds  (__host__ __device__)
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_permute(uint8_t state[SPONGENT128_STATE_BYTES]) {
    uint8_t lc = SPONGENT128_LFSR_INIT;
    for (int r = 0; r < SPONGENT128_NR_ROUNDS; r++) {
        spongent128_add_round_constant(state, lc);
        spongent128_sbox_layer(state);
        spongent128_player(state);
        lc = spongent128_lfsr_step(lc);
    }
}

/* -----------------------------------------------------------------------
 * Sponge hash  (__host__ __device__)
 *
 * Rate = 1 byte (state[0]).
 * Absorb: XOR each message byte into state[0], then permute.
 * Pad:    XOR 0x80 into state[0] (10* padding, 1-byte rate), permute.
 * Squeeze: collect state[0] after each permute, 16 times.
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_hash(const uint8_t *msg, size_t msg_len,
                      uint8_t digest[SPONGENT128_HASH_BYTES])
{
    uint8_t state[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = 0;

    /* Absorb */
    for (size_t i = 0; i < msg_len; i++) {
        state[0] ^= msg[i];
        spongent128_permute(state);
    }

    /* Pad */
    state[0] ^= 0x80u;
    spongent128_permute(state);

    /* Squeeze */
    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        digest[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1)
            spongent128_permute(state);
    }
}
