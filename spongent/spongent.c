/*
 * spongent/spongent.cu
 *
 * Spongent-128/128/8  —  all core functions tagged __host__ __device__
 * so the same code runs on both CPU threads and GPU threads.
 *
 * State bit ordering:
 *   bit 0   = MSB of state[0]
 *   bit 135 = LSB of state[16]
 *   i.e. bit i is at byte i/8, position (7 - i%8).
 *
 * pLayer: p(i) = (i * 34) mod 135  for i < 135;  p(135) = 135.
 *   Precomputed into a __device__ constant table on GPU,
 *   and a static table on CPU (initialized on first call).
 *
 * LFSR: 7-bit, poly x^7 + x + 1, init 0x7E = 0b1111110.
 *   Shift-left Fibonacci: new_bit = MSB XOR bit1.
 *
 * NOTE: verify hash output against official Spongent-128 test vectors
 *       before submitting. Test stubs are in tests/test_spongent.cu.
 */

#include <string.h>
#include "spongent.cuh"

/* -----------------------------------------------------------------------
 * S-box  (PRESENT 4-bit S-box)
 * Stored in __device__ constant memory for fast GPU access;
 * also usable as a plain array on CPU.
 * -------------------------------------------------------------------- */
static const uint8_t SBOX[16] = {
    0xE, 0xD, 0xB, 0x0,
    0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5,
    0x9, 0xC, 0x3, 0x6
};

/* -----------------------------------------------------------------------
 * pLayer permutation table  (b = 136)
 *   GPU: __device__ __constant__ — loaded once, lives in constant cache.
 *   CPU: static array, populated on first call.
 * -------------------------------------------------------------------- */
#define SPONGENT128_B 136

/* CPU-side table (lazily initialized). */
static uint8_t cpu_perm[SPONGENT128_B];
static int     cpu_perm_ready = 0;

static void init_cpu_perm(void) {
    for (int i = 0; i < SPONGENT128_B - 1; i++)
        cpu_perm[i] = (uint8_t)((i * 34) % (SPONGENT128_B - 1));
    cpu_perm[SPONGENT128_B - 1] = (uint8_t)(SPONGENT128_B - 1);
    cpu_perm_ready = 1;
}

/* -----------------------------------------------------------------------
 * Bit access helpers  (__host__ __device__)
 * -------------------------------------------------------------------- */
__host__ __device__ static inline
int get_bit(const uint8_t state[SPONGENT128_STATE_BYTES], int i) {
    return (state[i >> 3] >> (7 - (i & 7))) & 1;
}

__host__ __device__ static inline
void set_bit(uint8_t state[SPONGENT128_STATE_BYTES], int i, int v) {
    int byte = i >> 3;
    int bit  = 7 - (i & 7);
    state[byte] = (uint8_t)((state[byte] & ~(1 << bit)) | (v << bit));
}

/* -----------------------------------------------------------------------
 * LFSR step  (__host__ __device__)
 *
 * 7-bit, poly x^7 + x + 1 (Fibonacci shift-left).
 *   feedback = bit6 XOR bit0  (MSB XOR LSB)
 *   new_state = ((s << 1) & 0x7F) | feedback
 * -------------------------------------------------------------------- */
__host__ __device__
uint8_t spongent128_lfsr_step(uint8_t s) {
    uint8_t fb = (uint8_t)(((s >> 6) ^ s) & 1);
    return (uint8_t)(((s << 1) & 0x7F) | fb);
}

/* -----------------------------------------------------------------------
 * AddRoundConstant  (__host__ __device__)
 *
 * lc (7 bits) is XORed MSB-first into state bits 0..6:
 *   state[0] ^= lc << 1
 * bitrev7(lc) is XORed into state bits 129..135:
 *   state[16] ^= lc & 0x7F
 * (derivation: see Spongent paper §3.1)
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc)
{
    state[0]  ^= (uint8_t)(lc << 1);
    state[16] ^= (uint8_t)(lc & 0x7F);
}

/* -----------------------------------------------------------------------
 * sBoxLayer  (__host__ __device__)
 *
 * Apply PRESENT S-box to each nibble (34 nibbles in 17 bytes).
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]) {
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) {
        uint8_t hi = (state[i] >> 4) & 0xF;
        uint8_t lo =  state[i]       & 0xF;
        state[i] = (uint8_t)((SBOX[hi] << 4) | SBOX[lo]);
    }
}

/* -----------------------------------------------------------------------
 * pLayer  (__host__ __device__)
 *
 * Bit permutation: bit at position i moves to position perm[i].
 * On GPU: reads from d_perm (constant memory).
 * On CPU: reads from cpu_perm (static array).
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
 * Pad:    XOR 0x80 into state[0] (10*1 padding for 1-byte rate), permute.
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
    state[0] ^= 0x80;
    spongent128_permute(state);

    /* Squeeze */
    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        digest[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1)
            spongent128_permute(state);
    }
}
