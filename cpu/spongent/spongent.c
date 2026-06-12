#include <string.h>
#include "spongent.cuh"

static const uint8_t SBOX[16] = {
    0xE, 0xD, 0xB, 0x0,
    0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5,
    0x9, 0xC, 0x3, 0x6
};

#define SPONGENT128_B 136

static uint8_t cpu_perm[SPONGENT128_B];
static int     cpu_perm_ready = 0;

static void init_cpu_perm(void) {
    for (int i = 0; i < SPONGENT128_B - 1; i++)
        cpu_perm[i] = (uint8_t)((i * 34) % (SPONGENT128_B - 1));
    cpu_perm[SPONGENT128_B - 1] = (uint8_t)(SPONGENT128_B - 1);
    cpu_perm_ready = 1;
}

__host__ __device__ static inline
int get_bit(const uint8_t state[SPONGENT128_STATE_BYTES], int i) {
    return (state[i >> 3] >> (i & 7)) & 1;
}

__host__ __device__ static inline
void set_bit(uint8_t state[SPONGENT128_STATE_BYTES], int i, int v) {
    int byte = i >> 3;
    int bit  = i & 7;
    state[byte] = (uint8_t)((state[byte] & ~(1u << bit)) | ((v & 1u) << bit));
}

__host__ __device__ static inline
uint8_t reverse_byte(uint8_t b) {
    b = (uint8_t)(((b & 0xF0u) >> 4) | ((b & 0x0Fu) << 4));
    b = (uint8_t)(((b & 0xCCu) >> 2) | ((b & 0x33u) << 2));
    b = (uint8_t)(((b & 0xAAu) >> 1) | ((b & 0x55u) << 1));
    return b;
}

__host__ __device__
uint8_t spongent128_lfsr_step(uint8_t s) {
    uint8_t fb = (uint8_t)(((s >> 6) ^ (s >> 5)) & 1u);
    return (uint8_t)(((s << 1) & 0x7Fu) | fb);
}

__host__ __device__
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc)
{
    state[0]  ^= lc;
    state[16] ^= reverse_byte(lc);
}

__host__ __device__
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]) {
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) {
        uint8_t hi = (state[i] >> 4) & 0xFu;
        uint8_t lo =  state[i]       & 0xFu;
        state[i] = (uint8_t)((SBOX[hi] << 4) | SBOX[lo]);
    }
}

__host__ __device__
void spongent128_player(uint8_t state[SPONGENT128_STATE_BYTES]) {
    uint8_t tmp[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) tmp[i] = 0;

    if (!cpu_perm_ready) init_cpu_perm();
    for (int i = 0; i < SPONGENT128_B; i++)
        set_bit(tmp, cpu_perm[i], get_bit(state, i));

    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = tmp[i];
}

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

__host__ __device__
void spongent128_hash(const uint8_t *msg, size_t msg_len,
                      uint8_t digest[SPONGENT128_HASH_BYTES])
{
    uint8_t state[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = 0;

    for (size_t i = 0; i < msg_len; i++) {
        state[0] ^= msg[i];
        spongent128_permute(state);
    }

    state[0] ^= 0x80u;
    spongent128_permute(state);

    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        digest[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1)
            spongent128_permute(state);
    }
}
