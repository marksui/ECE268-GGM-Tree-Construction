/*
 * spongent/spongent_kernel.cu
 *
 * GPU kernel for the Spongent-128 GGM tree level expansion.
 *
 * This is the ONLY file in spongent/ that requires nvcc.
 * Everything else (spongent.c, spongent_prf.c) compiles with gcc.
 *
 * What lives here:
 *   - d_perm : GPU constant memory pLayer table (uploaded once via
 *              spongent128_upload_tables before any kernel launch)
 *   - spongent128_upload_tables() : host function to push table to GPU
 *   - spongent_expand_level : __global__ kernel, one thread per node
 *
 * Member C links this object into gpu/ggm_tree_gpu.cu.
 */

#include <stdint.h>
#include <stddef.h>
#include "spongent.cuh"
#include "spongent_prf.cuh"

#define SEED_BYTES    SPONGENT128_HASH_BYTES   /* 16 */
#define SPONGENT128_B 136

/* -----------------------------------------------------------------------
 * GPU constant memory tables
 * Uploading once avoids repeated recomputation on every thread.
 * -------------------------------------------------------------------- */
__device__ __constant__ uint8_t d_perm[SPONGENT128_B];
__device__ __constant__ uint8_t d_sbox[16];

void spongent128_upload_tables(void) {
    /* pLayer table: p(i) = (34 * i) mod 135, p(135) = 135 */
    uint8_t h_perm[SPONGENT128_B];
    for (int i = 0; i < SPONGENT128_B - 1; i++)
        h_perm[i] = (uint8_t)((i * 34) % (SPONGENT128_B - 1));
    h_perm[SPONGENT128_B - 1] = SPONGENT128_B - 1;
    cudaMemcpyToSymbol(d_perm, h_perm, sizeof(h_perm));

    /* PRESENT S-box */
    uint8_t h_sbox[16] = {0xE,0xD,0xB,0x0, 0x2,0x1,0x4,0xF,
                          0x7,0xA,0x8,0x5, 0x9,0xC,0x3,0x6};
    cudaMemcpyToSymbol(d_sbox, h_sbox, sizeof(h_sbox));
}

/* -----------------------------------------------------------------------
 * GPU-side bit helpers (identical to CPU versions in spongent.c)
 * -------------------------------------------------------------------- */
__device__ static inline
int gpu_get_bit(const uint8_t state[SPONGENT128_STATE_BYTES], int i) {
    return (state[i >> 3] >> (i & 7)) & 1;       /* LSB-first */
}
__device__ static inline
void gpu_set_bit(uint8_t state[SPONGENT128_STATE_BYTES], int i, int v) {
    int byte = i >> 3, bit = i & 7;               /* LSB-first */
    state[byte] = (uint8_t)((state[byte] & ~(1u << bit)) | ((v & 1u) << bit));
}

/* -----------------------------------------------------------------------
 * GPU-side permutation (uses constant-memory tables)
 * Mirrors spongent128_permute() in spongent.c but reads from d_perm/d_sbox.
 * -------------------------------------------------------------------- */
__device__
static void gpu_permute(uint8_t state[SPONGENT128_STATE_BYTES]) {
    uint8_t lc = SPONGENT128_LFSR_INIT;
    for (int r = 0; r < SPONGENT128_NR_ROUNDS; r++) {
        /* AddRoundConstant: lc into bits 0..6 of state[0]; reverse_byte(lc) into state[16] */
        uint8_t rlc = lc;
        rlc = (uint8_t)(((rlc & 0xF0u) >> 4) | ((rlc & 0x0Fu) << 4));
        rlc = (uint8_t)(((rlc & 0xCCu) >> 2) | ((rlc & 0x33u) << 2));
        rlc = (uint8_t)(((rlc & 0xAAu) >> 1) | ((rlc & 0x55u) << 1));
        state[0]  ^= lc;
        state[16] ^= rlc;

        /* sBoxLayer — use constant-memory S-box */
        for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) {
            uint8_t hi = (state[i] >> 4) & 0xF;
            uint8_t lo =  state[i]       & 0xF;
            state[i] = (uint8_t)((d_sbox[hi] << 4) | d_sbox[lo]);
        }

        /* pLayer — use constant-memory permutation table */
        uint8_t tmp[SPONGENT128_STATE_BYTES] = {0};
        for (int i = 0; i < SPONGENT128_B; i++)
            gpu_set_bit(tmp, d_perm[i], gpu_get_bit(state, i));
        for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = tmp[i];

        /* LFSR step: feedback = bit6 XOR bit5, shift-left */
        uint8_t fb = (uint8_t)(((lc >> 6) ^ (lc >> 5)) & 1u);
        lc = (uint8_t)(((lc << 1) & 0x7Fu) | fb);
    }
}

/* -----------------------------------------------------------------------
 * GPU-side hash (rate=1 byte, matches CPU spongent128_hash)
 * -------------------------------------------------------------------- */
__device__
static void gpu_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[SPONGENT128_HASH_BYTES])
{
    uint8_t state[SPONGENT128_STATE_BYTES] = {0};
    for (size_t i = 0; i < msg_len; i++) { state[0] ^= msg[i]; gpu_permute(state); }
    state[0] ^= 0x80;
    gpu_permute(state);
    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        digest[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1) gpu_permute(state);
    }
}

/* -----------------------------------------------------------------------
 * GPU-side expand (domain-separated, mirrors spongent128_expand in .c)
 * -------------------------------------------------------------------- */
__device__
static void gpu_expand(const uint8_t *seed,
                       uint8_t *out0, uint8_t *out1)
{
    uint8_t buf[1 + SEED_BYTES];
    for (int i = 0; i < SEED_BYTES; i++) buf[i + 1] = seed[i];

    buf[0] = 0x00;  gpu_hash(buf, 1 + SEED_BYTES, out0);
    buf[0] = 0x01;  gpu_hash(buf, 1 + SEED_BYTES, out1);
}

/* -----------------------------------------------------------------------
 * spongent_expand_level  (__global__)
 *
 * Grid: <<<(N + 255) / 256, 256>>>
 *   parents  : device ptr, N  × SEED_BYTES (level l)
 *   children : device ptr, 2N × SEED_BYTES (level l+1)
 *   N        : number of parent nodes = 2^l
 * -------------------------------------------------------------------- */
__global__
void spongent_expand_level(const uint8_t *parents,
                           uint8_t       *children,
                           size_t         N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const uint8_t *parent = parents  + i       * SEED_BYTES;
    uint8_t       *left   = children + (2*i)   * SEED_BYTES;
    uint8_t       *right  = children + (2*i+1) * SEED_BYTES;

    gpu_expand(parent, left, right);
}
