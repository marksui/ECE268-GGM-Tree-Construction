/*
 * keccak/keccak_f1600.cu
 *
 * Keccak-f[1600] permutation + SHA3-256 sponge wrapper.
 * All core functions tagged __host__ __device__ so the same code
 * runs on CPU threads and GPU threads.
 *
 * OPTIMISED: Rho+Pi step replaced with 25 literal assignments so that
 * tmp[25] uses only compile-time-constant indices → NVCC keeps tmp[]
 * in 25 registers instead of local memory (off-chip DRAM).
 * Same fix as the Spongent pLayer optimisation.
 *
 * Verified against NIST FIPS 202 KAT:
 *   SHA3-256("") == a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
 */

#include <string.h>
#include "keccak_f1600.cuh"

/* -----------------------------------------------------------------------
 * GPU constant memory — Round Constants only.
 * RHO and PI tables removed: all indices now hardcoded as literals.
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
__device__ __constant__ uint64_t d_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};
#endif

/* CPU-side round constants */
static const uint64_t h_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

/* Route RC to correct memory space at compile time */
#if defined(__CUDA_ARCH__)
  #define RC d_RC
#else
  #define RC h_RC
#endif

/* Rotate left — avoids UB when y == 0 */
#define ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

/* -----------------------------------------------------------------------
 * keccakf1600_permute  (__host__ __device__)
 *
 * 24 rounds of Theta / Rho+Pi / Chi / Iota.
 *
 * Rho+Pi: 25 literal assignments — no loop, no dynamic indexing.
 *   tmp[LITERAL] = ROTL64(state[LITERAL], LITERAL_SHIFT)
 *   All indices are compile-time constants → tmp[] stays in 25 registers.
 *   Zero local-memory (DRAM) traffic.
 *
 * Outer 24-round loop: NOT unrolled (#pragma unroll 1).
 *   24 copies of the ~200-instruction round body would overflow L1 I-cache.
 * -------------------------------------------------------------------- */
__host__ __device__
void keccakf1600_permute(uint64_t state[25])
{
    uint64_t C[5], D[5];

    #pragma unroll 1
    for (int r = 0; r < KECCAK1600_NR_ROUNDS; r++) {

        /* -- Theta ---------------------------------------------------- */
        #pragma unroll
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++)
            D[i] = C[(i+4)%5] ^ ROTL64(C[(i+1)%5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        /* -- Rho + Pi: literal indices → tmp[] in registers ----------- *
         * tmp[PI[i]] = ROTL64(state[i], RHO[i])  for i = 0..24        */
        uint64_t tmp[25];
        tmp[ 0] = ROTL64(state[ 0],  0);
        tmp[10] = ROTL64(state[ 1],  1);
        tmp[20] = ROTL64(state[ 2], 62);
        tmp[ 5] = ROTL64(state[ 3], 28);
        tmp[15] = ROTL64(state[ 4], 27);
        tmp[16] = ROTL64(state[ 5], 36);
        tmp[ 1] = ROTL64(state[ 6], 44);
        tmp[11] = ROTL64(state[ 7],  6);
        tmp[21] = ROTL64(state[ 8], 55);
        tmp[ 6] = ROTL64(state[ 9], 20);
        tmp[ 7] = ROTL64(state[10],  3);
        tmp[17] = ROTL64(state[11], 10);
        tmp[ 2] = ROTL64(state[12], 43);
        tmp[12] = ROTL64(state[13], 25);
        tmp[22] = ROTL64(state[14], 39);
        tmp[23] = ROTL64(state[15], 41);
        tmp[ 8] = ROTL64(state[16], 45);
        tmp[18] = ROTL64(state[17], 15);
        tmp[ 3] = ROTL64(state[18], 21);
        tmp[13] = ROTL64(state[19],  8);
        tmp[14] = ROTL64(state[20], 18);
        tmp[24] = ROTL64(state[21],  2);
        tmp[ 9] = ROTL64(state[22], 61);
        tmp[19] = ROTL64(state[23], 56);
        tmp[ 4] = ROTL64(state[24], 14);

        /* -- Chi: all indices compile-time constant after unroll ------ */
        #pragma unroll
        for (int j = 0; j < 25; j += 5)
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j+i] = tmp[j+i] ^ ((~tmp[j+(i+1)%5]) & tmp[j+(i+2)%5]);

        /* -- Iota ----------------------------------------------------- */
        state[0] ^= RC[r];
    }
}

/* -----------------------------------------------------------------------
 * keccak1600_hash  (__host__ __device__)
 *
 * SHA3-256: rate=136 bytes, output=32 bytes.
 * Handles single-block messages (msg_len < KECCAK1600_RATE_BYTES).
 * -------------------------------------------------------------------- */
__host__ __device__
void keccak1600_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[KECCAK1600_HASH_BYTES])
{
    uint64_t state[25];
    uint8_t *st = (uint8_t *)state;

    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    for (size_t i = 0; i < msg_len; i++) st[i] ^= msg[i];

    st[msg_len]                   ^= 0x06;
    st[KECCAK1600_RATE_BYTES - 1] ^= 0x80;

    keccakf1600_permute(state);

    for (int i = 0; i < KECCAK1600_HASH_BYTES; i++) digest[i] = st[i];
}

/* -----------------------------------------------------------------------
 * keccak_f1600_init_cuda  (host only)
 * Tables are compile-time initialised; this is a no-op kept for API compat.
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
void keccak_f1600_init_cuda(void) {}
#endif

#undef RC
