/*
 * keccak/keccak_f1600.cu
 *
 * Keccak-f[1600] permutation + SHA3-256 sponge wrapper.
 * All core functions tagged __host__ __device__ so the same code
 * runs on CPU threads and GPU threads.
 *
 * Verified against NIST FIPS 202 KAT:
 *   SHA3-256("") == a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
 */

#include <string.h>
#include "keccak_f1600.cuh"

/* -----------------------------------------------------------------------
 * GPU constant memory tables (declared here, extern'd by keccak_kernel.cu)
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
__device__ __constant__ uint64_t d_RC[24];
__device__ __constant__ int      d_RHO[25];
__device__ __constant__ int      d_PI[25];
#endif

/* -----------------------------------------------------------------------
 * Round constants  (FIPS 202, Section 3.2.5)
 * -------------------------------------------------------------------- */
static const uint64_t RC[24] = {
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

/* Rho rotation offsets */
static const int RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

/* Pi permutation indices */
static const int PI[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

/* -----------------------------------------------------------------------
 * Rotate left 64-bit — avoids UB when y == 0
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
  #define ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))
#else
  static inline uint64_t ROTL64(uint64_t x, int y) {
      return (y == 0) ? x : ((x << y) | (x >> (64 - y)));
  }
#endif

/* -----------------------------------------------------------------------
 * keccakf1600_permute  (__host__ __device__)
 *
 * Applies 24 rounds of Theta / Rho / Pi / Chi / Iota to state[25].
 * -------------------------------------------------------------------- */
__host__ __device__
void keccakf1600_permute(uint64_t state[25])
{
    uint64_t C[5], D[5], tmp[25];

    #pragma unroll
    for (int r = 0; r < KECCAK1600_NR_ROUNDS; r++) {
        /* Theta */
        #pragma unroll
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++)
            D[i] = C[(i+4)%5] ^ ROTL64(C[(i+1)%5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        /* Rho + Pi */
        #pragma unroll
        for (int i = 0; i < 25; i++)
            tmp[PI[i]] = ROTL64(state[i], RHO[i]);

        /* Chi */
        #pragma unroll
        for (int j = 0; j < 25; j += 5)
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j+i] = tmp[j+i] ^ ((~tmp[j+(i+1)%5]) & tmp[j+(i+2)%5]);

        /* Iota */
        state[0] ^= RC[r];
    }
}

/* -----------------------------------------------------------------------
 * keccak1600_hash  (__host__ __device__)
 *
 * SHA3-256: rate=136 bytes, capacity=64 bytes, output=32 bytes.
 * Handles messages that fit within one rate block (msg_len < 136),
 * which covers our GGM PRF input of 33 bytes (1 domain tag + 32 seed).
 *
 * For full multi-block support extend the absorb loop — not needed here.
 * -------------------------------------------------------------------- */
__host__ __device__
void keccak1600_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[KECCAK1600_HASH_BYTES])
{
    uint64_t state[25];
    uint8_t *st = (uint8_t *)state;

    /* Zero state */
    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    /* Absorb — XOR message bytes into state (fits in one block) */
    for (size_t i = 0; i < msg_len; i++)
        st[i] ^= msg[i];

    /* SHA3 padding: 0x06 at msg_len, 0x80 at rate-1 */
    st[msg_len]                    ^= 0x06;
    st[KECCAK1600_RATE_BYTES - 1]  ^= 0x80;

    keccakf1600_permute(state);

    /* Squeeze first 32 bytes */
    for (int i = 0; i < KECCAK1600_HASH_BYTES; i++)
        digest[i] = st[i];
}

/* -----------------------------------------------------------------------
 * keccak_f1600_init_cuda  (host only)
 *
 * Uploads RC, RHO, PI tables to GPU constant memory.
 * Call once before any GPU kernel launch.
 * -------------------------------------------------------------------- */
#ifdef __CUDACC__
void keccak_f1600_init_cuda(void)
{
    cudaMemcpyToSymbol(d_RC,  RC,  sizeof(RC));
    cudaMemcpyToSymbol(d_RHO, RHO, sizeof(RHO));
    cudaMemcpyToSymbol(d_PI,  PI,  sizeof(PI));
}
#endif
