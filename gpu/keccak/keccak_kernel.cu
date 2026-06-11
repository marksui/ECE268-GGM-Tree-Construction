/*
 * gpu/keccak/keccak_kernel.cu
 *
 * GPU kernel for Keccak-f1600 GGM tree level expansion.
 *
 * OPTIMISED: gpu_permute Rho+Pi step replaced with 25 literal assignments.
 *   The original loop  tmp[k_PI[i]] = ROTL64(state[i], k_RHO[i])
 *   used runtime values from constant memory as indices into tmp[],
 *   forcing tmp[] into local memory (off-chip DRAM, ~400 cycle latency).
 *   Same root cause as Spongent's pLayer regression.
 *
 *   Fix: all 25 dest indices and rotation amounts are now literals →
 *   NVCC keeps tmp[25] in 25 registers. Zero DRAM traffic in Rho+Pi.
 *   k_RHO and k_PI constant arrays removed (no longer needed).
 *
 * Outer 24-round loop: #pragma unroll 1 (prevents I-cache overflow).
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../../cpu/keccak/keccak_f1600.cuh"
#include "../../cpu/keccak/keccak_prf.cuh"

#define SEED_BYTES KECCAK1600_HASH_BYTES   /* 32 */

/* Round constants in GPU constant memory (compile-time initialised) */
__device__ __constant__ uint64_t k_RC[24] = {
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

#define GPU_ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

/* -----------------------------------------------------------------------
 * gpu_permute  (__device__)
 *
 * Keccak-f[1600]: 24 rounds, GPU register-resident tmp[25].
 *
 * Rho+Pi: 25 literal assignments — no dynamic indexing, no local memory.
 * Outer loop: #pragma unroll 1 — one loop body in I-cache, not 24 copies.
 * -------------------------------------------------------------------- */
__device__ static void gpu_permute(uint64_t state[25])
{
    uint64_t C[5], D[5];

    #pragma unroll 1
    for (int r = 0; r < 24; r++) {

        /* -- Theta ---------------------------------------------------- */
        #pragma unroll
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++)
            D[i] = C[(i+4)%5] ^ GPU_ROTL64(C[(i+1)%5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        /* -- Rho + Pi: literal indices → tmp[] in 25 registers -------- */
        uint64_t tmp[25];
        tmp[ 0] = GPU_ROTL64(state[ 0],  0);
        tmp[10] = GPU_ROTL64(state[ 1],  1);
        tmp[20] = GPU_ROTL64(state[ 2], 62);
        tmp[ 5] = GPU_ROTL64(state[ 3], 28);
        tmp[15] = GPU_ROTL64(state[ 4], 27);
        tmp[16] = GPU_ROTL64(state[ 5], 36);
        tmp[ 1] = GPU_ROTL64(state[ 6], 44);
        tmp[11] = GPU_ROTL64(state[ 7],  6);
        tmp[21] = GPU_ROTL64(state[ 8], 55);
        tmp[ 6] = GPU_ROTL64(state[ 9], 20);
        tmp[ 7] = GPU_ROTL64(state[10],  3);
        tmp[17] = GPU_ROTL64(state[11], 10);
        tmp[ 2] = GPU_ROTL64(state[12], 43);
        tmp[12] = GPU_ROTL64(state[13], 25);
        tmp[22] = GPU_ROTL64(state[14], 39);
        tmp[23] = GPU_ROTL64(state[15], 41);
        tmp[ 8] = GPU_ROTL64(state[16], 45);
        tmp[18] = GPU_ROTL64(state[17], 15);
        tmp[ 3] = GPU_ROTL64(state[18], 21);
        tmp[13] = GPU_ROTL64(state[19],  8);
        tmp[14] = GPU_ROTL64(state[20], 18);
        tmp[24] = GPU_ROTL64(state[21],  2);
        tmp[ 9] = GPU_ROTL64(state[22], 61);
        tmp[19] = GPU_ROTL64(state[23], 56);
        tmp[ 4] = GPU_ROTL64(state[24], 14);

        /* -- Chi ------------------------------------------------------ */
        #pragma unroll
        for (int j = 0; j < 25; j += 5)
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j+i] = tmp[j+i] ^ ((~tmp[j+(i+1)%5]) & tmp[j+(i+2)%5]);

        /* -- Iota ----------------------------------------------------- */
        state[0] ^= k_RC[r];
    }
}

/* -----------------------------------------------------------------------
 * gpu_hash  (__device__)
 * SHA3-256 single-block hash (msg_len < KECCAK1600_RATE_BYTES).
 * -------------------------------------------------------------------- */
__device__ static void gpu_hash(const uint8_t *msg, size_t msg_len,
                                uint8_t digest[SEED_BYTES])
{
    uint64_t state[25] = {0};
    uint8_t *st = (uint8_t *)state;

    for (size_t i = 0; i < msg_len; i++) st[i] ^= msg[i];
    st[msg_len]                   ^= 0x06;
    st[KECCAK1600_RATE_BYTES - 1] ^= 0x80;

    gpu_permute(state);

    for (int i = 0; i < SEED_BYTES; i++) digest[i] = st[i];
}

/* -----------------------------------------------------------------------
 * gpu_expand  (__device__)
 * Domain-separated expand matching keccak1600_expand().
 * -------------------------------------------------------------------- */
__device__ static void gpu_expand(const uint8_t *seed,
                                  uint8_t *out0, uint8_t *out1)
{
    uint8_t buf[1 + SEED_BYTES];
    for (int i = 0; i < SEED_BYTES; i++) buf[i + 1] = seed[i];
    buf[0] = 0x00;  gpu_hash(buf, 1 + SEED_BYTES, out0);
    buf[0] = 0x01;  gpu_hash(buf, 1 + SEED_BYTES, out1);
}

/* -----------------------------------------------------------------------
 * keccak_expand_level  (__global__)
 * One thread per parent node.
 * -------------------------------------------------------------------- */
__global__
void keccak_expand_level(const uint8_t *parents,
                         uint8_t       *children,
                         size_t         N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    gpu_expand(parents  + i       * SEED_BYTES,
               children + (2*i)   * SEED_BYTES,
               children + (2*i+1) * SEED_BYTES);
}

/* -----------------------------------------------------------------------
 * keccak_launch_expand_level  (host)
 * NOTE: cudaDeviceSynchronize() intentionally omitted.
 * Caller (ggm_tree_gpu.cu) issues a single sync after all levels.
 * -------------------------------------------------------------------- */
int keccak_launch_expand_level(const uint8_t *parents,
                               uint8_t       *children,
                               size_t         N,
                               int            threads_per_block)
{
    int blocks = (int)((N + threads_per_block - 1) / threads_per_block);
    keccak_expand_level<<<blocks, threads_per_block>>>(parents, children, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at keccak kernel launch: %s\n",
               cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
