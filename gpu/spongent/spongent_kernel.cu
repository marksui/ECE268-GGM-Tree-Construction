#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../../cpu/spongent/spongent.cuh"
#include "../../cpu/spongent/spongent_prf.cuh"

#define SEED_BYTES   SPONGENT128_HASH_BYTES
#define B            136
#define STATE_BYTES  SPONGENT128_STATE_BYTES
#define NR_ROUNDS    SPONGENT128_NR_ROUNDS

__device__ __constant__ uint8_t c_sbox2[256];

static const uint8_t SBOX4[16] = {
    0xE, 0xD, 0xB, 0x0, 0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5, 0x9, 0xC, 0x3, 0x6
};

void spongent128_upload_tables(void)
{
    uint8_t h_sbox2[256];
    for (int v = 0; v < 256; v++)
        h_sbox2[v] = (uint8_t)((SBOX4[v >> 4] << 4) | SBOX4[v & 0xF]);
    cudaMemcpyToSymbol(c_sbox2, h_sbox2, sizeof(h_sbox2));
}

__device__ static void gpu_permute(uint8_t state[STATE_BYTES])
{
    uint8_t lc = SPONGENT128_LFSR_INIT;

    #pragma unroll 1
    for (int r = 0; r < NR_ROUNDS; r++) {

        uint8_t rlc = lc;
        rlc = (uint8_t)(((rlc & 0xF0u) >> 4) | ((rlc & 0x0Fu) << 4));
        rlc = (uint8_t)(((rlc & 0xCCu) >> 2) | ((rlc & 0x33u) << 2));
        rlc = (uint8_t)(((rlc & 0xAAu) >> 1) | ((rlc & 0x55u) << 1));
        state[0]  ^= lc;
        state[16] ^= rlc;

        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++)
            state[i] = c_sbox2[state[i]];

        uint8_t tmp[STATE_BYTES];
        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) tmp[i] = 0;

        #pragma unroll
        for (int b = 0; b < STATE_BYTES; b++) {
            const uint8_t sv = state[b];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int bit_i = b * 8 + k;

                const int dst = (bit_i < B - 1) ? (bit_i * 34) % (B - 1)
                                                 : (B - 1);
                if (sv & (uint8_t)(1u << k))
                    tmp[dst >> 3] |= (uint8_t)(1u << (dst & 7));
            }
        }

        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) state[i] = tmp[i];

        uint8_t fb = (uint8_t)(((lc >> 6) ^ (lc >> 5)) & 1u);
        lc = (uint8_t)(((lc << 1) & 0x7Fu) | fb);
    }
}

__device__ static void gpu_hash(uint8_t                       domain,
                                 const uint8_t * __restrict__ seed,
                                 uint8_t       * __restrict__ out)
{
    uint8_t state[STATE_BYTES];
    #pragma unroll
    for (int i = 0; i < STATE_BYTES; i++) state[i] = 0;

    state[0] ^= domain;
    gpu_permute(state);

    for (int i = 0; i < SEED_BYTES; i++) {
        state[0] ^= seed[i];
        gpu_permute(state);
    }

    state[0] ^= 0x80u;
    gpu_permute(state);

    for (int j = 0; j < SEED_BYTES; j++) {
        out[j] = state[0];
        if (j < SEED_BYTES - 1) gpu_permute(state);
    }
}

__device__ static void gpu_expand(const uint8_t * __restrict__ seed,
                                  uint8_t       * __restrict__ out0,
                                  uint8_t       * __restrict__ out1)
{
    gpu_hash(0x00u, seed, out0);
    gpu_hash(0x01u, seed, out1);
}

__global__
void spongent_expand_level(const uint8_t * __restrict__ parents,
                           uint8_t       * __restrict__ children,
                           size_t                       N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    gpu_expand(parents  +  i        * SEED_BYTES,
               children + (2*i)     * SEED_BYTES,
               children + (2*i + 1) * SEED_BYTES);
}

int spongent_launch_expand_level(const uint8_t *parents,
                                 uint8_t       *children,
                                 size_t         N,
                                 int            threads_per_block)
{
    int blocks = (int)((N + threads_per_block - 1) / threads_per_block);
    spongent_expand_level<<<blocks, threads_per_block>>>(parents, children, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at spongent kernel launch: %s\n",
               cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
