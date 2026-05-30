/*
 * keccak/keccak_prf.c
 *
 * Implements the length-doubling wrapper layout for Member C infrastructure.
 */

#include <string.h>
#include "keccak_prf.cuh"
#include "keccak_f1600.cuh"

#define SEED_BYTES KECCAK1600_HASH_BYTES

__host__ __device__
void keccak1600_expand(const uint8_t *seed, size_t seed_len,
                       uint8_t       *out0,
                       uint8_t       *out1, size_t out_len)
{
    (void)out_len;
    uint8_t buf[1 + SEED_BYTES];
    size_t copy = (seed_len < SEED_BYTES) ? seed_len : SEED_BYTES;

    for (size_t i = 0; i < copy; i++)         buf[i + 1] = seed[i];
    for (size_t i = copy; i < SEED_BYTES; i++) buf[i + 1] = 0;

    buf[0] = 0x00;
    keccak1600_hash(buf, 1 + SEED_BYTES, out0);

    buf[0] = 0x01;
    keccak1600_hash(buf, 1 + SEED_BYTES, out1);
}

static void keccak1600_expand_cpu(const uint8_t *seed, size_t seed_len,
                                   uint8_t       *out0,
                                   uint8_t       *out1, size_t out_len)
{
    keccak1600_expand(seed, seed_len, out0, out1, out_len);
}

const prf_t KECCAK1600_PRF = {
    .name       = "Keccak-f1600",
    .seed_bytes = SEED_BYTES,
    .expand     = keccak1600_expand_cpu,
};/*
 * keccak/keccak_kernel.cu
 * GPU kernel processing for Keccak-f1600 GGM Level expansions.
 */

#include <stdint.h>
#include <stddef.h>
#include "keccak_f1600.cuh"
#include "keccak_prf.cuh"

#define SEED_BYTES KECCAK1600_HASH_BYTES

/* GPU Constant Memory mapping */
extern __device__ __constant__ uint64_t d_RC[24];
extern __device__ __constant__ int d_RHO[25];
extern __device__ __constant__ int d_PI[25];

void keccak1600_upload_tables(void) {
    uint64_t h_rc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
        0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
        0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    int h_rho[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
    int h_pi[25] = {0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4};
    
    cudaMemcpyToSymbol(d_RC, h_rc, sizeof(h_rc));
    cudaMemcpyToSymbol(d_RHO, h_rho, sizeof(h_rho));
    cudaMemcpyToSymbol(d_PI, h_pi, sizeof(h_pi));
}

#define GPU_ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

__device__ static void gpu_permute(uint64_t state[25]) {
    uint64_t C[5], D[5];
    #pragma unroll
    for (int r = 0; r < 24; r++) {
        #pragma unroll
        for (int i = 0; i < 5; i++) C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; i++) D[i] = C[(i + 4) % 5] ^ GPU_ROTL64(C[(i + 1) % 5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++) state[i] ^= D[i % 5];
        
        uint64_t temp[25];
        #pragma unroll
        for (int i = 0; i < 25; i++) temp[d_PI[i]] = GPU_ROTL64(state[i], d_RHO[i]);
        
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++) state[j + i] = temp[j + i] ^ ((~temp[j + ((i + 1) % 5)]) & temp[j + ((i + 2) % 5)]);
        }
        state[0] ^= d_RC[r];
    }
}

__device__ static void gpu_hash(const uint8_t *msg, size_t msg_len, uint8_t digest[SEED_BYTES]) {
    uint64_t state[25] = {0};
    uint8_t *st_bytes = (uint8_t *)state;

    for (size_t i = 0; i < msg_len; i++) st_bytes[i] ^= msg[i]; 
    st_bytes[msg_len] ^= 0x01;
    st_bytes[KECCAK1600_RATE_BYTES - 1] ^= 0x80;
    gpu_permute(state);

    for (int j = 0; j < SEED_BYTES; j++) digest[j] = st_bytes[j];
}

__device__ static void gpu_expand(const uint8_t *seed, uint8_t *out0, uint8_t *out1) {
    uint8_t buf[1 + SEED_BYTES];
    for (int i = 0; i < SEED_BYTES; i++) buf[i + 1] = seed[i];

    buf[0] = 0x00;  gpu_hash(buf, 1 + SEED_BYTES, out0);
    buf[0] = 0x01;  gpu_hash(buf, 1 + SEED_BYTES, out1);
}

__global__ void keccak_expand_level(const uint8_t *parents, uint8_t *children, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const uint8_t *parent = parents  + i       * SEED_BYTES;
    uint8_t       *left   = children + (2*i)   * SEED_BYTES;
    uint8_t       *right  = children + (2*i+1) * SEED_BYTES;

    gpu_expand(parent, left, right);
}