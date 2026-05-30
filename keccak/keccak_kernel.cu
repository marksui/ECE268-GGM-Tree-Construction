/*
 * keccak/keccak_kernel.cu
 *
 * GPU kernel for the Keccak-f1600 GGM tree level expansion.
 *
 * This is the ONLY file in keccak/ that must be compiled by nvcc.
 * keccak_f1600.cu and keccak_prf.cu compile with gcc for CPU targets.
 *
 * What lives here:
 *   - GPU-side permute/hash/expand (device-only, uses constant memory)
 *   - keccak1600_upload_tables()  : host fn to push tables to GPU
 *   - keccak_expand_level         : __global__ kernel, one thread per parent
 *
 * Member C links this into gpu/ggm_tree_gpu.cu for the Keccak GPU tree.
 */

#include <stdint.h>
#include <stddef.h>
#include "keccak_f1600.cuh"
#include "keccak_prf.cuh"

#define SEED_BYTES KECCAK1600_HASH_BYTES   /* 32 */

/* -----------------------------------------------------------------------
 * GPU constant memory tables
 * Declared in keccak_f1600.cu; extern'd here so the kernel can read them.
 * -------------------------------------------------------------------- */
extern __device__ __constant__ uint64_t d_RC[24];
extern __device__ __constant__ int      d_RHO[25];
extern __device__ __constant__ int      d_PI[25];

#define GPU_ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

/* -----------------------------------------------------------------------
 * gpu_permute  (__device__)
 *
 * Keccak-f[1600] using constant-memory RC/RHO/PI tables.
 * Mirrors keccakf1600_permute() but reads from GPU constant memory.
 * -------------------------------------------------------------------- */
__device__ static void gpu_permute(uint64_t state[25])
{
    uint64_t C[5], D[5], tmp[25];
    #pragma unroll
    for (int r = 0; r < 24; r++) {
        #pragma unroll
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++)
            D[i] = C[(i+4)%5] ^ GPU_ROTL64(C[(i+1)%5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        #pragma unroll
        for (int i = 0; i < 25; i++)
            tmp[d_PI[i]] = GPU_ROTL64(state[i], d_RHO[i]);

        #pragma unroll
        for (int j = 0; j < 25; j += 5)
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j+i] = tmp[j+i] ^ ((~tmp[j+(i+1)%5]) & tmp[j+(i+2)%5]);

        state[0] ^= d_RC[r];
    }
}

/* -----------------------------------------------------------------------
 * gpu_hash  (__device__)
 *
 * SHA3-256 sponge — same logic as keccak1600_hash() but device-only.
 * Handles single-block messages (msg_len < KECCAK1600_RATE_BYTES).
 * -------------------------------------------------------------------- */
__device__ static void gpu_hash(const uint8_t *msg, size_t msg_len,
                                uint8_t digest[SEED_BYTES])
{
    uint64_t state[25] = {0};
    uint8_t *st = (uint8_t *)state;

    for (size_t i = 0; i < msg_len; i++) st[i] ^= msg[i];

    /* SHA3 padding */
    st[msg_len]                   ^= 0x06;
    st[KECCAK1600_RATE_BYTES - 1] ^= 0x80;

    gpu_permute(state);

    for (int i = 0; i < SEED_BYTES; i++) digest[i] = st[i];
}

/* -----------------------------------------------------------------------
 * gpu_expand  (__device__)
 *
 * Domain-separated expand — same construction as keccak1600_expand().
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
 *
 * One thread per parent node.
 *   parents  : device ptr, N  × SEED_BYTES (level l nodes)
 *   children : device ptr, 2N × SEED_BYTES (level l+1 nodes)
 *   N        : number of parent nodes = 2^l
 *
 * Launch: keccak_expand_level<<<(N+255)/256, 256>>>(parents, children, N)
 * -------------------------------------------------------------------- */
__global__
void keccak_expand_level(const uint8_t *parents,
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
