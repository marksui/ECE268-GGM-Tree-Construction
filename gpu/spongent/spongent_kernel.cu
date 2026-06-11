/*
 * gpu/spongent/spongent_kernel.cu  —  Optimised v3
 *
 * Root cause of v2 regression (~5-6x slowdown vs baseline):
 *   v2 used constant-memory TABLE VALUES as runtime indices into tmp[].
 *   Dynamic array indexing (tmp[runtime_value]) forces NVCC to allocate
 *   tmp[] in LOCAL MEMORY (off-chip DRAM, ~400 cycle latency per access).
 *   70 rounds × ~170 local-mem ops per gpu_permute × 50+ calls per expand
 *   = ~595,000 DRAM ops per thread. Every thread stalled on memory.
 *
 * Fix — doubly-unrolled pLayer (this file, gpu_permute):
 *   With #pragma unroll on BOTH the b=0..16 and k=0..7 loops,
 *   bit_i = b*8+k is a compile-time constant after unrolling.
 *   dst = (bit_i * 34) % 135 is then also compile-time.
 *   tmp[dst>>3] uses a LITERAL index → NVCC keeps tmp[] in 17 registers.
 *   Zero local-memory traffic in pLayer.
 *
 * Also reverts shared-prefix gpu_expand from v2:
 *   Two simultaneous 17-byte state arrays increased register pressure.
 *   Reverted to sequential gpu_hash() calls — registers are reused.
 *
 * Retained from v2:
 *   - c_sbox2[256] byte-level S-box in __constant__ memory (no divergence)
 *   - #pragma unroll 1 on 70-round outer loop (I-cache friendly)
 *   - Single cudaDeviceSynchronize() after all levels in ggm_tree_gpu.cu
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../../cpu/spongent/spongent.cuh"
#include "../../cpu/spongent/spongent_prf.cuh"

#define SEED_BYTES   SPONGENT128_HASH_BYTES   /* 16 */
#define B            136
#define STATE_BYTES  SPONGENT128_STATE_BYTES  /* 17 */
#define NR_ROUNDS    SPONGENT128_NR_ROUNDS    /* 70 */

/* -----------------------------------------------------------------------
 * c_sbox2[v] = (SBOX4[v>>4]<<4) | SBOX4[v&0xF]
 *
 * Applies the 4-bit PRESENT S-box to both nibbles of a byte in one lookup.
 * Loaded from constant-memory cache; all warp lanes read the same addresses
 * (different state values as keys) → hardware broadcasts efficiently.
 * Eliminates the nibble-split + 16-case switch from the original code.
 *
 * Constant-memory footprint: 256 bytes (out of 64KB limit).
 * -------------------------------------------------------------------- */
__device__ __constant__ uint8_t c_sbox2[256];

static const uint8_t SBOX4[16] = {
    0xE, 0xD, 0xB, 0x0, 0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5, 0x9, 0xC, 0x3, 0x6
};

/* Upload c_sbox2 once before any kernel launch. */
void spongent128_upload_tables(void)
{
    uint8_t h_sbox2[256];
    for (int v = 0; v < 256; v++)
        h_sbox2[v] = (uint8_t)((SBOX4[v >> 4] << 4) | SBOX4[v & 0xF]);
    cudaMemcpyToSymbol(c_sbox2, h_sbox2, sizeof(h_sbox2));
}

/* -----------------------------------------------------------------------
 * gpu_permute  (__device__)
 *
 * 70-round Spongent-128 permutation.
 *
 * pLayer design:
 *   Outer loop b = 0..16 and inner loop k = 0..7 are BOTH fully unrolled.
 *   After unrolling, bit_i = b*8+k is a compile-time constant (0..135).
 *   dst = (bit_i * 34) % 135 is then also compile-time.
 *   Every tmp[dst>>3] access uses a LITERAL index → tmp[] stays in regs.
 *   The bit-test (sv & (1<<k)) with compile-time k becomes a predicated
 *   instruction — no branch divergence.
 *
 * Outer 70-round loop: NOT unrolled (#pragma unroll 1).
 *   70 copies of the loop body would far exceed the L1 instruction cache
 *   and cause fetch stalls. One loop body + branch is I-cache friendly.
 * -------------------------------------------------------------------- */
__device__ static void gpu_permute(uint8_t state[STATE_BYTES])
{
    uint8_t lc = SPONGENT128_LFSR_INIT;   /* 7-bit LFSR, init 0x7A */

    #pragma unroll 1
    for (int r = 0; r < NR_ROUNDS; r++) {

        /* -- AddRoundConstant ------------------------------------------ */
        uint8_t rlc = lc;
        rlc = (uint8_t)(((rlc & 0xF0u) >> 4) | ((rlc & 0x0Fu) << 4));
        rlc = (uint8_t)(((rlc & 0xCCu) >> 2) | ((rlc & 0x33u) << 2));
        rlc = (uint8_t)(((rlc & 0xAAu) >> 1) | ((rlc & 0x55u) << 1));
        state[0]  ^= lc;
        state[16] ^= rlc;

        /* -- sBoxLayer: byte-level constant-cache lookup --------------- */
        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++)
            state[i] = c_sbox2[state[i]];

        /* -- pLayer: doubly-unrolled, tmp[] lives in registers --------- *
         *
         * With BOTH loops unrolled:
         *   bit_i = b*8+k  →  compile-time constant (0..135)
         *   dst   = (bit_i*34)%135  →  compile-time constant
         *   tmp[dst>>3]   →  literal register index  (no DRAM)
         *   1u<<(dst&7)   →  compile-time mask
         *   1u<<k         →  compile-time mask
         *
         * Net result: 136 predicated register OR instructions per pLayer.
         * ---------------------------------------------------------------- */
        uint8_t tmp[STATE_BYTES];
        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) tmp[i] = 0;

        #pragma unroll
        for (int b = 0; b < STATE_BYTES; b++) {
            const uint8_t sv = state[b];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int bit_i = b * 8 + k;   /* compile-time constant */
                /* p(i) = (i*34)%135 for i<135, p(135)=135 */
                const int dst = (bit_i < B - 1) ? (bit_i * 34) % (B - 1)
                                                 : (B - 1);
                if (sv & (uint8_t)(1u << k))
                    tmp[dst >> 3] |= (uint8_t)(1u << (dst & 7));
            }
        }

        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) state[i] = tmp[i];

        /* -- LFSR step: feedback = bit6 XOR bit5 ----------------------- */
        uint8_t fb = (uint8_t)(((lc >> 6) ^ (lc >> 5)) & 1u);
        lc = (uint8_t)(((lc << 1) & 0x7Fu) | fb);
    }
}

/* -----------------------------------------------------------------------
 * gpu_hash  (__device__)
 *
 * Spongent128_Hash( domain_byte || seed[0..15] )
 *
 * Permute breakdown: 1 domain absorb + 16 seed absorbs + 1 pad + 15 squeeze
 *                  = 33 permute calls per hash.
 *
 * Only ONE 17-byte state array is live at any time.
 * -------------------------------------------------------------------- */
__device__ static void gpu_hash(uint8_t                       domain,
                                 const uint8_t * __restrict__ seed,
                                 uint8_t       * __restrict__ out)
{
    uint8_t state[STATE_BYTES];
    #pragma unroll
    for (int i = 0; i < STATE_BYTES; i++) state[i] = 0;

    /* Absorb domain byte */
    state[0] ^= domain;
    gpu_permute(state);

    /* Absorb 16 seed bytes */
    for (int i = 0; i < SEED_BYTES; i++) {
        state[0] ^= seed[i];
        gpu_permute(state);
    }

    /* Sponge 10* padding */
    state[0] ^= 0x80u;
    gpu_permute(state);

    /* Squeeze 16 output bytes */
    for (int j = 0; j < SEED_BYTES; j++) {
        out[j] = state[0];
        if (j < SEED_BYTES - 1) gpu_permute(state);
    }
}

/* -----------------------------------------------------------------------
 * gpu_expand  (__device__)
 *
 * Produces:
 *   out0 = Hash( 0x00 || seed )   left  child
 *   out1 = Hash( 0x01 || seed )   right child
 *
 * Sequential: left hash finishes and its registers are freed before right
 * hash begins. No two state arrays are live simultaneously.
 * Total permutes: 33 + 33 = 66 per expand.
 * -------------------------------------------------------------------- */
__device__ static void gpu_expand(const uint8_t * __restrict__ seed,
                                  uint8_t       * __restrict__ out0,
                                  uint8_t       * __restrict__ out1)
{
    gpu_hash(0x00u, seed, out0);
    gpu_hash(0x01u, seed, out1);
}

/* -----------------------------------------------------------------------
 * spongent_expand_level  (__global__)
 *
 * One thread per parent node.
 *   parents  : device ptr, N  × SEED_BYTES (level l)
 *   children : device ptr, 2N × SEED_BYTES (level l+1)
 *   N        : number of parent nodes = 2^l
 * -------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * spongent_launch_expand_level  (host)
 *
 * Launch the expand kernel for one level.
 * NOTE: cudaDeviceSynchronize() intentionally omitted.
 * The caller (ggm_tree_gpu.cu) issues a single sync after all levels.
 * -------------------------------------------------------------------- */
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
