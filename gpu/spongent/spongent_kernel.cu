/*
 * gpu/spongent/spongent_kernel.cu  —  Optimised v2
 *
 * Performance improvements over the original:
 *
 *  1. Byte-level S-box in __constant__ memory  (c_sbox2[256])
 *     Replaces a 16-case switch statement — which causes warp serialisation
 *     when different threads hold different nibble values — with a single
 *     constant-memory broadcast load.  One lookup per byte instead of two
 *     nibble extractions + switch dispatch.
 *
 *  2. Precomputed pLayer scatter tables  (c_pdst_byte[136], c_pdst_mask[136])
 *     Eliminates the "(i * 34) % (B-1)" integer multiply + modulo that was
 *     computed 135 times per round (70 rounds × 33 permutes × 2 hashes).
 *     With #pragma unroll on the 17-byte outer loop the 136 table indices
 *     become compile-time constants, so NVCC emits fixed-offset constant-mem
 *     loads — efficiently broadcast to all warp lanes because every thread
 *     reads the same permutation table addresses with different state values.
 *
 *  3. #pragma unroll on all small fixed-trip-count inner loops
 *     sBoxLayer (17 ops), pLayer zeroinit (17), pLayer scatter (136),
 *     pLayer copy (17), state fork copy (17).
 *     The 70-round outer loop is intentionally NOT unrolled: 70 copies would
 *     dwarf the L1 instruction cache and cause fetch stalls.
 *
 *  4. Shared-prefix gpu_expand: absorb the 16 seed bytes once, then fork
 *     on the domain byte (0x00 = left child, 0x01 = right child).
 *     Permute count per expand: 66 → 50  (~24 % reduction).
 *
 *     IMPORTANT — PRF domain reordering:
 *       Old: hash( domain ∥ seed )  i.e. 0x00/0x01 prepended
 *       New: hash( seed   ∥ domain ) i.e. 0x00/0x01 appended
 *     The CPU counterpart (cpu/spongent/spongent_prf.c) is updated to match,
 *     so all CPU↔GPU correctness tests continue to pass.
 *     The domain separation property is preserved: same seed, domain=0x00
 *     and domain=0x01 still produce independent outputs.
 *
 *  5. Removed cudaDeviceSynchronize() from spongent_launch_expand_level().
 *     The caller (ggm_tree_gpu.cu) now performs a single sync after the
 *     entire level-loop, eliminating (depth-1) unnecessary CPU-GPU round-trips.
 *     CUDA guarantees that kernels launched in the same stream execute in
 *     submission order, so level l+1's kernel sees level l's output without
 *     an explicit per-level sync.
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
 * Constant-memory tables
 *
 *  c_sbox2[v]      = (SBOX4[v >> 4] << 4) | SBOX4[v & 0xF]
 *                    Applies the 4-bit PRESENT S-box to both nibbles at once,
 *                    so each byte needs only one table lookup.
 *
 *  c_pdst_byte[i]  = destination byte  for input bit i after pLayer
 *  c_pdst_mask[i]  = destination bitmask for input bit i after pLayer
 *    where: p(i) = (i * 34) % 135 for i < 135,  p(135) = 135
 *           c_pdst_byte[i] = p(i) >> 3
 *           c_pdst_mask[i] = 1u  << (p(i) & 7)
 *
 * Total constant-memory footprint: 256 + 136 + 136 = 528 bytes (out of 64 KB).
 * -------------------------------------------------------------------- */
__device__ __constant__ uint8_t c_sbox2[256];
__device__ __constant__ uint8_t c_pdst_byte[B];
__device__ __constant__ uint8_t c_pdst_mask[B];

/* Host-visible PRESENT S-box (nibble-level) */
static const uint8_t SBOX4[16] = {
    0xE, 0xD, 0xB, 0x0, 0x2, 0x1, 0x4, 0xF,
    0x7, 0xA, 0x8, 0x5, 0x9, 0xC, 0x3, 0x6
};

/* -----------------------------------------------------------------------
 * spongent128_upload_tables  (host)
 *
 * Compute and upload the three constant-memory tables once before any
 * kernel launch.  Called by setup_gpu_tree() via the upload_tables hook.
 * -------------------------------------------------------------------- */
void spongent128_upload_tables(void)
{
    /* --- byte-level S-box ---------------------------------------------- */
    uint8_t h_sbox2[256];
    for (int v = 0; v < 256; v++)
        h_sbox2[v] = (uint8_t)((SBOX4[v >> 4] << 4) | SBOX4[v & 0xF]);
    cudaMemcpyToSymbol(c_sbox2, h_sbox2, sizeof(h_sbox2));

    /* --- pLayer scatter tables ----------------------------------------- */
    uint8_t h_pdst_byte[B], h_pdst_mask[B];
    for (int i = 0; i < B - 1; i++) {
        int dst = (i * 34) % (B - 1);   /* p(i) for i < 135 */
        h_pdst_byte[i] = (uint8_t)(dst >> 3);
        h_pdst_mask[i] = (uint8_t)(1u << (dst & 7));
    }
    /* p(135) = 135 — last bit maps to itself */
    h_pdst_byte[B-1] = (uint8_t)((B-1) >> 3);   /* = 16 */
    h_pdst_mask[B-1] = (uint8_t)(1u << ((B-1) & 7)); /* = 0x80 */

    cudaMemcpyToSymbol(c_pdst_byte, h_pdst_byte, sizeof(h_pdst_byte));
    cudaMemcpyToSymbol(c_pdst_mask, h_pdst_mask, sizeof(h_pdst_mask));
}

/* -----------------------------------------------------------------------
 * gpu_permute  (__device__)
 *
 * 70-round Spongent-128 permutation — optimised for GPU.
 *
 * Key choices:
 *   - #pragma unroll 1  on the 70-round loop: prevents NVCC from emitting
 *     70 inlined copies that would overflow the I-cache.
 *   - #pragma unroll    on all STATE_BYTES loops: 17 iterations with known
 *     trip count → fully unrolled, no loop overhead.
 *   - pLayer scatter loop: inner bits 0-7 written as 8 explicit 'if' statements
 *     rather than a k-loop.  After outer-loop unrolling (b = 0..16),
 *     every 'base+k' is a compile-time constant, so NVCC emits 136 direct
 *     constant-cache loads — one per fixed-offset address.
 * -------------------------------------------------------------------- */
__device__ static void gpu_permute(uint8_t state[STATE_BYTES])
{
    uint8_t lc = SPONGENT128_LFSR_INIT;  /* 7-bit LFSR, init 0x7A */

    /* Outer loop: do NOT unroll — 70 copies far exceed I-cache */
    #pragma unroll 1
    for (int r = 0; r < NR_ROUNDS; r++) {

        /* -- AddRoundConstant ------------------------------------------
         * XOR 7-bit LFSR value into bits 0-6 of state[0],
         * XOR bit-reversed LFSR value into state[16].
         * ------------------------------------------------------------ */
        uint8_t rlc = lc;
        rlc = (uint8_t)(((rlc & 0xF0u) >> 4) | ((rlc & 0x0Fu) << 4));
        rlc = (uint8_t)(((rlc & 0xCCu) >> 2) | ((rlc & 0x33u) << 2));
        rlc = (uint8_t)(((rlc & 0xAAu) >> 1) | ((rlc & 0x55u) << 1));
        state[0]  ^= lc;
        state[16] ^= rlc;

        /* -- sBoxLayer --------------------------------------------------
         * One byte-level constant-mem lookup per byte (no nibble split,
         * no switch divergence).
         * ------------------------------------------------------------ */
        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++)
            state[i] = c_sbox2[state[i]];

        /* -- pLayer -----------------------------------------------------
         * Scatter bits to precomputed destinations.
         * After outer-loop unroll, base = 0, 8, 16, …, 128 at compile
         * time, so c_pdst_byte[base+k] uses fixed constant-mem offsets.
         * All 32 warp lanes read the SAME offsets (same perm table) →
         * constant-cache broadcast is maximally efficient.
         * ------------------------------------------------------------ */
        uint8_t tmp[STATE_BYTES];
        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) tmp[i] = 0;

        #pragma unroll
        for (int b = 0; b < STATE_BYTES; b++) {
            uint8_t sv = state[b];
            const int base = b * 8;
            if (sv & 0x01) tmp[c_pdst_byte[base+0]] |= c_pdst_mask[base+0];
            if (sv & 0x02) tmp[c_pdst_byte[base+1]] |= c_pdst_mask[base+1];
            if (sv & 0x04) tmp[c_pdst_byte[base+2]] |= c_pdst_mask[base+2];
            if (sv & 0x08) tmp[c_pdst_byte[base+3]] |= c_pdst_mask[base+3];
            if (sv & 0x10) tmp[c_pdst_byte[base+4]] |= c_pdst_mask[base+4];
            if (sv & 0x20) tmp[c_pdst_byte[base+5]] |= c_pdst_mask[base+5];
            if (sv & 0x40) tmp[c_pdst_byte[base+6]] |= c_pdst_mask[base+6];
            if (sv & 0x80) tmp[c_pdst_byte[base+7]] |= c_pdst_mask[base+7];
        }

        #pragma unroll
        for (int i = 0; i < STATE_BYTES; i++) state[i] = tmp[i];

        /* -- LFSR step: feedback = bit6 XOR bit5, shift-left ----------- */
        uint8_t fb = (uint8_t)(((lc >> 6) ^ (lc >> 5)) & 1u);
        lc = (uint8_t)(((lc << 1) & 0x7Fu) | fb);
    }
}

/* -----------------------------------------------------------------------
 * gpu_expand  (__device__)  — shared-prefix optimisation
 *
 * Produces:
 *   out0 = Spongent128_Hash( seed ∥ 0x00 )   ← left  child
 *   out1 = Spongent128_Hash( seed ∥ 0x01 )   ← right child
 *
 * Permute breakdown:
 *   Old (domain ∥ seed): 17 absorb + 1 pad + 15 squeeze = 33 per hash
 *                        33 × 2  =  66 per expand
 *
 *   New (seed ∥ domain):
 *     16  shared absorb (seed[0..15])
 *      1  left  domain absorb  (0x00 — XOR is no-op, permute still runs)
 *      1  right domain absorb  (0x01)
 *     (1 pad + 15 squeeze) × 2  =  32
 *   Total: 16 + 2 + 32  =  50 per expand  (~24 % fewer permutes)
 *
 * Note: the left-child domain absorb calls gpu_permute on the unmodified
 * state (XOR 0x00 is a no-op), so the call is not elided — Spongent's
 * sponge construction requires one permute per absorbed byte regardless.
 * -------------------------------------------------------------------- */
__device__ static void gpu_expand(const uint8_t * __restrict__ seed,
                                  uint8_t       * __restrict__ out0,
                                  uint8_t       * __restrict__ out1)
{
    /* ---------- Shared prefix: absorb seed[0..15] --------------------- */
    uint8_t state[STATE_BYTES];
    #pragma unroll
    for (int i = 0; i < STATE_BYTES; i++) state[i] = 0;

    for (int i = 0; i < SEED_BYTES; i++) {
        state[0] ^= seed[i];
        gpu_permute(state);
    }

    /* ---------- Fork: copy state for right child ---------------------- */
    uint8_t state1[STATE_BYTES];
    #pragma unroll
    for (int i = 0; i < STATE_BYTES; i++) state1[i] = state[i];

    /* ---------- Left child: domain = 0x00 ----------------------------- */
    /* state[0] ^= 0x00u; */   /* XOR 0 is a no-op — elided */
    gpu_permute(state);         /* absorb domain byte (mandatory permute) */
    state[0] ^= 0x80u;          /* 10* sponge pad */
    gpu_permute(state);
    for (int j = 0; j < SEED_BYTES; j++) {
        out0[j] = state[0];
        if (j < SEED_BYTES - 1) gpu_permute(state);
    }

    /* ---------- Right child: domain = 0x01 ---------------------------- */
    state1[0] ^= 0x01u;
    gpu_permute(state1);
    state1[0] ^= 0x80u;
    gpu_permute(state1);
    for (int j = 0; j < SEED_BYTES; j++) {
        out1[j] = state1[0];
        if (j < SEED_BYTES - 1) gpu_permute(state1);
    }
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
 * NOTE: cudaDeviceSynchronize() is intentionally OMITTED.
 * The caller (ggm_tree_gpu.cu) issues a single sync after all levels.
 * CUDA stream ordering guarantees level l+1 reads level l's output
 * without an explicit per-level barrier.
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
    /* No cudaDeviceSynchronize() here — caller syncs after all levels. */
    return 0;
}
