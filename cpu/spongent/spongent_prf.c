/*
 * cpu/spongent/spongent_prf.c
 *
 * spongent128_expand — shared-prefix PRF, matching the optimised GPU kernel.
 *
 * Construction (v2 — matches gpu/spongent/spongent_kernel.cu):
 *   out0 = Spongent128_Hash( seed ∥ 0x00 )   ← left  child
 *   out1 = Spongent128_Hash( seed ∥ 0x01 )   ← right child
 *
 * vs original (v1):
 *   out0 = Spongent128_Hash( 0x00 ∥ seed )
 *   out1 = Spongent128_Hash( 0x01 ∥ seed )
 *
 * Reason for change: the GPU kernel now absorbs the 16 seed bytes once
 * (shared prefix) and forks only on the 1-byte domain tag.  The CPU must
 * use the same construction so that CPU↔GPU correctness tests pass.
 * Domain separation is preserved: same seed, domain 0x00 and 0x01 give
 * independent, unrelated outputs.
 *
 * CPU-side benefit: 66 → 50 permutes per expand (~24 % fewer calls to
 * spongent128_permute), which slightly improves CPU throughput too.
 */

#include <string.h>
#include "spongent_prf.cuh"
#include "spongent.cuh"

#define SEED_BYTES SPONGENT128_HASH_BYTES   /* 16 */

/* -----------------------------------------------------------------------
 * spongent128_expand  (__host__ __device__)
 *
 * Expand one 16-byte seed into two 16-byte children using the shared-prefix
 * construction above.
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len)
{
    (void)out_len;  /* always SEED_BYTES */

    size_t copy = (seed_len < SEED_BYTES) ? seed_len : (size_t)SEED_BYTES;

    /* ---- Shared prefix: absorb seed bytes ---------------------------- */
    uint8_t state[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = 0;

    for (size_t i = 0; i < copy; i++) {
        state[0] ^= seed[i];
        spongent128_permute(state);
    }
    /* Zero-pad if seed shorter than 16 bytes (rare; GPU always sends 16) */
    for (size_t i = copy; i < (size_t)SEED_BYTES; i++) {
        /* state[0] ^= 0x00u; */  /* XOR 0 is a no-op */
        spongent128_permute(state);
    }

    /* ---- Fork: copy shared state for right child --------------------- */
    uint8_t state1[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state1[i] = state[i];

    /* ---- Left child: absorb domain 0x00, pad, squeeze ---------------- */
    /* state[0] ^= 0x00u; */  /* XOR 0 is a no-op — elided */
    spongent128_permute(state);
    state[0] ^= 0x80u;        /* 10* sponge pad */
    spongent128_permute(state);
    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        out0[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1) spongent128_permute(state);
    }

    /* ---- Right child: absorb domain 0x01, pad, squeeze --------------- */
    state1[0] ^= 0x01u;
    spongent128_permute(state1);
    state1[0] ^= 0x80u;
    spongent128_permute(state1);
    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        out1[j] = state1[0];
        if (j < SPONGENT128_HASH_BYTES - 1) spongent128_permute(state1);
    }
}

/* -----------------------------------------------------------------------
 * CPU-side prf_t vtable wrapper for ggm_tree_build()
 * -------------------------------------------------------------------- */
static void spongent128_expand_cpu(const uint8_t *seed,  size_t seed_len,
                                   uint8_t       *out0,
                                   uint8_t       *out1,  size_t out_len)
{
    spongent128_expand(seed, seed_len, out0, out1, out_len);
}

const prf_t SPONGENT128_PRF = {
    .name       = "Spongent-128",
    .seed_bytes = SEED_BYTES,
    .expand     = spongent128_expand_cpu,
};
