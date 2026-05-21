/*
 * spongent/spongent_prf.cu
 *
 * spongent128_expand  — __host__ __device__, called by CPU tree and GPU kernel.
 * spongent_expand_level — __global__ kernel, one thread per parent node.
 */

#include <string.h>
#include "spongent_prf.cuh"
#include "spongent.cuh"

#define SEED_BYTES SPONGENT128_HASH_BYTES   /* 16 */

/* -----------------------------------------------------------------------
 * spongent128_expand  (__host__ __device__)
 *
 * Computes:
 *   out0 = Spongent128_Hash( 0x00 || seed )
 *   out1 = Spongent128_Hash( 0x01 || seed )
 *
 * The 1-byte domain tag is prepended to seed, giving a 17-byte input.
 * This ensures out0 != out1 even for the all-zero seed.
 * -------------------------------------------------------------------- */
__host__ __device__
void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len)
{
    (void)out_len;  /* always SEED_BYTES */

    /*
     * Build input buffer: [domain_tag (1 byte)] [seed (16 bytes)]
     * Stack-allocated — safe on both CPU and GPU (17 bytes).
     */
    uint8_t buf[1 + SEED_BYTES];
    size_t  copy = (seed_len < SEED_BYTES) ? seed_len : SEED_BYTES;

    for (size_t i = 0; i < copy; i++)       buf[i + 1] = seed[i];
    for (size_t i = copy; i < SEED_BYTES; i++) buf[i + 1] = 0;  /* zero-pad */

    buf[0] = 0x00;
    spongent128_hash(buf, 1 + SEED_BYTES, out0);

    buf[0] = 0x01;
    spongent128_hash(buf, 1 + SEED_BYTES, out1);
}

/* -----------------------------------------------------------------------
 * CPU-side prf_expand_fn wrapper (for ggm_tree_build vtable)
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
