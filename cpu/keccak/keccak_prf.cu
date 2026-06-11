/*
 * keccak/keccak_prf.cu
 *
 * Keccak-f1600 length-doubling PRG for the GGM tree.
 *
 * PRG construction (matching spongent_prf pattern):
 *   out0 = SHA3-256( 0x00 || seed )   <- left  child
 *   out1 = SHA3-256( 0x01 || seed )   <- right child
 *
 * keccak1600_expand is __host__ __device__ so it can be called from
 * CPU code (via the prf_t vtable) and from GPU device functions.
 *
 * The __global__ kernel lives in keccak_kernel.cu.
 */

#include <string.h>
#include "keccak_prf.cuh"
#include "keccak_f1600.cuh"

#define SEED_BYTES KECCAK1600_HASH_BYTES   /* 32 */

/* -----------------------------------------------------------------------
 * keccak1600_expand  (__host__ __device__)
 *
 * Expand one 32-byte seed into two 32-byte children.
 * Callable from CPU code and GPU device functions.
 * -------------------------------------------------------------------- */
__host__ __device__
void keccak1600_expand(const uint8_t *seed, size_t seed_len,
                       uint8_t       *out0,
                       uint8_t       *out1, size_t out_len)
{
    (void)out_len;  /* always SEED_BYTES */

    uint8_t buf[1 + SEED_BYTES];
    size_t copy = (seed_len < SEED_BYTES) ? seed_len : SEED_BYTES;

    for (size_t i = 0; i < copy; i++)          buf[i + 1] = seed[i];
    for (size_t i = copy; i < SEED_BYTES; i++) buf[i + 1] = 0;

    buf[0] = 0x00;
    keccak1600_hash(buf, 1 + SEED_BYTES, out0);

    buf[0] = 0x01;
    keccak1600_hash(buf, 1 + SEED_BYTES, out1);
}

/* -----------------------------------------------------------------------
 * CPU-side prf_expand_fn wrapper (for ggm_tree_build vtable)
 * -------------------------------------------------------------------- */
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
};
