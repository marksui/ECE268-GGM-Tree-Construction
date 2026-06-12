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

    for (size_t i = 0; i < copy; i++)          buf[i + 1] = seed[i];
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
};
