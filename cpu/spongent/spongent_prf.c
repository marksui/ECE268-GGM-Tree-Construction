#include <string.h>
#include "spongent_prf.cuh"
#include "spongent.cuh"

#define SEED_BYTES SPONGENT128_HASH_BYTES

static void spongent128_hash_cpu(uint8_t        domain,
                                  const uint8_t *seed,
                                  size_t         seed_len,
                                  uint8_t       *out)
{
    uint8_t state[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < SPONGENT128_STATE_BYTES; i++) state[i] = 0;

    state[0] ^= domain;
    spongent128_permute(state);

    size_t copy = (seed_len < (size_t)SEED_BYTES) ? seed_len : (size_t)SEED_BYTES;
    for (size_t i = 0; i < copy; i++) {
        state[0] ^= seed[i];
        spongent128_permute(state);
    }
    for (size_t i = copy; i < (size_t)SEED_BYTES; i++)
        spongent128_permute(state);

    state[0] ^= 0x80u;
    spongent128_permute(state);

    for (int j = 0; j < SPONGENT128_HASH_BYTES; j++) {
        out[j] = state[0];
        if (j < SPONGENT128_HASH_BYTES - 1) spongent128_permute(state);
    }
}

void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len)
{
    (void)out_len;
    spongent128_hash_cpu(0x00u, seed, seed_len, out0);
    spongent128_hash_cpu(0x01u, seed, seed_len, out1);
}

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

