#pragma once

#include <stdint.h>
#include <stddef.h>

#ifndef __CUDACC__
  #define __host__
  #define __device__
#endif

#define SPONGENT128_STATE_BYTES  17
#define SPONGENT128_RATE_BYTES    1
#define SPONGENT128_CAP_BYTES    16
#define SPONGENT128_HASH_BYTES   16
#define SPONGENT128_NR_ROUNDS    70
#define SPONGENT128_LFSR_INIT  0x7A

#ifdef __cplusplus
extern "C" {
#endif

__host__ __device__
uint8_t spongent128_lfsr_step(uint8_t state);

__host__ __device__
void spongent128_add_round_constant(uint8_t state[SPONGENT128_STATE_BYTES],
                                    uint8_t lc);

__host__ __device__
void spongent128_sbox_layer(uint8_t state[SPONGENT128_STATE_BYTES]);

__host__ __device__
void spongent128_player(uint8_t state[SPONGENT128_STATE_BYTES]);

__host__ __device__
void spongent128_permute(uint8_t state[SPONGENT128_STATE_BYTES]);

__host__ __device__
void spongent128_hash(const uint8_t *msg, size_t msg_len,
                      uint8_t digest[SPONGENT128_HASH_BYTES]);

#ifdef __cplusplus
}
#endif
