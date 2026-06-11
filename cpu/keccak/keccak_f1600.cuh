#pragma once
/*
 * keccak/keccak_f1600.cuh
 * CUDA-compatible header for Keccak-f1600 core functions.
 */

#include <stdint.h>
#include <stddef.h>

#ifndef __CUDACC__
  #define __host__
  #define __device__
#endif

#define KECCAK1600_STATE_BYTES  200
#define KECCAK1600_RATE_BYTES   136
#define KECCAK1600_CAP_BYTES     64
#define KECCAK1600_HASH_BYTES    32
#define KECCAK1600_NR_ROUNDS     24

__host__ __device__
void keccakf1600_permute(uint64_t state[25]);

__host__ __device__
void keccak1600_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[KECCAK1600_HASH_BYTES]);

#ifdef __CUDACC__
void keccak_f1600_init_cuda(void);
#endif