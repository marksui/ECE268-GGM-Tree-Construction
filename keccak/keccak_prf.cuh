#pragma once
#include <stdint.h>
#include <stddef.h>
#include "../common/prf_interface.h"

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__
#endif

__host__ __device__
void keccak1600_expand(const uint8_t *seed, size_t seed_len,
                       uint8_t *out0, uint8_t *out1, size_t out_len);

#ifdef __CUDACC__
__global__ void keccak_expand_level(const uint8_t *parents, uint8_t *children, size_t N);
void keccak1600_upload_tables(void);
#endif

extern const prf_t KECCAK1600_PRF;