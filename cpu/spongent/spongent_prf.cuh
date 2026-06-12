#pragma once

#include <stdint.h>
#include <stddef.h>
#include "../../common/prf_interface.h"

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__
#endif

#ifdef __cplusplus
extern "C" {
#endif

__host__ __device__
void spongent128_expand(const uint8_t *seed,  size_t seed_len,
                        uint8_t       *out0,
                        uint8_t       *out1,  size_t out_len);

#ifdef __CUDACC__
__global__
void spongent_expand_level(const uint8_t *parents,
                           uint8_t       *children,
                           size_t         N);

int spongent_launch_expand_level(const uint8_t *parents,
                                 uint8_t       *children,
                                 size_t         N,
                                 int            threads_per_block);
#endif

#ifdef __CUDACC__
void spongent128_upload_tables(void);
#endif

extern const prf_t SPONGENT128_PRF;

#ifdef __cplusplus
}
#endif
