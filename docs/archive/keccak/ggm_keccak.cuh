#ifndef GGM_KECCAK_CUH
#define GGM_KECCAK_CUH

#include <stdint.h>

struct Seed256 {
    uint64_t lanes[4];
};

__global__ void ggm_evaluate_kernel(
    const Seed256* d_roots,
    const uint32_t* d_x,
    Seed256* d_out,
    int depth,
    int n_evals
);

#endif
