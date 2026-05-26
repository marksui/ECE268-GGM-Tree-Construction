#ifndef GGM_KECCAK_CUH
#define GGM_KECCAK_CUH

#include <stdint.h>

// 256-bit seed represented as 4 x 64-bit lanes
struct Seed256 {
    uint64_t lanes[4];
};

// GGM Kernel declaration
// n_evals: Total number of GGM evaluations to perform
// depth: The depth of the GGM tree (number of bits in the input x)
// d_roots: Array of root seeds (length n_evals)
// d_x: Array of inputs x (length n_evals)
// d_out: Array to store the final evaluated seeds (length n_evals)
__global__ void ggm_evaluate_kernel(
    const Seed256* d_roots, 
    const uint32_t* d_x, 
    Seed256* d_out, 
    int depth, 
    int n_evals
);

#endif // GGM_KECCAK_CUH
