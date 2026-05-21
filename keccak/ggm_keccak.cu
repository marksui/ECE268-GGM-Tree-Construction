#include <stdio.h>
#include <stdlib.h>
#include "ggm_keccak.cuh"

#define KECCAK_ROUNDS 24

// --- GPU Constant Memory for Keccak ---
// Round constants (Iota)
__constant__ uint64_t d_RC[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation constants (Rho)
__constant__ int d_Rho[25] = {
     0,  1, 62, 28, 27, 36, 44,  6, 55, 20,
     3, 10, 43, 25, 39, 41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Pi permutation indices
__constant__ int d_Pi[25] = {
     0, 10, 20,  5, 15, 16,  1, 11, 21,  6,
     7, 17,  2, 12, 22, 23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

// --- Device Functions ---

__device__ __forceinline__ uint64_t ROTL64(uint64_t x, int y) {
    // Avoid undefined behavior when y == 0
    return (y == 0) ? x : ((x << y) | (x >> (64 - y)));
}

__device__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5];

    #pragma unroll
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            D[i] = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
        }
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // Rho and Pi
        uint64_t temp[25];
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            temp[d_Pi[i]] = ROTL64(state[i], d_Rho[i]);
        }

        // Chi
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                state[j + i] = temp[j + i] ^ ((~temp[j + ((i + 1) % 5)]) & temp[j + ((i + 2) % 5)]);
            }
        }

        // Iota
        state[0] ^= d_RC[round];
    }
}

__device__ void keccak_prg_double(const Seed256* input_seed, Seed256* out_L, Seed256* out_R) {
    uint64_t state[25] = {0}; 
    
    // 1. Absorb: Load 256-bit seed into the first 4 lanes
    state[0] = input_seed->lanes[0];
    state[1] = input_seed->lanes[1];
    state[2] = input_seed->lanes[2];
    state[3] = input_seed->lanes[3];
    
    // Apply Keccak padding (Domain separation / 10*1 padding)
    state[4] = 0x01; 
    state[16] ^= 0x8000000000000000ULL; 

    // 2. Permute
    keccak_f1600(state);

    // 3. Squeeze: Extract two 256-bit seeds (L and R)
    #pragma unroll
    for(int i = 0; i < 4; i++) out_L->lanes[i] = state[i];
    #pragma unroll
    for(int i = 0; i < 4; i++) out_R->lanes[i] = state[i + 4];
}

// --- GPU Kernel ---

__global__ void ggm_evaluate_kernel(const Seed256* d_roots, const uint32_t* d_x, Seed256* d_out, int depth, int n_evals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_evals) {
        Seed256 current_seed = d_roots[idx];
        uint32_t x = d_x[idx];
        Seed256 L, R;

        // Traverse the tree based on the bits of x
        for (int i = depth - 1; i >= 0; i--) {
            // Double the seed using our Keccak PRG
            keccak_prg_double(&current_seed, &L, &R);
            
            // Extract the i-th bit of x (MSB to LSB traversal)
            uint32_t bit = (x >> i) & 1;
            
            // Choose branch
            current_seed = (bit == 0) ? L : R;
        }
        
        // Write final leaf seed to global memory
        d_out[idx] = current_seed;
    }
}

// --- Host Code (Main) ---

int main() {
    // Configuration
    int n_evals = 100000; // Total GGM evaluations to perform
    int depth = 16;       // Depth of the GGM tree (16-bit input x)
    
    size_t seed_bytes = n_evals * sizeof(Seed256);
    size_t x_bytes = n_evals * sizeof(uint32_t);

    // Allocate Host Memory
    Seed256* h_roots = (Seed256*)malloc(seed_bytes);
    uint32_t* h_x = (uint32_t*)malloc(x_bytes);
    Seed256* h_out = (Seed256*)malloc(seed_bytes);

    // Initialize Host Data with dummy values for testing
    for (int i = 0; i < n_evals; i++) {
        h_roots[i].lanes[0] = i; // Varing root seeds
        h_roots[i].lanes[1] = 0xDEADBEEF;
        h_roots[i].lanes[2] = 0xCAFEBABE;
        h_roots[i].lanes[3] = 0x12345678;
        
        h_x[i] = i % (1 << depth); // Varing inputs x
    }

    // Allocate Device Memory
    Seed256 *d_roots, *d_out;
    uint32_t *d_x;
    cudaMalloc(&d_roots, seed_bytes);
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_out, seed_bytes);

    // Copy to Device
    cudaMemcpy(d_roots, h_roots, seed_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);

    // Configure and Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_evals + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching GGM Tree Traversal Kernel...\n");
    printf("Evaluations: %d | Tree Depth: %d\n", n_evals, depth);
    
    // Launch!
    ggm_evaluate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_roots, d_x, d_out, depth, n_evals);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy Result back to Host
    cudaMemcpy(h_out, d_out, seed_bytes, cudaMemcpyDeviceToHost);

    // Print a sample result to verify
    printf("\nSample Result (Index 0, Input x = %u):\n", h_x[0]);
    printf("Evaluated Seed Lanes:\n");
    printf("Lane 0: %016llx\n", (unsigned long long)h_out[0].lanes[0]);
    printf("Lane 1: %016llx\n", (unsigned long long)h_out[0].lanes[1]);
    printf("Lane 2: %016llx\n", (unsigned long long)h_out[0].lanes[2]);
    printf("Lane 3: %016llx\n", (unsigned long long)h_out[0].lanes[3]);

    // Free Memory
    cudaFree(d_roots);
    cudaFree(d_x);
    cudaFree(d_out);
    free(h_roots);
    free(h_x);
    free(h_out);

    return 0;
}