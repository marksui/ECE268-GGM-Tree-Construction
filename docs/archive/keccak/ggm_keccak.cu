#include <stdio.h>
#include <stdlib.h>
#include "ggm_keccak.cuh"

#define KECCAK_ROUNDS 24

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

__constant__ int d_Rho[25] = {
     0,  1, 62, 28, 27, 36, 44,  6, 55, 20,
     3, 10, 43, 25, 39, 41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

__constant__ int d_Pi[25] = {
     0, 10, 20,  5, 15, 16,  1, 11, 21,  6,
     7, 17,  2, 12, 22, 23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

__device__ __forceinline__ uint64_t ROTL64(uint64_t x, int y) {

    return (y == 0) ? x : ((x << y) | (x >> (64 - y)));
}

__device__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5];

    #pragma unroll
    for (int round = 0; round < KECCAK_ROUNDS; round++) {

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

        uint64_t temp[25];
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            temp[d_Pi[i]] = ROTL64(state[i], d_Rho[i]);
        }

        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                state[j + i] = temp[j + i] ^ ((~temp[j + ((i + 1) % 5)]) & temp[j + ((i + 2) % 5)]);
            }
        }

        state[0] ^= d_RC[round];
    }
}

__device__ void keccak_prg_double(const Seed256* input_seed, Seed256* out_L, Seed256* out_R) {
    uint64_t state[25] = {0};

    state[0] = input_seed->lanes[0];
    state[1] = input_seed->lanes[1];
    state[2] = input_seed->lanes[2];
    state[3] = input_seed->lanes[3];

    state[4] = 0x01;
    state[16] ^= 0x8000000000000000ULL;

    keccak_f1600(state);

    #pragma unroll
    for(int i = 0; i < 4; i++) out_L->lanes[i] = state[i];
    #pragma unroll
    for(int i = 0; i < 4; i++) out_R->lanes[i] = state[i + 4];
}

__global__ void ggm_evaluate_kernel(const Seed256* d_roots, const uint32_t* d_x, Seed256* d_out, int depth, int n_evals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_evals) {
        Seed256 current_seed = d_roots[idx];
        uint32_t x = d_x[idx];
        Seed256 L, R;

        for (int i = depth - 1; i >= 0; i--) {

            keccak_prg_double(&current_seed, &L, &R);

            uint32_t bit = (x >> i) & 1;

            current_seed = (bit == 0) ? L : R;
        }

        d_out[idx] = current_seed;
    }
}

int main() {

    int n_evals = 100000;
    int depth = 16;

    size_t seed_bytes = n_evals * sizeof(Seed256);
    size_t x_bytes = n_evals * sizeof(uint32_t);

    Seed256* h_roots = (Seed256*)malloc(seed_bytes);
    uint32_t* h_x = (uint32_t*)malloc(x_bytes);
    Seed256* h_out = (Seed256*)malloc(seed_bytes);

    for (int i = 0; i < n_evals; i++) {
        h_roots[i].lanes[0] = i;
        h_roots[i].lanes[1] = 0xDEADBEEF;
        h_roots[i].lanes[2] = 0xCAFEBABE;
        h_roots[i].lanes[3] = 0x12345678;

        h_x[i] = i % (1 << depth);
    }

    Seed256 *d_roots, *d_out;
    uint32_t *d_x;
    cudaMalloc(&d_roots, seed_bytes);
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_out, seed_bytes);

    cudaMemcpy(d_roots, h_roots, seed_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_evals + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching GGM Tree Traversal Kernel...\n");
    printf("Evaluations: %d | Tree Depth: %d\n", n_evals, depth);

    ggm_evaluate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_roots, d_x, d_out, depth, n_evals);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, seed_bytes, cudaMemcpyDeviceToHost);

    printf("\nSample Result (Index 0, Input x = %u):\n", h_x[0]);
    printf("Evaluated Seed Lanes:\n");
    printf("Lane 0: %016llx\n", (unsigned long long)h_out[0].lanes[0]);
    printf("Lane 1: %016llx\n", (unsigned long long)h_out[0].lanes[1]);
    printf("Lane 2: %016llx\n", (unsigned long long)h_out[0].lanes[2]);
    printf("Lane 3: %016llx\n", (unsigned long long)h_out[0].lanes[3]);

    cudaFree(d_roots);
    cudaFree(d_x);
    cudaFree(d_out);
    free(h_roots);
    free(h_x);
    free(h_out);

    return 0;
}
