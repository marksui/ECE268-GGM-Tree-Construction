/*
 * tests/test_keccak.cu
 *
 * Exhaustive test harness for Keccak-f1600.
 * Verifies NIST FIPS 202 KATs and CPU/GPU mathematical equivalence.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h> // Ensure you have this if using C-style booleans
// Include your Keccak interfaces
#include "../keccak/keccak_f1600.cuh"
#include "../keccak/keccak_prf.cuh"

#define NUM_EXHAUSTIVE_TESTS 10000

bool check_gpu_ready() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        printf("[DIAGNOSTIC] cudaGetDeviceCount failed: %s (Code: %d)\n", cudaGetErrorString(err), err);
        return false;
    }

    err = cudaFree(0);
    if (err != cudaSuccess) {
        printf("[DIAGNOSTIC] cudaFree(0) failed: %s (Code: %d)\n", cudaGetErrorString(err), err);
        return false;
    }

    return true; 
}


// ============================================================================
// 1. GPU KERNEL FOR EQUIVALENCE TESTING
// ============================================================================

__global__ void test_keccak_gpu_kernel(uint64_t* d_states, int num_tests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tests) {
        uint64_t local_state[25];
        
        // Coalesced-ish load from global memory into registers
        #pragma unroll
        for(int i = 0; i < 25; i++) {
            local_state[i] = d_states[idx * 25 + i];
        }
        
        // Execute the __device__ permutation (from keccak_f1600.c)
        keccakf1600_permute(local_state);
        
        // Store the permuted state back to global memory
        #pragma unroll
        for(int i = 0; i < 25; i++) {
            d_states[idx * 25 + i] = local_state[i];
        }
    }
}

// ============================================================================
// 2. NIST FIPS 202 KAT VERIFICATION
// ============================================================================

bool verify_nist_kat() {
    printf("[*] Running NIST FIPS 202 Known Answer Test (SHA3-256 Empty String)...\n");
    
    uint8_t digest[KECCAK1600_HASH_BYTES];
    uint8_t empty_msg[1] = {0}; // Empty message
    
    // Official NIST SHA3-256 output for an empty string
    const uint8_t nist_expected[32] = {
        0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 
        0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61, 0xd6, 0x62, 
        0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 
        0x82, 0xd8, 0x0a, 0x4b, 0x80, 0xf8, 0x43, 0x4a
    };

    // Run your hash wrapper
    keccak1600_hash(empty_msg, 0, digest);

    // Verify
    for(int i = 0; i < 32; i++) {
        if(digest[i] != nist_expected[i]) {
            printf("[FAILED] KAT mismatch at byte %d. Expected %02x, Got %02x\n", i, nist_expected[i], digest[i]);
            return false;
        }
    }
    printf("[SUCCESS] NIST FIPS 202 KAT Passed!\n");
    return true;
}

// ============================================================================
// 3. EXHAUSTIVE GPU VS CPU VERIFICATION
// ============================================================================

bool verify_exhaustive_equivalence() {
    printf("[*] Running exhaustive GPU vs CPU equivalence test (%d states)...\n", NUM_EXHAUSTIVE_TESTS);

    size_t states_bytes = NUM_EXHAUSTIVE_TESTS * 25 * sizeof(uint64_t);
    
    // Allocate Host memory
    uint64_t* h_states_in = (uint64_t*)malloc(states_bytes);
    uint64_t* h_states_cpu_out = (uint64_t*)malloc(states_bytes);
    uint64_t* h_states_gpu_out = (uint64_t*)malloc(states_bytes);

    // Seed randomness
    srand((unsigned)time(NULL));

    // Generate random 1600-bit states
    for(int i = 0; i < NUM_EXHAUSTIVE_TESTS * 25; i++) {
        uint64_t r1 = (uint64_t)rand();
        uint64_t r2 = (uint64_t)rand();
        h_states_in[i] = (r1 << 32) | r2;
        h_states_cpu_out[i] = h_states_in[i]; // Copy for CPU processing
    }

    // 1. Process on CPU (Sequential)
    for(int i = 0; i < NUM_EXHAUSTIVE_TESTS; i++) {
        keccakf1600_permute(&h_states_cpu_out[i * 25]);
    }

// 2. Process on GPU (Parallel)
    uint64_t* d_states;
    cudaError_t err = cudaMalloc(&d_states, states_bytes);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemcpy(d_states, h_states_in, states_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_EXHAUSTIVE_TESTS + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch Kernel
    test_keccak_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, NUM_EXHAUSTIVE_TESTS);
    
    // Check for synchronous launch errors (e.g., architecture mismatch)
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("[ERROR] Kernel LAUNCH failed: %s\n", cudaGetErrorString(launch_err));
        return false;
    }

    // Check for asynchronous execution crashes (e.g., memory out of bounds)
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("[ERROR] Kernel EXECUTION crashed: %s\n", cudaGetErrorString(sync_err));
        return false;
    }

    cudaMemcpy(h_states_gpu_out, d_states, states_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] Memcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_states);
        free(h_states_in); free(h_states_cpu_out); free(h_states_gpu_out);
        return false;
    }

    // 3. Compare Results
    bool passed = true;
    for(int i = 0; i < NUM_EXHAUSTIVE_TESTS; i++) {
        for(int j = 0; j < 25; j++) {
            if(h_states_cpu_out[i * 25 + j] != h_states_gpu_out[i * 25 + j]) {
                printf("[FAILED] Mismatch at test %d, lane %d!\n", i, j);
                printf("  CPU: %016llx\n", (unsigned long long)h_states_cpu_out[i * 25 + j]);
                printf("  GPU: %016llx\n", (unsigned long long)h_states_gpu_out[i * 25 + j]);
                passed = false;
                break;
            }
        }
        if(!passed) break;
    }

    if(passed) {
        printf("[SUCCESS] GPU parallelization perfectly matches CPU for all %d vectors!\n", NUM_EXHAUSTIVE_TESTS);
    }

    // Clean up
    cudaFree(d_states);
    free(h_states_in); free(h_states_cpu_out); free(h_states_gpu_out);
    
    return passed;
}

int main() {
    printf("==================================================\n");
    printf("        KECCAK-F1600 EXHAUSTIVE TEST SUITE        \n");
    printf("==================================================\n\n");

    // 1. Set your flag!
    bool gpu_is_ready = check_gpu_ready();

    if (!gpu_is_ready) {
        printf("[FATAL ERROR] GPU is either missing or the connection is stale.\n");
        printf("              Try restarting your DSMLP pod.\n");
        printf("==================================================\n");
        return -1; // Abort gracefully before crashing on cudaMalloc
    }

    printf("[SUCCESS] GPU is allocated, awake, and ready!\n\n");

    // 2. Only run GPU initialization if the flag is true
    keccak_f1600_init_cuda();

    bool kat_passed = verify_nist_kat();
    printf("\n");
    bool equiv_passed = verify_exhaustive_equivalence();

    printf("\n==================================================\n");
    if(kat_passed && equiv_passed) {
        printf("ALL TESTS PASSED! Ready for GGM Tree Integration.\n");
        return 0;
    } else {
        printf("TESTS FAILED. Check console output for details.\n");
        return -1;
    }
}