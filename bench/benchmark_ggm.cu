/*
 * Simple benchmark for GGM tree construction.
 * It prints CPU vs GPU timing for Spongent and Keccak.
 */

#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../cpu/ggm_tree_cpu.h"
#include "../gpu/ggm_tree_gpu.cuh"
#include "../cpu/spongent/spongent_prf.cuh"
#include "../cpu/keccak/keccak_prf.cuh"

static double get_time_ms(void) {
    auto t = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t.time_since_epoch());
    return ms.count();
}

static void make_seed(uint8_t *seed, int len, int start) {
    for (int i = 0; i < len; i++) {
        seed[i] = (uint8_t)((start + i * 13) & 0xff);
    }
}

static int check_gpu(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("# no gpu: %s\n", cudaGetErrorString(err));
        return 0;
    }
    if (count == 0) {
        printf("# no gpu found\n");
        return 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) {
        printf("# gpu: %s\n", prop.name);
    }

    err = cudaFree(0);
    if (err != cudaSuccess) {
        printf("# gpu init failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

static void print_result(const char *name, const char *where, int depth, double ms) {
    size_t leaves = (size_t)1 << depth;
    if (ms < 0.0) {
        printf("%s,%s,%d,%llu,FAIL,FAIL\n", name, where, depth, (unsigned long long)leaves);
        return;
    }

    double leaves_per_sec = 0.0;
    if (ms > 0.0) {
        leaves_per_sec = (double)leaves / (ms / 1000.0);
    }
    printf("%s,%s,%d,%llu,%.3f,%.2f\n", name, where, depth, (unsigned long long)leaves, ms, leaves_per_sec);
}

static double run_cpu(const prf_t *prf, const uint8_t *seed, int depth, int repeat) {
    double best = -1.0;

    for (int i = 0; i < repeat; i++) {
        ggm_tree_t tree;
        tree.data = NULL;
        tree.depth = 0;
        tree.seed_bytes = 0;
        tree.prf = NULL;

        double start = get_time_ms();
        int ok = ggm_tree_build(&tree, prf, seed, depth);
        double end = get_time_ms();

        if (ok != 0) {
            ggm_tree_free(&tree);
            return -1.0;
        }

        double used = end - start;
        if (best < 0.0 || used < best) {
            best = used;
        }
        ggm_tree_free(&tree);
    }

    return best;
}

static double run_gpu_spongent(const uint8_t *seed, int depth, int repeat) {
    double best = -1.0;

    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree;
        tree.d_data = NULL;
        tree.depth = 0;
        tree.seed_bytes = 0;

        cudaDeviceSynchronize();
        double start = get_time_ms();
        int ok = ggm_gpu_tree_build_spongent(&tree, seed, depth);
        cudaDeviceSynchronize();
        double end = get_time_ms();

        if (ok != 0) {
            ggm_gpu_tree_free(&tree);
            return -1.0;
        }

        double used = end - start;
        if (best < 0.0 || used < best) {
            best = used;
        }
        ggm_gpu_tree_free(&tree);
    }

    return best;
}

static double run_gpu_keccak(const uint8_t *seed, int depth, int repeat) {
    double best = -1.0;

    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree;
        tree.d_data = NULL;
        tree.depth = 0;
        tree.seed_bytes = 0;

        cudaDeviceSynchronize();
        double start = get_time_ms();
        int ok = ggm_gpu_tree_build_keccak(&tree, seed, depth);
        cudaDeviceSynchronize();
        double end = get_time_ms();

        if (ok != 0) {
            ggm_gpu_tree_free(&tree);
            return -1.0;
        }

        double used = end - start;
        if (best < 0.0 || used < best) {
            best = used;
        }
        ggm_gpu_tree_free(&tree);
    }

    return best;
}

int main(int argc, char **argv) {
    int repeat = 1;
    int depths[16] = {8, 12, 16, 20};
    int depth_count = 4;

    if (argc >= 2) {
        repeat = atoi(argv[1]);
        if (repeat <= 0) {
            repeat = 1;
        }
    }

    if (argc > 2) {
        depth_count = 0;
        for (int i = 2; i < argc && depth_count < 16; i++) {
            int d = atoi(argv[i]);
            if (d >= 0 && d <= GGM_MAX_DEPTH) {
                depths[depth_count] = d;
                depth_count++;
            }
        }
    }

    uint8_t seed16[16];
    uint8_t seed32[32];
    make_seed(seed16, 16, 3);
    make_seed(seed32, 32, 7);

    int has_gpu = check_gpu();
    printf("# repeat=%d\n", repeat);
    printf("prf,platform,depth,leaves,best_ms,leaves_per_sec\n");

    for (int i = 0; i < depth_count; i++) {
        int depth = depths[i];

        double ms = run_cpu(&SPONGENT128_PRF, seed16, depth, repeat);
        print_result("Spongent", "CPU", depth, ms);

        if (has_gpu) {
            ms = run_gpu_spongent(seed16, depth, repeat);
            print_result("Spongent", "GPU", depth, ms);
        }

        ms = run_cpu(&KECCAK1600_PRF, seed32, depth, repeat);
        print_result("Keccak", "CPU", depth, ms);

        if (has_gpu) {
            ms = run_gpu_keccak(seed32, depth, repeat);
            print_result("Keccak", "GPU", depth, ms);
        }
    }

    return 0;
}
