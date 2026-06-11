/*
 * bench/benchmark_ggm.cu — Comprehensive GGM tree benchmark
 *
 * Four sections:
 *   Section 0: Standard depths  {8,12,16,20}, best-of-1 (original data)
 *   Section 1: Dense depth sweep {4,6,10,14,18}, mean+stddev, repeat=5
 *   Section 2: Block-size sweep  Spongent GPU depth 16, tpb=32..512
 *   Section 3: Transfer breakdown build(H2D+compute) vs D2H, depths 12/16/20
 *
 * Usage:
 *   ./build/benchmark_ggm          — all sections
 *   ./build/benchmark_ggm 0        — section 0 only
 *   ./build/benchmark_ggm 1        — section 1 only
 *   ./build/benchmark_ggm 2        — section 2 only
 *   ./build/benchmark_ggm 3        — section 3 only
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../cpu/ggm_tree_cpu.h"
#include "../gpu/ggm_tree_gpu.cuh"
#include "../cpu/spongent/spongent_prf.cuh"
#include "../cpu/keccak/keccak_prf.cuh"

/* -----------------------------------------------------------------------
 * Timing helpers
 * -------------------------------------------------------------------- */
static double wall_ms(void) {
    auto t  = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t.time_since_epoch());
    return ms.count();
}

/* GPU-side elapsed time between two recorded events (ms) */
static float cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static void stats(double *samples, int n, double *mean, double *stddev) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += samples[i];
    *mean = s / n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = samples[i] - *mean;
        sq += d * d;
    }
    *stddev = (n > 1) ? sqrt(sq / (n - 1)) : 0.0;
}

static void make_seed(uint8_t *seed, int len, int start) {
    for (int i = 0; i < len; i++)
        seed[i] = (uint8_t)((start + i * 13) & 0xff);
}

/* -----------------------------------------------------------------------
 * GPU detection
 * -------------------------------------------------------------------- */
static int check_gpu(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("# no GPU found\n"); return 0;
    }
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess)
        printf("# gpu: %s\n", p.name);
    if (cudaFree(0) != cudaSuccess) {
        printf("# gpu init failed\n"); return 0;
    }
    return 1;
}

/* -----------------------------------------------------------------------
 * CPU runner — returns best_ms
 * -------------------------------------------------------------------- */
static double run_cpu(const prf_t *prf, const uint8_t *seed,
                      int depth, int repeat)
{
    double best = -1.0;
    for (int i = 0; i < repeat; i++) {
        ggm_tree_t tree = {0};
        double t0 = wall_ms();
        if (ggm_tree_build(&tree, prf, seed, depth) != 0) {
            ggm_tree_free(&tree); return -1.0;
        }
        double t1 = wall_ms();
        double ms = t1 - t0;
        if (best < 0.0 || ms < best) best = ms;
        ggm_tree_free(&tree);
    }
    return best;
}

/* CPU runner — returns all samples for mean/stddev */
static void run_cpu_samples(const prf_t *prf, const uint8_t *seed,
                             int depth, int repeat, double *out)
{
    for (int i = 0; i < repeat; i++) {
        ggm_tree_t tree = {0};
        double t0 = wall_ms();
        int ok = ggm_tree_build(&tree, prf, seed, depth);
        double t1 = wall_ms();
        out[i] = (ok == 0) ? (t1 - t0) : -1.0;
        ggm_tree_free(&tree);
    }
}

/* -----------------------------------------------------------------------
 * GPU Spongent runner — returns best_ms
 * -------------------------------------------------------------------- */
static double run_gpu_spongent(const uint8_t *seed, int depth, int repeat) {
    double best = -1.0;
    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree = {0};
        cudaDeviceSynchronize();
        double t0 = wall_ms();
        int ok = ggm_gpu_tree_build_spongent(&tree, seed, depth);
        cudaDeviceSynchronize();
        double t1 = wall_ms();
        if (ok == 0 && (best < 0.0 || (t1-t0) < best)) best = t1 - t0;
        ggm_gpu_tree_free(&tree);
    }
    return best;
}

/* GPU Spongent runner — returns all samples */
static void run_gpu_spongent_samples(const uint8_t *seed, int depth,
                                      int repeat, double *out)
{
    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree = {0};
        cudaDeviceSynchronize();
        double t0 = wall_ms();
        int ok = ggm_gpu_tree_build_spongent(&tree, seed, depth);
        cudaDeviceSynchronize();
        double t1 = wall_ms();
        out[i] = (ok == 0) ? (t1 - t0) : -1.0;
        ggm_gpu_tree_free(&tree);
    }
}

/* -----------------------------------------------------------------------
 * GPU Keccak runner — returns best_ms
 * -------------------------------------------------------------------- */
static double run_gpu_keccak(const uint8_t *seed, int depth, int repeat) {
    double best = -1.0;
    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree = {0};
        cudaDeviceSynchronize();
        double t0 = wall_ms();
        int ok = ggm_gpu_tree_build_keccak(&tree, seed, depth);
        cudaDeviceSynchronize();
        double t1 = wall_ms();
        if (ok == 0 && (best < 0.0 || (t1-t0) < best)) best = t1 - t0;
        ggm_gpu_tree_free(&tree);
    }
    return best;
}

/* GPU Keccak runner — returns all samples */
static void run_gpu_keccak_samples(const uint8_t *seed, int depth,
                                    int repeat, double *out)
{
    for (int i = 0; i < repeat; i++) {
        ggm_gpu_tree_t tree = {0};
        cudaDeviceSynchronize();
        double t0 = wall_ms();
        int ok = ggm_gpu_tree_build_keccak(&tree, seed, depth);
        cudaDeviceSynchronize();
        double t1 = wall_ms();
        out[i] = (ok == 0) ? (t1 - t0) : -1.0;
        ggm_gpu_tree_free(&tree);
    }
}

/* -----------------------------------------------------------------------
 * Print helpers
 * -------------------------------------------------------------------- */
static void print_best(const char *prf, const char *plat,
                        int depth, double ms)
{
    size_t leaves = (size_t)1 << depth;
    if (ms < 0.0) {
        printf("%s,%s,%d,%llu,FAIL,FAIL\n",
               prf, plat, depth, (unsigned long long)leaves);
        return;
    }
    double lps = (ms > 0.0) ? (double)leaves / (ms / 1000.0) : 0.0;
    printf("%s,%s,%d,%llu,%.3f,%.2f\n",
           prf, plat, depth, (unsigned long long)leaves, ms, lps);
}

static void print_mean(const char *prf, const char *plat,
                        int depth, double mean, double stddev)
{
    size_t leaves = (size_t)1 << depth;
    double lps = (mean > 0.0) ? (double)leaves / (mean / 1000.0) : 0.0;
    printf("%s,%s,%d,%llu,%.3f,%.3f,%.2f\n",
           prf, plat, depth, (unsigned long long)leaves,
           mean, stddev, lps);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SECTION 0: Standard depths — best-of-1 (matches original output)
 * ═══════════════════════════════════════════════════════════════════════ */
static void section0(int has_gpu,
                     const uint8_t *seed16, const uint8_t *seed32)
{
    printf("\n# === Section 0: Standard depths {8,12,16,20}, best-of-1 ===\n");
    printf("prf,platform,depth,leaves,best_ms,leaves_per_sec\n");

    int depths[] = {8, 12, 16, 20};
    int nd = 4;

    for (int i = 0; i < nd; i++) {
        int d = depths[i];
        print_best("Spongent", "CPU", d, run_cpu(&SPONGENT128_PRF, seed16, d, 1));
        if (has_gpu)
            print_best("Spongent", "GPU", d, run_gpu_spongent(seed16, d, 1));
        print_best("Keccak", "CPU", d, run_cpu(&KECCAK1600_PRF, seed32, d, 1));
        if (has_gpu)
            print_best("Keccak", "GPU", d, run_gpu_keccak(seed32, d, 1));
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SECTION 1: Dense depth sweep — mean ± stddev, repeat=5
 *
 * Intermediate depths {4,6,10,14,18} fill in the gaps between the four
 * standard depths so the speedup-vs-depth curve can be plotted smoothly.
 *
 * Spongent CPU is skipped for depth > 14 (would take > 8 min per run).
 * ═══════════════════════════════════════════════════════════════════════ */
static void section1(int has_gpu,
                     const uint8_t *seed16, const uint8_t *seed32)
{
    printf("\n# === Section 1: Dense depth sweep, repeat=5, mean+stddev ===\n");
    printf("# Spongent CPU skipped for depth > 14 (runtime > 8 min)\n");
    printf("prf,platform,depth,leaves,mean_ms,stddev_ms,leaves_per_sec\n");

    const int repeat  = 5;
    int depths[]      = {4, 6, 10, 14, 18};
    int nd            = 5;
    double samples[5];
    double mean, stddev;

    for (int i = 0; i < nd; i++) {
        int d = depths[i];

        /* Spongent CPU (skip depth 18) */
        if (d <= 14) {
            run_cpu_samples(&SPONGENT128_PRF, seed16, d, repeat, samples);
            stats(samples, repeat, &mean, &stddev);
            print_mean("Spongent", "CPU", d, mean, stddev);
        }

        /* Spongent GPU */
        if (has_gpu) {
            run_gpu_spongent_samples(seed16, d, repeat, samples);
            stats(samples, repeat, &mean, &stddev);
            print_mean("Spongent", "GPU", d, mean, stddev);
        }

        /* Keccak CPU */
        run_cpu_samples(&KECCAK1600_PRF, seed32, d, repeat, samples);
        stats(samples, repeat, &mean, &stddev);
        print_mean("Keccak", "CPU", d, mean, stddev);

        /* Keccak GPU */
        if (has_gpu) {
            run_gpu_keccak_samples(seed32, d, repeat, samples);
            stats(samples, repeat, &mean, &stddev);
            print_mean("Keccak", "GPU", d, mean, stddev);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SECTION 2: Block-size sensitivity sweep
 *
 * Runs Spongent GPU at depth 16 with threads-per-block in {32,64,128,256,512}.
 * Shows how occupancy tuning affects throughput on GTX 1080 Ti (sm_61,
 * 3584 CUDA cores, max 1024 threads/block).
 * ═══════════════════════════════════════════════════════════════════════ */
static void section2(int has_gpu, const uint8_t *seed16)
{
    printf("\n# === Section 2: Block-size sensitivity, Spongent GPU depth=16, repeat=5 ===\n");
    if (!has_gpu) { printf("# no GPU, skipping\n"); return; }
    printf("tpb,depth,leaves,mean_ms,stddev_ms,leaves_per_sec,speedup_vs_256\n");

    const int repeat  = 5;
    const int depth   = 16;
    int tpbs[]        = {32, 64, 128, 256, 512};
    int ntpb          = 5;
    double samples[5];
    double baseline   = -1.0;   /* mean at tpb=256 */

    for (int t = 0; t < ntpb; t++) {
        int tpb = tpbs[t];
        for (int i = 0; i < repeat; i++) {
            ggm_gpu_tree_t tree = {0};
            cudaDeviceSynchronize();
            double t0 = wall_ms();
            int ok = ggm_gpu_tree_build_spongent_tpb(&tree, seed16, depth, tpb);
            cudaDeviceSynchronize();
            double t1 = wall_ms();
            samples[i] = (ok == 0) ? (t1 - t0) : -1.0;
            ggm_gpu_tree_free(&tree);
        }
        double mean, stddev;
        stats(samples, repeat, &mean, &stddev);
        if (tpb == 256) baseline = mean;

        size_t leaves = (size_t)1 << depth;
        double lps    = (mean > 0.0) ? (double)leaves / (mean / 1000.0) : 0.0;
        double spdup  = (baseline > 0.0 && tpb != 256)
                        ? baseline / mean : 1.0;
        printf("%d,%d,%llu,%.3f,%.3f,%.2f,%.3f\n",
               tpb, depth, (unsigned long long)leaves,
               mean, stddev, lps, spdup);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SECTION 3: Memory transfer breakdown
 *
 * Separates GPU tree-build time (H2D seed copy + kernel compute) from
 * D2H copy time (reading the full tree back to host).  At large depths
 * the tree can be tens of MB; this shows whether transfer or compute
 * dominates.
 *
 * Depths: 12, 16, 20 for both PRFs.
 * ═══════════════════════════════════════════════════════════════════════ */
static void section3(int has_gpu,
                     const uint8_t *seed16, const uint8_t *seed32)
{
    printf("\n# === Section 3: Memory transfer breakdown (repeat=3) ===\n");
    printf("# build_ms = H2D seed + GPU compute\n");
    printf("# d2h_ms   = copy full tree device->host\n");
    if (!has_gpu) { printf("# no GPU, skipping\n"); return; }
    printf("prf,depth,leaves,tree_bytes,build_ms,d2h_ms,total_ms,d2h_pct\n");

    const int repeat  = 3;
    int depths[]      = {12, 16, 20};
    int nd            = 3;

    for (int i = 0; i < nd; i++) {
        int d = depths[i];

        /* ---- Spongent ---- */
        {
            size_t nodes      = ggm_gpu_tree_total_nodes(d);
            size_t tree_bytes = nodes * SPONGENT128_HASH_BYTES;
            uint8_t *h_buf    = (uint8_t *)malloc(tree_bytes);
            if (!h_buf) continue;

            double build_sum = 0, d2h_sum = 0;
            int ok_count = 0;

            for (int r = 0; r < repeat; r++) {
                ggm_gpu_tree_t tree = {0};
                cudaDeviceSynchronize();

                double b0 = wall_ms();
                int ok = ggm_gpu_tree_build_spongent(&tree, seed16, d);
                cudaDeviceSynchronize();
                double b1 = wall_ms();

                double c0 = wall_ms();
                if (ok == 0) ggm_gpu_tree_copy_to_host(&tree, h_buf, tree_bytes);
                double c1 = wall_ms();

                if (ok == 0) {
                    build_sum += (b1 - b0);
                    d2h_sum   += (c1 - c0);
                    ok_count++;
                }
                ggm_gpu_tree_free(&tree);
            }
            free(h_buf);

            if (ok_count > 0) {
                double bm = build_sum / ok_count;
                double dm = d2h_sum   / ok_count;
                double tm = bm + dm;
                printf("Spongent,%d,%llu,%llu,%.3f,%.3f,%.3f,%.1f%%\n",
                       d, (unsigned long long)((size_t)1 << d),
                       (unsigned long long)tree_bytes,
                       bm, dm, tm,
                       (tm > 0.0) ? 100.0 * dm / tm : 0.0);
            }
        }

        /* ---- Keccak ---- */
        {
            size_t nodes      = ggm_gpu_tree_total_nodes(d);
            size_t tree_bytes = nodes * KECCAK1600_HASH_BYTES;
            uint8_t *h_buf    = (uint8_t *)malloc(tree_bytes);
            if (!h_buf) continue;

            double build_sum = 0, d2h_sum = 0;
            int ok_count = 0;

            for (int r = 0; r < repeat; r++) {
                ggm_gpu_tree_t tree = {0};
                cudaDeviceSynchronize();

                double b0 = wall_ms();
                int ok = ggm_gpu_tree_build_keccak(&tree, seed32, d);
                cudaDeviceSynchronize();
                double b1 = wall_ms();

                double c0 = wall_ms();
                if (ok == 0) ggm_gpu_tree_copy_to_host(&tree, h_buf, tree_bytes);
                double c1 = wall_ms();

                if (ok == 0) {
                    build_sum += (b1 - b0);
                    d2h_sum   += (c1 - c0);
                    ok_count++;
                }
                ggm_gpu_tree_free(&tree);
            }
            free(h_buf);

            if (ok_count > 0) {
                double bm = build_sum / ok_count;
                double dm = d2h_sum   / ok_count;
                double tm = bm + dm;
                printf("Keccak,%d,%llu,%llu,%.3f,%.3f,%.3f,%.1f%%\n",
                       d, (unsigned long long)((size_t)1 << d),
                       (unsigned long long)tree_bytes,
                       bm, dm, tm,
                       (tm > 0.0) ? 100.0 * dm / tm : 0.0);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    uint8_t seed16[16], seed32[32];
    make_seed(seed16, 16, 3);
    make_seed(seed32, 32, 7);

    int has_gpu = check_gpu();

    /* Optional: run only one section if argv[1] is 0..3 */
    int only = -1;
    if (argc >= 2) only = atoi(argv[1]);

    if (only < 0 || only == 0) section0(has_gpu, seed16, seed32);
    if (only < 0 || only == 1) section1(has_gpu, seed16, seed32);
    if (only < 0 || only == 2) section2(has_gpu, seed16);
    if (only < 0 || only == 3) section3(has_gpu, seed16, seed32);

    return 0;
}
