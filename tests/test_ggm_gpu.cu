/*
 * tests/test_ggm_gpu.cu
 *
 * GPU GGM tree tests: builds trees on GPU and CPU, verifies they match.
 * Covers both Spongent-128 and Keccak-f1600 backends.
 *
 * Build:  make test_ggm_gpu  (requires nvcc)
 * Run:    ./build/test_ggm_gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../gpu/ggm_tree_gpu.cuh"
#include "../common/ggm_tree.h"
#include "../spongent/spongent.cuh"
#include "../spongent/spongent_prf.cuh"
#include "../keccak/keccak_f1600.cuh"
#include "../keccak/keccak_prf.cuh"

static int tests_run = 0, tests_pass = 0;
#define PASS(name) do { printf("  PASS  %s\n", name); tests_pass++; tests_run++; } while(0)
#define FAIL(name) do { printf("  FAIL  %s\n", name);               tests_run++; } while(0)
#define CHECK(cond, name) do { if(cond) PASS(name); else FAIL(name); } while(0)

/* Shared root seed used across tests */
static const uint8_t ROOT16[16] = {
    0xde,0xad,0xbe,0xef, 0x00,0x11,0x22,0x33,
    0x44,0x55,0x66,0x77, 0x88,0x99,0xaa,0xbb
};
static const uint8_t ROOT32[32] = {
    0xde,0xad,0xbe,0xef, 0x00,0x11,0x22,0x33,
    0x44,0x55,0x66,0x77, 0x88,0x99,0xaa,0xbb,
    0xca,0xfe,0xba,0xbe, 0xde,0xad,0xc0,0xde,
    0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef
};

/* -----------------------------------------------------------------------
 * SPONGENT: depth-1 spot-check against known CPU values
 * -------------------------------------------------------------------- */
static void test_spongent_depth1_known(void) {
    printf("\n[Spongent GPU depth-1 spot-check]\n");

    uint8_t expected_left[16], expected_right[16];
    spongent128_expand(ROOT16, 16, expected_left, expected_right, 16);

    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_spongent(&gpu_tree, ROOT16, 1) == 0,
          "spongent gpu build depth=1");

    size_t total = ggm_gpu_tree_total_nodes(1);
    uint8_t *h = (uint8_t *)malloc(total * 16);
    CHECK(h != NULL, "malloc host buffer");
    if (!h) { ggm_gpu_tree_free(&gpu_tree); return; }

    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, h, total * 16) == 0,
          "copy to host");
    CHECK(memcmp(h,      ROOT16,         16) == 0, "root preserved");
    CHECK(memcmp(h + 16, expected_left,  16) == 0, "left child correct");
    CHECK(memcmp(h + 32, expected_right, 16) == 0, "right child correct");

    free(h);
    ggm_gpu_tree_free(&gpu_tree);
}

/* -----------------------------------------------------------------------
 * SPONGENT: CPU vs GPU comparison at depth 4
 * -------------------------------------------------------------------- */
static void test_spongent_cpu_gpu_match(void) {
    printf("\n[Spongent CPU vs GPU — depth 4]\n");

    const int depth = 4;
    size_t total = ((size_t)2 << depth) - 1;

    ggm_tree_t cpu_tree;
    CHECK(ggm_tree_build(&cpu_tree, &SPONGENT128_PRF, ROOT16, depth) == 0,
          "spongent cpu build depth=4");

    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_spongent(&gpu_tree, ROOT16, depth) == 0,
          "spongent gpu build depth=4");

    uint8_t *gpu_h = (uint8_t *)malloc(total * 16);
    CHECK(gpu_h != NULL, "malloc gpu host buffer");
    if (!gpu_h) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); return; }
    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, gpu_h, total * 16) == 0,
          "copy gpu tree to host");

    int all_match = 1;
    for (int l = 0; l <= depth; l++) {
        size_t count = (size_t)1 << l;
        for (size_t i = 0; i < count; i++) {
            const uint8_t *cpu_node = ggm_tree_get_node(&cpu_tree, l, i);
            const uint8_t *gpu_node = gpu_h + (((size_t)1 << l) - 1 + i) * 16;
            if (memcmp(cpu_node, gpu_node, 16) != 0) {
                printf("  MISMATCH at level=%d i=%zu\n", l, (size_t)i);
                all_match = 0;
            }
        }
    }
    CHECK(all_match, "all spongent nodes match cpu==gpu");

    free(gpu_h);
    ggm_tree_free(&cpu_tree);
    ggm_gpu_tree_free(&gpu_tree);
}

/* -----------------------------------------------------------------------
 * KECCAK: depth-1 spot-check against known CPU values
 * -------------------------------------------------------------------- */
static void test_keccak_depth1_known(void) {
    printf("\n[Keccak GPU depth-1 spot-check]\n");

    uint8_t expected_left[32], expected_right[32];
    keccak1600_expand(ROOT32, 32, expected_left, expected_right, 32);

    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_keccak(&gpu_tree, ROOT32, 1) == 0,
          "keccak gpu build depth=1");

    size_t total = ggm_gpu_tree_total_nodes(1);
    uint8_t *h = (uint8_t *)malloc(total * 32);
    CHECK(h != NULL, "malloc host buffer");
    if (!h) { ggm_gpu_tree_free(&gpu_tree); return; }

    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, h, total * 32) == 0,
          "copy to host");
    CHECK(memcmp(h,      ROOT32,         32) == 0, "root preserved");
    CHECK(memcmp(h + 32, expected_left,  32) == 0, "left child correct");
    CHECK(memcmp(h + 64, expected_right, 32) == 0, "right child correct");

    free(h);
    ggm_gpu_tree_free(&gpu_tree);
}

/* -----------------------------------------------------------------------
 * KECCAK: CPU vs GPU comparison at depth 4
 * -------------------------------------------------------------------- */
static void test_keccak_cpu_gpu_match(void) {
    printf("\n[Keccak CPU vs GPU — depth 4]\n");

    const int depth = 4;
    size_t total = ((size_t)2 << depth) - 1;

    ggm_tree_t cpu_tree;
    CHECK(ggm_tree_build(&cpu_tree, &KECCAK1600_PRF, ROOT32, depth) == 0,
          "keccak cpu build depth=4");

    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_keccak(&gpu_tree, ROOT32, depth) == 0,
          "keccak gpu build depth=4");

    uint8_t *gpu_h = (uint8_t *)malloc(total * 32);
    CHECK(gpu_h != NULL, "malloc gpu host buffer");
    if (!gpu_h) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); return; }
    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, gpu_h, total * 32) == 0,
          "copy gpu tree to host");

    int all_match = 1;
    for (int l = 0; l <= depth; l++) {
        size_t count = (size_t)1 << l;
        for (size_t i = 0; i < count; i++) {
            const uint8_t *cpu_node = ggm_tree_get_node(&cpu_tree, l, i);
            const uint8_t *gpu_node = gpu_h + (((size_t)1 << l) - 1 + i) * 32;
            if (memcmp(cpu_node, gpu_node, 32) != 0) {
                printf("  MISMATCH at level=%d i=%zu\n", l, (size_t)i);
                all_match = 0;
            }
        }
    }
    CHECK(all_match, "all keccak nodes match cpu==gpu");

    free(gpu_h);
    ggm_tree_free(&cpu_tree);
    ggm_gpu_tree_free(&gpu_tree);
}

/* -----------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------- */
int main(void) {
    printf("=== GPU GGM Tree Tests (Spongent-128 + Keccak-f1600) ===\n");

    test_spongent_depth1_known();
    test_spongent_cpu_gpu_match();
    test_keccak_depth1_known();
    test_keccak_cpu_gpu_match();

    printf("\n=== Results: %d / %d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
