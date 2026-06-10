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
#include "../cpu/ggm_tree_cpu.h"
#include "../cpu/spongent/spongent.cuh"
#include "../cpu/spongent/spongent_prf.cuh"
#include "../cpu/keccak/keccak_f1600.cuh"
#include "../cpu/keccak/keccak_prf.cuh"

static int tests_run = 0, tests_pass = 0;
#define PASS(name) do { printf("  PASS  %s\n", name); tests_pass++; tests_run++; } while(0)
#define FAIL(name) do { printf("  FAIL  %s\n", name);               tests_run++; } while(0)
#define CHECK(cond, name) do { if(cond) PASS(name); else FAIL(name); } while(0)

/* -----------------------------------------------------------------------
 * Fixed seeds used across tests
 * -------------------------------------------------------------------- */

/* Primary seed (used by original tests) */
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

/* KAT seeds: all-zeros, all-ones, incrementing bytes */
static const uint8_t SEED_ZEROS[16] = {
    0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00
};
static const uint8_t SEED_ONES[16] = {
    0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff
};
static const uint8_t SEED_INC[16] = {
    0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b, 0x0c,0x0d,0x0e,0x0f
};

static const uint8_t *KAT_SEEDS[3]   = { SEED_ZEROS, SEED_ONES, SEED_INC };
static const char    *KAT_NAMES[3]   = { "all-zeros", "all-ones", "incrementing" };

/* -----------------------------------------------------------------------
 * Helper: print a 16-byte value as hex
 * -------------------------------------------------------------------- */
static void print_hex(const char *label, const uint8_t *v, size_t n) {
    printf("    %s: ", label);
    for (size_t i = 0; i < n; i++) printf("%02x", v[i]);
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SPONGENT: depth-1 spot-check against known CPU values
 * -------------------------------------------------------------------- */
static void test_spongent_depth1_known(void) {
    printf("\n[Spongent GPU depth-1 spot-check]\n");

    uint8_t expected_left[16], expected_right[16];
    spongent128_expand(ROOT16, 16, expected_left, expected_right, 16);

    ggm_gpu_tree_t gpu_tree = {0};
    int rc = ggm_gpu_tree_build_spongent(&gpu_tree, ROOT16, 1);
    CHECK(rc == 0, "spongent gpu build depth=1");
    if (rc != 0) { ggm_gpu_tree_free(&gpu_tree); return; }

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
    int rc = ggm_gpu_tree_build_spongent(&gpu_tree, ROOT16, depth);
    CHECK(rc == 0, "spongent gpu build depth=4");
    if (rc != 0) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); return; }

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
 * SPONGENT KAT: expand() with 3 fixed seeds — CPU is reference
 *
 * For each seed (all-zeros, all-ones, incrementing), build a depth-1 GPU
 * tree and verify the two children match the CPU expand() output exactly.
 * Also verifies domain separation (left != right) and independence
 * (output != seed) for each seed.
 * -------------------------------------------------------------------- */
static void test_spongent_kat_expand(void) {
    printf("\n[Spongent KAT — expand() with 3 fixed seeds]\n");

    for (int s = 0; s < 3; s++) {
        const uint8_t *seed = KAT_SEEDS[s];
        printf("  seed: %s\n", KAT_NAMES[s]);

        /* CPU reference */
        uint8_t cpu_left[16], cpu_right[16];
        spongent128_expand(seed, 16, cpu_left, cpu_right, 16);

        print_hex("    left ", cpu_left,  16);
        print_hex("    right", cpu_right, 16);

        /* GPU */
        ggm_gpu_tree_t gpu_tree = {0};
        int rc = ggm_gpu_tree_build_spongent(&gpu_tree, seed, 1);

        char buf[64];
        snprintf(buf, sizeof(buf), "gpu build depth=1 [%s]", KAT_NAMES[s]);
        CHECK(rc == 0, buf);
        if (rc != 0) { ggm_gpu_tree_free(&gpu_tree); continue; }

        size_t total = ggm_gpu_tree_total_nodes(1);
        uint8_t *h = (uint8_t *)malloc(total * 16);
        if (!h) { ggm_gpu_tree_free(&gpu_tree); continue; }
        ggm_gpu_tree_copy_to_host(&gpu_tree, h, total * 16);

        snprintf(buf, sizeof(buf), "left child cpu==gpu [%s]",  KAT_NAMES[s]);
        CHECK(memcmp(h + 16, cpu_left,  16) == 0, buf);
        snprintf(buf, sizeof(buf), "right child cpu==gpu [%s]", KAT_NAMES[s]);
        CHECK(memcmp(h + 32, cpu_right, 16) == 0, buf);
        snprintf(buf, sizeof(buf), "domain separation [%s]",    KAT_NAMES[s]);
        CHECK(memcmp(cpu_left, cpu_right, 16) != 0, buf);
        snprintf(buf, sizeof(buf), "left != seed [%s]",         KAT_NAMES[s]);
        CHECK(memcmp(cpu_left, seed, 16) != 0, buf);

        free(h);
        ggm_gpu_tree_free(&gpu_tree);
    }
}

/* -----------------------------------------------------------------------
 * SPONGENT KAT: full tree at depth 8 with 3 fixed seeds
 *
 * For each seed, builds a complete 511-node tree on both CPU and GPU and
 * verifies every node matches byte-for-byte.  Depth 8 = 256 leaves,
 * enough to stress all kernel launch paths.
 * -------------------------------------------------------------------- */
static void test_spongent_kat_depth8(void) {
    printf("\n[Spongent KAT — depth 8, 3 seeds, all nodes cpu==gpu]\n");

    const int   depth     = 8;
    const size_t NODE_SZ  = 16;
    size_t total          = ((size_t)2 << depth) - 1;  /* 511 nodes */

    for (int s = 0; s < 3; s++) {
        const uint8_t *seed = KAT_SEEDS[s];
        printf("  seed: %s\n", KAT_NAMES[s]);

        ggm_tree_t cpu_tree;
        char buf[64];
        snprintf(buf, sizeof(buf), "cpu build depth=8 [%s]", KAT_NAMES[s]);
        CHECK(ggm_tree_build(&cpu_tree, &SPONGENT128_PRF, seed, depth) == 0, buf);

        ggm_gpu_tree_t gpu_tree = {0};
        int rc = ggm_gpu_tree_build_spongent(&gpu_tree, seed, depth);
        snprintf(buf, sizeof(buf), "gpu build depth=8 [%s]", KAT_NAMES[s]);
        CHECK(rc == 0, buf);
        if (rc != 0) {
            ggm_tree_free(&cpu_tree);
            ggm_gpu_tree_free(&gpu_tree);
            continue;
        }

        uint8_t *gpu_h = (uint8_t *)malloc(total * NODE_SZ);
        if (!gpu_h) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); continue; }
        ggm_gpu_tree_copy_to_host(&gpu_tree, gpu_h, total * NODE_SZ);

        int all_match = 1;
        for (int l = 0; l <= depth && all_match; l++) {
            size_t count = (size_t)1 << l;
            for (size_t i = 0; i < count; i++) {
                const uint8_t *cpu_node = ggm_tree_get_node(&cpu_tree, l, i);
                const uint8_t *gpu_node = gpu_h +
                    (((size_t)1 << l) - 1 + i) * NODE_SZ;
                if (memcmp(cpu_node, gpu_node, NODE_SZ) != 0) {
                    printf("  MISMATCH level=%d i=%zu\n", l, i);
                    all_match = 0;
                    break;
                }
            }
        }
        snprintf(buf, sizeof(buf), "all 511 nodes match cpu==gpu [%s]", KAT_NAMES[s]);
        CHECK(all_match, buf);

        free(gpu_h);
        ggm_tree_free(&cpu_tree);
        ggm_gpu_tree_free(&gpu_tree);
    }
}

/* -----------------------------------------------------------------------
 * SPONGENT KAT: depth-0 edge case (single root node, no expansion)
 *
 * A depth-0 tree contains exactly one node: the root seed itself.
 * No PRF calls should be made; the GPU must preserve the seed unchanged.
 * Tested with all three KAT seeds.
 * -------------------------------------------------------------------- */
static void test_spongent_depth0(void) {
    printf("\n[Spongent KAT — depth 0 edge case]\n");

    for (int s = 0; s < 3; s++) {
        const uint8_t *seed = KAT_SEEDS[s];

        ggm_gpu_tree_t gpu_tree = {0};
        char buf[64];
        snprintf(buf, sizeof(buf), "gpu build depth=0 [%s]", KAT_NAMES[s]);
        int rc = ggm_gpu_tree_build_spongent(&gpu_tree, seed, 0);
        CHECK(rc == 0, buf);
        if (rc != 0) { ggm_gpu_tree_free(&gpu_tree); continue; }

        uint8_t h[16];
        CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, h, 16) == 0,
              "copy depth-0 tree to host");
        snprintf(buf, sizeof(buf), "root preserved at depth 0 [%s]", KAT_NAMES[s]);
        CHECK(memcmp(h, seed, 16) == 0, buf);

        ggm_gpu_tree_free(&gpu_tree);
    }
}

/* -----------------------------------------------------------------------
 * KECCAK: depth-1 spot-check against known CPU values
 * -------------------------------------------------------------------- */
static void test_keccak_depth1_known(void) {
    printf("\n[Keccak GPU depth-1 spot-check]\n");

    uint8_t expected_left[32], expected_right[32];
    keccak1600_expand(ROOT32, 32, expected_left, expected_right, 32);

    ggm_gpu_tree_t gpu_tree = {0};
    int rc = ggm_gpu_tree_build_keccak(&gpu_tree, ROOT32, 1);
    CHECK(rc == 0, "keccak gpu build depth=1");
    if (rc != 0) { ggm_gpu_tree_free(&gpu_tree); return; }

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
    int rc = ggm_gpu_tree_build_keccak(&gpu_tree, ROOT32, depth);
    CHECK(rc == 0, "keccak gpu build depth=4");
    if (rc != 0) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); return; }

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

    /* Original tests */
    test_spongent_depth1_known();
    test_spongent_cpu_gpu_match();

    /* Spongent KATs */
    test_spongent_kat_expand();
    test_spongent_kat_depth8();
    test_spongent_depth0();

    /* Keccak tests */
    test_keccak_depth1_known();
    test_keccak_cpu_gpu_match();

    printf("\n=== Results: %d / %d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
