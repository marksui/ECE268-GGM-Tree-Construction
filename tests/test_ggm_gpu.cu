/*
 * tests/test_ggm_gpu.cu
 *
 * GPU GGM tree test: builds a depth-4 tree on GPU and on CPU,
 * then checks they produce identical node values.
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

static int tests_run = 0, tests_pass = 0;
#define PASS(name) do { printf("  PASS  %s\n", name); tests_pass++; tests_run++; } while(0)
#define FAIL(name) do { printf("  FAIL  %s\n", name);               tests_run++; } while(0)
#define CHECK(cond, name) do { if(cond) PASS(name); else FAIL(name); } while(0)

/* -----------------------------------------------------------------------
 * Depth-1 spot-check against known-good values
 * (expected values computed from the fixed CPU implementation)
 * -------------------------------------------------------------------- */
static void test_depth1_known_values(void) {
    printf("\n[GPU depth-1 spot-check]\n");

    uint8_t root[16] = {
        0xde,0xad,0xbe,0xef, 0x00,0x11,0x22,0x33,
        0x44,0x55,0x66,0x77, 0x88,0x99,0xaa,0xbb
    };
    /* expected = CPU spongent128_expand(root) — verified vs official TV */
    uint8_t expected_left[16] = {
        0xda,0x8c,0xce,0xd4, 0xb0,0xc4,0x51,0x9f,
        0x80,0xaf,0xa1,0x56, 0xd2,0x04,0x24,0xc1
    };
    uint8_t expected_right[16] = {
        0xff,0x0f,0x7f,0x86, 0xc9,0xdd,0xcc,0xb3,
        0x9c,0x4b,0x84,0xe0, 0x7f,0x32,0xc2,0xcb
    };

    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_spongent(&gpu_tree, root, 1) == 0, "gpu build depth=1");

    size_t total = ggm_gpu_tree_total_nodes(1);
    uint8_t *h = (uint8_t *)malloc(total * 16);
    CHECK(h != NULL, "malloc host buffer");
    if (!h) { ggm_gpu_tree_free(&gpu_tree); return; }

    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, h, total * 16) == 0, "copy to host");
    CHECK(memcmp(h,      root,           16) == 0, "root preserved");
    CHECK(memcmp(h + 16, expected_left,  16) == 0, "left child correct");
    CHECK(memcmp(h + 32, expected_right, 16) == 0, "right child correct");

    free(h);
    ggm_gpu_tree_free(&gpu_tree);
}

/* -----------------------------------------------------------------------
 * CPU vs GPU comparison — depth 4
 * Both trees must produce bit-identical output for every node.
 * -------------------------------------------------------------------- */
static void test_cpu_gpu_match(void) {
    printf("\n[CPU vs GPU comparison — depth 4]\n");

    uint8_t root[16] = {
        0xde,0xad,0xbe,0xef, 0x00,0x11,0x22,0x33,
        0x44,0x55,0x66,0x77, 0x88,0x99,0xaa,0xbb
    };
    const int depth = 4;
    size_t total = ((size_t)2 << depth) - 1;  /* 2^(depth+1)-1 = 31 nodes */

    /* --- CPU tree --- */
    ggm_tree_t cpu_tree;
    CHECK(ggm_tree_build(&cpu_tree, &SPONGENT128_PRF, root, depth) == 0,
          "cpu tree build depth=4");

    /* --- GPU tree --- */
    ggm_gpu_tree_t gpu_tree = {0};
    CHECK(ggm_gpu_tree_build_spongent(&gpu_tree, root, depth) == 0,
          "gpu tree build depth=4");

    /* Copy GPU tree to host */
    uint8_t *gpu_h = (uint8_t *)malloc(total * 16);
    CHECK(gpu_h != NULL, "malloc gpu host buffer");
    if (!gpu_h) { ggm_tree_free(&cpu_tree); ggm_gpu_tree_free(&gpu_tree); return; }
    CHECK(ggm_gpu_tree_copy_to_host(&gpu_tree, gpu_h, total * 16) == 0,
          "copy gpu tree to host");

    /* Compare every node level by level */
    int all_match = 1;
    for (int l = 0; l <= depth; l++) {
        size_t count = (size_t)1 << l;
        for (size_t i = 0; i < count; i++) {
            const uint8_t *cpu_node = ggm_tree_get_node(&cpu_tree, l, i);
            /* GPU flat index: (2^l - 1) + i */
            const uint8_t *gpu_node = gpu_h + (((size_t)1 << l) - 1 + i) * 16;
            if (memcmp(cpu_node, gpu_node, 16) != 0) {
                printf("  MISMATCH at level=%d i=%zu\n", l, i);
                all_match = 0;
            }
        }
    }
    CHECK(all_match, "all nodes match cpu==gpu");

    /* Print a few nodes for visual confirmation */
    printf("  Root     : ");
    for (int b = 0; b < 16; b++) printf("%02x", root[b]); puts("");
    const uint8_t *leaf0 = ggm_tree_get_node(&cpu_tree, depth, 0);
    printf("  Leaf[0]  : ");
    for (int b = 0; b < 16; b++) printf("%02x", leaf0[b]); puts("");
    printf("  GPU[0]   : ");
    for (int b = 0; b < 16; b++) printf("%02x", gpu_h[((1<<depth)-1)*16+b]); puts("");

    free(gpu_h);
    ggm_tree_free(&cpu_tree);
    ggm_gpu_tree_free(&gpu_tree);
}

int main(void) {
    printf("=== GPU GGM Tree Tests (Spongent-128) ===\n");
    test_depth1_known_values();
    test_cpu_gpu_match();
    printf("\n=== Results: %d / %d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
