/*
 * tests/test_spongent.c
 *
 * Unit tests for Spongent-128 + GGM tree integration.
 * Compiled with gcc (no nvcc needed) — the CUDA shim in spongent.cuh
 * strips __host__/__device__ tags so everything links as plain C.
 *
 * Build:  make test_spongent
 * Run:    ./build/test_spongent
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "../spongent/spongent.cuh"
#include "../spongent/spongent_prf.cuh"
#include "../common/ggm_tree.h"
#include "../common/utils.h"

/* -----------------------------------------------------------------------
 * Test harness
 * -------------------------------------------------------------------- */
static int tests_run = 0, tests_pass = 0;

#define EXPECT(cond, name) do {                                         \
    tests_run++;                                                         \
    if (cond) { printf("  PASS  %s\n", name); tests_pass++; }          \
    else      { printf("  FAIL  %s  (line %d)\n", name, __LINE__); }   \
} while(0)

#define EXPECT_BYTES_EQ(a, b, len, name) \
    EXPECT(bytes_eq((a),(b),(len)) == 0, name)

/* -----------------------------------------------------------------------
 * LFSR sequence
 * -------------------------------------------------------------------- */
static void test_lfsr(void) {
    printf("\n[LFSR]\n");
    uint8_t s = SPONGENT128_LFSR_INIT;
    printf("  Sequence from 0x%02X: ", s);
    for (int i = 0; i < 12; i++) { printf("0x%02X ", s); s = spongent128_lfsr_step(s); }
    printf("\n");
    /* Verify first 5 steps against reference Python (joostrijneveld/readable-crypto) */
    const uint8_t ref[5] = { 0x7A, 0x74, 0x68, 0x50, 0x21 };
    uint8_t lc = SPONGENT128_LFSR_INIT; int ok = 1;
    for (int i = 0; i < 5; i++) { if (lc != ref[i]) ok = 0; lc = spongent128_lfsr_step(lc); }
    EXPECT(ok, "lfsr_step matches reference sequence");
}

/* -----------------------------------------------------------------------
 * sBoxLayer — spot-check two nibbles
 * -------------------------------------------------------------------- */
static void test_sbox(void) {
    printf("\n[sBoxLayer]\n");
    /* SBOX[0xE]=0x3, SBOX[0xD]=0xC  =>  0xED -> 0x3C */
    uint8_t state[SPONGENT128_STATE_BYTES] = {0};
    state[0] = 0xED;
    spongent128_sbox_layer(state);
    EXPECT(state[0] == 0x3C, "sBoxLayer: 0xED -> 0x3C");
}

/* -----------------------------------------------------------------------
 * pLayer — bijectivity check
 * -------------------------------------------------------------------- */
static void test_player_bijective(void) {
    printf("\n[pLayer bijectivity]\n");
    int ok = 1;
    uint8_t state[SPONGENT128_STATE_BYTES], tmp[SPONGENT128_STATE_BYTES];
    for (int i = 0; i < 136 && ok; i++) {
        for (int j = 0; j < SPONGENT128_STATE_BYTES; j++) state[j] = 0;
        state[i / 8] |= (uint8_t)(1 << (7 - (i % 8)));
        for (int j = 0; j < SPONGENT128_STATE_BYTES; j++) tmp[j] = state[j];
        spongent128_player(tmp);
        int bits = 0;
        for (int j = 0; j < SPONGENT128_STATE_BYTES; j++) {
            uint8_t b = tmp[j]; while (b) { bits += b & 1; b >>= 1; }
        }
        if (bits != 1) { ok = 0; printf("  bit %d -> %d bits set\n", i, bits); }
    }
    EXPECT(ok, "pLayer: each input bit maps to exactly one output bit");
}

/* -----------------------------------------------------------------------
 * Hash — determinism + sensitivity
 * -------------------------------------------------------------------- */
static void test_hash(void) {
    printf("\n[Hash]\n");
    uint8_t d1[16], d2[16], d3[16];
    const uint8_t m1[] = "hello", m2[] = "Hello";

    spongent128_hash(m1, 5, d1);
    spongent128_hash(m1, 5, d2);
    spongent128_hash(m2, 5, d3);

    EXPECT_BYTES_EQ(d1, d2, 16, "hash(hello) is deterministic");
    EXPECT(bytes_eq(d1, d3, 16) != 0, "hash(hello) != hash(Hello)");
    print_hex("  hash(\"hello\")", d1, 16);

    uint8_t empty[16];
    spongent128_hash(NULL, 0, empty);
    print_hex("  hash(empty) ", empty, 16);
    /* Official Spongent-128 test vector (CHES 2011 paper + spongent website) */
    const uint8_t tv_msg[] = "Sponge + Present = Spongent";
    const uint8_t tv_exp[16] = {
        0x6B,0x7B,0xA3,0x5E,0xB0,0x9D,0xE0,0xF8,
        0xDE,0xF0,0x6A,0xE5,0x55,0x69,0x4C,0x53
    };
    uint8_t tv_got[16];
    spongent128_hash(tv_msg, sizeof(tv_msg)-1, tv_got);
    EXPECT_BYTES_EQ(tv_got, tv_exp, 16, "hash matches official test vector");
}

/* -----------------------------------------------------------------------
 * PRG expand — domain separation + stability
 * -------------------------------------------------------------------- */
static void test_expand(void) {
    printf("\n[PRG expand]\n");
    uint8_t seed[16] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                        0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    uint8_t o0a[16], o1a[16], o0b[16], o1b[16];

    spongent128_expand(seed, 16, o0a, o1a, 16);
    spongent128_expand(seed, 16, o0b, o1b, 16);

    EXPECT_BYTES_EQ(o0a, o0b, 16, "expand deterministic (out0)");
    EXPECT_BYTES_EQ(o1a, o1b, 16, "expand deterministic (out1)");
    EXPECT(bytes_eq(o0a, o1a, 16) != 0, "out0 != out1 (domain separation)");
    EXPECT(bytes_eq(o0a, seed, 16) != 0, "out0 != seed");
    print_hex("  seed ", seed, 16);
    print_hex("  out0 ", o0a,  16);
    print_hex("  out1 ", o1a,  16);
}

/* -----------------------------------------------------------------------
 * GGM tree — structure + correctness
 * -------------------------------------------------------------------- */
static void test_ggm_tree(void) {
    printf("\n[GGM tree depth=4]\n");
    uint8_t root[16] = {0xDE,0xAD,0xBE,0xEF,0x00,0x11,0x22,0x33,
                        0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB};
    ggm_tree_t tree;
    int rc = ggm_tree_build(&tree, &SPONGENT128_PRF, root, 4);
    EXPECT(rc == 0, "ggm_tree_build(depth=4) succeeds");
    if (rc != 0) return;

    EXPECT(ggm_tree_num_leaves(&tree) == 16, "16 leaves");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&tree, 0, 0), root, 16, "root matches seed");

    uint8_t exp0[16], exp1[16];
    spongent128_expand(root, 16, exp0, exp1, 16);
    EXPECT_BYTES_EQ(ggm_tree_get_node(&tree, 1, 0), exp0, 16, "node(1,0)==expand(root)[0]");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&tree, 1, 1), exp1, 16, "node(1,1)==expand(root)[1]");

    ggm_tree_print(&tree, 2);
    ggm_tree_free(&tree);
}

static void test_ggm_tree_depth8(void) {
    printf("\n[GGM tree depth=8]\n");
    uint8_t root[16] = {0};
    ggm_tree_t tree;
    int rc = ggm_tree_build(&tree, &SPONGENT128_PRF, root, 8);
    EXPECT(rc == 0, "build depth=8 succeeds");
    if (rc == 0) {
        EXPECT(ggm_tree_num_leaves(&tree) == 256, "256 leaves");
        ggm_tree_free(&tree);
    }
}

/* -----------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------- */
int main(void) {
    printf("=== Spongent-128 + GGM tree tests ===\n");
    test_lfsr();
    test_sbox();
    test_player_bijective();
    test_hash();
    test_expand();
    test_ggm_tree();
    test_ggm_tree_depth8();
    printf("\n=== Results: %d / %d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
