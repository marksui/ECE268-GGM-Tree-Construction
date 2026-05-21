/*
 * tests/test_ggm.c
 *
 * GGM tree framework tests using a trivial dummy PRF.
 * No dependency on Spongent or Keccak — Member C can run this alone.
 *
 * Build:  make test_ggm
 * Run:    ./build/test_ggm
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../common/ggm_tree.h"
#include "../common/utils.h"

static void dummy_expand(const uint8_t *seed, size_t seed_len,
                         uint8_t *out0, uint8_t *out1, size_t out_len) {
    size_t n = seed_len < out_len ? seed_len : out_len;
    for (size_t i = 0; i < n; i++) { out0[i] = seed[i] ^ 0xAA; out1[i] = seed[i] ^ 0x55; }
}
static const prf_t DUMMY_PRF = { "Dummy-XOR", 4, dummy_expand };

static int tests_run = 0, tests_pass = 0;
#define EXPECT(cond, name) do { \
    tests_run++; \
    if (cond) { printf("  PASS  %s\n", name); tests_pass++; } \
    else      { printf("  FAIL  %s  (line %d)\n", name, __LINE__); } \
} while(0)
#define EXPECT_BYTES_EQ(a,b,len,name) EXPECT(bytes_eq(a,b,len)==0,name)

static void test_depth0(void) {
    printf("\n[depth=0]\n");
    uint8_t root[4] = {1,2,3,4};
    ggm_tree_t t;
    EXPECT(ggm_tree_build(&t, &DUMMY_PRF, root, 0) == 0, "build succeeds");
    EXPECT(ggm_tree_num_leaves(&t) == 1, "1 leaf");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,0,0), root, 4, "root == seed");
    ggm_tree_free(&t);
}

static void test_depth1(void) {
    printf("\n[depth=1]\n");
    uint8_t root[4] = {0xFF,0xFF,0xFF,0xFF};
    ggm_tree_t t;
    ggm_tree_build(&t, &DUMMY_PRF, root, 1);
    uint8_t e0[4], e1[4]; dummy_expand(root,4,e0,e1,4);
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,1,0), e0, 4, "node(1,0) correct");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,1,1), e1, 4, "node(1,1) correct");
    ggm_tree_free(&t);
}

static void test_depth3(void) {
    printf("\n[depth=3]\n");
    uint8_t root[4] = {0x12,0x34,0x56,0x78};
    ggm_tree_t t;
    ggm_tree_build(&t, &DUMMY_PRF, root, 3);
    EXPECT(ggm_tree_num_leaves(&t) == 8, "8 leaves");
    uint8_t n10[4],n11[4],n20[4],n21[4],n22[4],n23[4];
    dummy_expand(root,4,n10,n11,4);
    dummy_expand(n10,4,n20,n21,4);
    dummy_expand(n11,4,n22,n23,4);
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,2,0),n20,4,"node(2,0)");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,2,1),n21,4,"node(2,1)");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,2,2),n22,4,"node(2,2)");
    EXPECT_BYTES_EQ(ggm_tree_get_node(&t,2,3),n23,4,"node(2,3)");
    ggm_tree_print(&t, 3);
    ggm_tree_free(&t);
}

static void test_oob(void) {
    printf("\n[out-of-bounds]\n");
    uint8_t root[4] = {0};
    ggm_tree_t t; ggm_tree_build(&t, &DUMMY_PRF, root, 2);
    EXPECT(ggm_tree_get_node(&t, 3, 0)  == NULL, "level > depth -> NULL");
    EXPECT(ggm_tree_get_node(&t, 2, 4)  == NULL, "index >= 2^level -> NULL");
    EXPECT(ggm_tree_get_node(&t, -1, 0) == NULL, "negative level -> NULL");
    ggm_tree_free(&t);
}

static void test_bad_inputs(void) {
    printf("\n[bad inputs]\n");
    uint8_t root[4] = {0};
    ggm_tree_t t;
    EXPECT(ggm_tree_build(&t, &DUMMY_PRF, root, -1)            == -1, "depth=-1");
    EXPECT(ggm_tree_build(&t, &DUMMY_PRF, root, GGM_MAX_DEPTH+1) == -1, "depth>MAX");
    EXPECT(ggm_tree_build(NULL, &DUMMY_PRF, root, 2)           == -1, "null tree");
    EXPECT(ggm_tree_build(&t, NULL, root, 2)                   == -1, "null prf");
}

static void test_leaves_ptr(void) {
    printf("\n[get_leaves pointer]\n");
    uint8_t root[4] = {0xAB,0xCD,0xEF,0x01};
    ggm_tree_t t; ggm_tree_build(&t, &DUMMY_PRF, root, 3);
    EXPECT_BYTES_EQ(ggm_tree_get_leaves(&t), ggm_tree_get_node(&t,3,0), 4,
                    "get_leaves == get_node(depth,0)");
    ggm_tree_free(&t);
}

int main(void) {
    printf("=== GGM tree framework tests (Dummy PRF) ===\n");
    test_depth0(); test_depth1(); test_depth3();
    test_oob(); test_bad_inputs(); test_leaves_ptr();
    printf("\n=== Results: %d / %d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
