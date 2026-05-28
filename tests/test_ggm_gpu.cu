/* Small GPU check for the GGM tree. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../gpu/ggm_tree_gpu.cuh"
#include "../spongent/spongent.cuh"

int main(void) {
    uint8_t root[16] = {
        0xde,0xad,0xbe,0xef, 0x00,0x11,0x22,0x33,
        0x44,0x55,0x66,0x77, 0x88,0x99,0xaa,0xbb
    };
    uint8_t expected_left[16] = {
        0x60,0x51,0x9b,0x61, 0x14,0xe8,0xfe,0x2a,
        0x91,0xc5,0xe1,0x00, 0xb7,0x66,0xc7,0x7c
    };
    uint8_t expected_right[16] = {
        0xb0,0x94,0x31,0xb2, 0x2f,0xdd,0x74,0x7b,
        0x4a,0x76,0x67,0x73, 0x0b,0x1d,0xa4,0x85
    };
    int depth = 1;
    ggm_gpu_tree_t gpu_tree = {0};

    printf("=== GPU GGM tree test (Spongent) ===\n");

    if (ggm_gpu_tree_build_spongent(&gpu_tree, root, depth) != 0) {
        printf("FAIL gpu tree build\n");
        return 1;
    }

    size_t bytes = ggm_gpu_tree_total_nodes(depth) * SPONGENT128_HASH_BYTES;
    uint8_t *gpu_copy = (uint8_t *)malloc(bytes);
    if (!gpu_copy) {
        printf("FAIL malloc\n");
        ggm_gpu_tree_free(&gpu_tree);
        return 1;
    }

    if (ggm_gpu_tree_copy_to_host(&gpu_tree, gpu_copy, bytes) != 0) {
        printf("FAIL copy gpu tree back\n");
        free(gpu_copy);
        ggm_gpu_tree_free(&gpu_tree);
        return 1;
    }

    if (memcmp(gpu_copy, root, 16) != 0 ||
        memcmp(gpu_copy + 16, expected_left, 16) != 0 ||
        memcmp(gpu_copy + 32, expected_right, 16) != 0) {
        printf("FAIL gpu tree bytes do not match expected depth 1 tree\n");
        free(gpu_copy);
        ggm_gpu_tree_free(&gpu_tree);
        return 1;
    }

    printf("PASS gpu tree depth 1 matches expected bytes\n");
    free(gpu_copy);
    ggm_gpu_tree_free(&gpu_tree);
    return 0;
}
