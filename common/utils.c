/*
 * common/utils.c
 */

#include <stdio.h>
#include <string.h>
#include "utils.h"

void print_hex(const char *label, const uint8_t *data, size_t len) {
    if (label) printf("%s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

void bytes_xor(uint8_t *dst, const uint8_t *src, size_t len) {
    for (size_t i = 0; i < len; i++) dst[i] ^= src[i];
}

int bytes_eq(const uint8_t *a, const uint8_t *b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff == 0 ? 0 : 1;
}

void bytes_reverse(const uint8_t *in, uint8_t *out, size_t len) {
    if (in == out) {
        for (size_t i = 0; i < len / 2; i++) {
            uint8_t tmp = out[i];
            out[i] = out[len - 1 - i];
            out[len - 1 - i] = tmp;
        }
    } else {
        for (size_t i = 0; i < len; i++) out[i] = in[len - 1 - i];
    }
}

uint8_t bits_reverse(uint8_t x, int nbits) {
    uint8_t r = 0;
    for (int i = 0; i < nbits; i++)
        r |= (uint8_t)(((x >> i) & 1) << (nbits - 1 - i));
    return r;
}
