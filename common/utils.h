#pragma once
/*
 * common/utils.h
 */

#include <stdint.h>
#include <stddef.h>

void    print_hex(const char *label, const uint8_t *data, size_t len);
void    bytes_xor(uint8_t *dst, const uint8_t *src, size_t len);
int     bytes_eq(const uint8_t *a, const uint8_t *b, size_t len);
void    bytes_reverse(const uint8_t *in, uint8_t *out, size_t len);
uint8_t bits_reverse(uint8_t x, int nbits);
