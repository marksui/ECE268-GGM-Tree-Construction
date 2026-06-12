#pragma once

#include <stdint.h>
#include <stddef.h>

typedef void (*prf_expand_fn)(
    const uint8_t *seed,
    size_t         seed_len,
    uint8_t       *out0,
    uint8_t       *out1,
    size_t         out_len
);

typedef struct {
    const char    *name;
    size_t         seed_bytes;
    prf_expand_fn  expand;
} prf_t;
