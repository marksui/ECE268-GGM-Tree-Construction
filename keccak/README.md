# keccak/ — Keccak-f1600 (Member A)

## Files to create

```
keccak/keccak_f1600.cuh     header  — __host__ __device__ declarations
keccak/keccak_f1600.cu      impl    — permutation, all functions __host__ __device__
keccak/keccak_prf.cuh       header  — expand + kernel declaration
keccak/keccak_prf.cu        impl    — keccak_expand (__host__ __device__)
                                      keccak_expand_level (__global__ kernel)
                                      KECCAK_PRF vtable
```

## Pattern to follow

`spongent/spongent_prf.cu` is the exact pattern to replicate.
Your `__global__ keccak_expand_level` will have the same signature as
`spongent_expand_level` — Member C calls both identically.

## Interface

```c
extern const prf_t KECCAK_PRF;   // plug into ggm_tree_build()

__global__ void keccak_expand_level(
    const uint8_t *parents,   // N × 16 bytes
    uint8_t       *children,  // 2N × 16 bytes
    size_t         N
);
```

## Makefile

Uncomment the `test_keccak` block in the top-level Makefile once ready.
