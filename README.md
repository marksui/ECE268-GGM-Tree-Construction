# ECE268 GGM Tree Construction

Group 2, Spring 2026

Anjana Manoj A69041661; Nikhita Neelakanta A69041742; Mark Sui A18080143

## Project Layout

```text
cpu/                 CPU GGM tree and CPU PRF code
  spongent/          Spongent CPU implementation
  keccak/            Keccak CPU implementation
gpu/                 CUDA GGM tree and GPU PRF kernels
  spongent/          Spongent GPU kernel
  keccak/            Keccak GPU kernel
common/              Shared PRF interface and utility helpers
tests/               CPU and GPU test programs
docs/                Reports, results, and archived files
```

## Run

CPU tests:

```bash
make test_ggm
make test_spongent
```

GPU tests on a CUDA machine:

```bash
make test_ggm_gpu
make test_keccak
```

Build outputs are generated under `build/` and are not part of the source tree.
