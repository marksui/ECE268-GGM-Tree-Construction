# ECE268 GGM Tree Construction

Group 2, Spring 2026

Anjana Manoj A69041661  
Nikhita Neelakanta A69041742  
Mark Sui A18080143

This project builds a GGM tree using two PRFs: Spongent-128 and Keccak-f1600.
Both PRFs have CPU and CUDA GPU versions. The benchmark compares runtime,
throughput, and GPU speedup for different tree depths.

## File List

- `common/` shared PRF interface and small helper functions
- `cpu/` CPU GGM tree code, CPU Spongent, and CPU Keccak
- `gpu/` CUDA GGM tree code and CUDA kernels for Spongent and Keccak
- `tests/` correctness tests for CPU, GPU, Spongent, and Keccak
- `bench/benchmark_ggm.cu` benchmark for CPU vs GPU throughput
- `output/` saved terminal outputs from testing and benchmarking
- `docs/` report materials, figures, and archived older code
- `Makefile` build, test, and benchmark commands

## How to Use

The normal testing order I used was:

```bash
make clean
make all
make gpu_all
```

`make all` runs the CPU tests. `make gpu_all` runs the GPU tests, including the
CPU vs GPU GGM tree comparison.

Individual tests can also be run like this:

```bash
make test_ggm        # basic GGM tree CPU test
make test_spongent   # Spongent CPU test
make test_ggm_gpu    # CPU vs GPU GGM tree test
make test_keccak     # Keccak CPU vs GPU test
```

For benchmark results:

```bash
make benchmark
```

To choose the tested depths manually:
```bash
./build/benchmark_ggm 1 8 12 16 20
```

Here `1` is the repeat count. `8 12 16 20` are the GGM tree depths.
Depth 20 can take longer, especially for the Spongent CPU run.

## GPU Arch

The default GPU arch is `sm_61`, which is for GTX 1080 Ti in datahub.
For another GPU, pass `GPU_ARCH` in the make command:
```bash
make gpu_all GPU_ARCH=sm_89   # RTX 4070 Ti
```
