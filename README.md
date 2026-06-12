# ECE268 GGM Tree Construction

Group 2, Spring 2026

Anjana Manoj A69041661
Nikhita Neelakanta A69041742
Mark Sui A18080143

This project builds a GGM tree using two PRFs: Spongent-128 and Keccak-f1600.
Both PRFs have CPU and CUDA GPU versions. The benchmark compares runtime,
throughput, and GPU speedup for different tree depths.

## File Tree

```text
.
|-- .gitignore                        ignored build and notebook files
|-- README.md                         project overview and run notes
|-- Makefile                          build, test, and benchmark commands
|
|-- common/                           shared code used by CPU and GPU tests
|   |-- prf_interface.h               common PRF function interface
|   |-- utils.c                       small helper functions
|   `-- utils.h                       helper function header
|
|-- cpu/                              CPU implementation
|   |-- ggm_tree_cpu.c                CPU GGM tree construction
|   |-- ggm_tree_cpu.h                CPU GGM tree header
|   |
|   |-- keccak/                       CPU Keccak files
|   |   |-- keccak_f1600.cu           Keccak-f1600 permutation and hash
|   |   |-- keccak_f1600.cuh          Keccak header
|   |   |-- keccak_prf.cu             Keccak PRF expand function
|   |   `-- keccak_prf.cuh            Keccak PRF header
|   |
|   `-- spongent/                     CPU Spongent files
|       |-- spongent.c                Spongent permutation and hash
|       |-- spongent.cuh              Spongent constants and header
|       |-- spongent_prf.c            Spongent PRF expand function
|       `-- spongent_prf.cuh          Spongent PRF header
|
|-- gpu/                              CUDA implementation
|   |-- ggm_tree_gpu.cu               GPU GGM tree construction
|   |-- ggm_tree_gpu.cuh              GPU GGM tree header
|   |
|   |-- keccak/
|   |   `-- keccak_kernel.cu          Keccak CUDA kernel
|   |
|   `-- spongent/
|       `-- spongent_kernel.cu        Spongent CUDA kernel
|
|-- tests/                            correctness tests
|   |-- test_ggm.c                    basic CPU GGM tree test
|   |-- test_spongent.c               Spongent CPU test
|   |-- test_ggm_gpu.cu               CPU vs GPU GGM tree test
|   `-- test_keccak.cu                Keccak test and CPU/GPU comparison
|
|-- bench/                            benchmark and plotting code
|   |-- benchmark_ggm.cu              CPU/GPU benchmark program
|   `-- plot_results.py               makes result figures from benchmark output
|
|-- output/                           saved terminal outputs
|   |-- benchmark_ggm_output.txt      benchmark output
|   |-- results.txt                   result copy
|   |-- test_ggm.txt                  CPU GGM test output
|   |-- test_ggm_gpu.txt              GPU GGM test output
|   |-- test_keccak.txt               Keccak test output
|   `-- test_spongent.txt             Spongent test output
|
`-- docs/                             report and result materials
    |-- report/
    |   `-- ECE268 Progress Report.pdf
    |
    |-- archive/
    |   `-- keccak/
    |       |-- ggm_keccak.cu         older Keccak/GGM code
    |       `-- ggm_keccak.cuh        older Keccak/GGM header
    |
    `-- results/
        |-- benchmark_chart.ipynb     notebook for benchmark charts
        |-- benchmark_plots.py        script for extra plots
        |-- sim_spongent.txt          Spongent simulation notes/output
        |
        |-- figures/
        |   |-- runtime_vs_depth.png
        |   |-- speedup_vs_depth.png
        |   |-- testing_10000vectors.jpg
        |   `-- throughput_comparison_depth_20.png
        |
        `-- plots/
            |-- plot1_speedup.png
            |-- plot2_throughput.png
            |-- plot3_runtime.png
            |-- plot4_blocksize.png
            |-- plot5_memory.png
            `-- plot6_crossover.png
```

## How to Use

This is the normal order I used:

```bash
make clean
make all
make gpu_all
```

Some single commands:

```bash
make test_ggm
make test_spongent
make test_ggm_gpu
make test_keccak
```

Benchmark:

```bash
make benchmark
```

Run only the basic depth benchmark:

```bash
./build/benchmark_ggm 0
```

The benchmark includes depths 8, 12, 16, and 20. Depth 20 is slow for Spongent CPU.

## GPU Arch

Default is `sm_61` for GTX 1080 Ti. For a different GPU:

```bash
make gpu_all GPU_ARCH=sm_89 #RTX 4070Ti
```
