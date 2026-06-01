# Makefile — GGM PRF Tree Project (ECE268)
#
# CPU tests (gcc, no CUDA needed):
#   make all             run both CPU test suites
#   make test_ggm        GGM tree framework (Dummy PRF)
#   make test_spongent   Spongent-128 + GGM tree
#
# GPU builds (nvcc required):
#   make test_ggm_gpu    Spongent + Keccak CPU vs GPU comparison
#   make test_keccak     Keccak NIST KAT + 10k CPU/GPU equivalence
#   make benchmark       CPU/GPU throughput benchmark
#   make gpu_all         build all GPU test binaries
#
# Override GPU arch (default sm_70 = Datahub V100):
#   make gpu_all GPU_ARCH=sm_80

CC        = gcc
NVCC      = nvcc
CFLAGS    = -std=c11 -Wall -Wextra -Wpedantic -O2 \
            -Icpu -Icommon -Icpu/spongent -Icpu/keccak

GPU_ARCH ?= sm_70
NVCCFLAGS = -std=c++11 -O2 -rdc=true \
            -Icpu -Icommon -Icpu/spongent -Icpu/keccak -Igpu -Igpu/spongent -Igpu/keccak \
            --generate-code arch=compute_$(subst sm_,,$(GPU_ARCH)),code=$(GPU_ARCH) \
            --generate-code arch=compute_$(subst sm_,,$(GPU_ARCH)),code=compute_$(subst sm_,,$(GPU_ARCH))

BUILD      = build
COMMON     = cpu/ggm_tree_cpu.c common/utils.c
COMMON_OBJS = $(BUILD)/ggm_tree_cpu.o $(BUILD)/utils.o
SPONG_CPU  = cpu/spongent/spongent.c cpu/spongent/spongent_prf.c
SPONG_GPU  = gpu/spongent/spongent_kernel.cu
SPONG_OBJS = $(BUILD)/spongent.o $(BUILD)/spongent_prf.o $(BUILD)/spongent_kernel.o
GGM_GPU    = gpu/ggm_tree_gpu.cu
BENCH      = bench/benchmark_ggm.cu
KECCAK_CPU = cpu/keccak/keccak_f1600.cu cpu/keccak/keccak_prf.cu
KECCAK_GPU = gpu/keccak/keccak_kernel.cu
KECCAK_OBJS = $(BUILD)/keccak_f1600.o $(BUILD)/keccak_prf.o $(BUILD)/keccak_kernel.o

.PHONY: all gpu_all test_ggm test_spongent test_ggm_gpu test_keccak benchmark clean

all: test_ggm test_spongent

gpu_all: test_ggm_gpu test_keccak

$(BUILD):
	mkdir -p $(BUILD)

# -----------------------------------------------------------------------
# CPU: GGM tree framework (Dummy PRF)
# -----------------------------------------------------------------------
$(BUILD)/test_ggm: tests/test_ggm.c $(COMMON) | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test_ggm: $(BUILD)/test_ggm
	./$(BUILD)/test_ggm

# -----------------------------------------------------------------------
# CPU: Spongent-128 + GGM tree
# -----------------------------------------------------------------------
$(BUILD)/test_spongent: tests/test_spongent.c $(COMMON) $(SPONG_CPU) | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test_spongent: $(BUILD)/test_spongent
	./$(BUILD)/test_spongent

# -----------------------------------------------------------------------
# GPU: Spongent + Keccak CPU vs GPU comparison
# -----------------------------------------------------------------------
$(BUILD)/ggm_tree_cpu.o: cpu/ggm_tree_cpu.c cpu/ggm_tree_cpu.h common/prf_interface.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/utils.o: common/utils.c common/utils.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/spongent.o: cpu/spongent/spongent.c cpu/spongent/spongent.cuh | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/spongent_prf.o: cpu/spongent/spongent_prf.c cpu/spongent/spongent_prf.cuh | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/spongent_kernel.o: $(SPONG_GPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/keccak_f1600.o: cpu/keccak/keccak_f1600.cu cpu/keccak/keccak_f1600.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/keccak_prf.o: cpu/keccak/keccak_prf.cu cpu/keccak/keccak_prf.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/keccak_kernel.o: $(KECCAK_GPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/ggm_tree_gpu.o: $(GGM_GPU) gpu/ggm_tree_gpu.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/test_ggm_gpu: tests/test_ggm_gpu.cu \
                        $(BUILD)/ggm_tree_gpu.o \
                        $(COMMON_OBJS) \
                        $(SPONG_OBJS) \
                        $(KECCAK_OBJS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_ggm_gpu: $(BUILD)/test_ggm_gpu
	./$(BUILD)/test_ggm_gpu

# -----------------------------------------------------------------------
# GPU: Keccak NIST KAT + 10k CPU/GPU equivalence
# -----------------------------------------------------------------------
$(BUILD)/test_keccak: tests/test_keccak.cu $(KECCAK_OBJS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_keccak: $(BUILD)/test_keccak
	./$(BUILD)/test_keccak

# -----------------------------------------------------------------------
# Benchmark: CPU/GPU throughput across depths
# -----------------------------------------------------------------------
$(BUILD)/benchmark_ggm: $(BENCH) \
                        $(BUILD)/ggm_tree_gpu.o \
                        $(COMMON_OBJS) \
                        $(SPONG_OBJS) \
                        $(KECCAK_OBJS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

benchmark: $(BUILD)/benchmark_ggm
	./$(BUILD)/benchmark_ggm

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD)
