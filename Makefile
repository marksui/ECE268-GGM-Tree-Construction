# Makefile — GGM PRF Tree Project (EC268)
#
# CPU tests (gcc, no CUDA needed):
#   make all             run both CPU test suites
#   make test_ggm        GGM tree framework (Dummy PRF)
#   make test_spongent   Spongent-128 + GGM tree
#
# GPU builds (nvcc required):
#   make test_ggm_gpu    Spongent CPU vs GPU comparison test
#   make test_keccak     Keccak NIST KAT + 10k CPU/GPU equivalence test
#   make gpu_all         build all GPU test binaries
#
# Override GPU arch (default sm_70 = Datahub V100):
#   make gpu_all GPU_ARCH=sm_80

CC        = gcc
NVCC      = nvcc
CFLAGS    = -std=c11 -Wall -Wextra -Wpedantic -O2 -Icommon -Ispongent -Ikeccak

GPU_ARCH ?= sm_70
NVCCFLAGS = -std=c++11 -O2 -Icommon -Ispongent -Ikeccak -Igpu \
            --generate-code arch=compute_$(subst sm_,,$(GPU_ARCH)),code=$(GPU_ARCH)

BUILD      = build
COMMON     = common/ggm_tree.c common/utils.c
SPONG_CPU  = spongent/spongent.c spongent/spongent_prf.c
SPONG_GPU  = spongent/spongent_kernel.cu
GGM_GPU    = gpu/ggm_tree_gpu.cu
KECCAK_SRC = keccak/keccak_f1600.cu keccak/keccak_prf.cu

.PHONY: all gpu_all test_ggm test_spongent test_ggm_gpu test_keccak clean

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
# GPU: Spongent CPU vs GPU comparison (Member B + C)
# -----------------------------------------------------------------------
$(BUILD)/spongent_gpu.o: $(SPONG_GPU) $(SPONG_CPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

$(BUILD)/ggm_tree_gpu.o: $(GGM_GPU) gpu/ggm_tree_gpu.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD)/test_ggm_gpu: tests/test_ggm_gpu.cu \
                        $(BUILD)/ggm_tree_gpu.o \
                        $(BUILD)/spongent_gpu.o \
                        $(COMMON) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_ggm_gpu: $(BUILD)/test_ggm_gpu
	./$(BUILD)/test_ggm_gpu

# -----------------------------------------------------------------------
# GPU: Keccak NIST KAT + 10k CPU/GPU equivalence (Member A)
# -----------------------------------------------------------------------
$(BUILD)/test_keccak: tests/test_keccak.cu $(KECCAK_SRC) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_keccak: $(BUILD)/test_keccak
	./$(BUILD)/test_keccak

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD)
