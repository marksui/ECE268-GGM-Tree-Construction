# Makefile — GGM PRF Tree Project (EC268)
#
# CPU tests (gcc, no CUDA needed):
#   make test_ggm        GGM tree framework — Dummy PRF, no PRF deps
#   make test_spongent   Spongent-128 + GGM tree
#
# GPU build (nvcc required):
#   make spongent_gpu    Compile spongent .cu files to a GPU-ready object
#
# Member A:
#   make test_keccak     Uncomment block below once keccak/ is implemented

CC     = gcc
NVCC   = nvcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O2 -Icommon -Ispongent -Ikeccak

# nvcc flags: -x cu forces CUDA compilation, -dc for device-code linking
NVCCFLAGS = -std=c++11 -O2 -Icommon -Ispongent -Ikeccak \
            --generate-code arch=compute_70,code=sm_70

BUILD  = build
COMMON = common/ggm_tree.c common/utils.c
SPONG_CPU = spongent/spongent.c spongent/spongent_prf.c
SPONG_GPU = spongent/spongent_kernel.cu
GGM_GPU = gpu/ggm_tree_gpu.cu

.PHONY: all test_ggm test_spongent spongent_gpu ggm_gpu test_ggm_gpu clean

all: test_ggm test_spongent

$(BUILD):
	mkdir -p $(BUILD)

# -----------------------------------------------------------------------
# GGM tree tests — pure C, no PRF dependency
# -----------------------------------------------------------------------
$(BUILD)/test_ggm: tests/test_ggm.c $(COMMON) | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test_ggm: $(BUILD)/test_ggm
	./$(BUILD)/test_ggm

# -----------------------------------------------------------------------
# Spongent tests — compiled with gcc (CUDA shim strips __host__/__device__)
# -----------------------------------------------------------------------
$(BUILD)/test_spongent: tests/test_spongent.c $(COMMON) $(SPONG_CPU) | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test_spongent: $(BUILD)/test_spongent
	./$(BUILD)/test_spongent

# -----------------------------------------------------------------------
# Spongent GPU object (nvcc) — Member C links this into the GPU tree binary
# -----------------------------------------------------------------------
$(BUILD)/spongent_gpu.o: $(SPONG_GPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

spongent_gpu: $(BUILD)/spongent_gpu.o
	@echo "Spongent GPU object built: $(BUILD)/spongent_gpu.o"

$(BUILD)/ggm_tree_gpu.o: $(GGM_GPU) gpu/ggm_tree_gpu.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

ggm_gpu: $(BUILD)/ggm_tree_gpu.o $(BUILD)/spongent_gpu.o
	@echo "GGM GPU objects built in $(BUILD)/"

$(BUILD)/test_ggm_gpu: tests/test_ggm_gpu.cu $(BUILD)/ggm_tree_gpu.o $(BUILD)/spongent_gpu.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_ggm_gpu: $(BUILD)/test_ggm_gpu
	./$(BUILD)/test_ggm_gpu

# -----------------------------------------------------------------------
# Keccak tests — uncomment once keccak/ is implemented (Member A)
# -----------------------------------------------------------------------
# KECCAK = keccak/keccak_f1600.cu keccak/keccak_prf.cu
# $(BUILD)/test_keccak: tests/test_keccak.c $(COMMON) $(KECCAK) | $(BUILD)
# 	$(CC) $(CFLAGS) $^ -o $@
# test_keccak: $(BUILD)/test_keccak
# 	./$(BUILD)/test_keccak

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD)
