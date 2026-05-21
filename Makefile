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

.PHONY: all test_ggm test_spongent spongent_gpu clean

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
$(BUILD)/spongent_gpu.o: $(SPONG_GPU) $(SPONG_CPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

spongent_gpu: $(BUILD)/spongent_gpu.o
	@echo "Spongent GPU object built: $(BUILD)/spongent_gpu.o"

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
