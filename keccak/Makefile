# Makefile — GGM PRF Tree Project (EC268)

CC     = gcc
NVCC   = nvcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O2 -Icommon -Ispongent -Ikeccak

# Compile for 1080 Ti (61), Volta (70), Turing/2080 Ti (75), and Ampere/3090 (80/86)
NVCCFLAGS = -std=c++11 -O2 -Icommon -Ispongent -Ikeccak -x cu -rdc=true \
            -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_86,code=compute_86

BUILD  = build

COMMON = 
SPONG_CPU = spongent/spongent.c spongent/spongent_prf.c
SPONG_GPU = spongent/spongent_kernel.cu
KECCAK_CPU = keccak/keccak_f1600.c keccak/keccak_prf.c
KECCAK_GPU = keccak/keccak_kernel.cu

.PHONY: all test_spongent spongent_gpu test_keccak clean

all: test_keccak

$(BUILD):
	mkdir -p $(BUILD)

# -----------------------------------------------------------------------
# Spongent Targets
# -----------------------------------------------------------------------
$(BUILD)/test_spongent: tests/test_spongent.c $(COMMON) $(SPONG_CPU) | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test_spongent: $(BUILD)/test_spongent
	./$(BUILD)/test_spongent

$(BUILD)/spongent_gpu.o: $(SPONG_GPU) $(SPONG_CPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

spongent_gpu: $(BUILD)/spongent_gpu.o
	@echo "Spongent GPU object built: $(BUILD)/spongent_gpu.o"

# -----------------------------------------------------------------------
# Keccak Targets 
# -----------------------------------------------------------------------
$(BUILD)/keccak_gpu.o: $(KECCAK_GPU) $(KECCAK_CPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

$(BUILD)/test_keccak: tests/test_keccak.cu $(COMMON) $(KECCAK_CPU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_keccak: $(BUILD)/test_keccak
	./$(BUILD)/test_keccak

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD) tests/run_keccak_tests
