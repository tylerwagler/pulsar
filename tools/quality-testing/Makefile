# The official-continuation scorer, linked against the engine's objects.
# `make -C tools/quality-testing` after a `make pulsar-bench` at the repo root.
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NATIVE_CPU_FLAG ?= -march=native
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lpng -ljpeg
ROOT := ../..
OBJS = $(ROOT)/src/engine/*.o $(ROOT)/src/cuda/*.o $(ROOT)/src/cuda/mmq/*.o $(ROOT)/src/lib/pulsar_help.o $(ROOT)/src/lib/pulsar_json.o $(ROOT)/src/tp/*.o $(ROOT)/src/vendor/*.o

score_official: score_official.cc
	$(MAKE) -C $(ROOT) pulsar-bench
	$(NVCC) $(NVCCFLAGS) -I$(ROOT)/src -o $@ score_official.cc $(OBJS) $(LDLIBS)

clean:
	rm -f score_official
.PHONY: clean
