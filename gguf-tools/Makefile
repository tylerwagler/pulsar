CC ?= cc
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt
# post-restructure: link against the split engine/cuda TUs (built via the
# pulsar-bench target, which forces every core object)
QUALITY_TARGETS := pulsar-bench
QUALITY_OBJS = ../src/engine/*.o ../src/cuda/*.o ../src/cuda/mmq/*.o ../src/lib/pulsar_help.o ../src/vendor/*.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I../src -o

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

QUANTIZE_SRCS := $(wildcard quantize/dsq_*.c)
QUANTIZE_OBJS := $(QUANTIZE_SRCS:.c=.o)
QUANTS_SRCS := quants_common.c quants_fp.c quants_kquants.c
QUANTS_OBJS := $(QUANTS_SRCS:.c=.o)

.PHONY: all clean quality-score test test-py test-cutlass-mxfp4 test-mxfp8-lt sfcmp

all: deepseek4-quantize

test_quants: test_quants.c $(QUANTS_OBJS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -o $@ test_quants.c $(QUANTS_OBJS) -lm -pthread

# test also rebuilds the production binary: a stale deepseek4-quantize
# once shipped sign-stripped MXFP4 donors while test_quants (rebuilt) passed.
test: test_quants deepseek4-quantize
	./test_quants

# Self-tests for the python maintenance tools (reflow_artifact.py and
# splice_bf16_plain.py). Kept out of `test` because they need the HF source
# checkpoint on disk; reflow's F32 codec path also needs numpy, so prefer the
# quant venv when it exists.
PYTEST_PYTHON ?= $(shell test -x $(HOME)/.venvs/pulsar-quant/bin/python && \
	echo $(HOME)/.venvs/pulsar-quant/bin/python || echo python3)
test-py:
	$(PYTEST_PYTHON) test_reflow_artifact.py
	$(PYTEST_PYTHON) test_splice_bf16.py

# Byte-compare the quantizer's native cutlass_mxfp4 (type 40) output against
# a reference GGUF whose type-40 tensors were produced by the validated
# CUTLASS packer (the retired converter/splice_cutlass_mxfp4.py pipeline).
# Needs the QAT HF snapshot and a reference GGUF on disk; override the paths
# as needed. Checks one gate and one down tensor (different N/K orientation).
DSV4_HF ?= $(HOME)/models/dsv4-flash-dspark
CUTLASS_REF_GGUF ?= ../gguf/dspark.gguf
test-cutlass-mxfp4: deepseek4-quantize
	./deepseek4-quantize --hf $(DSV4_HF) --template $(CUTLASS_REF_GGUF) \
	    --compare-tensor dspark.0.ffn_gate_exps.weight | tee /tmp/dsq-t40-gate.txt
	grep -q '^byte_compare: OK' /tmp/dsq-t40-gate.txt
	./deepseek4-quantize --hf $(DSV4_HF) --template $(CUTLASS_REF_GGUF) \
	    --compare-tensor dspark.0.ffn_down_exps.weight | tee /tmp/dsq-t40-down.txt
	grep -q '^byte_compare: OK' /tmp/dsq-t40-down.txt
	@echo "cutlass_mxfp4 byte-compare: OK"

# Prove the native MXFP8_LT (type 41) packer is byte-identical to the numpy
# reference repack (tools/mxfp8_prestore/repack_mxfp8_lt.py) that the retired
# post-hoc bandaid used. Self-contained: synthesizes a type-38 workhorse block,
# packs it both ways, and diffs. No model, no GPU.
quantize/tests/mxfp8_lt_emit: quantize/tests/mxfp8_lt_emit.c $(QUANTS_OBJS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -o $@ quantize/tests/mxfp8_lt_emit.c $(QUANTS_OBJS) -lm -pthread

test-mxfp8-lt: quantize/tests/mxfp8_lt_emit
	./quantize/tests/mxfp8_lt_emit 256 128 /tmp/mxfp8lt-38.bin /tmp/mxfp8lt-c.bin
	python3 quantize/tests/mxfp8_lt_ref.py 256 128 /tmp/mxfp8lt-38.bin /tmp/mxfp8lt-py.bin
	cmp /tmp/mxfp8lt-c.bin /tmp/mxfp8lt-py.bin
	./quantize/tests/mxfp8_lt_emit 1024 256 /tmp/mxfp8lt-38b.bin /tmp/mxfp8lt-cb.bin
	python3 quantize/tests/mxfp8_lt_ref.py 1024 256 /tmp/mxfp8lt-38b.bin /tmp/mxfp8lt-pyb.bin
	cmp /tmp/mxfp8lt-cb.bin /tmp/mxfp8lt-pyb.bin
	@echo "mxfp8_lt byte-compare: OK"

# IQ2_XXS (16) -> IQ2_XXS_MMQ (43) permutation proof. Same shape of test as
# test-mxfp8-lt: the C packer and an independently-written Python reference are
# run over identical raw bytes and diffed. 6291456 is the shipped per-tensor
# block count (192 experts x 2048 x 4096 / 256).
test-iq2-mmq: quantize/tests/iq2_mmq_emit
	./quantize/tests/iq2_mmq_emit 32 /tmp/iq2mmq-16.bin /tmp/iq2mmq-c.bin
	python3 quantize/tests/iq2_mmq_ref.py 32 /tmp/iq2mmq-16.bin /tmp/iq2mmq-py.bin
	cmp /tmp/iq2mmq-c.bin /tmp/iq2mmq-py.bin
	./quantize/tests/iq2_mmq_emit 8192 /tmp/iq2mmq-16b.bin /tmp/iq2mmq-cb.bin
	python3 quantize/tests/iq2_mmq_ref.py 8192 /tmp/iq2mmq-16b.bin /tmp/iq2mmq-pyb.bin
	cmp /tmp/iq2mmq-cb.bin /tmp/iq2mmq-pyb.bin
	@echo "iq2_xxs_mmq byte-compare: OK"

quantize/tests/iq2_mmq_emit: quantize/tests/iq2_mmq_emit.c $(QUANTS_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm -pthread

# CUTLASS Sm120 block-scaled SFB/SFA layout proof (the artifact that pins the
# swizzle to CUTLASS's own tile_atom_to_shape_SFB). Needs nvcc + the CUTLASS
# pin; override CUTLASS_DIR if it lives elsewhere. Host-only run (no kernel).
CUTLASS_DIR ?= $(abspath ..)/cutlass
CUTLASS_INC ?= -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
sfcmp: quantize/tests/sfcmp.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
	    $(CUTLASS_INC) -o quantize/tests/sfcmp quantize/tests/sfcmp.cu
	@echo "built quantize/tests/sfcmp (run it to dump the SFB/SFA layouts)"

quantize/%.o: quantize/%.c quantize/dsq_internal.h quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -Iquantize -c -o $@ $<

%.o: %.c quants_internal.h quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -c -o $@ $<

deepseek4-quantize: $(QUANTIZE_OBJS) $(QUANTS_OBJS)
	$(CC) $(CFLAGS) -o $@ $(QUANTIZE_OBJS) $(QUANTS_OBJS) -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.cc $(QUALITY_OBJS) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize test_quants quality-testing/score_official quantize/*.o *.o
