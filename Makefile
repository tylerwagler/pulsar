CC ?= cc
NATIVE_CPU_FLAG ?= -march=native

DEBUG_FLAGS ?= -g
# Header dependency generation; see the ALL_OBJS/PULSAR_DEPS block below for why.
PULSAR_DEPFLAGS := -MMD -MP

CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only

# C++ port (pulsar): host TUs migrate .c -> .cpp one at a time. The FP and
# optimization flags MUST stay identical to CFLAGS so a ported TU generates
# the same math as its C predecessor (the per-TU bit-exact gate depends on
# it). -fno-exceptions/-fno-rtti is the project style: hot paths never throw,
# fatal errors keep the pulsar_die() contract.
CXX ?= g++
CXXFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c++23
CXXFLAGS += -D_GNU_SOURCE -fno-finite-math-only -fno-exceptions -fno-rtti
# Partial designated initializers (fields filled right after, or {0} zero-init)
# are a pervasive idiom here; C's -Wextra accepts them silently, C++'s warns.
# Keep the warning surface identical to the C build.
CXXFLAGS += -Wno-missing-field-initializers
CXXFLAGS += -DPULSAR_VERSION_STR='"$(PULSAR_VERSION_STR)"'
CXXFLAGS += $(PULSAR_DEPFLAGS)

# Version string reported by /version, /health and the startup banner. Derived
# from git so it never goes stale (e.g. "v0.2.3-8-gec51fb2", "-dirty" if the
# tree has uncommitted changes); falls back to "unknown" outside a git checkout.
PULSAR_VERSION_STR := $(shell git describe --tags --dirty --always 2>/dev/null || echo unknown)
CFLAGS += -DPULSAR_VERSION_STR='"$(PULSAR_VERSION_STR)"'
CFLAGS += $(PULSAR_DEPFLAGS)

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math --default-stream per-thread $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
NVCCFLAGS += $(PULSAR_DEPFLAGS)
# Append-only hook for probe builds (e.g. L003's tile sweep:
#   make pulsar NVCC_EXTRA='-DPULSAR_MXFP4_TILE_M=64')
# Deliberately APPEND rather than let a caller override NVCCFLAGS wholesale:
# overriding it silently drops --use_fast_math and -arch, which would make a
# probe measure a different compiler configuration than the one that ships.
override NVCCFLAGS += $(NVCC_EXTRA)

# HC_F32=1 (-DPULSAR_HC_F32) used to restore f32 residual carriers: the control
# build that proved the BF16 storage narrowing was a pure no-op. That flip
# shipped and nothing set the switch afterwards -- no target, no gate, no test,
# nothing at all (its last mention, a comment in the since-deleted
# attn_indexed_bench.cu, went with that file; L106 K7) -- so it went on 2026-08-17.
# The carriers are BF16, full stop, and pulsar_hc_t has one definition.

# An object is stale when the FLAGS that produced it change, not only when a
# source does -- and nothing in make's dependency graph sees CUDA_ARCH.  That
# gap is how an sm_75 object silently linked into an sm_120f binary
# (2026-08-21): make saw every .o newer than its .cu and did nothing, and the
# only recovery was `rm src/cuda/*.o` by hand.  It is also why the gates reach
# for `-B`: a blunt rebuild-everything, because there was no way to say
# "rebuild if the compiler configuration moved".
#
# This stamp says it.  It holds the full nvcc invocation signature, so arch,
# --use_fast_math and NVCC_EXTRA probe flags all invalidate objects; it is
# rewritten ONLY when that signature actually changes, so it does not churn
# timestamps on every make.  Deliberately NOT applied to the host compiler:
# CXXFLAGS carries PULSAR_VERSION_STR, which moves every commit and would
# rebuild the world for nothing.
CUDA_FLAG_STAMP := .build/cuda-flags.stamp
CUDA_FLAG_SIG   := $(NVCC) $(NVCCFLAGS)
$(shell mkdir -p $(dir $(CUDA_FLAG_STAMP)) 2>/dev/null; \
        [ "$$(cat $(CUDA_FLAG_STAMP) 2>/dev/null)" = "$(CUDA_FLAG_SIG)" ] || \
        printf '%s' "$(CUDA_FLAG_SIG)" > $(CUDA_FLAG_STAMP))

# CUTLASS: EXTERNAL, header-only, and deliberately NOT a git submodule -- see
# cutlass.pin for why (a symlinked submodule path made `git checkout` fail or
# silently clobber, which cost three gate runs and a bad dev push on
# 2026-08-23).  Override CUTLASS_DIR to point anywhere; `make cutlass` clones
# the pinned sha if you have nothing.
CUTLASS_DIR ?= $(CURDIR)/cutlass
CUTLASS_INC ?= -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_PIN_SHA := $(shell sed -n 's/^sha=//p' $(CURDIR)/cutlass.pin 2>/dev/null)
CUTLASS_PIN_URL := $(shell sed -n 's/^url=//p' $(CURDIR)/cutlass.pin 2>/dev/null)

# Fail EARLY and by name.  Without this the failure is a wall of
# "cutlass/cutlass.h: No such file" from the middle of a parallel build, which
# is how an empty cutlass/ directory went unnoticed repeatedly.
.PHONY: cutlass-check cutlass
cutlass-check:
	@[ -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" ] || { \
	  echo "ERROR: CUTLASS headers not found at CUTLASS_DIR=$(CUTLASS_DIR)"; \
	  echo "       expected $(CUTLASS_DIR)/include/cutlass/cutlass.h"; \
	  echo "       run 'make cutlass', or set CUTLASS_DIR=/path/to/cutlass"; \
	  echo "       pinned sha: $(CUTLASS_PIN_SHA)"; exit 1; }
	@if [ -e "$(CUTLASS_DIR)/.git" ] && [ -n "$(CUTLASS_PIN_SHA)" ]; then \
	  have=$$(git -C "$(CUTLASS_DIR)" rev-parse HEAD 2>/dev/null); \
	  [ "$$have" = "$(CUTLASS_PIN_SHA)" ] || \
	    echo "WARNING: CUTLASS at $(CUTLASS_DIR) is $$have, pin says $(CUTLASS_PIN_SHA)"; \
	fi

# Clone/checkout the pinned CUTLASS into CUTLASS_DIR (default ./cutlass).
cutlass:
	@if [ ! -e "$(CUTLASS_DIR)/.git" ]; then \
	  git clone --filter=blob:none "$(CUTLASS_PIN_URL)" "$(CUTLASS_DIR)"; fi
	@git -C "$(CUTLASS_DIR)" fetch --depth=1 origin "$(CUTLASS_PIN_SHA)" 2>/dev/null || \
	 git -C "$(CUTLASS_DIR)" fetch origin
	@git -C "$(CUTLASS_DIR)" checkout -q "$(CUTLASS_PIN_SHA)"
	@echo "CUTLASS at $(CUTLASS_DIR) -> $(CUTLASS_PIN_SHA)"
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt

PULSAR_INC = -Isrc -Isrc/lib -Isrc/vendor

ENGINE_SRCS = $(wildcard src/engine/*.c) $(wildcard src/engine/*.cpp)
ENGINE_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(ENGINE_SRCS)))
AGENT_SRCS = $(wildcard src/agent/*.c) $(wildcard src/agent/*.cpp)
AGENT_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(AGENT_SRCS)))
SERVER_SRCS = $(wildcard src/server/*.c) $(wildcard src/server/*.cpp)
SERVER_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SERVER_SRCS)))
# CUTLASS TUs need the CUTLASS include path + c++17; they build via dedicated rules below,
# so keep them out of the generic src/cuda/%.o rule.
CUTLASS_CUDA_OBJS = src/cuda/pulsar_mxfp4_cutlass.o
CUDA_SRCS = $(filter-out src/cuda/pulsar_mxfp4_cutlass.cu,$(wildcard src/cuda/*.cu))
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
# Vendored llama.cpp MMQ (plan 41b, see src/cuda/mmq/VENDOR.md).  The wildcard is
# empty unless the tree has actually been vendored, in which case PULSAR_HAVE_MMQ
# lights up the routed gate/up MMQ arm in pulsar_cuda_moe.cu.  No tree -> stock build.
# mmid.cu is deliberately NOT compiled on its own: ds4_mmid.cu #includes it so
# that upstream's file-static launch_mm_ids_helper<N> is in scope (L008).
# Compiling both would duplicate every symbol in it.
MMQ_SRCS = $(filter-out src/cuda/mmq/mmid.cu,$(wildcard src/cuda/mmq/*.cu))
MMQ_OBJS = $(MMQ_SRCS:.cu=.o)
ifneq ($(strip $(MMQ_SRCS)),)
MMQ_CPPFLAGS = -DPULSAR_HAVE_MMQ -Isrc/cuda/mmq
endif
LIB_HDRS = src/lib/pulsar_help.h src/lib/pulsar_kvstore.h
CORE_OBJS = $(ENGINE_OBJS) $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)

# ---------------------------------------------------------------------------
# AUTOMATIC HEADER DEPENDENCIES  (-MMD -MP)
#
# Every rule below used to hand-list its header prerequisites, and the tree paid
# for that three separate times on 2026-08-18 alone:
#   * five tests were stale against a KV format change, the worst of them feeding
#     f32 rows to a packed reader -- 598016 NaNs the instant it was rebuilt, days
#     late, and only because deleting the vendored MMQ headers forced a rebuild;
#   * an attn fixture had been degenerate for days behind those stale objects;
#   * deleting common.cuh needed a manual `make clean`, because a hand-listed
#     prerequisite cannot express "this header no longer exists".
# The note on the prefill gate below PREDICTED this in as many words -- "make
# would then see moe.o as up to date after a .cuh-only edit, relink the OLD
# kernel, and print PASS -- the gate would certify a kernel it never compiled" --
# and called project-wide -MMD -MP the real fix, deferring it because this
# Makefile was shared with a parallel branch.  That branch (flashinfer-attn) is
# retired, so the reason is gone and the fix lands.
#
# -MMD writes a .d beside each .o listing the headers it actually opened
# (skipping system headers); -MP adds a phony target per header so a DELETED
# header does not wedge the build with "No rule to make target".  nvcc accepts
# both directly and, given -o, names the target with the object's full path --
# checked, not assumed.
#
# Hand-listed prerequisites are kept underneath as a floor, not removed: they
# cost nothing, and on a first build (no .d yet) they are all make has.
ALL_OBJS = $(CORE_OBJS) $(AGENT_OBJS) $(SERVER_OBJS) \
           $(patsubst %.cpp,%.o,$(wildcard src/cli/*.cpp)) \
           $(patsubst %.c,%.o,$(wildcard src/cli/*.c)) \
           $(patsubst %.cpp,%.o,$(wildcard src/lib/*.cpp)) \
           $(patsubst %.c,%.o,$(wildcard src/lib/*.c)) \
           $(patsubst %.cpp,%.o,$(wildcard src/vendor/*.cpp)) \
           $(patsubst %.cpp,%.o,$(wildcard tests/*.cpp))
PULSAR_DEPS = $(ALL_OBJS:.o=.d)
PULSAR_LINK ?= $(NVCC) $(NVCCFLAGS)
PULSAR_LINK_LIBS ?= $(CUDA_LDLIBS)

# A failed recipe must not leave a half-written target: a partially-linked
# test binary from a failed build was later executed by a gate run as if it
# were current (make compares mtimes, not build success -- 2026-08-19).
.DELETE_ON_ERROR:

.PHONY: gates gates-quick cuda-spec-width-gate all help clean test seam-check cuda-spark cuda-regression cuda-attn-gates cuda-frontier-gate cuda-multiseq-gate cuda-multiseq-gate-nodspark cuda-bank-spec-gate cuda-accounting-gate cuda-evict-restore-gate cuda-fork-gate cuda-session-payload-gate cuda-algo-stability-gate cuda-mixed-prefill-gate cuda-mixed-neutrality-gate cuda-mixed-neutrality-gate-wide cuda-prefill-gate cuda-prefill-gate-baseline cuda-spec-sampling-gate warm-fork-3way warm-partial-fork-3way sse-decode-bench decode-floor-gate decode-floor-baseline context-coherence-probe

all: help

help:
	@echo "Pulsar build targets (CUDA-only fork for DGX Spark / GB10):"
	@echo "  make cuda-spark          Build for the DGX Spark / GB10 (sm_120f)"
	@echo "  make test                Build and run tests"
	@echo "  make cuda-regression     Kernel smokes vs synthetic slabs (modelless)"
	@echo "  make cuda-attn-gates     fp16 attention correctness gates: kernel oracle,"
	@echo "                           banked KV-leak isolation, split-KV merge (modelless)"
	@echo "  make cuda-frontier-gate  Multiseq frontier-isolation gate (needs the model;"
	@echo "                           FRONTIER_MODEL=./ds4flash.gguf by default)"
	@echo "  make cuda-multiseq-gate  Multiseq-vs-solo token-stream gate + aggregate"
	@echo "                           throughput at N=1..3 (needs the model)"
	@echo "  make cuda-multiseq-gate-nodspark"
	@echo "                           The same gate with speculation disabled"
	@echo "                           (--no-dspark config; needs the model)"
	@echo "  make decode-floor-gate   5-workload decode floor, scored on the WORST"
	@echo "                           workload (needs a RUNNING server; see"
	@echo "                           make decode-floor-baseline to record the floor)"
	@echo "  make sse-decode-bench    Decode rate off the SSE stream, alongside"
	@echo "                           wall-clock (needs a RUNNING server)"
	@echo "  make context-coherence-probe"
	@echo "                           Begin/middle/end fact recall at depth, free"
	@echo "                           prose + checksum (needs a RUNNING server)"
	@echo "  make cuda-prefill-gate   Prefill bit-exactness gate: full-vocab frontier"
	@echo "                           logits byte-compared against a baseline build"
	@echo "                           (needs the model + a baseline blob)"
	@echo "  make cuda-prefill-gate-baseline"
	@echo "                           Build the baseline blob from PREFILL_BASELINE_REF"
	@echo "                           in a git worktree (needs the model)"
	@echo "  make cuda-spec-sampling-gate"
	@echo "                           Speculative-sampling chi-square exactness"
	@echo "                           oracle + acceptance alpha (needs the model)"
	@echo "  make gates               Run EVERY release-blocking gate and print a"
	@echo "                           pass/fail summary (needs the GB10 + model:"
	@echo "                           make gates FRONTIER_MODEL=/srv/models/x.gguf)"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B pulsar-server CUDA_ARCH=sm_120f

pulsar-server: $(SERVER_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

# Development tools, not part of the shipped release. The release is just
# pulsar-server (the HTTP API). Source is kept; build these by name.
pulsar: src/cli/pulsar_cli.o src/lib/pulsar_help.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-bench: src/cli/pulsar_bench.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-eval: src/cli/pulsar_eval.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-agent: $(AGENT_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# fp16 attention correctness gates (standalone .cu, no model needed):
# kernel-vs-f64 oracle, banked cross-sequence isolation (the KV-leak oracle
# for the default-on fp16 tier), and split-KV decode merge vs the single-walk
# golden across every staging branch.  Each file's header documents its scope;
# these were previously build-by-hand only.
# ARCH MUST BE PINNED HERE, and the failure mode if it is not is a LIE rather
# than an error.  pulsar_cuda_attn_f16.cu compiles its whole MMA body behind
# `__CUDA_ARCH__ >= 800`; below that the kernel is a no-op stub that voids its
# arguments and writes nothing.  NVCC_ARCH_FLAGS is EMPTY unless CUDA_ARCH is
# passed (see the top of this file), so these three built for nvcc's DEFAULT
# arch, the stub ran, and the oracle graded a kernel that never executed --
# reporting FAIL with `never-written = 655360` and every value at the -12345
# fill sentinel.  Measured 2026-08-15 on clean cf33e3e: FAIL without the flag,
# PASS (never-written = 0, rel L2 5.2e-04) with it, i.e. the gate had been
# red for reasons that had nothing to do with the code under test.
# Presumed introduced by the sparky nvcc 13.0 -> 13.3 bump moving the default
# arch; pin it so the default can never decide whether a gate is honest.
ATTN_GATE_ARCH ?= sm_120f

# ⚠ THESE THREE #include A SHIPPED .cu AND MUST DEPEND ON IT.
#
# Each of these tests compiles one .cu that pulls in the real kernel source, so
# the engine is genuinely a prerequisite -- and until 2026-08-18 the rules said
# only "<test>.cu Makefile".  Every kernel change since the KV format was
# unified therefore left them UP TO DATE, and all three sat stale for days:
# attn_f16_kernel_test was feeding f32 rows to a packed reader and would have
# reported 598016 NaNs the moment it was rebuilt.  It took deleting the vendored
# MMQ headers to force that rebuild.
#
# This is the failure the -B note further down predicts almost word for word --
# "make would then see moe.o as up to date ... and print PASS -- the gate would
# certify a kernel it never compiled" -- arriving on the TEST side rather than
# the engine side.
#
# ✅ Project-wide -MMD -MP landed 2026-08-18, so these lists are no longer what
# keeps these three correct -- the generated .d files are.  They are KEPT as the
# floor: on a first build, or immediately after `make clean`, no .d exists yet
# and a hand-listed prerequisite is all make has.
tests/attn_f16_kernel_test: tests/attn_f16_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_attn_f16.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/attn_pack_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

tests/attn_f16_banked_test: tests/attn_f16_banked_test.cu Makefile \
                            src/cuda/pulsar_cuda_attn_f16.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/attn_pack_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

tests/attn_decode_split_test: tests/attn_decode_split_test.cu Makefile \
                            src/cuda/pulsar_cuda_attention.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/attn_pack_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# attn_f16_kernel_test takes [n_tokens window n_head bench n_comp ratio top_k
# raw_cap] and its own header argues the compressed and indexed halves matter --
# "wiring the kernel in against only the n_comp=0 path would have shipped that
# half untested" -- but the gate ran the default (n_comp=0) shape ONLY, so both
# halves were untested exactly as that note warned.  Running them found a stale
# oracle: it clamped an out-of-visible top-k selection to row 0, mirroring the
# f32 kernel, while the f16 kernel masks the row to -INF (row 0 substitution
# double-counts row 0 whenever row 0 was also legitimately selected).  Every
# top_k>0 shape disagreed by ~8e-1 and nothing was running to notice.
# attn_f16_banked_test took a "p" argument selecting ATTN_PACK comp banks over
# f32 ones; the comp format parameter is gone from the kernels (2026-08-18), so
# there is one mode and one invocation.
cuda-attn-gates: tests/attn_f16_kernel_test tests/attn_f16_banked_test tests/attn_decode_split_test
	./tests/attn_f16_kernel_test
	./tests/attn_f16_kernel_test 40 24 32 x 8 4          # compressed tail
	./tests/attn_f16_kernel_test 40 24 32 x 8 4 3        # indexed top-k selection
	./tests/attn_f16_kernel_test 48 16 32 x 12 4 5 20    # indexed + ring raw rows
	./tests/attn_f16_kernel_test 48 16 32 x 12 4 0 20    # decode-batch, no topk table
	./tests/attn_f16_banked_test
	./tests/attn_decode_split_test

# Backend-seam enforcement (see the contract atop src/pulsar_gpu.h): nothing
# outside src/cuda/ may touch CUDA APIs directly. tools/seam_check.py strips
# comments/strings first, so docs may explain the backend; code may not call it.
seam-check:
	@python3 tools/seam_check.py

# Multiseq frontier-isolation gate (engine-side wrong-bank wiring; see the
# header of tests/multiseq_frontier_gate.c).  MODEL-DEPENDENT — run manually
# on the GB10, not part of `make test`.  Discipline before running: no
# pulsar-server/pulsar_test process, `sync; echo 3 > /proc/sys/vm/drop_caches`,
# check `free -g` headroom (the model is ~87 GB).
FRONTIER_MODEL ?= ./ds4flash.gguf
cuda-frontier-gate: tests/multiseq_frontier_gate
	PULSAR_MSEQ_BANKS=2 ./tests/multiseq_frontier_gate $(FRONTIER_MODEL)

# Multiseq-vs-solo token-stream gate + first aggregate-throughput measurement
# (see the header of tests/multiseq_decode_gate.c).  MODEL-DEPENDENT — run
# manually on the GB10 with the same memory discipline as the frontier gate.
cuda-multiseq-gate: tests/multiseq_decode_gate
	PULSAR_MSEQ_BANKS=3 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 3 512

# The same gate with speculation DISABLED — the pulsar-bench/pulsar-eval/agent and
# `pulsar-server --no-dspark` config, and a different allocation shape (no DSpark
# graph state).  The driver must work there; it used to reject every step.
# Shorter (N=2, 64 steps): this is a config gate, not a throughput run.
cuda-multiseq-gate-nodspark: tests/multiseq_decode_gate
	PULSAR_MSEQ_BANKS=2 PULSAR_GATE_NO_DSPARK=1 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 2 64

# Tier-2 PATH A / Option F: fused DSpark speculation on a BANK (N=1 spec-on-
# bank == classic, N=2 spec-time-slice with warm per-bank rings, mseq_dirty
# cheap resume).  See tests/bank_spec_gate.c.  MODEL-DEPENDENT, drafter-merged
# model, same memory discipline as the gates above; hold temp/gpu.lock.
cuda-bank-spec-gate: tests/bank_spec_gate
	PULSAR_MSEQ_BANKS=2 ./tests/bank_spec_gate $(FRONTIER_MODEL) 128

# Tier-2 overcommit accounting-exactness gate (task #55, increment 1): the
# exact-frontier touched-KV number the eviction guard triggers on must track the
# real cudaMemGetInfo physical delta of the demand-paged comp/index growth.  See
# tests/accounting_gate.c.  MODEL-DEPENDENT, same memory discipline as the gates
# above (hold temp/gpu.lock, drop_caches, no other pulsar process); fills stay
# modest (peak ~64k tokens) and do not exercise eviction.
cuda-accounting-gate: tests/accounting_gate
	PULSAR_MSEQ_BANKS=2 ./tests/accounting_gate $(FRONTIER_MODEL)

# Tier-2 increment 2b bank evict/restore bit-identity + reclaim gate (the
# memory-safety core; no OOM risk). See tests/bank_evict_restore_gate.c.
cuda-evict-restore-gate: tests/bank_evict_restore_gate tests/bank_fork_gate
	PULSAR_MSEQ_BANKS=2 ./tests/bank_evict_restore_gate $(FRONTIER_MODEL)

# Tier-2 PATH-A full-prefix fork gate (plan-33 inc A): fork==cold oracle. See
# tests/bank_fork_gate.c. MODEL-DEPENDENT, needs PULSAR_MSEQ_BANKS>=3.
cuda-fork-gate: tests/bank_fork_gate
	PULSAR_MSEQ_BANKS=3 ./tests/bank_fork_gate $(FRONTIER_MODEL)

# Session payload SAVE -> LOAD round trip. Nothing covered this before
# 2026-08-18, and the gap hid a raw-ring stride bug that read past the end of
# the ring allocation. Compares the comp caches byte-for-byte and then decodes
# one token from both sessions -- a byte-copy payload owes exact equality.
cuda-session-payload-gate: tests/session_payload_gate
	./tests/session_payload_gate $(FRONTIER_MODEL)

# plan-34 phase-2 inc 2: cuBLASLt algo-stability. A decode bank's step logits must
# be byte-identical across batched-step widths M (incl. the M=4->5 custom->cuBLASLt
# boundary) so a co-scheduled big prefill (inc 4) cannot perturb it. MODEL-DEPENDENT,
# needs PULSAR_MSEQ_BANKS>=8. Run pack on/off + idx-fp4 on/off under GPU discipline.
cuda-algo-stability-gate: tests/algo_stability_gate
	PULSAR_MSEQ_BANKS=8 ./tests/algo_stability_gate $(FRONTIER_MODEL)

# plan-34 phase-2 inc 3: K-row single-bank prefill through the mixed entry —
# coherence vs classic, K>ratio boundary, tensor-core speed. MODEL-DEPENDENT.
cuda-mixed-prefill-gate: tests/mixed_prefill_gate
	PULSAR_MSEQ_BANKS=2 ./tests/mixed_prefill_gate $(FRONTIER_MODEL)

# Artifact-only, no GPU, seconds: does the ROUTER agree with the artifact's own
# REAP declaration? A router left in source expert order against compacted
# expert weights routes every token to the wrong expert and NOTHING FAILS --
# in range, no NaN, still fluent, just missing most of what the model knows.
# srcfmt-v1 shipped exactly that on 40 of 43 layers. Cheap enough to run on
# every artifact before it is ever loaded.
cuda-reap-router-audit:
	python3 gguf-tools/reap/audit_reap_router.py $(FRONTIER_MODEL)

# plan-34 phase-2 inc 4: TRUE mixed step — decode banks + one K-row prefill run
# fused. Gate 4 co-scheduling neutrality (decode logits byte-identical with/without
# a co-scheduled prefill), gate 2 prefill correctness, gate 3 MoE two-pass split.
cuda-mixed-neutrality-gate: tests/mixed_neutrality_gate
	PULSAR_MSEQ_BANKS=3 ./tests/mixed_neutrality_gate $(FRONTIER_MODEL)

# Wide variant: 12 decode banks exercises the armed M-neutral kernel range past
# 8 (the l048-ntcap-16 coverage) — NT instantiations 9..16 and the MoE
# non-grouped path at those widths, same byte-identity demand as the 2-bank run.
cuda-mixed-neutrality-gate-wide: tests/mixed_neutrality_gate
	PULSAR_GATE_NDEC=12 PULSAR_MSEQ_BANKS=13 ./tests/mixed_neutrality_gate $(FRONTIER_MODEL)

# plan-33 inc B: 3-way output-equality harness (server-level; see the script).
warm-fork-3way: pulsar-server
	bash tests/warm_fork_3way.sh $(FRONTIER_MODEL)

# plan-33 inc D: partial-prefix fork 3-way output-equality harness (server-level).
warm-partial-fork-3way: pulsar-server
	bash tests/warm_partial_fork_3way.sh $(FRONTIER_MODEL)

# ---- client-side serving gates (stdlib Python; need a RUNNING pulsar-server) ----
# These measure what a client experiences, so they talk HTTP to an already-started
# server rather than linking the engine.  Start it with --no-kv-disk: a warm disk
# checkpoint skips prefill outright, which makes TTFT and wall-clock t/s
# incomparable between runs.  Override the endpoint with SERVE_HOST/SERVE_PORT.
SERVE_HOST ?= 127.0.0.1
SERVE_PORT ?= 8080

# Decode rate as the stream delivers it, reported alongside wall-clock.
sse-decode-bench:
	python3 tests/sse_decode_bench.py --host $(SERVE_HOST) --port $(SERVE_PORT) \
	  --kv-disk-state disabled

# Five workloads, 512 tokens, C1; scores the WORST one against a recorded floor.
# Capture the floor once on a known-good build:
#   make decode-floor-baseline
decode-floor-gate:
	python3 tests/decode_floor_gate.py --host $(SERVE_HOST) --port $(SERVE_PORT) \
	  --kv-disk-state disabled

decode-floor-baseline:
	python3 tests/decode_floor_gate.py --host $(SERVE_HOST) --port $(SERVE_PORT) \
	  --kv-disk-state disabled --write-baseline

# Facts planted at begin/middle/end of a long context, asked back in free prose
# (no grammar mask) plus a checksum that needs all three at once.
context-coherence-probe:
	python3 tests/context_coherence_probe.py --host $(SERVE_HOST) --port $(SERVE_PORT)

# Prefill bit-exactness gate (the D2R acceptance gate; see the header of
# tests/prefill_bitexact_gate.c).  MODEL-DEPENDENT — run manually on the GB10,
# not part of `make test`.  Discipline before running: no pulsar-server/pulsar_test
# process, `sync; echo 3 > /proc/sys/vm/drop_caches`, check `free -g` headroom
# (the model is ~87 GB).  The GPU is shared: hold temp/gpu.lock.
#
# Two-step, and the baseline is the slow half:
#   1. `make cuda-prefill-gate-baseline`  — ONCE per baseline ref.  Checks out
#      PREFILL_BASELINE_REF into a detached worktree, copies THIS tree's gate
#      source + Makefile into it (the harness must be identical; only the
#      engine/kernels may differ), builds there, and dumps the blob.
#      Since 2026-08-19 (L046) the blob is COMMITTED under tests/test-vectors/,
#      so step 1 is needed only when RE-ANCHORING — a fresh clone runs step 2
#      directly.  Caveat that killed the old bootstrap: copying THIS Makefile
#      into the ref's worktree only links if the ref's vendor/object layout is
#      compatible with it, so keep the anchor recent (it should be anyway: the
#      anchor is the last shipped numerics change).
#   2. `make cuda-prefill-gate`           — after every D2R increment.
# Each step loads the model once (~35 s) and prefills 2*(512+2048+4096) tokens.
# Re-baselined 2026-07-26 to the v0.3.1 shipped commit: the old 8aa9d35 blob
# predated the v0.2.3 type-40/MXFP8_LT model repack, so its header no longer
# matched the shipped ds4flash.gguf ("baseline header mismatch — different
# model") AND the 8aa9d35 engine cannot even run the repacked tensors. The
# baseline now protects drift from the current shipped line.
#
# Re-baselined again 2026-08-14 (L005), same failure mode one artifact later:
# 536466c predates type-43 (IQ2_XXS_MMQ), so the baseline build cannot load the
# shipped artifact and the gate could not run at all -- two releases went out
# with it silently rotted.
#
# WHY 3324bf4 SPECIFICALLY: it is the LAST DELIBERATE NUMERICS CHANGE, which is
# the only correct anchor for this gate.  "Earliest commit that can still load
# the artifact" is the intuitive choice and it is WRONG -- measured 2026-08-14:
# baselining at df3a0d8 (2026-08-08, the fp16-tier flip) makes the gate run but
# FAIL at every depth with all 129280 logits differing, because e09563e
# (2026-08-11) narrowed the ATTN_PACK rope tail from f32 to bf16 and 3324bf4
# fixed that loader the same day.  Both are shipped-on-purpose precision
# changes; a baseline older than them reports intended work as a regression and
# the gate becomes noise you learn to ignore -- which is how it rotted before.
#
# So: re-baseline whenever a numerics change SHIPS, and only then.  Between such
# changes the gate is a true bit-exactness assertion, which is what makes a FAIL
# worth acting on.
#
# MOVED 3324bf4 -> 26e7569 on 2026-08-15.  Sixteen numerics changes shipped
# between them and the gate was RED at every depth throughout, so it asserted
# nothing -- exactly the rot the note above warns about, arrived at from the
# other direction (not a too-old anchor, but a too-long deferral).  The shipped
# changes: MXFP8 expert activations as the only arm, the E4M3 norm epilogues,
# one comp-cache format each for attention (packed 584 B) and the indexer
# (MXFP4 68 B), --quality and its five slow paths gone, TF32 unconditional,
# and the n_tokens >= 128 attention floor removed at all three sites once the
# stale-gact bug it was hiding was fixed (26e7569 itself).
# MOVED 26e7569 -> a695c73 on 2026-08-16, for a reason the earlier moves did not
# have: the old anchor CANNOT LOAD the new artifact at all. 26e7569 predates BF16
# weight support, and the source-format migration puts token_embd, the output
# head, the norms, the compressors and the drafter markov head in BF16 -- so the
# baseline worktree dies at load and the gate cannot run, rather than reporting a
# difference. That is a structural block, not a rotted anchor.
#
# What ships with it (all deliberate numerics changes, per the policy above):
# every tensor stored in its SOURCE format -- bf16 where the checkpoint is bf16,
# f32 where it is f32, e4m3 where it is e4m3 -- which removes the last F16 tensor
# from the artifact. Plus the engine work that made it loadable and neutral: the
# BF16 weight path restored, the f32/bf16 matmul arms given the inc-4
# prefix/suffix split they never had (the mixed-neutrality break), and the fused
# HC norm+mix GEMV generalised off F16.
#
# Deliberately baselined BEFORE the F16 deletion sweep, not after: that sweep
# removes only paths the artifact can no longer reach, so it must be bit-exact,
# and anchoring here makes the gate PROVE that rather than assume it.
# MOVED a695c73 -> 06d0b3e on 2026-08-17, an ordinary application of the policy:
# numerics shipped, so the anchor moves. What shipped is the A8 campaign's decode
# half -- the five dense GEMVs (a6aafc2), then the attn-output 'a' and 'b'
# projections (960d439, 4e3aa31) -- which puts decode on E4M3 activations end to
# end, plus the REAP router repair. Against a695c73 the gate was red at every
# depth and so asserted nothing.
#
# ⚠ THE PRICE IS ON THE RECORD AND THE ANCHOR MOVES BACK IF IT IS PAID DOWN:
# that conversion costs -12.1% of SERVED decode (22.28 -> 19.59 t/s), entirely
# through speculative acceptance (0.5227 -> 0.4091); per-step time is 0.5%
# better. Baselining here records those numerics as intended, which is a claim a
# revert would overturn. See L057.
#
# Baselining at HEAD makes the run trivially green -- both sides are the same
# commit -- so today's PASS is not evidence about the engine. Its value is the
# anchor for tomorrow's drift, plus a harness self-test: a commit compared
# against ITSELF failing would mean the gate is broken rather than the engine.
# MOVED 06d0b3e -> 4d1ee81 on 2026-08-17, second move today and the same rule
# both times: numerics shipped, so the anchor follows. What shipped since the
# morning anchor is the KV unification -- every cache on PULSAR_ATTN_PACK, which
# moves the rope tail f16 -> bf16 -- plus the fp4 GEMV reading the producer's
# E4M3 instead of re-deriving it, and the plain matmul's BF16 activations.
#
# Against 06d0b3e this gate is red at every depth, which is the rot condition
# the note above warns about: an assertion that always fires asserts nothing.
#
# PRICE ON THE RECORD, as with the last move: the KV work costs -4.6% decode and
# -4.9 acceptance points (L061), kept deliberately for one KV format end to end.
# If the drafter half is reverted the anchor moves again.
#
# MOVED 4d1ee81 -> 055239b on 2026-08-18, ordinary application of the policy.
# NOTE THE REF: it is the last commit that changes COMPUTED numerics, not HEAD.
# Everything after it (payload v5, the MXKV deletion, the cuda-regression repair)
# changes stored formats, dead code and tests -- nothing prefill computes -- so
# anchoring at HEAD would have recorded a numerics event that did not happen.
#
# What shipped since 4d1ee81:
#   - the tensor-core tier back on f16 after a one-commit bf16 experiment
#     (832c2d8). P is in [0,1] by construction, so bf16's exponent range is
#     unusable while its 3 lost mantissa bits are pure cost.
#   - the online softmax normalising by the weights the MMA actually multiplies
#     (315c435): it summed the unrounded p while storing the rounded one, so the
#     weights did not sum to one.
#   - the packed-comp prefill migration reverted (a4c5b83) and re-landed
#     correctly (d5dab83, 1e0517b, 9e9e48a, 1e65c81) -- see L063; the broken
#     version zeroed EVERY prefill logits row.
#   - the double-quantise removal (f2c3040, 055239b): three KV paths quantised
#     the same rows twice, and the ring now takes the bytes attention read.
#
# ⚠ PRICE ON THE RECORD: decode acceptance 0.4362 -> 0.4237 (-2.9% rel), and
# prefill throughput MEASURED FLAT (918.89 -> 917.68 t/s, inside noise), so the
# consistency gain is not paid for by speed. Tyler decided to keep it (L064).
# That decision is what this anchor records; a revert would overturn it and the
# anchor would move back.
#
# The 2026-08-18 move also has a provenance the earlier ones lacked: L063 and
# L064 record exactly which bytes moved and the experiment that proved it -- a
# single-chunk prompt stayed bit-identical while a ring-reading one did not,
# which is what established that the fast-math scale bucket is not idempotent.
# MOVED d50777bd -> HEAD-of-this-change on 2026-08-24, and for a NEW REASON the
# earlier moves did not have: not because the old anchor could not load the
# artifact, but because a change that DISAGREES with the old anchor was measured
# CLOSER TO THE SOURCE.
#
# Tyler, 2026-08-24: "I would love to disagree with our past if it puts us
# closer to the original model.  In fact, we probably need to completely
# re-evaluate all decisions we made regarding being bit-exact to ourselves and
# start evaluating against the source logits."
#
# The change: hc_expand's gather lets --use_fast_math reassociate four products
# into a tree.  Pairwise summation has O(log n) error growth vs sequential O(n),
# so the tree is the MORE ACCURATE arithmetic, and graded against the B300
# reference it is closer at 6 of 9 depths -- including ALL THREE known-high
# outliers (story 512 0.643->0.569, story 30464 0.255->0.173, code 3840
# 0.196->0.190).  See pulsar-notes/bit-exact-vs-source-2026-08-24.md.
#
# ⇒ THE RULE THIS ESTABLISHES, alongside "re-baseline whenever a numerics change
# SHIPS": a byte-gate FAIL is not by itself a reason to revert.  Grade the change
# against cuda-reference-gate first.  If it moves us CLOSER to the source, adopt
# it and move this anchor deliberately; if it moves us away, reject it.  The byte
# gate keeps its whole value -- catching UNINTENDED change -- exactly as long as
# every move of this anchor is deliberate and says why.
# MOVED f742688 -> the L074 consolidation on 2026-08-24, the second deliberate
# move under the source standard -- and the first one that CORRECTED AN EARLIER
# VERDICT OF MINE.
#
# L074 (one authority for the tail-rope math) failed this byte gate, was graded,
# and was REJECTED on an all-depths NET of +21.4% "further".  That metric was
# wrong: [[L080]] then measured that the three known-high depths are FLAT
# reference positions (entropy ~1.7 nats, p(top1) 0.42-0.70) against ~0.0000 and
# p(top1)=1.0000 everywhere else, so summing across all depths let three noisy
# rows swamp six informative ones by five orders.  Re-graded on CONFIDENT depths
# only it is **-15.14% CLOSER** to the source (4102 -90.2%, 6144 -75.2%).
#
# Controls run at the same time: the tree itself grades +0.00%, and the
# SUPERSEDED arithmetic grades +48.88% FURTHER and FAILS -- so the gate rejects
# regression as well as accepting improvement.
PREFILL_BASELINE_REF ?= 5d45142
# The baseline blob is COMMITTED (L046): a fresh clone can run cuda-prefill-gate
# with no bootstrap step, and the gate's guarantee no longer depends on a loose
# file surviving in somebody's tree. The name carries the anchor ref, and the
# default composes from PREFILL_BASELINE_REF so the two cannot drift apart:
# re-anchoring = update the REF above, `make cuda-prefill-gate-baseline` (dumps
# straight to the new tracked name), then commit the blob + this line together.
# The blob is ~2.5 MB and self-describing (ref stamp + model header + token FNV;
# see tests/prefill_bitexact_gate.cpp and the .md next to the blob).
PREFILL_BASELINE     ?= tests/test-vectors/prefill_bitexact_baseline-$(PREFILL_BASELINE_REF).bin
PREFILL_BASELINE_WT  ?= temp/wt-prefill-baseline
# The blob stamps `git rev-parse --short HEAD` as resolved INSIDE the baseline
# worktree; normalise the expected ref through the same abbreviation so the
# provenance compare is not defeated by a differing --short length.
PREFILL_BASELINE_REF_SHORT := $(shell git rev-parse --short $(PREFILL_BASELINE_REF) 2>/dev/null || echo $(PREFILL_BASELINE_REF))

# Both sides of a byte-compare must be built for the SAME arch, so pin sm_120f
# here exactly as the baseline sub-make does.  CUDA_ARCH ?= defaults to EMPTY
# (top of this file), so only an explicit sm_120f produces the objects the gate
# must measure; passing it on the sub-make command line also overrides any
# CUDA_ARCH inherited from the caller's environment.
#
# -B COVERS THE WHOLE LINK, NOT JUST THE HARNESS, AND THAT IS THE POINT.  There
# is no -MMD/-MP anywhere in this Makefile: src/cuda/%.o hand-lists its headers.
# That list is complete today, but D2R is expected to land a new header (e.g.
# src/cuda/pulsar_cuda_d2r.cuh) included by pulsar_cuda_moe.cu; make would then see
# moe.o as up to date after a .cuh-only edit, relink the OLD kernel, and print
# PASS — the gate would certify a kernel it never compiled.  Forcing the whole
# tree costs a rebuild per run, which is acceptable: the gate is manual and
# spends ~3 GPU-minutes regardless, and an acceptance gate must never certify
# stale objects.  Forcing the link also rebuilds every object at the sm_120f
# pinned above, which retires the old "make clean first" arch caveat FOR THIS
# TARGET: a tree previously built at another arch can no longer leak a
# mismatched object into the compare.
#
# ✅ `-MMD -MP` LANDED 2026-08-18 (see the ALL_OBJS/PULSAR_DEPS block at the top).
# The parallel branch this was deferred for -- flashinfer-attn -- is retired, so
# the reason expired.  Make now sees .cuh edits by itself.
#
# -B STAYS ANYWAY, and the reason is worth being precise about, because "we have
# dependency tracking now" is exactly the argument that would drop it wrongly:
# -MMD -MP tracks WHICH FILES a TU read, not WHICH FLAGS it was compiled with.  A
# tree previously built at another arch leaves objects whose headers are all
# up to date and whose CODE is for the wrong GPU, and make cannot tell.  That
# half of the hazard is untouched, and it is the half that made 6d41502's arch
# pin appear not to work.  So: dependency tracking removed the .cuh-staleness
# reason for -B; the arch-mismatch reason remains, and it is sufficient on its
# own for an acceptance gate.
# DIAGNOSTIC, deliberately NOT in GATE_TARGETS.  Spec-vs-greedy byte equality is
# NOT an invariant of this engine: measured 2026-08-14 on clean dev, the set of
# diverging draft depths is stable across generation LENGTH but changes with the
# PROMPT -- one prompt gives {4,5}, another gives all of {1..8}.  Not even depth 1
# is safe, because spec verify uses the batch output head while non-spec decode
# uses the single-row head.  So a fixed known-diverging list would fail on main.
# Still the right tool for an A/B against a known-good binary (it is what proved
# the L035 act-cache bug); making it a gate needs a recorded per-depth baseline
# blob from a reference commit, on the cuda-prefill-gate pattern.
# The two-minute semantic check that was missing on 2026-08-22: greedy chat
# answer through the engine, asserting content no us-vs-us comparison can fake.
# See tests/chat_smoke_gate.py's header for the salad it exists to catch.
cuda-chat-smoke-gate: pulsar
	python3 tests/chat_smoke_gate.py $(FRONTIER_MODEL) ./pulsar

cuda-spec-width-gate: pulsar
	python3 tests/spec_verify_width_gate.py $(FRONTIER_MODEL) --binary ./pulsar

cuda-prefill-gate:
	# NOT -B any more.  Staleness is fully expressed now: headers via -MMD -MP
	# + `-include $(PULSAR_DEPS)`, compiler configuration via CUDA_FLAG_STAMP.
	# -B rebuilt the entire engine twice per suite to cover the flag axis alone.
	$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check $(PREFILL_BASELINE) \
		$(PREFILL_BASELINE_REF_SHORT)

# Cross-engine fidelity gate: grade prefill logits against the vLLM reference
# blobs from the B300 capture.  This is the only gate that measures us against
# something we did not produce -- every other gate compares us to ourselves,
# which cannot catch an error we make consistently.
#
# The blobs are ~4.6 MB and live OUTSIDE this repo (pulsar-notes is private and
# stays that way), so the path comes from PULSAR_REF_DIR.  When it is unset or
# the blobs are missing this SKIPS LOUDLY rather than passing: a gate that
# quietly succeeds when its fixture is absent is worse than no gate, which is
# exactly the hole L078's golden fixture fell into.
#
# Tolerance 1e-4 clears the six mid rows with 10-100x headroom (measured
# 2026-08-21 on a provenanced binary: 3.5e-7 .. 9.9e-6, top-1 matched).
# --known-high names the three rows whose KL is a documented, not-yet-explained
# outlier -- shallow and file-end positions where a flat next-token
# distribution amplifies small numeric differences (ledger L080).  They keep
# the TOP-1 contract and lose only the KL ceiling; a blanket tolerance loose
# enough to pass them would be ~1e4x too loose for the mid rows and would
# protect nothing.
PULSAR_REF_TOL ?= 1e-4
# Per-depth KL budgets: the gate grades DIRECTION against these (closer to the
# source = pass).  Absent, it falls back to the absolute-ceiling check only,
# which cannot see direction -- see the 2026-08-24 note in
# pulsar-notes/bit-exact-vs-source-2026-08-24.md.
KL_BUDGET_STORY ?= tests/test-vectors/kl-budget-story.txt
KL_BUDGET_CODE  ?= tests/test-vectors/kl-budget-code.txt
# ⚠ ONE SHELL, DELIBERATELY.  Each make recipe LINE gets its own shell, so an
# `exit 0` in a guard on the first line exits only that line and make runs the
# rest anyway -- which is exactly how the first version of this target failed
# `make gates` with "cannot read reference blob /story.ref.bin" whenever
# PULSAR_REF_DIR was unset (i.e. by default).  The guard and the work must sit
# in the same shell for the skip to be a skip.
cuda-reference-gate:
	@if [ -z "$(PULSAR_REF_DIR)" ] || [ ! -f "$(PULSAR_REF_DIR)/story.ref.bin" ]; then \
		echo "  SKIP  cuda-reference-gate: set PULSAR_REF_DIR to the reference-capture dir"; \
		echo "        (blobs live outside the repo; without them this gate grades nothing)"; \
	else \
		set -e; \
		$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f; \
		./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
			$(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin \
			$(PULSAR_REF_TOL) --known-high 512,30464 --known-flip 512 \
			$(if $(wildcard $(KL_BUDGET_STORY)),--kl-baseline $(KL_BUDGET_STORY),); \
		./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
			$(PULSAR_REF_DIR)/code.ref.bin $(PULSAR_REF_DIR)/code.tokens.bin \
			$(PULSAR_REF_TOL) --known-high 3840 \
			$(if $(wildcard $(KL_BUDGET_CODE)),--kl-baseline $(KL_BUDGET_CODE),); \
	fi

# Re-record the KL budgets from the CURRENT tree.  Same discipline as
# PREFILL_BASELINE_REF: do this only when a change has been GRADED CLOSER to the
# source and adopted, and say why in the commit.  Re-recording to silence a
# regression is the one thing that makes this gate worthless.
.PHONY: cuda-reference-gate-budget
cuda-reference-gate-budget:
	@if [ -z "$(PULSAR_REF_DIR)" ] || [ ! -f "$(PULSAR_REF_DIR)/story.ref.bin" ]; then \
		echo "REFUSING: set PULSAR_REF_DIR to the reference-capture dir"; exit 1; fi
	$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
		$(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin \
		$(PULSAR_REF_TOL) --known-high 512,30464 --known-flip 512 \
		--dump-kl $(KL_BUDGET_STORY)
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
		$(PULSAR_REF_DIR)/code.ref.bin $(PULSAR_REF_DIR)/code.tokens.bin \
		$(PULSAR_REF_TOL) --known-high 3840 --dump-kl $(KL_BUDGET_CODE)
	@echo "KL budgets re-recorded -- COMMIT THEM with the reason"

# NOTE: CUTLASS is an external header-only include path (never a submodule, and
# never populated by `git worktree add`), so the baseline build is pointed at
# THIS tree's CUTLASS_DIR.
cuda-prefill-gate-baseline:
	git worktree remove --force $(PREFILL_BASELINE_WT) 2>/dev/null || true
	rm -rf $(PREFILL_BASELINE_WT)
	git worktree prune
	git worktree add --detach $(PREFILL_BASELINE_WT) $(PREFILL_BASELINE_REF)
	cp tests/prefill_bitexact_gate.cpp $(PREFILL_BASELINE_WT)/tests/
	cp Makefile $(PREFILL_BASELINE_WT)/
	cp tests/long_context_story_prompt.txt $(PREFILL_BASELINE_WT)/tests/
	$(MAKE) -C $(PREFILL_BASELINE_WT) tests/prefill_bitexact_gate \
		CUDA_ARCH=sm_120f CUTLASS_DIR=$(abspath $(CUTLASS_DIR))
	cd $(PREFILL_BASELINE_WT) && ./tests/prefill_bitexact_gate $(abspath $(FRONTIER_MODEL)) \
		--dump $(abspath $(PREFILL_BASELINE))
	git worktree remove --force $(PREFILL_BASELINE_WT)
	@echo "baseline ($(PREFILL_BASELINE_REF)) -> $(PREFILL_BASELINE)"
# Speculative-sampling exactness oracle: chi-square of plain-sampled vs
# speculative per-position marginals, plus the greedy prefix gate and the
# acceptance rate alpha (see the header of tests/spec_sampling_gate.c).
# Proposal-agnostic, so it gates both the deterministic and the
# temperature-matched (p/q) accept rules.  MODEL-DEPENDENT — same memory
# discipline as the gates above; not part of `make test`.
# Defaults to FRONTIER_MODEL so one variable points every gate at the same
# artifact.  It used to default to a different path, which is a trap: passing
# FRONTIER_MODEL alone made this gate "fail" instantly on a missing file and
# look like a real regression (hit 2026-08-14 during the L031 sweep).
SPEC_GATE_MODEL ?= $(FRONTIER_MODEL)
SPEC_GATE_ARGS ?= 0.95
cuda-spec-sampling-gate: tests/spec_sampling_gate
	./tests/spec_sampling_gate $(SPEC_GATE_MODEL) $(SPEC_GATE_ARGS)

# Every release-blocking gate, in one command.
#
# WHY THIS EXISTS: gates are separate make targets, and a gate nobody invokes is
# a gate that silently stops compiling.  On 2026-08-14 a sweep found the prefill
# gate rotted (its baseline ref predated type-43, so it could not run at all),
# two attention gates uncompilable since an API parameter was removed that
# morning, and a REAL production bug -- fp16 attention breaking mixed-batch
# prefill -- that had shipped six days earlier.  None of it was visible to the
# product build.  "Run each and record pass/fail" in a checklist is not a
# mechanism; this is.
#
# Continues past failures so one broken gate does not hide the rest, prints a
# summary, and exits non-zero if any failed.  Needs the GB10 and the model:
#   make gates FRONTIER_MODEL=/srv/models/<artifact>.gguf
GATE_TARGETS = cuda-reap-router-audit cuda-regression cuda-chat-smoke-gate \
	cuda-attn-gates cuda-prefill-gate \
	cuda-reference-gate \
               cuda-frontier-gate cuda-multiseq-gate cuda-multiseq-gate-nodspark \
               cuda-bank-spec-gate cuda-accounting-gate cuda-evict-restore-gate \
               cuda-fork-gate cuda-algo-stability-gate cuda-mixed-prefill-gate \
               cuda-mixed-neutrality-gate cuda-mixed-neutrality-gate-wide \
               cuda-spec-sampling-gate

# The numerics-critical subset, for the ITERATION loop.  `make gates` is a
# pre-merge instrument -- 17 gates, each loading ~76 GiB of weights, with
# spec_sampling alone running 2x2500 trajectories; it is the wrong tool to sit
# and watch after every edit.  These four are the ones that actually catch a
# numerics or dispatch regression:
#   cuda-prefill-gate        full-vocab logits byte-identical at 5 depths
#   cuda-frontier-gate       frontier logits across banks
#   cuda-mixed-neutrality-gate  the small-n_tok / m-neutral split shapes
#   cuda-attn-gates          attention kernel + bank isolation
# SPEC_TRAJ shortens the sampling gate's chi-square when it is included; the
# full TRAJ is required for statistical power, so quick mode leaves it out
# rather than run it under-powered and call it a pass.
# NOT a substitute for `make gates` before a merge.
GATES_QUICK_TARGETS = cuda-attn-gates cuda-prefill-gate cuda-frontier-gate \
                      cuda-mixed-neutrality-gate

gates-quick:
	@rc=0; passed=""; failed=""; \
	for g in $(GATES_QUICK_TARGETS); do \
	  printf '\n\033[1m=== %s ===\033[0m\n' "$$g"; \
	  if $(MAKE) --no-print-directory $$g; then passed="$$passed $$g"; \
	  else rc=1; failed="$$failed $$g"; fi; \
	done; \
	printf '\n\033[1m=== gates-quick summary ===\033[0m\n'; \
	for g in $$passed; do printf '  \033[32mPASS\033[0m  %s\n' "$$g"; done; \
	for g in $$failed; do printf '  \033[31mFAIL\033[0m  %s\n' "$$g"; done; \
	if [ $$rc -eq 0 ]; then printf '\nQUICK GATES PASS (not a substitute for `make gates`)\n'; \
	else printf '\nQUICK GATES FAILED\n'; fi; \
	exit $$rc

# The sub-makes below get -j explicitly: a recipe's $(MAKE) does not inherit a
# jobserver through the shell for-loop, so without this every compile a gate
# triggers is SERIAL.
GATE_JOBS ?= $(shell nproc 2>/dev/null || echo 4)

gates:
	@rc=0; passed=""; failed=""; times=""; suite0=$$(date +%s); \
	for g in $(GATE_TARGETS); do \
	  printf '\n\033[1m=== %s ===\033[0m\n' "$$g"; \
	  t0=$$(date +%s); \
	  if $(MAKE) -j$(GATE_JOBS) --no-print-directory "$$g" CUDA_ARCH=sm_120f \
	        FRONTIER_MODEL="$(FRONTIER_MODEL)" SPEC_GATE_MODEL="$(SPEC_GATE_MODEL)" \
	        CUTLASS_DIR="$(CUTLASS_DIR)"; then \
	    passed="$$passed $$g"; \
	  else \
	    failed="$$failed $$g"; rc=1; \
	  fi; \
	  times="$$times $$g:$$(( $$(date +%s) - t0 ))"; \
	done; \
	printf '\n===================== GATE SUMMARY =====================\n'; \
	for g in $$passed; do printf '  PASS  %s\n' "$$g"; done; \
	for g in $$failed; do printf '  FAIL  %s\n' "$$g"; done; \
	printf '\n  seconds per gate (slowest first):\n'; \
	for e in $$times; do printf '    %6s  %s\n' "$${e##*:}" "$${e%%:*}"; done \
	  | sort -rn; \
	printf '\n  suite total: %s s\n' "$$(( $$(date +%s) - suite0 ))"; \
	if [ $$rc -eq 0 ]; then printf '\nALL GATES PASS\n'; \
	else printf '\nGATES FAILED:%s\n' "$$failed"; fi; \
	exit $$rc

src/engine/%.o: src/engine/%.c src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

# Ported C++ TUs (pulsar). One rule per source dir, mirroring the .c rules.
src/engine/%.o: src/engine/%.cpp src/engine/pulsar_engine_internal.h src/engine/cursor.hpp src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/server/%.o: src/server/%.cpp src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.cpp src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.cpp src/pulsar.h src/lib/pulsar_help.h src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.cpp src/pulsar.h $(LIB_HDRS) src/lib/sha1.hpp
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.c src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/server/%.o: src/server/%.c src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.c src/pulsar.h src/lib/pulsar_help.h src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.c src/pulsar.h $(LIB_HDRS)
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/vendor/%.o: src/vendor/%.cpp src/vendor/linenoise.h src/vendor/rax.h src/vendor/rax_malloc.h
	$(CXX) $(CXXFLAGS) -Wno-write-strings -c -o $@ $<

tests/pulsar_test.o: tests/pulsar_test.cpp $(SERVER_SRCS) src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Wno-unused-function -c -o $@ tests/pulsar_test.cpp

tests/pulsar_agent_test.o: tests/pulsar_agent_test.cpp $(AGENT_SRCS) src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Wno-unused-function -c -o $@ tests/pulsar_agent_test.cpp

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ tests/cuda_long_context_smoke.cpp

tests/multiseq_frontier_gate.o: tests/multiseq_frontier_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/multiseq_frontier_gate.cpp

tests/multiseq_decode_gate.o: tests/multiseq_decode_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/multiseq_decode_gate.cpp

tests/bank_spec_gate.o: tests/bank_spec_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_spec_gate.cpp

tests/accounting_gate.o: tests/accounting_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/accounting_gate.cpp

tests/bank_evict_restore_gate.o: tests/bank_evict_restore_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_evict_restore_gate.cpp

tests/bank_fork_gate.o: tests/bank_fork_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_fork_gate.cpp

tests/session_payload_gate.o: tests/session_payload_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/session_payload_gate.cpp

tests/algo_stability_gate.o: tests/algo_stability_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/algo_stability_gate.cpp

tests/mixed_prefill_gate.o: tests/mixed_prefill_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mixed_prefill_gate.cpp

tests/mixed_neutrality_gate.o: tests/mixed_neutrality_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mixed_neutrality_gate.cpp

# Public-API only (pulsar.h): the gate must build unchanged against the baseline
# ref's tree, so it must not depend on engine internals that may have drifted.
# PULSAR_GATE_BUILD_REF stamps the blob with the git HEAD that built the dumper, so
# a baseline re-dumped from the tree under test cannot pass vacuously (see the
# file header).  cuda-prefill-gate force-rebuilds this object so the stamp is
# never stale.
GATE_BUILD_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
tests/prefill_bitexact_gate.o: tests/prefill_bitexact_gate.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -DPULSAR_GATE_BUILD_REF='"$(GATE_BUILD_REF)"' \
		-c -o $@ tests/prefill_bitexact_gate.cpp
tests/spec_sampling_gate.o: tests/spec_sampling_gate.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/spec_sampling_gate.cpp

src/cuda/%.o: src/cuda/%.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h src/cuda/pulsar_iq2_tables_cuda.inc src/cuda/pulsar_cuda_mx.cuh $(CUDA_FLAG_STAMP)
	$(NVCC) $(NVCCFLAGS) $(MMQ_CPPFLAGS) -Isrc -c -o $@ $<

# Vendored llama.cpp MMQ TUs: templated C++17, own include root, and they do not
# depend on the pulsar CUDA headers -- keep them off the generic src/cuda rule.
# The vendored MMQ objects listed NO header prerequisites at all, so any .cuh
# edit left every one of them stale -- the same hazard the prefill gate solves
# with -B ("an acceptance gate must never certify stale objects") and the same
# one that made 6d41502's arch pin appear not to work.  A wildcard over the
# vendored headers is cheap here: these rebuild in seconds relative to a gate run.
MMQ_HDRS := $(wildcard src/cuda/mmq/*.cuh) $(wildcard src/cuda/mmq/*.h)

src/cuda/mmq/%.o: src/cuda/mmq/%.cu $(MMQ_HDRS) $(CUDA_FLAG_STAMP)
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
		-diag-suppress 20012 -diag-suppress 177 -Isrc -Isrc/cuda/mmq -c -o $@ $<

# CUTLASS MXFP4 tensor-core expert FFN (GB10/sm_120f). Requires -arch=sm_120f (family mode) for the
# mxf4 block-scale MMA; build the whole engine with CUDA_ARCH=sm_120f so all objects match arch.
src/cuda/pulsar_mxfp4_cutlass.o: src/cuda/pulsar_mxfp4_cutlass.cu src/pulsar_gpu.h $(CUDA_FLAG_STAMP) | cutlass-check
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 20012 -diag-suppress 177 -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/pulsar_mxfp4_cutlass.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_frontier_gate: tests/multiseq_frontier_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_decode_gate: tests/multiseq_decode_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_spec_gate: tests/bank_spec_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/accounting_gate: tests/accounting_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_evict_restore_gate: tests/bank_evict_restore_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_fork_gate: tests/bank_fork_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/session_payload_gate: tests/session_payload_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/algo_stability_gate: tests/algo_stability_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mixed_prefill_gate: tests/mixed_prefill_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mixed_neutrality_gate: tests/mixed_neutrality_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/prefill_bitexact_gate: tests/prefill_bitexact_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/spec_sampling_gate: tests/spec_sampling_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

pulsar_test: tests/pulsar_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

pulsar_agent_test: tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: pulsar_test seam-check
	./pulsar_test

clean:
	rm -rf .build
	rm -f pulsar pulsar-server pulsar-bench pulsar-eval pulsar-agent pulsar_test pulsar_agent_test src/engine/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cuda/mmq/*.o src/cuda/mmq/test/*.o src/cli/*.o src/lib/*.o src/vendor/*.o tests/*.o src/engine/*.d src/agent/*.d src/server/*.d src/cuda/*.d src/cuda/mmq/*.d src/cuda/mmq/test/*.d src/cli/*.d src/lib/*.d src/vendor/*.d tests/*.d tests/cuda_long_context_smoke tests/multiseq_frontier_gate tests/multiseq_decode_gate tests/prefill_bitexact_gate tests/bank_spec_gate tests/spec_sampling_gate tests/accounting_gate tests/bank_evict_restore_gate tests/bank_fork_gate tests/session_payload_gate tests/algo_stability_gate tests/mixed_prefill_gate tests/mixed_neutrality_gate tests/attn_f16_kernel_test tests/attn_f16_banked_test tests/attn_decode_split_test

# Pull in the generated header dependencies.  `-include` (not `include`) so a
# tree with no .d files yet -- a fresh clone, or right after `make clean` -- is
# silent rather than fatal.  This line is what makes every rule above correct
# after the first build; the hand-listed prerequisites are only the floor.
-include $(PULSAR_DEPS)
