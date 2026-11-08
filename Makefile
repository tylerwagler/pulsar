# Machine-local settings: model, fixture and reference paths live in local.mk (gitignored), never here.  Every
# such variable below defaults to a refusal that names it (need_path), raised only when a recipe that needs it runs:
# set it in local.mk (see local.mk.example), on the command line, or in the environment.
-include local.mk
need_path = $(error $(1) is not set -- give it the path (local.mk, the command line or the environment))

NATIVE_CPU_FLAG ?= -march=native

DEBUG_FLAGS ?= -g
# Header dependency generation; see the ALL_OBJS/PULSAR_DEPS block below for why.
PULSAR_DEPFLAGS := -MMD -MP

# The .c -> .cpp port is COMPLETE: no C translation unit remains, so the C
# compiler, CFLAGS and the five src/*/%.o-from-%.c pattern rules are gone, along
# with the instruction to keep CFLAGS and CXXFLAGS identical.  That rule existed
# so a freshly ported TU generated the same math as its C predecessor, and there
# are no predecessors left to match.
#
# -ffast-math with -fno-finite-math-only is still deliberate and still load
# bearing: it is the FP contract the numerics gates were calibrated against, so
# it is not a knob to turn casually.  -fno-exceptions/-fno-rtti is project
# style -- hot paths never throw, and fatal errors keep the pulsar_die()
# contract.
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
# The version string bakes into objects via -D, so an incremental build keeps
# serving the OLD string from any unrebuilt object -- the listen banner said
# g70141e0f through three later same-day deploys (2026-08-25). Rebuild the
# consuming TUs whenever HEAD moves. --git-path resolves correctly in
# worktrees, where .git is a file.
PULSAR_GIT_HEAD := $(shell git rev-parse --git-path HEAD 2>/dev/null)
ifneq ($(PULSAR_GIT_HEAD),)
src/server/cli_main.o src/server/http_server.o: $(PULSAR_GIT_HEAD)
endif

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -Xcompiler -Wall,-Wextra -O3 -g -lineinfo --use_fast_math --default-stream per-thread $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
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
CUTLASS_INC ?= -isystem $(CUTLASS_DIR)/include -isystem $(CUTLASS_DIR)/tools/util/include
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
# libpng + libjpeg-turbo are the DECODER for the vision path (L216): the
# reference preprocesses with Pillow, which decodes PNG with libpng and JPEG
# with libjpeg-turbo at its defaults, so using the same two libraries is what
# makes the decoded pixels match.  vision-pixel-gate proves it per format.
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lpng -ljpeg

PULSAR_INC = -Isrc -Isrc/lib -Isrc/vendor -I$(CUDA_HOME)/include

ENGINE_SRCS = $(wildcard src/engine/*.cpp)
# L295: the family modules (src/families/<name>/, the plugin layout) are engine objects; a new family adds a
# directory, not a line here.
FAMILY_SRCS = $(wildcard src/families/*/*.cpp)
ENGINE_OBJS = $(ENGINE_SRCS:.cpp=.o) $(FAMILY_SRCS:.cpp=.o)
# Two-rank TP transport + host-pinned GPU slab (slice 4b).  Only the pieces the
# engine calls: pulsar_tp.o (pure host, g++) and pulsar_tp_gpu.o (needs the CUDA
# runtime, so nvcc).  The gate scheduler / verify modules are wired by the decode
# and verify slices and stay out until then (no dead code).
TP_OBJS = src/tp/pulsar_tp.o src/tp/pulsar_tp_gpu.o
AGENT_SRCS = $(wildcard src/agent/*.cpp)
AGENT_OBJS = $(AGENT_SRCS:.cpp=.o)
SERVER_SRCS = $(wildcard src/server/*.cpp)
SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)
SERVER_HDRS = $(wildcard src/server/*.h)
# CUTLASS TUs need the CUTLASS include path + c++17; they build via dedicated rules below,
# so keep them out of the generic src/cuda/%.o rule.
CUTLASS_CUDA_OBJS = src/cuda/pulsar_mxfp4_cutlass.o src/cuda/pulsar_cuda_fmha.o
CUDA_SRCS = $(filter-out src/cuda/pulsar_mxfp4_cutlass.cu src/cuda/pulsar_cuda_fmha.cu,$(wildcard src/cuda/*.cu))
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
LIB_HDRS = src/lib/pulsar_help.h src/lib/pulsar_kvtext.h src/lib/pulsar_segstore.h src/lib/pulsar_kvchain.h src/lib/pulsar_utf8.h src/lib/pulsar_think_scan.hpp src/lib/pulsar_dsml.h src/lib/pulsar_ctxmem.h src/lib/pulsar_json.h \
           src/lib/pulsar_kvdigest.h src/lib/pulsar_kvpool.h
# pulsar_json.o rides in CORE_OBJS rather than being listed per target like the
# other src/lib objects: the ENGINE objects reference it (the safetensors reader
# scans shard headers and __metadata__ as JSON), so every target that links
# $(CORE_OBJS) needs it.  ALL_OBJS globs src/lib/*.cpp but is only used to
# derive .d files, and link rules use $^, so there is no duplicate object.
# The frontend (L251 S5; the shared kit since L295 F5) rides here too -- the BPE engine, the output-parser
# kit, the Qwen and Inkling renderers and parser configurations: each family loads its tokenizer at open and the engine's
# tokenizer entries dispatch to it, so every target linking $(CORE_OBJS) references it.
FRONTEND_OBJS = src/lib/bpe_tokenizer.o src/lib/pyjson.o src/lib/qwen_chat.o src/lib/qwen_output.o src/lib/out_parse_kit.o \
                src/lib/inkling_chat.o src/lib/nemotron_chat.o src/lib/chat_render_kit.o src/lib/gemma4_chat.o src/lib/gemma4_output.o
# L347 I3a: the disk KV payload digest (the engine's payload streams, the segment store and the TP workers share it)
# and the store's host memory pool ride here too: the engine references the digest, the store both.
KVIO_OBJS = src/lib/pulsar_kvdigest.o src/lib/pulsar_kvpool.o
# The digest's four lanes are four scalar multiply chains: GCC's vectorizers pack them into AVX2, which has no 64-bit
# multiply and emulates it (7.9 GB/s cache-resident on CT 124's Xeon 8259CL); scalar, they run 18.9 GB/s (v1: 5.1).
src/lib/pulsar_kvdigest.o: CXXFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize
CORE_OBJS = $(ENGINE_OBJS) $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) src/lib/pulsar_json.o $(FRONTEND_OBJS) $(TP_OBJS) \
            $(KVIO_OBJS)
# L303: grammar-constrained decoding.  xgrammar's C++ core is vendored VERBATIM under src/vendor/xgrammar (the pin
# and the copy recipe: src/vendor/xgrammar/UPSTREAM), so a refresh is a copy.  Its flags are confined to it and to
# the one boundary TU, src/lib/pulsar_grammar.cpp: -fexceptions (xgrammar reports every error by throwing; the
# boundary catches everything -- pulsar_grammar.h), no -ffast-math (it parses schema numbers), and upstream's own
# -Wno-unused-parameter for the vendored files only (35 sites; our boundary keeps the warning).  The server and the
# server-linked tests take GRAMMAR_OBJS; nothing else links it.
XGRAMMAR_SRCS = $(wildcard src/vendor/xgrammar/cpp/*.cc src/vendor/xgrammar/cpp/support/*.cc src/vendor/xgrammar/cpp/converter_ext/*.cc)
XGRAMMAR_OBJS = $(XGRAMMAR_SRCS:.cc=.o)
XGRAMMAR_INC = -Isrc/vendor/xgrammar/include -isystem src/vendor/xgrammar/3rdparty
GRAMMAR_CXXFLAGS = $(filter-out -ffast-math -fno-exceptions,$(CXXFLAGS)) -fexceptions \
                   -DXGRAMMAR_ENABLE_CPPTRACE=0 -DXGRAMMAR_ENABLE_INTERNAL_CHECK=0
GRAMMAR_OBJS = src/lib/pulsar_grammar.o $(XGRAMMAR_OBJS)

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
           $(patsubst %.cpp,%.o,$(wildcard src/lib/*.cpp)) \
           $(patsubst %.cpp,%.o,$(wildcard src/vendor/*.cpp)) $(XGRAMMAR_OBJS) \
           $(patsubst %.cpp,%.o,$(wildcard tests/*.cpp))
PULSAR_DEPS = $(ALL_OBJS:.o=.d)
PULSAR_LINK ?= $(NVCC) $(NVCCFLAGS)
PULSAR_LINK_LIBS ?= $(CUDA_LDLIBS)

# A failed recipe must not leave a half-written target: a partially-linked
# test binary from a failed build was later executed by a gate run as if it
# were current (make compares mtimes, not build success -- 2026-08-19).
.DELETE_ON_ERROR:

.PHONY: gates gates-preflight gates-quick agent-test-gate host-checks expert-stream-probe decode-kernel-census cuda-runner-gate cuda-spec-width-gate all help clean test seam-check cuda-spark cuda-regression cuda-kv-rows-pack-gate cuda-attn-gates cuda-seam-gate cuda-multiseq-gate cuda-multiseq-gate-nodspark cuda-algo-stability-gate cuda-algo-stability-gate-deep cuda-mixed-zero-prefill-gate cuda-fused-step-gate decode-reference-gate verify-width-probe cuda-prefill-gate cuda-prefill-gate-baseline cuda-prefill-decode-gate cuda-prefill-decode-gate-baseline cuda-spec-sampling-gate spec-teacher-forced-probe cuda-row-neutrality-gate cuda-row-neutrality-gate-deep cuda-row-neutrality-gate-deeper sse-decode-bench decode-floor-gate decode-floor-baseline context-coherence-probe tp-core-test tp-command-free-test tp-transport-test tp-mesh-test tp-slab-probe tp-dmabuf-probe

all: help

help:
	@echo "Pulsar build targets (CUDA-only fork for DGX Spark / GB10):"
	@echo "  make cuda-spark          Build for the DGX Spark / GB10 (sm_120f)"
	@echo "  make test                Build and run tests"
	@echo "  make cuda-regression     Kernel smokes vs synthetic slabs (modelless)"
	@echo "  make cuda-attn-gates     fp16 attention correctness gates: kernel oracle,"
	@echo "                           banked KV-leak isolation, one-row launch (modelless)"
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
	@echo "  make gates               = make release: EVERY release-blocking test, from"
	@echo "                           tests/manifest.toml (tests/run.py --list release),"
	@echo "                           with a pass/fail and budget summary (needs the GB10"
	@echo "                           + the cached models: local.mk)"
	@echo "  make check | kernels     The no-GPU tier; the GPU-no-model tier"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B pulsar-server CUDA_ARCH=sm_120f

pulsar-server: $(SERVER_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

# Development tools, not part of the shipped release. The release is just
# pulsar-server (the HTTP API). Source is kept; build these by name.
pulsar: src/cli/pulsar_cli.o src/lib/pulsar_help.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-bench: src/cli/pulsar_bench.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-eval: src/cli/pulsar_eval.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-agent: $(AGENT_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

# L278: the family kernel gates that need a GPU and no model run here, for every family -- Qwen's GDN, QSA and
# S4 (router / MoE / GR / PLE) beside DeepSeek's and the shared EXL3 arms.
CUDA_REGRESSION_BINS = tests/cuda_long_context_smoke tests/moe_route_bounds_gate tests/expert_table_gate \
                       tests/exl3_moe_epilogue_gate tests/exl3_dense_gate tests/exl3_moe_prefill_gate tests/exl3_half_gate \
                       tests/gdn_gate tests/qsa_attn_gate tests/qsa_yarn_test tests/qwen_s4_gate tests/router_gate tests/gqa_attn_gate tests/sconv_gate tests/head_gate \
                       tests/bf16_rows_gate tests/nvfp4_rows_gate tests/mamba2_gate tests/rowops_gate tests/lora_rows_gate tests/arch_gate tests/pool_gate tests/f16_rows_gate tests/fmha_gate \
                       tests/kernel_contract_gate
# 2026-10-14: they run CONCURRENTLY, each one's output kept together (-O).  They are model-free and mostly
# host-bound (their references run on the CPU: measured alone, 258 s in series, the longest 114 s of which 113 s
# user time), so in series they were 258 s of the battery's serial front.  No gate here grades a time.
# The sub-make gets its own -j: the parent's job-slot flags are dropped from MAKEFLAGS (command-line variables are
# kept), else make warns "-jN forced in submake" -- a warning in the battery log, whose baseline is zero.
CUDA_REGRESSION_RUNS = $(CUDA_REGRESSION_BINS:tests/%=cuda-regression-run/%)
cuda-regression: $(CUDA_REGRESSION_BINS)
	@MAKEFLAGS='$(filter-out -j% --jobserver%,$(MAKEFLAGS))' \
	  $(MAKE) --no-print-directory -O -k -j$(words $(CUDA_REGRESSION_BINS)) $(CUDA_REGRESSION_RUNS)
# router_gate runs through its own target.
.PHONY: $(CUDA_REGRESSION_RUNS)
$(filter-out cuda-regression-run/router_gate,$(CUDA_REGRESSION_RUNS)): cuda-regression-run/%: tests/%
	./tests/$* $(if $(filter exl3_half_gate,$*),$(EXL3_HALF_SCRATCH))
cuda-regression-run/router_gate: router-gate

# L295 F7: BF16 weights x raw bf16 rows -- the dense arm (rows kernel at decode widths, the fixed-order cuBLASLt GEMM
# for prompt chunks) and the routed arm, vs double, plus their M / chunk invariance.  Needs a GPU, no model.
tests/bf16_rows_gate: tests/bf16_rows_gate.cu src/cuda/pulsar_cuda_bf16_rows.cuh src/cuda/pulsar_cuda_qwen.h $(CUDA_OBJS) \
                      $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/bf16_rows_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)
.PHONY: bf16-rows-gate
bf16-rows-gate: tests/bf16_rows_gate
	./tests/bf16_rows_gate

# L298: NVFP4 W4A16 x raw bf16 rows -- bit-identical to the BF16 rows kernel on the dequantized weight (per K slice),
# graded vs double, M-independent, the front door and its refusals.  Needs a GPU, no model.  `nvfp4-rows-bench`: the
# LiLiCorr drafter's shapes at M = 1 / 4 / 8, weights cold, vs the BF16 rows kernel (information).
tests/nvfp4_rows_gate: tests/nvfp4_rows_gate.cu src/cuda/pulsar_cuda_nvfp4.h src/cuda/pulsar_cuda_bf16_rows.cuh \
                       src/cuda/pulsar_cuda_qwen.h $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/nvfp4_rows_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)
.PHONY: nvfp4-rows-gate nvfp4-rows-bench
nvfp4-rows-gate: tests/nvfp4_rows_gate
	./tests/nvfp4_rows_gate
nvfp4-rows-bench: tests/nvfp4_rows_gate
	./tests/nvfp4_rows_gate --bench

# L298: the LiLiCorr candidate-lattice head + greedy walk vs vLLM's own draft rounds (the fixtures
# tools/lilicorr/reference.py `fixture` writes from tools/lilicorr/vllm_dump.py's goldens).  Needs a GPU, the drafter
# checkpoint and the fixtures: LILICORR_DRAFTER (default /path/to/models/nemotron35-super-lilicorr), LILICORR_FIXTURES.
LILICORR_DRAFTER ?= $(call need_path,LILICORR_DRAFTER)
LILICORR_FIXTURES ?= $(call need_path,LILICORR_FIXTURES)
tests/lilicorr_head_gate: tests/lilicorr_head_gate.cu src/cuda/pulsar_cuda_lilicorr.h src/cuda/pulsar_cuda_nvfp4.h \
                          src/cuda/pulsar_cuda_qwen.h $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/lilicorr_head_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)
.PHONY: lilicorr-head-gate
lilicorr-head-gate: tests/lilicorr_head_gate
	./tests/lilicorr_head_gate $(LILICORR_DRAFTER) $(LILICORR_FIXTURES)

# L218: the two KV row packers (window E4M3/E8M0, main E2M1/E4M3) byte-exact
# against the host replica in tests/kv_row_fixture.h, plus the ring slot rule.
# Needs a GPU, no model.
cuda-kv-rows-pack-gate: tests/kv_rows_pack_gate tests/kv_rows_pack_gate_fastmath
	./tests/kv_rows_pack_gate
	./tests/kv_rows_pack_gate_fastmath

# L302: THE sampler is on the device; it must give the host sampler's bytes (frozen in tests/sampler_host_ref.cpp,
# built with the engine's own flags) -- token, distribution, logprobs, walk -- over every mode and edge row.
# Needs a GPU, no model.
cuda-sampler-device-gate: tests/sampler_device_gate
	./tests/sampler_device_gate

# fp16 attention correctness gates (standalone .cu, no model needed):
# kernel-vs-f64 oracle across the dense, compressed, indexed, decode-batch and
# one-row shapes, and banked cross-sequence isolation (the KV-leak oracle for
# the one attention kernel).  Each file's header documents its scope; these
# were previously build-by-hand only.
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
# Presumed introduced by the GB10's nvcc 13.0 -> 13.3 bump moving the default
# arch; pin it so the default can never decide whether a gate is honest.
ATTN_GATE_ARCH ?= sm_120f

# ⚠ THESE TWO #include A SHIPPED .cu AND MUST DEPEND ON IT.
#
# Each of these tests compiles one .cu that pulls in the real kernel source, so
# the engine is genuinely a prerequisite -- and until 2026-08-18 the rules said
# only "<test>.cu Makefile".  Every kernel change since the KV format was
# unified therefore left them UP TO DATE, and they sat stale for days:
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
# keeps these tests correct -- the generated .d files are.  They are KEPT as the
# floor: on a first build, or immediately after `make clean`, no .d exists yet
# and a hand-listed prerequisite is all make has.
# ---- standalone probes ---------------------------------------------------
# Modelless, single-TU investigations that answer one question each.  Several
# are cited from SHIPPED source as the provenance for a kernel's layout
# (pulsar_cuda_attn_f16.cu -> attn_mma_probe, pulsar_cuda_indexer_mxfp4.cu and
# docs/history/indexer-mxfp4-scorer.md -> idx_mxfp4_probe, docs/history/engine-perf-map.md ->
# idx_mma_issue_bench), so they are documentation with a compiler attached.
#
# They had NO targets until now, which is exactly how two of them rotted in
# silence.  `make probes` builds every one that is known to build, so the next
# rename breaks a build instead of a claim nobody re-checks.
#
# IDX_PROBE_GENCODE pins the VIRTUAL and real arch to the 'a' variant, and the
# -gencode form is required: -arch=sm_121a still emits a compute_121 PTX
# fallback pass, and ptxas rejects .block_scale / .kind::mxf4nvf4 there just as
# it does at compute_120.  The shipped indexer TU dodges the whole problem by
# guarding on __CUDA_ARCH_FAMILY_SPECIFIC__; these standalone probes have no
# such guard, so the fallback pass must not exist at all.  This is the recipe
# each probe's own header documents -- do not "simplify" it to -arch.
IDX_PROBE_GENCODE ?= -gencode arch=compute_121a,code=sm_121a

PROBES = tests/attn_mma_probe tests/fp16_fold_probe tests/mxfp8_mma_probe \
         tests/idx_mxfp4_probe tests/idx_mma_issue_bench \
         tests/idx_mxfp4_kernel_test tests/idx_mxfp4_kernel_test_64 \
         tests/csa2_compressor_kernel_test \
         tests/indexer_hadamard_kernel_test \
         tests/candidate_kernel_test

.PHONY: probes
probes: $(PROBES)

tests/attn_mma_probe: tests/attn_mma_probe.cu Makefile
	$(NVCC) -O3 --use_fast_math -arch=$(ATTN_GATE_ARCH) -o $@ $<

# --use_fast_math is the POINT here: it asks whether the optimizer folds the
# fp16 round-trip to identity, which it only can under these flags.
tests/fp16_fold_probe: tests/fp16_fold_probe.cu Makefile
	$(NVCC) -O3 --use_fast_math -arch=$(ATTN_GATE_ARCH) -o $@ $<

tests/mxfp8_mma_probe: tests/mxfp8_mma_probe.cu Makefile \
                            src/cuda/mmq/ds4_mxfp8_mma.cuh
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda/mmq -o $@ $<

tests/idx_mxfp4_probe: tests/idx_mxfp4_probe.cu Makefile
	$(NVCC) -O3 $(IDX_PROBE_GENCODE) -o $@ $<

tests/idx_mma_issue_bench: tests/idx_mma_issue_bench.cu Makefile
	$(NVCC) -O3 $(IDX_PROBE_GENCODE) -o $@ $<

# The correctness gate for the shipped block-scaled indexer scorer.  It drives
# the REAL kernel by #including it, so it needs the same include path the
# engine TU gets.  Repaired and mutation-validated 2026-09-01 (it had not
# compiled since f9fbafe); see the ledger row for the one open question it
# surfaced -- the tier reproduces the storage semantics to f32 precision up to a
# uniform factor of exactly 4.
tests/idx_mxfp4_kernel_test: tests/idx_mxfp4_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_indexer_mxfp4.cu src/cuda/pulsar_cuda_internal.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# The same probe at V4's 64 heads.  The tier is templated on the count, so the
# 32-head build proves nothing about the other instantiation -- its M/N split
# (4 m-tiles x 2 n-splits) and its smem sizes are different code, and the
# per-split register block is 8 n-tiles instead of 4.
tests/idx_mxfp4_kernel_test_64: tests/idx_mxfp4_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_indexer_mxfp4.cu src/cuda/pulsar_cuda_internal.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -DIDX_TEST_HEADS=PULSAR_IDX_MXFP4_HEADS_V4 -o $@ $<

# L218: the CSA2 compressor kernels (pool + bf16 + RMSNorm + bf16, the pending
# slot store) against the reference's Compressor.forward written out in C.
tests/csa2_compressor_kernel_test: tests/csa2_compressor_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_csa2.cu src/cuda/pulsar_cuda_internal.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# L218: the CSA2 candidate pool (block-max, radix top-k, mask) against the
# reference's select_candidate_blocks written out in C.
tests/candidate_kernel_test: tests/candidate_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_candidates.cu src/cuda/pulsar_cuda_internal.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# V4's indexer rotation (the 128-point Hadamard, L218's deletion, restored): the
# ORDER and NORMALISATION are pinned against a from-scratch host oracle, byte for
# byte, including every E8M0 scale byte.  Host side built -fno-fast-math so the
# oracle's arithmetic is IEEE and the comparison can be exact.
tests/indexer_hadamard_kernel_test.o: tests/indexer_hadamard_kernel_test.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -c -o $@ tests/indexer_hadamard_kernel_test.cpp

tests/indexer_hadamard_kernel_test: tests/indexer_hadamard_kernel_test.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: indexer-hadamard-kernel-check
indexer-hadamard-kernel-check: tests/indexer_hadamard_kernel_test
	./tests/indexer_hadamard_kernel_test

# L269: the tail rope on every stored element format -- f16 and bf16 against the f32 rotation rounded to their
# format, at V4.1's Q shape and rope sets, plus the refusal of a two-byte buffer that is neither.  The rope used to
# pick its kernel by width alone and ran V4.1's f16 Q as bf16.  Host side -fno-fast-math (IEEE tolerances).  Runs
# in cuda-attn-gates (battery, and gates-dev's attention tier); the target below is for iterating on it alone.
tests/rope_tail_format_gate.o: tests/rope_tail_format_gate.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -c -o $@ tests/rope_tail_format_gate.cpp

tests/rope_tail_format_gate: tests/rope_tail_format_gate.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: rope-tail-format-gate
rope-tail-format-gate: tests/rope_tail_format_gate
	./tests/rope_tail_format_gate

# DELIBERATELY NOT in PROBES: tests/flashinfer_sparse_mla_bench.cu needs
# FlashInfer's csrc/sparse_mla_sm120_prefill.cu, which is not vendored here.

tests/attn_f16_kernel_test: tests/attn_f16_kernel_test.cu Makefile \
                            src/cuda/pulsar_cuda_attn_f16.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h \
                            tests/kv_row_fixture.h tests/attn_pack_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

tests/attn_f16_banked_test: tests/attn_f16_banked_test.cu Makefile \
                            src/cuda/pulsar_cuda_attn_f16.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/kv_row_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# The two KV row packers vs the host replica, byte for byte (L218).  Compiled
# WITHOUT --use_fast_math on purpose: the packers must be exact under either
# setting, and the engine builds them with it -- run both (the gate target
# below builds the fast-math variant too).
tests/kv_rows_pack_gate: tests/kv_rows_pack_gate.cu Makefile \
                         src/cuda/pulsar_cuda_kvrows.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/kv_row_fixture.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<
tests/kv_rows_pack_gate_fastmath: tests/kv_rows_pack_gate.cu Makefile \
                         src/cuda/pulsar_cuda_kvrows.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h tests/kv_row_fixture.h
	$(NVCC) -O3 --use_fast_math -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# PLAN 94 phase 1 (L217): the routed-expert ROUTE BOUNDS rule.  Model-free -- it
# includes the builders' TU and drives the three kernels on synthetic arrays with
# canary words on BOTH sides of every one, so an expert id that escapes [0,
# n_total) trips a canary here instead of silently corrupting a scratch
# reservation in the field.  Removing the clamp fails this gate.
tests/moe_route_bounds_gate: tests/moe_route_bounds_gate.cu Makefile \
                             src/cuda/pulsar_cuda_moe_pairs.cu src/cuda/pulsar_cuda_internal.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# PLAN 94 phase 1 (L217): the per-expert ADDRESS TABLE's contract.  Model-free:
# it includes the table's TU and pins the arithmetic every consumer used to do
# itself, the one-table-per-stack cache identity, the refusal on bad input, and
# the type-44 layout against the shipped artifact's OWN tensor size.  A wrong
# expert address looks like fluent text, which is why this is a gate and not a
# comment (plan s4.1).
# L241 4g-2: the routed-expert decode GEMV microbenchmark -- the pair's
# production shape (every expert at half the intermediate width, 6 slots) on
# one GPU, where ncu can profile it.  Prints us/call, GB/s and an output
# checksum (a bit-exact kernel change must keep it).  Not a gate.
tests/expert_gemv_bench: tests/expert_gemv_bench.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -Isrc -Isrc/cuda -o $@ $< $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

tests/expert_table_gate: tests/expert_table_gate.cu Makefile \
                         src/cuda/pulsar_cuda_expert_table.cu src/cuda/pulsar_cuda_internal.h \
                         src/engine/exl3_trellis.h
	$(NVCC) -O3 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

# L245 step 3 / task #46: the EXL3 routed arm's projections (the decode arm) and its epilogues -- the fold
# and the sum, incl. their K-split partials -- graded against the host authority (exl3_trellis.h + double
# arithmetic) on random tiles at the V4.1 and Qwen expert shapes.  Model-free; needs a device.  Includes the
# kernel TUs directly, as the table gate does.
tests/exl3_moe_epilogue_gate: tests/exl3_moe_epilogue_gate.cu tests/exl3_moe_epilogue_gate_qwen.inc \
                              tests/exl3_moe_epilogue_gate_uponly.inc Makefile \
                              src/cuda/mmq/exl3_moe_epilogue.cu src/cuda/mmq/exl3_moe_epilogue.cuh \
                              src/cuda/mmq/exl3_moe_prefill.cu src/cuda/mmq/exl3_moe_prefill.cuh \
                              src/cuda/mmq/ds4_exl3_dev.cuh src/cuda/pulsar_cuda_mx.cuh src/engine/exl3_trellis.h
	$(NVCC) -O3 -std=c++17 -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -Isrc/cuda/mmq -o $@ $<

.PHONY: exl3-moe-epilogue-gate
exl3-moe-epilogue-gate: tests/exl3_moe_epilogue_gate
	./tests/exl3_moe_epilogue_gate

# L251: the EXL3 dense-Linear arm -- the complete Linear (input rotation,
# split-K trellis GEMV, output Hadamard + svh) at the Qwen dense shapes and a
# DeepSeek one, K = 2..5, M = 1..16 and a 40-row prefill, graded against the
# host authority in double; M-neutrality bit-exact; mutations; refusals.
# Links the PRODUCTION object (the MMQ rule, the engine's NVCCFLAGS), so pass
# the served arch: make exl3-dense-gate CUDA_ARCH=sm_120f.  Model-free.
tests/exl3_dense_gate: tests/exl3_dense_gate.cu tests/exl3_dense_ref.h $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile \
                       src/cuda/mmq/ds4_exl3_dense.cuh src/cuda/pulsar_cuda_mx.cuh src/engine/exl3_trellis.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_dense_gate.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: exl3-dense-gate
exl3-dense-gate: tests/exl3_dense_gate
	./tests/exl3_dense_gate

# L287 / task #46: the routed experts' tensor-core arms -- the prompt-chunk GEMM and the
# decode GEMM, one each for every family -- vs an f32-weight reference: DeepSeek V4 / V4.1
# and Qwen shapes, both producer formats (E4M3 slot, bf16 rows), every rate, 1..8192 tokens
# (decode: 1..40), ragged loads, chunk (decode: batch) invariance bit-exact, refusals.  Links the PRODUCTION object, so pass the served arch:
# make exl3-moe-prefill-gate CUDA_ARCH=sm_120f.  Model-free.
tests/exl3_moe_prefill_gate: tests/exl3_moe_prefill_gate.cu src/cuda/mmq/exl3_moe_prefill.o Makefile \
                             src/cuda/mmq/exl3_moe_prefill.cuh src/cuda/pulsar_cuda_mx.cuh src/engine/exl3_trellis.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_moe_prefill_gate.cu src/cuda/mmq/exl3_moe_prefill.o

.PHONY: exl3-moe-prefill-gate
exl3-moe-prefill-gate: tests/exl3_moe_prefill_gate
	./tests/exl3_moe_prefill_gate

# L269 W1: a rank's EXL3 expert half (gate / up by output columns, down by input
# rows) is a byte-exact piece of the whole, through the production builder; the
# routed entry under expert_split on both halves == the whole (z and the folded
# mid's E4M3 bit for bit, out_0 + out_1 within f32 reassociation) at decode and
# prefill widths, every rate; refusals.  V4.1 shape, model-free, one GPU; the
# stacks go to an unlinked scratch file in EXL3_HALF_SCRATCH.
EXL3_HALF_SCRATCH ?= .
tests/exl3_half_gate: tests/exl3_half_gate.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_half_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: exl3-half-gate
exl3-half-gate: tests/exl3_half_gate
	./tests/exl3_half_gate $(EXL3_HALF_SCRATCH)

# L287: one DeepSeek routed EXL3 layer on a prompt chunk through the production
# front door (ms/layer, tok/s at 512 / 4096 / 8192 tokens; V4 or V4.1 shapes), and
# the D2R arm vs decode-once fp16 + cuBLASLt per expert.  Not a gate.
tests/exl3_moe_prefill_bench: tests/exl3_moe_prefill_bench.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_moe_prefill_bench.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# Task #46: one routed EXL3 layer at DECODE / VERIFY widths (S streams x R rows, expert overlap between a
# stream's consecutive rows) through the production front door, Qwen / V4 / V4.1 shapes, DRAM-cold.  Not a gate.
tests/exl3_moe_decode_bench: tests/exl3_moe_decode_bench.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_moe_decode_bench.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# L251: the dense arm on a REAL exllamav3-quantized weight, driven by
# the private notes' research/l251/exl3-dense/xcheck.py (which compares it with
# exllamav3's own forward on the same activations).  Not a gate.
tests/exl3_dense_xcheck: tests/exl3_dense_xcheck.cu tests/exl3_dense_ref.h $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile \
                         src/cuda/mmq/ds4_exl3_dense.cuh src/cuda/pulsar_cuda_mx.cuh src/engine/exl3_trellis.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_dense_xcheck.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# L251: the dense arm's microbenchmark -- EXL3 K=2..5 vs the engine's MXFP8
# decode GEMV (the production wrapper, decode rows declared, the A8 slot armed)
# and a streaming-read roofline, DRAM-cold, M = 1..16.  Not a gate.
tests/exl3_dense_bench: tests/exl3_dense_bench.cu $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/exl3_dense_bench.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# L251 S4: Qwen3.8-Flash-Next's router, MoE block (EXL3 routed + shared), Gated
# Residual and PLE injection -- the production objects vs the double references in
# tests/qwen_ref.h, model-free: decisions, ties, NaN routing, M-neutrality, the
# PLE conv state carried (batch == single-token steps), mutations.  Pass the served
# arch: make qwen-s4-gate CUDA_ARCH=sm_120f.
QWEN_S4_HDRS = tests/qwen_ref.h tests/exl3_dense_ref.h src/cuda/pulsar_cuda_qwen.h src/cuda/pulsar_cuda_mx.cuh \
               src/engine/exl3_trellis.h
tests/qwen_s4_gate: tests/qwen_s4_gate.cu $(QWEN_S4_HDRS) $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/qwen_s4_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)
.PHONY: qwen-s4-gate
qwen-s4-gate: tests/qwen_s4_gate
	./tests/qwen_s4_gate

# L295 F2: the shared MoE router (src/cuda/pulsar_cuda_router.cu) -- every router a family declares (Qwen's, DeepSeek's
# through its tensor binding, Inkling-Small's) vs the double reference in the gate: picks, weights, sink / sigmoid
# gammas, batch-width neutrality, rule 9, refusals.  Model-free.
tests/router_gate: tests/router_gate.cu src/cuda/pulsar_cuda_router.h src/cuda/pulsar_cuda_qwen.h $(CUDA_OBJS) \
                   $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/router_gate.cu \
		$(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)
# L336: writes src/cuda/pulsar_cuda_lt_picks.inc (pulsar_lt_fixed's committed kernel choices) -- a measurement tool,
# not a gate: run it on a quiet device, `tests/lt_picks --from-log <server log>` (the "fixed GEMM" lines name the shapes).
# L353 no. 2: the store's index under ThreadSanitizer: client threads walk its trailers while the worker changes it.
# Host only; built from source with its own flags, so it shares no objects with the server.
tests/segstore_tsan: tests/segstore_tsan.cpp src/lib/pulsar_segstore.cpp src/lib/pulsar_segstore.h \
                     src/lib/pulsar_kvdigest.cpp src/lib/pulsar_kvpool.cpp Makefile
	$(CXX) -std=c++23 -O1 -g -fsanitize=thread -D_GNU_SOURCE -Isrc -Isrc/lib -o $@ tests/segstore_tsan.cpp \
	  src/lib/pulsar_segstore.cpp src/lib/pulsar_kvdigest.cpp src/lib/pulsar_kvpool.cpp -lpthread
.PHONY: segstore-tsan
segstore-tsan: tests/segstore_tsan
	TSAN_OPTIONS=halt_on_error=1 ./tests/segstore_tsan
tests/lt_picks: tests/lt_picks.cu src/cuda/pulsar_cuda_lt_fixed.o src/cuda/pulsar_cuda_lt_fixed.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/lt_picks.cu src/cuda/pulsar_cuda_lt_fixed.o $(CUDA_LDLIBS)
.PHONY: router-gate
router-gate: tests/router_gate
	./tests/router_gate

# L251: the Gated DeltaNet kernels (Qwen3.8-Flash-Next) -- conv + gates + the
# delta-rule recurrence + gated RMSNorm between the block's projections.  Vs the
# double host authority; decode == prefill, chunk and batch neutrality
# bit-exact; the A8 slot vs the canonical encoder; mutations; refusals.  Links
# the PRODUCTION object; pass the served arch (CUDA_ARCH=sm_120f).  Model-free.
src/cuda/pulsar_cuda_gdn.o: src/cuda/pulsar_cuda_gdn.h
tests/gdn_gate: tests/gdn_gate.cu tests/gdn_ref.h src/cuda/pulsar_cuda_gdn.o src/cuda/pulsar_cuda_gdn.h \
                src/cuda/pulsar_cuda_mx.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/gdn_gate.cu src/cuda/pulsar_cuda_gdn.o

.PHONY: gdn-gate
gdn-gate: tests/gdn_gate
	./tests/gdn_gate

# L295 F1: the family-neutral GQA attention block (runtime geometry, BF16 KV rows,
# full / window-ring spans, q/k norm, relative-bias + temperature score mod,
# output gate) vs the double host authority (tests/gqa_ref.h), graded; decode ==
# prefill == chunked == verify == batched bit-exact; ring rollback; mutations;
# a long position past the temperature floor; refusals.  Links the PRODUCTION
# object; pass the served arch (CUDA_ARCH=sm_120f).  Model-free.
# L298: the block drafter's small ops (src/cuda/pulsar_cuda_blockdraft.h: the grouped dynamic conv, SwiGLU, the tap
# row copy, top-k + log-softmax; L299 the stream-mean tap, the top-k id map) and (L299) the CandidateSelector chain
# head (src/cuda/pulsar_cuda_candsel.h) against the host authority (tests/blockdraft_gate.cu).  Model-free.
tests/blockdraft_gate: tests/blockdraft_gate.cu src/cuda/pulsar_cuda_blockdraft.o src/cuda/pulsar_cuda_blockdraft.h \
                       src/cuda/pulsar_cuda_candsel.o src/cuda/pulsar_cuda_candsel.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/blockdraft_gate.cu src/cuda/pulsar_cuda_blockdraft.o \
		src/cuda/pulsar_cuda_candsel.o
.PHONY: blockdraft-gate
blockdraft-gate: tests/blockdraft_gate
	./tests/blockdraft_gate
src/cuda/pulsar_cuda_attn_gqa.o: src/cuda/pulsar_cuda_attn_gqa.h
tests/gqa_attn_gate: tests/gqa_attn_gate.cu tests/gqa_ref.h src/cuda/pulsar_cuda_attn_gqa.o src/cuda/pulsar_cuda_attn_gqa.h src/lib/pulsar_yarn.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/gqa_attn_gate.cu src/cuda/pulsar_cuda_attn_gqa.o

.PHONY: gqa-attn-gate
gqa-attn-gate: tests/gqa_attn_gate
	./tests/gqa_attn_gate

# The block's decode / verify / prefill timings at Inkling's shapes against a measured read-bandwidth and
# mma.sync roofline (L295 F1).  Not a gate.
tests/gqa_attn_bench: tests/gqa_attn_bench.cu src/cuda/pulsar_cuda_attn_gqa.o src/cuda/pulsar_cuda_attn_gqa.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/gqa_attn_bench.cu src/cuda/pulsar_cuda_attn_gqa.o


# L251: the GDN microbenchmark -- decode steps at 1/4/8/16 sequences, DRAM-cold
# (state slots rotated past L2), prefill throughput, and a streaming-copy
# roofline measured in the same process.  Not a gate.
tests/gdn_bench: tests/gdn_bench.cu src/cuda/pulsar_cuda_gdn.o src/cuda/pulsar_cuda_gdn.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/gdn_bench.cu src/cuda/pulsar_cuda_gdn.o

# L295 E3: the family-neutral short convolution block (depthwise causal, K taps,
# bf16, a [K-1][C] state slot a sequence) vs the host replay of its arithmetic
# (bit-exact) and the double reference (graded); decode == prefill == chunked ==
# verify == batched bit-exact; per-row capture and rollback; mutations; the
# 4-stream launch; refusals.  Links the PRODUCTION object; pass the served arch
# (CUDA_ARCH=sm_120f).  Model-free.
src/cuda/pulsar_cuda_sconv.o: src/cuda/pulsar_cuda_sconv.h src/cuda/pulsar_cuda_sconv.cuh
tests/sconv_gate: tests/sconv_gate.cu tests/sconv_ref.h src/cuda/pulsar_cuda_sconv.o src/cuda/pulsar_cuda_sconv.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/sconv_gate.cu src/cuda/pulsar_cuda_sconv.o

.PHONY: sconv-gate
sconv-gate: tests/sconv_gate
	./tests/sconv_gate

# The block's launches at Inkling-Small's shapes (K+V, the two residual deltas;
# decode, verify with capture, a 4096-row prefill chunk).  Not a gate.
tests/sconv_bench: tests/sconv_bench.cu src/cuda/pulsar_cuda_sconv.o src/cuda/pulsar_cuda_sconv.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/sconv_bench.cu src/cuda/pulsar_cuda_sconv.o


# L296 N2: the family-neutral Mamba-2 (SSD) mixer block (runtime heads / groups / norm group / conv kernel; the
# conv is E3's block) vs a double reference with its bf16 rounding points; decode == prefill == chunked == verify
# == batched bit-exact; DELTA + SNAPSHOT verify capture, commit and rollback to any accepted row; mutations;
# refusals.  Links the PRODUCTION objects; pass the served arch (CUDA_ARCH=sm_120f).  Model-free.
src/cuda/pulsar_cuda_mamba2.o: src/cuda/pulsar_cuda_mamba2.h src/cuda/pulsar_cuda_sconv.h src/cuda/pulsar_cuda_sconv.cuh
tests/mamba2_gate: tests/mamba2_gate.cu tests/mamba2_ref.h tests/sconv_ref.h src/cuda/pulsar_cuda_mamba2.o src/cuda/pulsar_cuda_sconv.o src/cuda/pulsar_cuda_mamba2.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/mamba2_gate.cu src/cuda/pulsar_cuda_mamba2.o src/cuda/pulsar_cuda_sconv.o

.PHONY: mamba2-gate
mamba2-gate: tests/mamba2_gate
	./tests/mamba2_gate

# The block's launches at Nemotron 3.5 Super's shapes (decode 1 / 16 sequences, a verify with capture + commit,
# a 4096-row prefill) against a measured streaming roofline.  Not a gate.
tests/mamba2_bench: tests/mamba2_bench.cu src/cuda/pulsar_cuda_mamba2.o src/cuda/pulsar_cuda_sconv.o src/cuda/pulsar_cuda_mamba2.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/mamba2_bench.cu src/cuda/pulsar_cuda_mamba2.o src/cuda/pulsar_cuda_sconv.o


# L295 E5: the family head options -- the embedding RMSNorm (graded vs double,
# batch-independent bytes), the logits finish (scale bit-exact, padding -inf,
# argmax never a pad), the identity head launches nothing; refusals.  Model-free.
src/cuda/pulsar_cuda_head.o: src/cuda/pulsar_cuda_head.h
tests/head_gate: tests/head_gate.cu tests/sconv_ref.h src/cuda/pulsar_cuda_head.o src/cuda/pulsar_cuda_head.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/head_gate.cu src/cuda/pulsar_cuda_head.o

.PHONY: head-gate
head-gate: tests/head_gate
	./tests/head_gate

# L295: the row operations between a layer plan's blocks (pulsar_cuda_rowops.h) -- the RMSNorm and the fused residual
# add bit-exact against a host replica of their one arithmetic and graded vs double, the f32 -> bf16 emit bit-exact,
# a row's bytes independent of its batch; refusals.  Model-free.
src/cuda/pulsar_cuda_rowops.o: src/cuda/pulsar_cuda_rowops.h
tests/rowops_gate: tests/rowops_gate.cu tests/sconv_ref.h src/cuda/pulsar_cuda_rowops.o src/cuda/pulsar_cuda_rowops.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/rowops_gate.cu src/cuda/pulsar_cuda_rowops.o

.PHONY: rowops-gate
rowops-gate: tests/rowops_gate
	./tests/rowops_gate
# L338: the runtime LoRA primitive (pulsar_rows_lora_launch) against the f64 reference at Gemma 4's shapes; row
# independence, determinism, the padded pitches untouched, refusals.  Model-free.
tests/lora_rows_gate: tests/lora_rows_gate.cu src/cuda/pulsar_cuda_f16_rows.o src/cuda/pulsar_cuda_lt_fixed.o \
                      src/cuda/pulsar_cuda_f16_rows.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/lora_rows_gate.cu src/cuda/pulsar_cuda_f16_rows.o \
	    src/cuda/pulsar_cuda_lt_fixed.o -lcublasLt
.PHONY: lora-rows-gate
lora-rows-gate: tests/lora_rows_gate
	./tests/lora_rows_gate


# L251: the same TU as a shared object for the real-weight harness in
# the private notes' research/l251/gdn/ (ctypes from torch, against transformers'
# own Qwen4ExpTextGatedDeltaNet).  The engine's NVCCFLAGS plus -fPIC.  Not a gate.
tests/libpulsar_gdn.so: src/cuda/pulsar_cuda_gdn.cu src/cuda/pulsar_cuda_gdn.h src/cuda/pulsar_cuda_mx.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -Isrc -o $@ src/cuda/pulsar_cuda_gdn.cu

# L321: CUTLASS's memory-efficient attention (pulsar_cuda_fmha) vs a double reference (model-free).
tests/fmha_gate: tests/fmha_gate.cu src/cuda/pulsar_cuda_fmha.o src/cuda/pulsar_cuda_fmha.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/fmha_gate.cu src/cuda/pulsar_cuda_fmha.o

.PHONY: fmha-gate
fmha-gate: tests/fmha_gate
	./tests/fmha_gate

# L321: the f16 rows arm (fp16 tensor cores, the fixed-algorithm GEMM) + the LayerNorm / bias / GELU row ops vs host
# references, row independence bit for bit (model-free).
tests/f16_rows_gate: tests/f16_rows_gate.cu src/cuda/pulsar_cuda_f16_rows.o src/cuda/pulsar_cuda_lt_fixed.o \
                     src/cuda/pulsar_cuda_rowops.o src/cuda/pulsar_cuda_f16_rows.h src/cuda/pulsar_cuda_rowops.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/f16_rows_gate.cu src/cuda/pulsar_cuda_f16_rows.o \
	        src/cuda/pulsar_cuda_lt_fixed.o src/cuda/pulsar_cuda_rowops.o -lcublasLt

.PHONY: f16-rows-gate
f16-rows-gate: tests/f16_rows_gate
	./tests/f16_rows_gate

# L321: the encoders' mean pool -- f32, row order, bit-exact to the host loop (model-free).
tests/pool_gate: tests/pool_gate.cu src/cuda/pulsar_cuda_pool.o src/cuda/pulsar_cuda_pool.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/pool_gate.cu src/cuda/pulsar_cuda_pool.o

.PHONY: pool-gate
pool-gate: tests/pool_gate
	./tests/pool_gate

# L321: lanes that exist only for some devices decide by the DEVICE, not the compiled arch (model-free).
tests/arch_gate: tests/arch_gate.cu Makefile src/pulsar_gpu.h src/cuda/pulsar_cuda_arch.cuh src/cuda/mmq/ds4_mmq.h \
                 $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/arch_gate.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: arch-gate
arch-gate: tests/arch_gate
	./tests/arch_gate

# L335: the kernel contracts the L335 CUDA fixes restored (exact MX exponent, refusals, bounds, NaN rules), one case
# per fix (model-free; the case list is the file header).  Links the PRODUCTION objects.
tests/kernel_contract_gate: tests/kernel_contract_gate.cu Makefile src/pulsar_gpu.h src/cuda/pulsar_cuda_mx.cuh \
                            $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/kernel_contract_gate.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: kernel-contract-gate
kernel-contract-gate: tests/kernel_contract_gate
	./tests/kernel_contract_gate

# L251 S3: the Qwen full-attention + QSA layer (pulsar_cuda_qsa.cu) against a
# double host reference -- cache encode, top-512 selection incl. exact ties,
# output, the o_proj E4M3 slot, decode == prefill, mutations.  Model-free;
# needs a device.  Links the PRODUCTION objects.
tests/qsa_attn_gate: tests/qsa_attn_gate.cu Makefile src/pulsar_gpu.h src/cuda/pulsar_cuda_mx.cuh \
                     $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/qsa_attn_gate.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: qsa-attn-gate
qsa-attn-gate: tests/qsa_attn_gate
	./tests/qsa_attn_gate

# L280: the QSA rope table (native and YaRN) against transformers' own rope init, element-wise
# (tests/qsa_yarn_golden.h, made by tools/qwen/yarn_golden.py).  Host only; links the production objects
# for pulsar_qsa_rope_init.
tests/qsa_yarn_test: tests/qsa_yarn_test.cu tests/qsa_yarn_golden.h Makefile src/pulsar_gpu.h src/lib/pulsar_yarn.h \
                     $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -Itests -o $@ tests/qsa_yarn_test.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

.PHONY: qsa-yarn-test
qsa-yarn-test: tests/qsa_yarn_test
	./tests/qsa_yarn_test

# L251 S3: replay one real attention layer's captured projections through the
# same entry point, for the offline grade against transformers (see the tool's
# header).  Not a gate by itself: the grade lives with the capture script.
tests/qsa_layer_replay: tests/qsa_layer_replay.cu Makefile src/pulsar_gpu.h src/cuda/pulsar_cuda_mx.cuh \
                        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/qsa_layer_replay.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# L251 S3: the layer's decode (M = 1/4/8 at 8K/64K/256K, DRAM-cold) and prefill
# chunk timings on GB10.  Not a gate.
tests/qsa_attn_bench: tests/qsa_attn_bench.cu Makefile src/pulsar_gpu.h src/cuda/pulsar_cuda_mx.cuh \
                      $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Isrc -Isrc/cuda -o $@ tests/qsa_attn_bench.cu \
	        $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS) $(CUDA_LDLIBS)

# The restored 0731 unified NVFP4 row CODEC ORACLE -- HOST ONLY, no device, so
# it runs anywhere the tree builds.  tests/attn_pack_fixture.h mirrors the row
# the device packer writes; this binary pins the geometry and the recipe
# arithmetic the kernel must reproduce.
tests/attn_pack_fixture_test: tests/attn_pack_fixture_test.cpp tests/attn_pack_fixture.h Makefile src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -o $@ tests/attn_pack_fixture_test.cpp

# The DEVICE gate for that packer: runs the real kernel and grades its bytes and
# its f32 writeback against an INDEPENDENT CPU transcription (deliberately not
# the fixture -- an oracle shared with the code under test can be wrong in the
# same direction).  Host side built -fno-fast-math so the reference is exact;
# the tolerances exist because the DEVICE is fast-math.  No model needed.
tests/attn_pack_gate.o: tests/attn_pack_gate.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -c -o $@ tests/attn_pack_gate.cpp

tests/attn_pack_gate: tests/attn_pack_gate.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: cuda-attn-pack-gate
cuda-attn-pack-gate: tests/attn_pack_gate
	./tests/attn_pack_gate


.PHONY: attn-pack-fixture-check
attn-pack-fixture-check: tests/attn_pack_fixture_test
	./tests/attn_pack_fixture_test

# The Engram n-gram hash gate (L218 phase 0c) -- HOST ONLY, no device and no
# model, so it runs anywhere the tree builds.  The hash is integer arithmetic
# over a fixed layout, which makes it one of the few pieces of the V4.1 work
# that can be proven correct before a GPU exists.  tests/engram_hash_fixture.h
# is GENERATED from the real layout (see gen_engram_fixture.py); engram.cpp is
# compiled beside log.cpp because pulsar_die() lives there and the gate checks
# the refusals, not just the arithmetic.
tests/engram_hash_test: tests/engram_hash_test.cpp tests/engram_hash_fixture.h \
                        src/engine/engram.cpp src/engine/log.cpp src/engine/alloc.cpp Makefile \
                        src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -o $@ tests/engram_hash_test.cpp \
	    src/engine/engram.cpp src/engine/log.cpp src/engine/alloc.cpp

.PHONY: engram-hash-check
engram-hash-check: tests/engram_hash_test
	./tests/engram_hash_test

# L245: the EXL3 trellis arithmetic (src/engine/exl3_trellis.h -- tile bit
# layout, mul1 codebook, Hadamard basis) graded byte-exact against vectors
# exllamav3's own reconstruct produced on real MiaAI V4.1 checkpoint blocks
# and random-word tiles at every rate.  Host-only and instant; the header is
# the one authority the transcoder and the device kernels are held to.
tests/exl3_dequant_gate: tests/exl3_dequant_gate.cpp src/engine/exl3_trellis.h \
                         src/pulsar_gpu.h Makefile
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -o $@ tests/exl3_dequant_gate.cpp

.PHONY: exl3-dequant-gate
exl3-dequant-gate: tests/exl3_dequant_gate
	./tests/exl3_dequant_gate tests/test-vectors/exl3

# The adaptive draft depth rule (src/engine/spec_depth.h), shared by DSpark and the Qwen MTP drafter.
.PHONY: spec-depth-gate
tests/spec_depth_gate: tests/spec_depth_gate.cpp src/engine/spec_depth.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ tests/spec_depth_gate.cpp
spec-depth-gate: tests/spec_depth_gate
	./tests/spec_depth_gate


# L295 F3: the declared state kinds (src/engine/state_kinds.h) over two stub declarations -- an Inkling-shaped one (SConv
# conv state, window rings, global pools) and one grading every capture mode -- model-free, seconds
# (tests/state_kinds_gate.cpp).  In gates-dev's family tier and the battery.
.PHONY: state-kinds-gate
state-kinds-gate: tests/state_kinds_gate
	./tests/state_kinds_gate
# L272 P5: the session contract every family meets, on REAL weights (tests/session_contract_gate.cpp: chunk
# neutrality C1-C5 incl. the interruptible sync, the bank surface B1-B5), one target per family's model.  The
# zero-weight family gate above proves wiring only.  A new family adds one line here.
# the served pack (L305): U-e4-d5 with the DFlash2 block drafter (B6 needs a drafter) and the EXL3 K5 head, no MTP
QWEN_GATE_MODEL ?= $(call need_path,QWEN_GATE_MODEL)
# L278 contract 4: Qwen's reference anchors, graded by the runner's cuda-reference-gate-* on the hosted Qwen model.
# The capture is the CONTAINER's (HF qwen4_exp modules on these quantized weights, layer-streamed; L251
# 2026-09-28) -- matched precision, as DeepSeek's B300 capture is, so a KL here is OUR arithmetic's divergence,
# not the quantization's.
# No quotes in these values: gates-dev passes them through a shell variable.
QWEN_REF_ANCHOR_DIR ?= $(call need_path,QWEN_REF_ANCHOR_DIR)
# Re-recorded 2026-10-21 on the SERVED pack (L305: QWEN_GATE_MODEL, the EXL3 K5 head; the anchors' capture and the
# 2026-10-07 budgets, e21a0868, were the MTP pack's BF16 head -- code 512 0.076 then), report-only `--dump-kl`, the same
# numbers at dev ec5bd400 and on the Flare series (gpuq #420):
#   story  512 2.4e-4 | 2048 0.88 FLIP (ref 710, ours <|im_end|>; ref margin 2.14 -- L251's documented mismatch)
#          4096 1.96 | 4102 1.35 | 6144 0.55 | 16388 1.1e-5 | 31183 1.5e-6        top-1 6/7
#   code   512 0.086 | 2048 6.0e-4 | 3957 0.039                                      top-1 3/3
# The high rows are the positions L251 traced to the model's own sensitivity (a random 0.67%-norm nudge of the
# REFERENCE's own state flips story 4096 three times in four) -- known-high: top-1 enforced, KL informational and
# graded for DIRECTION by the budgets.  The tolerance holds the confident rows with 3-5x headroom.
# Story re-recorded 2026-11-05 at dev d671badc (L336: the router's prompt logits on the bf16 tensor cores), graded
# CLOSER and adopted (Tyler: "Confirm, then accept"; battery #623, the 07aa3a19 control gpuq #625 = the old budget
# to the bit, so the move is the router's alone):
#   story  512 3.5e-4 | 2048 0.14 MATCH (its flip gone) | 4096 2.43 FLIP (ours 449, ref margin 1.28 in ours)
#          4102 0.42 | 6144 3.13 FLIP (ours 883, gap 2.69) | 16388 1.4e-6 | 31183 8.2e-8   top-1 5/7
#   confident depths net 1.25e-5 -> 1.47e-6 (-88 %); 2048 -84 %, 4102 -69 %; 512 +47 %, 4096 +24 %, 6144 +463 %.
#   The flips sit at the known-high (model-sensitive) depths; the code budget is unchanged (it passed).
QWEN_REF_TOL ?= 1e-3
QWEN_KL_BUDGET_STORY ?= tests/test-vectors/kl-budget-qwen-story.txt
QWEN_KL_BUDGET_CODE  ?= tests/test-vectors/kl-budget-qwen-code.txt
QWEN_REF_ANCHORS = --ref-dir $(QWEN_REF_ANCHOR_DIR) --ref-tol $(QWEN_REF_TOL) \
	--kl-story $(QWEN_KL_BUDGET_STORY) --kl-code $(QWEN_KL_BUDGET_CODE) \
	--story-known-high 2048,4096,4102,6144 --story-known-flip 4096,6144 --code-known-high 512,3957
.PHONY: session-contract-gate-qwen session-contract-gate-ds session-contract-gate-nemotron
session-contract-gate-qwen: tests/session_contract_gate
	PULSAR_MSEQ_BANKS=4 ./tests/session_contract_gate $(QWEN_GATE_MODEL)
# L296 N7: Nemotron-H's BF16 mini (N9: with its MTP drafter, B6 / F / S run)
session-contract-gate-nemotron: tests/session_contract_gate
	PULSAR_MSEQ_BANKS=4 ./tests/session_contract_gate $(NEMOTRON_GATE_MODEL)
session-contract-gate-ds: tests/session_contract_gate
	PULSAR_MSEQ_BANKS=4 ./tests/session_contract_gate $(FRONTIER_MODEL)
# L284: a session payload at an off-grid, past-the-prefill frontier round-trips through the public API
# (tests/payload_frontier_gate.cpp): the same logits and decode rows, the same greedy speculation, a byte-identical
# re-save, the carried checkpoint's resume == cold, the digest.  The one kv-state payload, one target per family.
.PHONY: payload-frontier-gate-qwen payload-frontier-gate-ds
payload-frontier-gate-qwen: tests/payload_frontier_gate
	./tests/payload_frontier_gate $(QWEN_GATE_MODEL)
payload-frontier-gate-ds: tests/payload_frontier_gate
	./tests/payload_frontier_gate $(FRONTIER_MODEL)
# L289: the prompt end (tests/prompt_end_gate.cpp): a sync of the prompt the bank last prefilled resumes at its end,
# no row evaluated, and its logits, session payload and greedy speculation are a cold prefill's byte for byte; the
# record survives its own reuse; a different prompt of the same length is not served by it.  One target per family.
.PHONY: prompt-end-gate-qwen prompt-end-gate-ds
prompt-end-gate-qwen: tests/prompt_end_gate
	./tests/prompt_end_gate $(QWEN_GATE_MODEL)
prompt-end-gate-ds: tests/prompt_end_gate
	./tests/prompt_end_gate $(FRONTIER_MODEL)
# L313: bank-independent checkpoints (tests/bank_adopt_gate.cpp): a bank that adopts another bank's published state
# -- its pool rows below T and its grid checkpoint or record at T -- and prefills the rest is the cold prefill byte for
# byte (logits + greedy tokens), for the next turn, a side request and the same prompt; the lender keeps its own.
.PHONY: bank-adopt-gate-qwen bank-adopt-gate-ds bank-adopt-gate-nemotron
bank-adopt-gate-qwen: tests/bank_adopt_gate
	./tests/bank_adopt_gate $(QWEN_GATE_MODEL)
bank-adopt-gate-ds: tests/bank_adopt_gate
	./tests/bank_adopt_gate $(FRONTIER_MODEL)
bank-adopt-gate-nemotron: tests/bank_adopt_gate
	./tests/bank_adopt_gate $(NEMOTRON_GATE_MODEL)
# L284 #3: a family pool's bank physical residency (tests/bank_residency_gate.cpp): free_physical returns an idle
# bank's KV (touched + MemAvailable), alloc_physical + its segment chain bring it back, and its greedy continuation
# is byte-identical to a bank never freed.
.PHONY: bank-residency-gate-qwen
bank-residency-gate-qwen: tests/bank_residency_gate
	PULSAR_MSEQ_BANKS=4 ./tests/bank_residency_gate $(QWEN_GATE_MODEL)
# L357: the family-neutral ports of DeepSeek's retired frontier, accounting and head-graph-replay gates (runner
# gates bank-isolation-gate, kv-accounting-gate, graph-eager-gate; these targets run one by hand): an idle bank
# untouched by its neighbours' batched steps, batch order and a batchmate's position invisible to a row (the declared
# state, byte for byte); touched KV == the physical a fill takes and a bank free returns (cudaMemGetInfo); a replayed
# decode / verify step graph == the eager step at the same row count.
.PHONY: bank-isolation-gate-qwen kv-accounting-gate-qwen graph-eager-gate-nemotron
bank-isolation-gate-qwen: tests/bank_isolation_gate
	./tests/bank_isolation_gate $(QWEN_GATE_MODEL)
kv-accounting-gate-qwen: tests/kv_accounting_gate
	./tests/kv_accounting_gate $(QWEN_GATE_MODEL)
# Qwen's driver does not stage its steps (they never graph), so the graph-eager gate grades a family whose driver does
# (the runner hosts it on the LiLiCorr pack)
graph-eager-gate-nemotron: tests/graph_eager_gate
	./tests/graph_eager_gate $(LILICORR_GATE_MODEL)
# L278 contract 2, the loader, for every family: each served container's sparse HEADER CLONE must open, and each
# mutant of it (unknown arch, a missing key / tensor, a transposed shape, a dtype no reader takes, a stack layout no
# arm reads) must be REFUSED BY NAME with the open returning -- never an exit (tests/loader_contract.py,
# tests/loader_open.cpp).  Host-only: headers and zeros, no weight byte read, no GPU.
LOADER_CONTRACT_WORK ?= /var/tmp/pulsar-loader-contract
.PHONY: loader-contract-gate
loader-contract-gate: tests/loader_open
	python3 tests/loader_contract.py --open ./tests/loader_open --work $(LOADER_CONTRACT_WORK) $(FRONTIER_MODEL) $(QWEN_GATE_MODEL)
# L338 E2: the Gemma 4 family's load on a built container (tests/gemma4_family_gate.cpp): every tensor bound by its
# bind record, every byte priced, the plan (PULSAR_FFN_DENSE_MOE), the stored widths, the facts, the session refusals;
# then the loader contract's neutral faults on it.  Host-only.  The container: tools/container/build.py emit --hf <the
# BF16 checkpoint> --exl3 <its exllamav3 1.6.5 K4 pack> --recipe format-maps/gemma4-exl3-k4.json --all (L338 E1).
GEMMA4_GATE_MODEL ?= $(call need_path,GEMMA4_GATE_MODEL)
.PHONY: gemma4-family-gate
gemma4-family-gate: tests/gemma4_family_gate tests/loader_open
	./tests/gemma4_family_gate $(GEMMA4_GATE_MODEL)
	python3 tests/loader_contract.py --open ./tests/loader_open --work $(LOADER_CONTRACT_WORK) $(GEMMA4_GATE_MODEL)
# L338 P2.3: the runtime JEV adapter on a container built with --adapter jev (tests/gemma4_jev_gate.cpp): the bank
# switch, every decision prompt synced on an adapter bank, reuse == cold, the plain bank untouched; then the decision
# head on the captured hidden rows against the BF16 reference (tests/gemma4/jev_grade.py).  Needs the full model (the
# RTX 6000).  GEMMA4_JEV_REF = decision_ref.pt (L338 P0), GEMMA4_JEV_SNAP = the JEV snapshot (head + calibration).
GEMMA4_JEV_MODEL ?= $(call need_path,GEMMA4_JEV_MODEL)
GEMMA4_JEV_REF ?= $(call need_path,GEMMA4_JEV_REF)
GEMMA4_JEV_SNAP ?= $(call need_path,GEMMA4_JEV_SNAP)
GEMMA4_JEV_WORK ?= /tmp/gemma4-jev
.PHONY: gemma4-jev-gate
gemma4-jev-gate: tests/gemma4_jev_gate
	mkdir -p $(GEMMA4_JEV_WORK)
	python3 tests/gemma4/jev_grade.py prep $(GEMMA4_JEV_REF) $(GEMMA4_JEV_WORK)/cases.bin
	./tests/gemma4_jev_gate $(GEMMA4_JEV_MODEL) $(GEMMA4_JEV_WORK)/cases.bin $(GEMMA4_JEV_WORK)/out.bin
	python3 tests/gemma4/jev_grade.py grade $(GEMMA4_JEV_REF) $(GEMMA4_JEV_WORK)/out.bin $(GEMMA4_JEV_SNAP)
# L296 N1: the Nemotron-H family's load on a built container (tests/nemotron_family_gate.{cpp,py}), as Inkling's above:
# every tensor bound by its bind record, every byte priced, the grouped plan ({Mamba-2 | attention} x {MoE | none}), the
# drafter, the facts, the tokenizer + chat front, the session refusals -- on the container and its header clone; then
# each Nemotron-H mutant refused by name, and the loader contract's neutral faults.  Host-only.  The container: the
# 5-block BF16 mini (tools/nemotron/make_mini.py), built by tools/container/build.py emit --hf
# /path/to/models/nemotron35-mini --recipe format-maps/nemotron_h-bf16.json --all.
NEMOTRON_GATE_MODEL ?= $(call need_path,NEMOTRON_GATE_MODEL)
.PHONY: nemotron-family-gate
nemotron-family-gate: tests/nemotron_family_gate tests/loader_open
	python3 tests/nemotron_family_gate.py --probe ./tests/nemotron_family_gate --open ./tests/loader_open \
	    --work $(LOADER_CONTRACT_WORK)/nemotron $(NEMOTRON_GATE_MODEL)
	python3 tests/loader_contract.py --open ./tests/loader_open --work $(LOADER_CONTRACT_WORK) $(NEMOTRON_GATE_MODEL)
# L296 N7: the Nemotron-H forward's byte invariants on the core driver (tests/nemotron_fwd_gate.cpp exact): a prompt
# cut anywhere == the cold prefill, a batched decode == each bank's own, prefix reuse from a grid checkpoint, payload
# and snapshot round trips, the bank carry, the decode step graphed == eager (L296 perf X7) -- bytes; decode vs
# prefill graded.  Needs the GPU and the container.  `./tests/nemotron_fwd_gate bench` times the decode step (info).
.PHONY: nemotron-fwd-gate
nemotron-fwd-gate: tests/nemotron_fwd_gate
	./tests/nemotron_fwd_gate exact $(NEMOTRON_GATE_MODEL) synthetic
# L298: the same gate on a container whose drafter is the LiLiCorr block drafter (spec_block.h): spec == plain greedy
# and seeded-sampled, the forced-reject stress (the drafter's context rings == the plain twin's), the batched lane and
# toolsample's per-position modes.  L317: the battery runs it in the runner on the hosted engines (block-spec-gate on
# Qwen's DFlash2 and Nemotron's LiLiCorr, 128 tokens a stream); these two targets are the gates-dev depth (256) on a
# drafter path change, and the hand run.
LILICORR_GATE_MODEL ?= $(call need_path,LILICORR_GATE_MODEL)
MTP_GATE_TOKENS ?= 256
.PHONY: lilicorr-spec-gate
lilicorr-spec-gate: tests/mtp_spec_gate
	./tests/mtp_spec_gate $(LILICORR_GATE_MODEL) $(MTP_GATE_TOKENS)
# L299: the same gate on the DFlash2 block drafter (Qwen3.8-Flash-Next, the CandidateSelector chain head, the HC-mean
# taps); the served pack, which holds no MTP module since L305 (the gate names the drafter anyway).
DFLASH2_GATE_MODEL ?= $(QWEN_GATE_MODEL)
.PHONY: dflash2-spec-gate
dflash2-spec-gate: tests/mtp_spec_gate
	PULSAR_GATE_DRAFTER=dflash2 ./tests/mtp_spec_gate $(DFLASH2_GATE_MODEL) $(MTP_GATE_TOKENS)
# L306: a refused block-drafter pass fails the single-bank lane's request by name, never a plain round
# (tests/draft_refusal_gate.cpp; the refusal forced by detaching the session's block lane for one call).
.PHONY: draft-refusal-gate
draft-refusal-gate: tests/draft_refusal_gate
	PULSAR_GATE_DRAFTER=dflash2 ./tests/draft_refusal_gate $(DFLASH2_GATE_MODEL)
# L307: poisoned disposable scratch changes no output (tests/scratch_poison_gate.cpp) -- the batched speculative lane
# over 4 banks, the bank subset and the verify width alternating every tick, unpoisoned == unpoisoned == POISONED byte
# for byte, on the fixture of each graphed lane: DeepSeek (the output head's graph; DSpark), Qwen3.8 DFlash2 (the
# block drafter's pass graph) and Nemotron-H LiLiCorr (the decode and verify step graphs, the drafter's pass graph).
# Also a runner gate (`scratch-poison-gate`: the battery's hosted models with speculative rounds, no extra load).
SCRATCH_POISON_TOKENS ?= 64
.PHONY: scratch-poison-gate scratch-poison-gate-mutant
scratch-poison-gate: tests/scratch_poison_gate
	./tests/scratch_poison_gate $(FRONTIER_MODEL) $(SCRATCH_POISON_TOKENS)
	PULSAR_GATE_DRAFTER=dflash2 ./tests/scratch_poison_gate $(DFLASH2_GATE_MODEL) $(SCRATCH_POISON_TOKENS)
	./tests/scratch_poison_gate $(LILICORR_GATE_MODEL) $(SCRATCH_POISON_TOKENS)
# ... and its teeth: the same gate linked against a Qwen trunk step that clears its MoE non-finite flag on each arena's
# first step only (tests/scratch_poison_mutant.py) -- right by luck unpoisoned, a stale read poisoned.  PASS = the gate
# FAILED on it with its premise (the unpoisoned runs agree) intact.
scratch-poison-gate-mutant: tests/scratch_poison_gate_mutant
	@log=$${TMPDIR:-/var/tmp}/scratch-poison-mutant-$$$$.log; \
	PULSAR_GATE_DRAFTER=dflash2 ./tests/scratch_poison_gate_mutant $(DFLASH2_GATE_MODEL) 24 > $$log 2>&1; rc=$$?; \
	cat $$log; \
	if [ $$rc -ne 0 ] && grep -q 'SCRATCH-POISON GATE: FAIL' $$log && grep -q 'ok .*the unpoisoned runs agree' $$log; \
	then printf 'SCRATCH-POISON MUTANT: CAUGHT (the gate failed on the stale read; the unpoisoned runs agreed)\n'; \
	else printf 'SCRATCH-POISON MUTANT: NOT CAUGHT -- the gate has no teeth (or its premise broke: see the log)\n'; \
	  exit 1; fi


# L251 S4: Qwen3.8-Flash-Next's PLE n-gram ids vs transformers' own module, bit-exact,
# whole-sequence and chunked (decode steps), plus the row gather over the checkpoint's
# 128 shard tensors through the Engram pool -- HOST ONLY.  Vectors from
# tools/qwen/gen_ngram_vectors.py (the corpus file lives beside the checkpoint on
# the GB10; the small in-tree one covers 4 corpus prefixes + the EOS cases, ids only).
# the container's PENGRAM1 v2 row file (tools/container/ple_rows.py); empty = read the
# checkpoint's shard parts named in the vectors instead
QWEN_NGRAM_ROWFILE ?=
tests/qwen_ngram_test: tests/qwen_ngram_test.cpp src/engine/qwen_ngram.cpp src/engine/qwen_ngram.h \
                       src/engine/engram.cpp src/engine/log.cpp src/engine/alloc.cpp Makefile \
                       src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -o $@ tests/qwen_ngram_test.cpp src/engine/qwen_ngram.cpp \
		src/engine/engram.cpp src/engine/log.cpp src/engine/alloc.cpp -lpthread
.PHONY: qwen-ngram-test
qwen-ngram-test: tests/qwen_ngram_test
	./tests/qwen_ngram_test tests/test-vectors/qwen-ngram-small.vec --no-rows

# L251 S5: the Qwen3.8-Flash-Next tokenizer, chat renderer, client-span map and
# output parser, graded against HF (goldens from tests/qwen/gen_qwen_goldens.py
# -- transformers' apply_chat_template + tokenizers).  Model-free; the inputs are
# the checkpoint's tokenizer.json + generation_config.json, the L216 corpus case
# files and the 466-row calibration corpus HF rendered from them.
QWEN_TOK_DIR ?= $(call need_path,QWEN_TOK_DIR)
QWEN_RENDER_CASES ?= $(call need_path,QWEN_RENDER_CASES)
QWEN_CALIB ?= $(call need_path,QWEN_CALIB)
QWEN_CHAT_SRCS = src/lib/bpe_tokenizer.cpp src/lib/pyjson.cpp src/lib/qwen_chat.cpp src/lib/qwen_output.cpp src/lib/out_parse_kit.cpp
tests/qwen_chat_gate: tests/qwen_chat_gate.cpp tests/bpe_gate_util.h $(QWEN_CHAT_SRCS) src/lib/bpe_tokenizer.h \
                      src/lib/bpe_unicode_tables.inc src/lib/pyjson.h src/lib/qwen_chat.h src/lib/out_parse_kit.h src/lib/pulsar_utf8.h src/lib/chat_refuse.h \
                      src/lib/sha1.hpp Makefile
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/lib -o $@ tests/qwen_chat_gate.cpp $(QWEN_CHAT_SRCS)
.PHONY: qwen-chat-gate
qwen-chat-gate: tests/qwen_chat_gate
	./tests/qwen_chat_gate --tok $(QWEN_TOK_DIR) --vectors tests/test-vectors/qwen38 \
	    --cases $(QWEN_RENDER_CASES) --calib $(QWEN_CALIB)


# L296 N8: the Nemotron-H frontend -- the shared BPE engine in its HF "tekken" configuration (no normaliser,
# ignore_merges) graded against HF tokenizers, the ChatML + <think> renderer against the checkpoint's
# chat_template.jinja through transformers, the output-parser kit's Nemotron configuration against vLLM's nemotron_v3
# parser engine (goldens from tests/nemotron/gen_nemotron_goldens.py).  Model-free; the inputs are the checkpoint's
# tokenizer files and the L216 corpus.
NEMOTRON_TOK_DIR ?= $(call need_path,NEMOTRON_TOK_DIR)
NEMOTRON_CHAT_SRCS = src/lib/bpe_tokenizer.cpp src/lib/pyjson.cpp src/lib/nemotron_chat.cpp src/lib/chat_render_kit.cpp src/lib/qwen_output.cpp \
                     src/lib/out_parse_kit.cpp
tests/nemotron_chat_gate: tests/nemotron_chat_gate.cpp tests/bpe_gate_util.h $(NEMOTRON_CHAT_SRCS) src/lib/bpe_tokenizer.h \
                          src/lib/bpe_unicode_tables.inc src/lib/pyjson.h src/lib/nemotron_chat.h src/lib/qwen_chat.h \
                          src/lib/out_parse_kit.h src/lib/pulsar_utf8.h src/lib/chat_refuse.h src/lib/sha1.hpp Makefile
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/lib -o $@ tests/nemotron_chat_gate.cpp $(NEMOTRON_CHAT_SRCS)
.PHONY: nemotron-chat-gate
nemotron-chat-gate: tests/nemotron_chat_gate
	./tests/nemotron_chat_gate --tok $(NEMOTRON_TOK_DIR) --vectors tests/test-vectors/nemotron_h \
	    --cases $(QWEN_RENDER_CASES) --calib $(QWEN_CALIB)

# L321: the Nemotron-3.5 encoders' tokenizer (tekken + the U+200C Replace pair) graded against HF tokenizers -- T1-T4 of
# the chat gate (adversarial, span guard, every scalar value, the calibration corpus); goldens from
# tests/nemotron_enc/gen_tok_goldens.py.  Model-free; NEMOTRON_ENC_TOK_DIR holds the checkpoint's tokenizer.json +
# config.json (Embed and Rerank ship the same tokenizer).
NEMOTRON_ENC_TOK_DIR ?= $(call need_path,NEMOTRON_ENC_TOK_DIR)
tests/nemotron_enc_tok_gate: tests/nemotron_enc_tok_gate.cpp tests/bpe_gate_util.h src/lib/bpe_tokenizer.cpp \
                             src/lib/pyjson.cpp src/lib/bpe_tokenizer.h src/lib/bpe_unicode_tables.inc src/lib/pyjson.h \
                             src/lib/sha1.hpp Makefile
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/lib -o $@ tests/nemotron_enc_tok_gate.cpp src/lib/bpe_tokenizer.cpp src/lib/pyjson.cpp
.PHONY: nemotron-enc-tok-gate
nemotron-enc-tok-gate: tests/nemotron_enc_tok_gate
	./tests/nemotron_enc_tok_gate --tok $(NEMOTRON_ENC_TOK_DIR) --vectors tests/test-vectors/nemotron_enc --calib $(QWEN_CALIB)

# L338: the Gemma 4 frontend -- the shared BPE engine's SentencePiece-BPE row (Replace(' ' -> U+2581), one piece a run,
# a symbol per character with <0xXX> byte fallback) graded against HF tokenizers (T1-T6), the renderer of the
# checkpoint's chat_template.jinja against transformers' apply_chat_template (R1 hand cases, R2 the L216 corpus) and the
# turn-by-turn front's composition (C1); goldens from tests/gemma4/gen_tok_goldens.py and gen_render_goldens.py.
# Model-free; GEMMA4_TOK_DIR holds the checkpoint's tokenizer.json + generation_config.json (the container's copies).
GEMMA4_TOK_DIR ?= $(call need_path,GEMMA4_TOK_DIR)
GEMMA4_CHAT_SRCS = src/lib/bpe_tokenizer.cpp src/lib/pyjson.cpp src/lib/gemma4_chat.cpp src/lib/gemma4_output.cpp \
                   src/lib/chat_render_kit.cpp src/lib/out_parse_kit.cpp
tests/gemma4_chat_gate: tests/gemma4_chat_gate.cpp tests/bpe_gate_util.h $(GEMMA4_CHAT_SRCS) src/lib/bpe_tokenizer.h \
                        src/lib/bpe_unicode_tables.inc src/lib/pyjson.h src/lib/gemma4_chat.h src/lib/chat_render_kit.h src/lib/out_parse_kit.h \
                        src/lib/chat_refuse.h src/lib/pulsar_utf8.h src/lib/sha1.hpp Makefile
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/lib -o $@ tests/gemma4_chat_gate.cpp $(GEMMA4_CHAT_SRCS)
.PHONY: gemma4-chat-gate
gemma4-chat-gate: tests/gemma4_chat_gate
	./tests/gemma4_chat_gate --tok $(GEMMA4_TOK_DIR) --vectors tests/test-vectors/gemma4 --calib $(QWEN_CALIB) \
	    --cases $(QWEN_RENDER_CASES)

# The attention layout table gate (two profiles, one engine) -- HOST ONLY.  The
# table is a pure function of the artifact's declared metadata, so both profiles'
# mode rows are checkable with no model and no device.  The arrays in the test are
# read out of the real artifacts, not copied from the shape profiles.
tests/attn_layout_test: tests/attn_layout_test.cpp src/engine/model_layout.cpp \
                        src/engine/shape_profiles.cpp src/engine/log.cpp Makefile \
                        src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -o $@ tests/attn_layout_test.cpp \
	    src/engine/model_layout.cpp src/engine/shape_profiles.cpp src/engine/log.cpp

.PHONY: attn-layout-check
attn-layout-check: tests/attn_layout_test
	./tests/attn_layout_test

# The 0731 Compressor state machine + pooling oracle (PLAN 96 s18.2) -- HOST
# ONLY.  Written before the kernels, the way the row codec and the row packer
# were: the state machine is the part of the compressor port most likely to be
# wrong QUIETLY, and it is pure host arithmetic.  Built -fno-fast-math so the
# comparison against the f64 reference vectors is meaningful.
# tests/compressor_pool_fixture.h is GENERATED by compressor_ref.py.
tests/compressor_pool_test: tests/compressor_pool_test.cpp tests/compressor_pool_fixture.h Makefile
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -Itests -o $@ tests/compressor_pool_test.cpp

.PHONY: compressor-pool-check
compressor-pool-check: tests/compressor_pool_test
	./tests/compressor_pool_test

# The V4 indexer SCORING oracle -- HOST ONLY.  The indexer's own compressor and
# the Hadamard are device work (rows/L218.md s29), but the scoring's combine, the
# future mask and the top-k sentinel are pure arithmetic over rows and are where
# its semantics live.  tests/indexer_score_fixture.h is GENERATED by
# gate-baseline/l218-v41/indexer_score_ref.py from the shipped reference.
tests/indexer_score_test: tests/indexer_score_test.cpp tests/indexer_score_fixture.h Makefile
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -Itests -o $@ tests/indexer_score_test.cpp

.PHONY: indexer-score-check
indexer-score-check: tests/indexer_score_test
	./tests/indexer_score_test

# Single-pass mHC hand-over (L218): the fused split collapses with the pre it
# was handed and leaves its own behind; two chained calls vs a host oracle.
# Built the way the engine builds the TU (--use_fast_math).  L295 F2: the TU's router select is the shared router's
# (pulsar_router_select_launch), linked from its production object -- with lt_fixed's, which its logits call (L336).
tests/hc_carry_kernel_test: tests/hc_carry_kernel_test.cu Makefile src/cuda/pulsar_cuda_router.o \
                            src/cuda/pulsar_cuda_lt_fixed.o \
                            src/cuda/pulsar_cuda_hc_router.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h
	$(NVCC) -O3 --use_fast_math -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $< src/cuda/pulsar_cuda_router.o \
	  src/cuda/pulsar_cuda_lt_fixed.o -lcublasLt

# attn_f16_kernel_test takes [n_tokens window n_head bench n_comp ratio top_k
# raw_cap] and its own header argues the compressed and indexed halves matter --
# "wiring the kernel in against only the n_comp=0 path would have shipped that
# half untested" -- but the gate ran the default (n_comp=0) shape ONLY, so both
# halves were untested exactly as that note warned.  Running them found a stale
# oracle: it clamped an out-of-visible top-k selection to row 0, mirroring the
# f32 kernel of the time, while the f16 kernel masks the row to -INF (row 0 substitution
# double-counts row 0 whenever row 0 was also legitimately selected).  Every
# top_k>0 shape disagreed by ~8e-1 and nothing was running to notice.
# attn_f16_banked_test took a "p" argument selecting packed comp banks over
# f32 ones; the comp format parameter is gone from the kernels (2026-08-18), so
# there is one mode and one invocation.
cuda-attn-gates: tests/gqa_attn_gate tests/attn_f16_kernel_test tests/attn_f16_banked_test tests/kv_rows_pack_gate tests/hc_carry_kernel_test tests/sampler_device_gate tests/nt_crossover_sweep tests/exl3_dense_gate tests/exl3_dense_bench tests/exl3_dense_xcheck tests/vision_router_gate tests/vision_hc_gate tests/rope_tail_format_gate
	./tests/attn_f16_kernel_test
	./tests/attn_f16_kernel_test 40 24 32 x 8 4          # compressed tail
	./tests/attn_f16_kernel_test 40 24 32 x 8 4 3        # indexed top-k selection
	./tests/attn_f16_kernel_test 48 16 32 x 12 4 5 20    # indexed + ring raw rows
	./tests/attn_f16_kernel_test 48 16 32 x 12 4 0 20    # decode-batch, no topk table
	./tests/attn_f16_kernel_test 1 16 32 x 12 1 5 20     # ONE-row indexed launch (L166: n_tokens==1 is the same kernel)
	# The SAME kernel on V4's row family.  Every line above builds V4.1 WINDOW/MAIN
	# rows, and the kernel carried no UNIFIED arm at all until L218 s46 -- so the
	# whole gate passed while every V4 attention decoded garbage.  A gate built
	# from one family's fixture cannot fail for the other; these two lines are the
	# other family, and the second is the shape V4 production actually runs.
	./tests/attn_f16_kernel_test 40 24 32 x 8 4 0 0 unified   # compressed tail, UNIFIED rows
	./tests/attn_f16_kernel_test 36 128 64 x 0 0 0 0 unified  # the V4 production shape
	./tests/attn_f16_kernel_test 48 16 32 x 0 0 0 0 vis   # L216 image-span visibility, dense prefill plan
	./tests/attn_f16_kernel_test 48 16 32 x 0 0 0 48 vis  # L216 image-span visibility, scalar-ring plan
	./tests/attn_f16_banked_test
	./tests/hc_carry_kernel_test
	./tests/vision_router_gate $(VISION_ROUTER_GOLDENS)
	./tests/vision_hc_gate
	./tests/rope_tail_format_gate   # L269: the tail rope on f32 / f16 / bf16 storage
	./tests/gqa_attn_gate           # L295 F1: the family-neutral GQA attention block

# The host-only MODEL-FREE oracles, plus the eval extractor self-test.  Every
# one of these was a loose target that no battery ran -- the shape L224 found
# rotted (the agent suite had been ABORTING for days because no battery ran it).
# They are seconds each, need no GPU and no model, and their binaries compile
# their own sources one-shot (no shared objects), so the background sub-make
# cannot race the runner's build.  pulsar-eval is a prerequisite of `gates` for
# the same reason: the background sub-make may only RUN it.  container-tests
# (L279) is the artifact builder's suite: no GPU, the synthetic builds' golden
# SHA-256s, and the oracle suites that read the mounted checkpoints' headers.
host-checks: attn-layout-check engram-hash-check compressor-pool-check \
             indexer-score-check attn-pack-fixture-check tp-core-test tp-command-free-test container-tests pulsar-eval
	./pulsar-eval --self-test-extractors
	python3 tools/coverage_matrix.py   # L278: every contract x family has a battery gate or a declared gap

# L199/L200 candidate #3 picked up for L210: does the expert GEMV's ADDRESS
# ORDER cost bandwidth?  Model-free and standalone -- it allocates one real IQ2
# gate/up tensor's worth of experts and streams it linearly, plane-at-a-time in
# ascending expert order (the kernel's order), and interleaved at 2 KB (what an
# unsorted grid would issue), with and without the per-iteration barrier.  See
# the file header for the comparisons and what each outcome means.
tests/expert_stream_probe: tests/expert_stream_probe.cu Makefile
	$(NVCC) -O3 --use_fast_math -arch=$(ATTN_GATE_ARCH) -Isrc -Isrc/cuda -o $@ $<

.PHONY: expert-stream-probe
expert-stream-probe: tests/expert_stream_probe
	./tests/expert_stream_probe

# Per-kernel census of one single-stream decode step (L210): two nsys runs on the
# same prompt differing by N decode steps, so load + prefill cancel.  This is the
# instrument behind the decode bucket table in rows/L210.md; run it on the Spark
# (needs nsys and the production artifact) with no other pulsar process live.
.PHONY: decode-kernel-census
decode-kernel-census:
	python3 tests/decode_kernel_census.py --binary ./pulsar \
		--model $(FRONTIER_MODEL)

# Backend-seam enforcement (see the contract atop src/pulsar_gpu.h): nothing
# outside src/cuda/ may touch CUDA APIs directly. tools/seam_check.py strips
# comments/strings first, so docs may explain the backend; code may not call it.
seam-check:
	@python3 tools/seam_check.py

# The DeepSeek model the standalone DeepSeek targets below run on.  MODEL-DEPENDENT -- run manually on the GB10, not
# part of `make test`.  Discipline before running: no pulsar-server/pulsar_test process,
# `sync; echo 3 > /proc/sys/vm/drop_caches`, check `free -g` headroom (the model is ~87 GB).
#
# VISION-EXP, not 0731 (2026-09-21, Tyler: "we should NOT be using that as the
# battery model.  we haven't been using that model in a long time").  The
# default was `./ds4flash.gguf`, a symlink to the 0731 REAP25 artifact
# `v5mx4-0731-srcfmt-v1-reapfix-lt.gguf` -- superseded, and overridden with
# `FRONTIER_MODEL=` on every recent battery anyway, so the default named a model
# nothing ran.  It was not harmless: the prefill-baseline comment below records a
# blob dumped through the `ds4flash.gguf` symlink silently comparing two
# DIFFERENT artifacts' logits whenever the battery named the Vision-Exp one,
# reporting 129280/129280 differing at every depth -- a FAIL with no information
# in it.  Naming the served artifact makes the default and the practice one fact.
# This one variable names the DeepSeek artifact for the gates that still load it.  `unit-test-gate` and gates-dev's
# pulsar_test runs load the served Qwen pack (QWEN_GATE_MODEL) instead since L357 (DeepSeek is smoke-only, 2026-11-07).
#
# SAFETENSORS, 2026-09-21: the GGUF container is retired, so this names the
# artifact itself -- the per-layer-shard safetensors checkpoint the engine reads
# and the promotion lane swaps shards in.  Every gate takes this as a PATH, so
# one line moved the whole battery off GGUF.
FRONTIER_MODEL ?= $(call need_path,FRONTIER_MODEL)

# L115 token-seam gate: sampled-vs-canonical boundary drift keeps the live
# KV (see the header of tests/token_seam_gate.cpp).  MODEL-DEPENDENT.
cuda-seam-gate: tests/token_seam_gate
	PULSAR_MSEQ_BANKS=3 ./tests/token_seam_gate $(FRONTIER_MODEL)

# Tier-2 multiseq gate: co-scheduling neutrality, mixed-entry byte identity,
# ALL_ROWS head mode, stale-classic guard (see the header of
# tests/multiseq_decode_gate.cpp).  MODEL-DEPENDENT — run manually on the GB10
# with the memory discipline above.
cuda-multiseq-gate: tests/multiseq_decode_gate
	PULSAR_MSEQ_BANKS=3 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 3 64

# The same gate with speculation DISABLED — the pulsar-bench/pulsar-eval/agent and
# `pulsar-server --no-dspark` config, and a different allocation shape (no DSpark
# graph state).  The driver must work there; it used to reject every step.
# Shorter (N=2, 64 steps): this is a config gate, not a throughput run.
cuda-multiseq-gate-nodspark: tests/multiseq_decode_gate
	PULSAR_MSEQ_BANKS=2 PULSAR_GATE_NO_DSPARK=1 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 2 64

# L151: measurement, not a gate -- the M-independent nt kernels vs the
# tensor-core arms per dense shape at M = 1..16 (real weights). Decides the
# default row cap.
cuda-nt-sweep: tests/nt_crossover_sweep
	./tests/nt_crossover_sweep $(FRONTIER_MODEL) 4

# L216: the Vision-Exp image-layout math (grid/shrink/block) graded against
# goldens generated by RUNNING the checkpoint's inference/image_processor.py.
# Host-only and instant -- no model, no GPU -- so it runs in the battery.
vision-layout-gate: tests/vision_layout_gate
	./tests/vision_layout_gate tests/test-vectors/vision-layout-goldens.txt

# L216: the pixel half -- Pillow-faithful resample/contain/pad, normalisation,
# bf16 rounding and patchify -- graded against patches the reference produced
# from the same decoded RGB.  Host-only; the codec is deliberately out of scope.
vision-pixel-gate: tests/vision_pixel_gate
	./tests/vision_pixel_gate tests/test-vectors/vision-pixel-goldens.bin

# L216: our decoder (libpng + libjpeg-turbo) vs Pillow's, on the same bytes --
# the same two libraries the reference decodes with, checked per format.
vision-codec-gate: tests/vision_codec_gate
	./tests/vision_codec_gate tests/test-vectors/vision-codec-goldens.bin

# L216: the composed per-image path (decode + preprocess + sentinel span) vs the
# reference's load_image + build_image_block.  EXACT -- no tolerance.
VISION_SPAN_GOLDENS ?= tests/test-vectors/vision-span-goldens.bin
vision-span-gate: tests/vision_span_gate
	./tests/vision_span_gate $(VISION_SPAN_GOLDENS)

# L216: the ViT + aligner forward vs the checkpoint's own vision.py.  This one
# needs a VISION-EXP artifact (the battery's FRONTIER_MODEL is 0731, which has no
# tower), so it is NOT in tests/manifest.toml and refuses to silently pass when
# VISION_MODEL is unset.
VISION_MODEL ?=
# L216: the WHOLE per-image lane, bytes in -> HC carrier out, vs the reference.
# This is the instrument that proves the image lane ran: decode, preprocess,
# span discovery, tower, and placement in one call.  Needs a VISION-EXP artifact.
VISION_IMAGE_GOLDENS ?= tests/test-vectors/vision-image-goldens.bin
vision-image-gate: tests/vision_image_gate
	@if [ -z "$(VISION_MODEL)" ]; then \
		echo "  SKIP  vision-image-gate: set VISION_MODEL=/path/to/a/vision-exp/checkpoint"; \
	else \
		echo "  running against $(VISION_MODEL)"; \
		./tests/vision_image_gate $(VISION_MODEL) $(VISION_IMAGE_GOLDENS); \
	fi

# L216: an image request through the PUBLIC API (pulsar_session_sync_mm), the
# whole engine on a real artifact.  Needs a VISION-EXP artifact.
VISION_IMAGE_GOLDENS_SYNC ?= tests/test-vectors/vision-image-goldens.bin
vision-image-sync-gate: tests/vision_image_sync_gate
	@if [ -z "$(VISION_MODEL)" ]; then \
		echo "  SKIP  vision-image-sync-gate: set VISION_MODEL=/path/to/a/vision-exp/checkpoint"; \
	else \
		echo "  running against $(VISION_MODEL)"; \
		./tests/vision_image_sync_gate $(VISION_MODEL) $(VISION_IMAGE_GOLDENS_SYNC); \
	fi

vision-merge-gate: tests/vision_merge_gate
	@if [ -z "$(VISION_MODEL)" ]; then \
		echo "  SKIP  vision-merge-gate: set VISION_MODEL=/path/to/a/vision-exp/checkpoint"; \
	else \
		echo "  running against $(VISION_MODEL)"; \
		./tests/vision_merge_gate $(VISION_MODEL) tests/test-vectors/vision-merge-goldens.bin; \
	fi

# L216: the MoE router's image-slot bias (bias_vl) vs a line-for-line
# transcription of the reference's Gate.forward.  Needs a GPU, NO model: the
# bias / bias_vl / tid2eid tensors are synthetic and laid out in one model_map.
# L216: where a MERGED image span lands in the HC carrier -- the replication
# across every HC stream and the offset write.  GPU, no model.
cuda-vision-hc-gate: tests/vision_hc_gate
	./tests/vision_hc_gate

VISION_ROUTER_GOLDENS ?= tests/test-vectors/vision-router-goldens.bin
cuda-vision-router-gate: tests/vision_router_gate
	./tests/vision_router_gate $(VISION_ROUTER_GOLDENS)

# L216: image-span VISIBILITY -- the reference's get_image_visible /
# get_window_topk_idxs_visible, which is what makes a span bidirectional.
# Pure integer arithmetic, EXACT, no GPU and no model, so it runs in the battery.
vision-visible-gate: tests/vision_visible_gate
	./tests/vision_visible_gate tests/test-vectors/vision-visible-goldens.bin

# L216: the sentinel-id PRODUCER -- placeholder token in, `vocab_size + role`
# block out, and the reader must recover exactly what the writer wrote.  No GPU
# and no model, so it runs in the battery.
vision-placeholder-gate: tests/vision_placeholder_gate
	./tests/vision_placeholder_gate tests/test-vectors/vision-image-goldens.bin

# L268: Qwen's image preprocessing vs HF's own processor (tests/vision_qwen_pixel_goldens.py), bit for bit; host-only.
vision-qwen-pixel-gate: tests/vision_qwen_pixel_gate
	./tests/vision_qwen_pixel_gate tests/test-vectors/vision-qwen-pixel-goldens.bin

# L268: Qwen's tower + merger vs HF's own modules (tests/vision_qwen_tower_goldens.py), graded against the reference's
# bf16-vs-fp32 floor like vision-tower-gate; runs on the served Qwen container (it carries model.visual.*).
vision-qwen-tower-gate: tests/vision_qwen_tower_gate
	./tests/vision_qwen_tower_gate $(QWEN_GATE_MODEL) tests/test-vectors/vision-qwen-tower-goldens.bin

# L331: Nemotron 3.5 Super VL's image path.  The fixture is tests/nemotron_vision_goldens.py's (the checkpoint's own
# processor / C-RADIO / projector code, FP32 on any CPU).
#   nemotron-vision-pixel-gate   HOST, model-free: the tile rule (every size), the pixels bit-exact to torch's
#                                processor (f32 and the bf16 the tower reads), the patch order, the image front
#                                (a tower-less engine refuses by name; the expansion's block)
#   nemotron-vision-tower-gate   GPU: the device tower's LLM rows vs the FP32 reference, each image within 3x the
#                                reference's own bf16-vs-fp32 floor (L268's rule); NEMOTRON_VISION_MODEL holds the tower
#   nemotron-vision-smoke        GPU: one image question through the engine (tests/image_chat_smoke.cpp): "red"
#   nemotron-vision-live         GPU: a real server (tests/nemotron_vision_live.py): V1-V8 name the images' content,
#                                and the prefix cache resumes past the same image, not past a different one
NEMOTRON_VISION_MODEL ?= /srv/models/nemotron35-k4-lilicorr-vision-pulsar
.PHONY: nemotron-vision-pixel-gate nemotron-vision-tower-gate nemotron-vision-smoke nemotron-vision-live
nemotron-vision-pixel-gate: tests/nemotron_vision_pixel_gate
	./tests/nemotron_vision_pixel_gate tests/test-vectors/nemotron-vision-goldens.bin
nemotron-vision-tower-gate: tests/nemotron_vision_tower_gate
	./tests/nemotron_vision_tower_gate $(NEMOTRON_VISION_MODEL) tests/test-vectors/nemotron-vision-goldens.bin
nemotron-vision-smoke: tests/image_chat_smoke
	python3 -c "import sys;sys.path.insert(0,'tests');import nemotron_vision_live as L;open('/var/tmp/nemotron-vision-smoke.png','wb').write(L.disc())"
	./tests/image_chat_smoke $(NEMOTRON_VISION_MODEL) /var/tmp/nemotron-vision-smoke.png "What colour is the circle in this image? Answer with one word." red
nemotron-vision-live: pulsar-server
	python3 tests/nemotron_vision_live.py $(NEMOTRON_VISION_MODEL) --binary ./pulsar-server
tests/nemotron_vision_pixel_gate.o: tests/nemotron_vision_pixel_gate.cpp src/families/nemotron_h/nemotron_h.h \
                                    src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/nemotron_vision_pixel_gate.cpp
tests/nemotron_vision_pixel_gate: tests/nemotron_vision_pixel_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/nemotron_vision_tower_gate.o: tests/nemotron_vision_tower_gate.cpp src/families/nemotron_h/nemotron_h.h \
                                    src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/nemotron_vision_tower_gate.cpp
tests/nemotron_vision_tower_gate: tests/nemotron_vision_tower_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

vision-tower-gate: tests/vision_tower_gate
	@if [ -z "$(VISION_MODEL)" ]; then \
		echo "  SKIP  vision-tower-gate: set VISION_MODEL=/path/to/a/vision-exp/checkpoint"; \
	else \
		echo "  running against $(VISION_MODEL)"; \
		./tests/vision_tower_gate $(VISION_MODEL) tests/test-vectors/vision-tower-goldens.bin; \
	fi

# plan-34 phase-2 inc 2: cuBLASLt algo-stability. A decode bank's step logits must
# be byte-identical across batched-step widths M (incl. the M=4->5 custom->cuBLASLt
# boundary) so a co-scheduled big prefill (inc 4) cannot perturb it. MODEL-DEPENDENT,
# needs PULSAR_MSEQ_BANKS>=8. Run pack on/off + idx-fp4 on/off under GPU discipline.
cuda-algo-stability-gate: tests/algo_stability_gate
	PULSAR_MSEQ_BANKS=16 ./tests/algo_stability_gate $(FRONTIER_MODEL)
# L175: bank 0 at ~2200 tokens (indexed lane engaged) under the same width sweep.
cuda-algo-stability-gate-deep: tests/algo_stability_gate
	PULSAR_MSEQ_BANKS=16 ./tests/algo_stability_gate $(FRONTIER_MODEL) deep

# L260 fusion phase A: a FROM-ZERO prefill run co-scheduled with a decode row in
# one mixed step -- coherent vs classic from zero, decode row neutral.
cuda-mixed-zero-prefill-gate: tests/mixed_zero_prefill_gate
	PULSAR_MSEQ_BANKS=2 ./tests/mixed_zero_prefill_gate $(FRONTIER_MODEL)

# L260 fusion phase B: verify rows + prompt chunks in one fused step -- verify rows
# byte-identical to a decode-only step, the prompt's final row and drafter ring
# matching a classic prefill.
cuda-fused-step-gate: tests/fused_step_gate
	PULSAR_MSEQ_BANKS=4 ./tests/fused_step_gate $(FRONTIER_MODEL)
# L260: decode rows at widths 1/4/9/16 graded against the reference blobs (report only).
decode-reference-gate: tests/decode_reference_gate
	PULSAR_MSEQ_BANKS=2 ./tests/decode_reference_gate $(FRONTIER_MODEL) $(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin
	PULSAR_MSEQ_BANKS=2 ./tests/decode_reference_gate $(FRONTIER_MODEL) $(PULSAR_REF_DIR)/code.ref.bin $(PULSAR_REF_DIR)/code.tokens.bin
# L284 probe (measurement, never fails on a number): EVERY row of a verify-shaped step at widths
# 1..32, one bank and four banks a step, against the same rows decoded one token at a time, and the
# base row against the B300 reference -- which arm is closer to the source.
verify-width-probe: tests/verify_width_probe
	PULSAR_MSEQ_BANKS=2 ./tests/verify_width_probe $(FRONTIER_MODEL) $(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin --max-depth 6144 --extra 1024,3072,5120
	PULSAR_MSEQ_BANKS=4 ./tests/verify_width_probe $(FRONTIER_MODEL) $(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin --banks 4 --widths 4,8,16,24,32 --max-depth 6144 --extra 1024,3072,5120
	PULSAR_MSEQ_BANKS=2 ./tests/verify_width_probe $(FRONTIER_MODEL) $(PULSAR_REF_DIR)/code.ref.bin $(PULSAR_REF_DIR)/code.tokens.bin --extra 1024,1536,2560,3072
# L288 C1: Qwen's wide verify (32 rows: 8 streams x 4) the same way -- every row against the one-token decode
# (plain decode steps: Qwen's verify leaves its bank for the commit to move; expected identical, the decode arms
# run 16-row blocks) and the container capture.
.PHONY: verify-width-probe-qwen
verify-width-probe-qwen: tests/verify_width_probe
	PULSAR_MSEQ_BANKS=8 ./tests/verify_width_probe $(QWEN_GATE_MODEL) $(QWEN_REF_ANCHOR_DIR)/story.ref.bin $(QWEN_REF_ANCHOR_DIR)/story.tokens.bin --banks 8 --widths 8,16,24,32 --max-depth 6144 --extra 1024,3072,5120 --serial plain
	PULSAR_MSEQ_BANKS=8 ./tests/verify_width_probe $(QWEN_GATE_MODEL) $(QWEN_REF_ANCHOR_DIR)/code.ref.bin $(QWEN_REF_ANCHOR_DIR)/code.tokens.bin --banks 8 --widths 8,16,24,32 --extra 1024,1536,2560,3072,3500 --serial plain

# Artifact-only, no GPU, seconds: does the ROUTER agree with the artifact's own
# REAP declaration? A router left in source expert order against compacted
# expert weights routes every token to the wrong expert and NOTHING FAILS --
# in range, no NaN, still fluent, just missing most of what the model knows.
# srcfmt-v1 shipped exactly that on 40 of 43 layers. Cheap enough to run on
# every artifact before it is ever loaded.
cuda-reap-router-audit:
	python3 tools/reap/audit_reap_router.py $(FRONTIER_MODEL)

# The artifact builder's module suites (tools/container, L247), no GPU, minutes:
# names/policy against every declaration of the served Vision-Exp artifact
# (shard, gguf_name, layout, dims_ne), kv entry-for-entry, every producer
# byte-identical to the served bytes; the qwen4_exp family (L251) end to end on a
# synthetic miniature checkpoint (recipe refusals, emit twice byte-identical,
# verify --roundtrip, the PLE row file); the DeepSeek family (L279) on its own
# synthetic miniature.  Both synthetic suites hold GOLDEN SHA-256s of every
# emitted file, fixed before the L279 refactor: the builder's byte-identity
# instrument.  The oracle paths live in the tests and
# refuse when the checkpoints are not mounted; the codecs need numpy, so pass
# the interpreter that has it (CONTAINER_PY=.../.venv/bin/python).
CONTAINER_PY ?= python3
# The oracles test_names / test_kv / test_producers read (local.mk): the served DeepSeek container, the Vision-Exp
# and V4.1 HF snapshots (a glob matching one directory is fine) and a glob of the V4.1 EXL3 shards.
.PHONY: container-tests
container-tests:
	cd tools/container && for t in test_exl3_rates.py test_deepseek.py test_qwen.py test_nemotron_h.py test_nemotron_enc.py test_nemotron_parse.py test_gemma4.py; do \
	  $(CONTAINER_PY) $$t || exit 1; done

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
# 0.196->0.190).  See the private notes' bit-exact-vs-source-2026-08-24.md.
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
# MOVED 13bcbab1 -> 5fe1831d on 2026-11-06 (L354), BY TYLER'S OVERRIDE: the CUDA 13.5 + CUTLASS v4.8.1 toolchain moved the
# depths with a small tail (see the blob's .md and the dated note at cuda-reference-gate-budget).
# MOVED f5bea7aa -> 13bcbab1 on 2026-10-13 (L292), BY TYLER'S OVERRIDE, not by
# the CLOSER rule above.  The 0731 NVFP4 KV row's two divides are IEEE
# (__fdiv_rn) instead of fast-math MUFU.RCP x FMUL, which makes the device row
# exactly the host recipe (rule 7) and moves a rare KV code.  The reference gate
# graded it a noise-level AWAY from an FP8-KV B300 capture that cannot arbitrate
# NVFP4 rounding; see the dated note at cuda-reference-gate-budget.  13bcbab1 is
# the pre-squash tip of the fix; the landed commit's engine sources equal it.
PREFILL_BASELINE_REF ?= d3e488fe
# L357: the byte baselines are Qwen's (the production model), recorded by BASELINE_MODEL's dump at the ref.
BASELINE_MODEL ?= $(QWEN_GATE_MODEL)
# The baseline blob is COMMITTED (L046): a fresh clone can run cuda-prefill-gate
# with no bootstrap step, and the gate's guarantee no longer depends on a loose
# file surviving in somebody's tree. The name carries the anchor ref, and the
# default composes from PREFILL_BASELINE_REF so the two cannot drift apart:
# re-anchoring = update the REF above, `make cuda-prefill-gate-baseline` (dumps
# straight to the new tracked name), then commit the blob + this line together.
# The blob is ~2.5 MB and self-describing (ref stamp + model header + token FNV;
# see tests/prefill_bitexact_gate.cpp and the .md next to the blob).
#
# ⚠ A BLOB IS ONLY MEANINGFUL FOR THE ARTIFACT IT WAS DUMPED ON (L218 s124).
# The header pins the logits width and the prompt FNV -- NOT the weights -- so a
# blob dumped through a DIFFERENT artifact silently compared two models' logits,
# reporting 129280/129280 differing at every depth -- a FAIL with no information
# in it.  Specifically: `ds4flash.gguf` used to symlink to the 0731 REAP25
# artifact while the battery named the Vision-Exp one.
#
# RESOLVED 2026-09-21, and this is now structural rather than a warning.  The
# 0731 artifact is deleted, `FRONTIER_MODEL` names the Vision-Exp artifact
# directly (no symlink indirection), and the `ds4flash.gguf` symlink points at
# that same file -- so the default, the battery's model and the symlink are ONE
# fact and the trap cannot be re-entered by taking a different route to the
# default.  The 684fa8d blob of the deleted 0731 artifact was deleted 2026-10-30
# (git history keeps it; its note is docs/history/).  13bcbab1 is
# the Vision-Exp blob the battery names (the f5bea7aa one it replaced was dumped
# by a clean dev checkout's own binary; this one by `cuda-prefill-gate-baseline` at the
# ref, see its .md).
PREFILL_BASELINE     ?= tests/test-vectors/prefill_bitexact_baseline-qwen-$(PREFILL_BASELINE_REF).bin
PREFILL_DECODE_BASELINE_REF ?= d3e488fe
PREFILL_DECODE_BASELINE     ?= tests/test-vectors/prefill_decode_baseline-qwen-$(PREFILL_DECODE_BASELINE_REF).bin
PREFILL_DECODE_BASELINE_WT  ?= temp/wt-prefill-decode-baseline
PREFILL_DECODE_BASELINE_REF_SHORT := $(shell git rev-parse --short $(PREFILL_DECODE_BASELINE_REF) 2>/dev/null || echo $(PREFILL_DECODE_BASELINE_REF))
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
# DIAGNOSTIC, deliberately NOT in tests/manifest.toml.  Spec-vs-greedy byte equality is
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

# L278 contract 6: the live server tier -- a real pulsar-server per family on a scratch port, driven over HTTP
# (tests/server_live.py: plain answer, forced / streamed / required tool calls, the tool_result continuation, the
# undeclared-tool refusal).  L317: the battery runs it as a leg of live-legs (one shared server per model); this target
# runs it alone, on its own server.  SERVER_LIVE_MODELS to choose.
SERVER_LIVE_MODELS ?= $(FRONTIER_MODEL) $(QWEN_GATE_MODEL)
# L303: structured outputs live (tests/constrained_live.py): every constrained output valid over seeds at T 0 and
# T 0.8, thinking off and on; spec == plain byte-exact for each (logprobs requests decode without speculation);
# the protocols' format fields; the cost against the same prompt unconstrained; a matcher failure (injected,
# PULSAR_GRAMMAR_FAIL_AFTER) fails only its request.  L317: the battery runs it as a leg of live-legs; this target runs
# it alone (two server loads per family) on CONSTRAINED_LIVE_MODELS.
CONSTRAINED_LIVE_MODELS ?= $(SERVER_LIVE_MODELS)
.PHONY: constrained-live-gate
constrained-live-gate: pulsar-server
	@rc=0; for m in $(CONSTRAINED_LIVE_MODELS); do \
	  sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	  python3 tests/constrained_live.py $$m --binary ./pulsar-server || rc=1; \
	done; exit $$rc

.PHONY: server-live-gate
server-live-gate: pulsar-server
	@rc=0; for m in $(SERVER_LIVE_MODELS); do \
	  sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	  python3 tests/server_live.py $$m --binary ./pulsar-server || rc=1; \
	  sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	  python3 tests/server_abandon_live.py $$m --binary ./pulsar-server || rc=1; \
	done; exit $$rc

# L316: an agent loop's reuse == cold on the production Qwen pack (tests/prefix_turns_live.py --reply 600: eight
# Anthropic tool turns with ~150-400-token replies, thinking echoed; the last turn in the loop and re-sent alone ==
# a fresh server's, byte for byte, on one prompt).  L317: the battery runs it as a leg of live-legs (Qwen's shared
# server); this target runs it alone (two server loads).  ~4 min.
PREFIX_TURNS_LIVE_MODEL ?= $(QWEN_GATE_MODEL)
PREFIX_TURNS_LIVE_WORK ?= $(CURDIR)/.build/prefix-turns-live
.PHONY: prefix-turns-live-gate
prefix-turns-live-gate: pulsar-server
	sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	python3 tests/prefix_turns_live.py $(PREFIX_TURNS_LIVE_MODEL) --binary ./pulsar-server \
	  --work $(PREFIX_TURNS_LIVE_WORK) --tag gate --reply 600
# L317: the battery's live legs on ONE server per model (tests/live_legs.py): constrained-live, server-live and the
# abandon leg on DeepSeek and Qwen, the L312 stall check, prefix-turns (L316's gate and the side shape), preempt-exact,
# (L318) resume-deepest, (L322) sidecall-turns and (L329) fanout-prefix on Qwen,
# opencode-turns on Nemotron's LiLiCorr pack.  Per model: the shared server, then -- only if a leg needs it -- one
# instance with the test hooks (the injected grammar failure, the KV-budget override), then one fresh instance for
# every leg's reuse == cold reference.  Each leg keeps its own checks and PASS/FAIL; the targets above still run a
# leg alone.  The last section of `make gates`.  LIVE_LEGS="a,b" runs only those.
# L278 battery cut (2026-11-03, Tyler): the battery's live legs run on the served families (Qwen, Nemotron); DeepSeek's
# (not served; ~985 s) are live-legs-deepseek, which gates-dev runs on a DeepSeek path change.
LIVE_LEGS ?=
LIVE_LEGS_WORK ?= $(CURDIR)/.build/live-legs
LIVE_LEGS_MODELS ?= --model qwen=$(QWEN_GATE_MODEL) --model nemotron=$(LILICORR_GATE_MODEL)
.PHONY: live-legs live-legs-deepseek
live-legs: pulsar-server
	python3 tests/live_legs.py --binary ./pulsar-server --work $(LIVE_LEGS_WORK) $(LIVE_LEGS_MODELS) \
	  $(if $(LIVE_LEGS),--legs $(LIVE_LEGS),)
live-legs-deepseek: pulsar-server
	python3 tests/live_legs.py --binary ./pulsar-server --work $(LIVE_LEGS_WORK)-deepseek \
	  --model deepseek=$(FRONTIER_MODEL) $(if $(LIVE_LEGS),--legs $(LIVE_LEGS),)

# L357: the published-model smoke (tests/model_smoke_live.py): every model we publish starts on a pulsar-server of its
# own, one at a time, and answers the one request its kind serves -- a chat model "7+5" (thinking off) with 12, an
# embedder a vector of its advertised dimension, a reranker Paris above bananas for "capital of France", the parser a
# 200 on a text-free PNG.  A model that does not start FAILS by name; no skips.  DeepSeek V4 (smoke-only since
# 2026-11-07) gets its release check here.  The SMOKE_* models are the published retrieval and Gemma containers.
SMOKE_GEMMA_MODEL ?= $(call need_path,SMOKE_GEMMA_MODEL)
SMOKE_EMBED_MODEL ?= $(call need_path,SMOKE_EMBED_MODEL)
SMOKE_RERANK_MODEL ?= $(call need_path,SMOKE_RERANK_MODEL)
SMOKE_PARSE_MODEL ?= $(call need_path,SMOKE_PARSE_MODEL)
# Qwen and Nemotron (QWEN_GATE_MODEL, LILICORR_GATE_MODEL) take the same check as a live leg on live-legs' warm instance
# (L357: no server of their own for it); name them here to smoke them alone.
SMOKE_MODELS ?= $(FRONTIER_MODEL) $(SMOKE_GEMMA_MODEL) $(SMOKE_EMBED_MODEL) $(SMOKE_RERANK_MODEL) $(SMOKE_PARSE_MODEL)
MODEL_SMOKE_WORK ?= $(CURDIR)/.build/model-smoke
model-smoke: pulsar-server
	python3 tests/model_smoke_live.py $(SMOKE_MODELS) --binary ./pulsar-server --work $(MODEL_SMOKE_WORK)

# L280: a prompt's KV is admitted against the 2b guard's budget before it lands (spill first / preempt the newest
# decoder, L314 / 503), never past it into memory pressure.  A real server on a SMALL model (the guard's budget lowered
# through its test hook to 2.2 conversations), so it fits beside a running production server; ~2 min.  Family-neutral.
# The guarded servers' logs are kept (L326) in .build/kv-admission/servers.log.
KV_ADMISSION_MODEL ?= $(call need_path,KV_ADMISSION_MODEL)
.PHONY: server-kv-admission-live
server-kv-admission-live: pulsar-server
	mkdir -p .build/kv-admission
	python3 tests/server_kv_admission_live.py $(KV_ADMISSION_MODEL) --binary ./pulsar-server \
	  --log .build/kv-admission/servers.log

# L319: the touched-KV ceiling is the box's measured headroom (min of the ledger's share and MemAvailable - floor at the
# post-warmup reading), so long sessions spill / wait at the floor instead of running the machine dry (the work hang:
# Nemotron at --ctx 1M allowed 9.05 GiB of touched KV against 5.16 GiB of headroom; MemAvailable reached zero and the
# GPU stalled).  A balloon leaves the server --headroom-gib over the floor, then conversations of ~2.4x that land on
# fresh banks; MemAvailable is sampled throughout and a dip past the floor stops the run as a FAIL.  Needs the box to
# itself (a quiet gpuq job, production stopped); Nemotron Super (the published container), ~5 min, under a timeout.  gates-dev runs it on an
# admission / guard / bank path.
KV_HEADROOM_MODEL ?= $(call need_path,KV_HEADROOM_MODEL)
# L357: every test program too (each tests/<name> rule, read from `make -qp`, and the battery mutants): the test sources
# had gathered 68 warnings while the gate covered only the engine.  One build, so the engine's objects are compiled once.
# L354 (F23): the landing rule's zero warnings and zero doxygen warnings, as gates that can fail.  warnings-gate builds a copy
# of this tree (tracked and new files, so uncommitted edits count) from nothing, so an object already built can't hide a
# warning, and greps the log for both formats (gcc's `: warning:`, nvcc's front-end `: warning #NNN-D:`).  CUTLASS is
# -isystem: its own headers raise about 16,000 under -Wextra and aren't ours.  Both run before the runner: a build's memory
# beside the accounting gate would fail it.
.PHONY: warnings-gate doxygen-gate
warnings-gate:
	@d=$$(mktemp -d $${TMPDIR:-/var/tmp}/pulsar-warnings-XXXXXX) || exit 1; \
	trap 'rm -rf "$$d"' EXIT; \
	git ls-files -co --exclude-standard -z | tar --null -T - -cf - | tar -xf - -C "$$d" || exit 1; \
	[ ! -f local.mk ] || cp local.mk "$$d/"; \
	progs=$$($(MAKE) -C "$$d" -qp CUTLASS_DIR=$(abspath $(CUTLASS_DIR)) 2>/dev/null | grep -oE '^tests/[A-Za-z0-9_]+:' \
	  | tr -d : | sort -u); \
	$(MAKE) -C "$$d" -j$(GATE_JOBS) --no-print-directory pulsar-server pulsar pulsar_test pulsar_agent_test \
	  $$progs $(BATTERY_MUTANT_NAMES:%=tests/gates_runner_mutant_%) \
	  CUDA_ARCH=$(or $(CUDA_ARCH),sm_120f) CUTLASS_DIR=$(abspath $(CUTLASS_DIR)) > "$$d/build.log" 2>&1; rc=$$?; \
	grep -E ': warning( #[0-9]+-D)?:' "$$d/build.log" | grep -v '^make' | sort -u > "$$d/warnings"; n=$$(wc -l < "$$d/warnings"); \
	head -40 "$$d/warnings"; \
	[ $$rc -eq 0 ] || { tail -30 "$$d/build.log"; echo "WARNINGS GATE: FAIL (the build failed, rc $$rc)"; exit 1; }; \
	[ "$$n" -eq 0 ] || { echo "WARNINGS GATE: FAIL ($$n warnings)"; exit 1; }; \
	echo "WARNINGS GATE: PASS (a build from nothing, the engine and $$(echo $$progs | wc -w) test programs + the mutants, 0 warnings)"
doxygen-gate:
	@rm -f docs/api/doxygen-warnings.log; \
	(cat docs/Doxyfile; echo "HAVE_DOT = NO") | doxygen - > /dev/null 2>&1 || { echo "DOXYGEN GATE: FAIL (doxygen did not run)"; exit 1; }; \
	[ -f docs/api/doxygen-warnings.log ] || { echo "DOXYGEN GATE: FAIL (no warnings log: the Doxyfile moved it?)"; exit 1; }; \
	if [ -s docs/api/doxygen-warnings.log ]; then head -40 docs/api/doxygen-warnings.log; \
	  echo "DOXYGEN GATE: FAIL ($$(wc -l < docs/api/doxygen-warnings.log) lines)"; exit 1; fi; \
	echo "DOXYGEN GATE: PASS (an empty warnings log)"

# L355: what a request resumes from after a disconnect, a spill and a restart.  In the release tier it is a live leg
# (L357: live-legs' hooks instance is the warm one's restart and takes budgets through the test route); this runs it alone,
# on instances of its own.
.PHONY: resume-fixes-live
resume-fixes-live: pulsar-server
	python3 tests/resume_fixes_live.py $(QWEN_GATE_MODEL) --binary ./pulsar-server

# L353 no. 5: a parked decoder holds no KV, so a prompt that can never fit gets its 503 and the decoder comes back.  In
# the release tier it is a live leg (L357: its budget through the test route on live-legs' hooks instance); this runs it
# alone.
.PHONY: preempt-wedge-live
preempt-wedge-live: pulsar-server
	python3 tests/preempt_live.py $(QWEN_GATE_MODEL) wedge --binary ./pulsar-server

.PHONY: server-kv-headroom-live
server-kv-headroom-live: pulsar-server
	python3 tests/server_kv_headroom_live.py $(KV_HEADROOM_MODEL) --binary ./pulsar-server

# 2026-10-29, the work hang as a leg: Nemotron 3.5 Super (K4 + LiLiCorr, 16 banks, --ctx 1M), three conversations grown
# by ~15k-token continued prefills toward 200k under a balloon that leaves the derived floor + 3 GiB.  PASS: no request
# hangs, the stall watchdog stays quiet, MemAvailable never goes more than 0.25 GiB under the floor, persists happen or
# are skipped loudly, every answer is 200 or a named 503, no uvm / page-allocation line in dmesg.  ~1-2 h, the box to
# itself (a quiet gpuq job, production stopped).  Not in gates-dev: run it for a guard / floor / persist change.
MEM_SQUEEZE_MODEL ?= $(call need_path,MEM_SQUEEZE_MODEL)
.PHONY: nemotron-mem-squeeze-live
nemotron-mem-squeeze-live: pulsar-server
	python3 tests/nemotron_mem_squeeze_live.py $(MEM_SQUEEZE_MODEL) --binary ./pulsar-server

# L318: an Open WebUI conversation (the history rewritten inside the previous turn, ~30k-token prompts) resumes within
# one segment of its byte match from the deepest valid state -- own bank, adopt, disk chain -- a prefill its client
# abandons keeps its computed prefix for the re-send, and both stay reuse == cold against a fresh server.  Production's
# Qwen pack and flags (262k, 16 banks, 64 GB disk tier); two loads, ~8 min.  gates-dev runs it on a routing / kv path.
RESUME_LIVE_MODEL ?= $(QWEN_GATE_MODEL)
RESUME_LIVE_WORK ?= $(or $(TMPDIR),/var/tmp)/pulsar-l318-$(USER)
.PHONY: resume-deepest-live
resume-deepest-live: pulsar-server
	python3 tests/resume_deepest_live.py $(RESUME_LIVE_MODEL) --binary ./pulsar-server --work $(RESUME_LIVE_WORK)

# L322: Claude Code's SUGGESTION MODE side call extends a conversation's bank; the next real turn routes in place on
# that bank, rewinds to its own checkpoint below the divergence and prefills the delta (not a fresh bank + a disk
# chain), and stays reuse == cold against a fresh server.  The Qwen pack, 4 banks, 64k; two loads, ~3 min.  The battery
# runs it as a leg of live-legs; gates-dev runs it on a routing / kv path.
SIDECALL_LIVE_WORK ?= $(CURDIR)/.build/sidecall-turns-live
.PHONY: sidecall-turns-live
sidecall-turns-live: pulsar-server
	sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	python3 tests/sidecall_turns_live.py $(RESUME_LIVE_MODEL) --binary ./pulsar-server --work $(SIDECALL_LIVE_WORK)

# L324: the duplicate-prefix census (/metrics pulsar:kv_dup_*): three fan-out siblings sharing a ~14k-token system
# prompt, each on a fresh bank that adopts the prefix, show their shared pages as duplicates.  The Qwen pack, 4 banks,
# 64k; one load, ~2 min.  The battery runs it as a leg of live-legs (Qwen, Nemotron); gates-dev on a routing / kv path.
KV_DUP_LIVE_WORK ?= $(CURDIR)/.build/kv-dup-live
.PHONY: kv-dup-live
kv-dup-live: pulsar-server
	sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	python3 tests/kv_dup_live.py $(RESUME_LIVE_MODEL) --binary ./pulsar-server --work $(KV_DUP_LIVE_WORK)

# L329 (a): the shared-prefix wait -- ten fan-out siblings sent 0.25 s apart over a NEW ~20k-token system prompt
# prefill it once (the first stops at the shared cut, the rest wait queued and adopt it), and each reply (one token,
# its top-5 logprobs) == the request alone on a fresh server.  The Qwen pack, 16 banks, 64k; two loads (three with
# FANOUT_AB=1: the same burst with PULSAR_SERVER_PREFIX_WAIT_MS=0, the last first token on vs off), ~3-5 min.  The
# battery runs it as a leg of live-legs (Qwen); gates-dev on a routing / kv path.
FANOUT_LIVE_WORK ?= $(CURDIR)/.build/fanout-prefix-live
FANOUT_AB ?=
.PHONY: fanout-prefix-live
fanout-prefix-live: pulsar-server
	sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	python3 tests/fanout_prefix_live.py $(RESUME_LIVE_MODEL) --binary ./pulsar-server --work $(FANOUT_LIVE_WORK) \
	  $(if $(FANOUT_AB),--ab,)

# L330: a disk KV restore loads its chain a slice of segments per worker turn, so a decoding stream beside it stalls
# about one slice, not the whole read.  A ~200k-token prompt abandoned at 90% of its prefill (its prefix stored as a
# chain), then re-sent from the dropped page cache while a greedy stream decodes; the slice's test hook
# (PULSAR_SERVER_KV_RESTORE_SLICE_MS=250) cuts the ~2.4 GiB chain into many slices.  The Qwen pack, 262k, 4 banks; one
# load, ~5 min.  gates-dev runs it on a routing / kv path.
RESTORE_STALL_LIVE_WORK ?= $(CURDIR)/.build/restore-stall-live
.PHONY: restore-stall-live
restore-stall-live: pulsar-server
	sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	python3 tests/restore_stall_live.py $(RESUME_LIVE_MODEL) --binary ./pulsar-server --work $(RESTORE_STALL_LIVE_WORK)

cuda-spec-width-gate: pulsar
	python3 tests/spec_verify_width_gate.py $(FRONTIER_MODEL) --binary ./pulsar

cuda-prefill-gate:
	# NOT -B any more.  Staleness is fully expressed now: headers via -MMD -MP
	# + `-include $(PULSAR_DEPS)`, compiler configuration via CUDA_FLAG_STAMP.
	# -B rebuilt the entire engine twice per suite to cover the flag axis alone.
	$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(BASELINE_MODEL) --check $(PREFILL_BASELINE) \
		$(PREFILL_BASELINE_REF_SHORT)


# NOTE the shape above (2026-09-21): the conditional WRAPS the recipe.  It used
# to be `@if ...; then echo SKIP; exit 0; fi` followed by the build and the run as
# SEPARATE recipe lines, and that does not skip anything -- `exit 0` ends only
# that one recipe line's shell, so the next line ran anyway and the gate graded
# an EMPTY model path.  It never showed because the model always existed, so the
# branch had never executed; removing the 0731 default is what fired it.  Any
# future SKIP in this file must wrap, not precede.

# L181: decode-step byte gate (see PREFILL_DECODE_BASELINE above).
cuda-prefill-decode-gate:
	$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(BASELINE_MODEL) --check-decode $(PREFILL_DECODE_BASELINE) \
		$(PREFILL_DECODE_BASELINE_REF_SHORT)
cuda-prefill-decode-gate-baseline:
	git worktree remove --force $(PREFILL_DECODE_BASELINE_WT) 2>/dev/null || true
	rm -rf $(PREFILL_DECODE_BASELINE_WT)
	git worktree prune
	git worktree add --detach $(PREFILL_DECODE_BASELINE_WT) $(PREFILL_DECODE_BASELINE_REF)
	cp tests/prefill_bitexact_gate.cpp $(PREFILL_DECODE_BASELINE_WT)/tests/
	cp Makefile $(PREFILL_DECODE_BASELINE_WT)/
	cp tests/long_context_story_prompt.txt $(PREFILL_DECODE_BASELINE_WT)/tests/
	$(MAKE) -C $(PREFILL_DECODE_BASELINE_WT) tests/prefill_bitexact_gate \
		CUDA_ARCH=sm_120f CUTLASS_DIR=$(abspath $(CUTLASS_DIR))
	cd $(PREFILL_DECODE_BASELINE_WT) && ./tests/prefill_bitexact_gate $(abspath $(BASELINE_MODEL)) \
		--dump-decode $(abspath $(PREFILL_DECODE_BASELINE))
	git worktree remove --force $(PREFILL_DECODE_BASELINE_WT)
	@echo "decode baseline ($(PREFILL_DECODE_BASELINE_REF)) -> $(PREFILL_DECODE_BASELINE)"

# Cross-engine fidelity gate: grade prefill logits against the vLLM reference
# blobs from the B300 capture.  This is the only gate that measures us against
# something we did not produce -- every other gate compares us to ourselves,
# which cannot catch an error we make consistently.
#
# The blobs are ~4.6 MB and live OUTSIDE this repo (the private notes are private and
# stays that way), so the path comes from PULSAR_REF_DIR.  When it is unset or
# the blobs are missing this SKIPS LOUDLY rather than passing: a gate that
# quietly succeeds when its fixture is absent is worse than no gate, which is
# exactly the hole L078's golden fixture fell into.
#
# Per-depth KL budgets: the gate grades DIRECTION against these (closer to the
# source = pass).  Absent, it falls back to the absolute-ceiling check only,
# which cannot see direction -- see the 2026-08-24 note in
# the private notes' bit-exact-vs-source-2026-08-24.md.  RECORDED FROM THE SERVED
# MODEL (Vision-Exp, dev 4499f93d, L240, 2026-09-23, `--dump-kl`); the 0731
# budgets they replaced graded a different model's numbers.  RE-RECORDED
# 2026-10-13 at 13bcbab1 (L292 NVFP4 exact divide) under Tyler's documented
# OVERRIDE -- see the dated note at cuda-reference-gate-budget.
KL_BUDGET_STORY ?= tests/test-vectors/kl-budget-story.txt
KL_BUDGET_CODE  ?= tests/test-vectors/kl-budget-code.txt
# The reference capture the battery grades against BY DEFAULT (L240): the
# Vision-Exp B300 capture staged on the Spark (blobs + token files, 4.7 MB;
# source: the 2026-09-23 reference-capture archive (vexp/out) + the shared
# *.tokens.bin, same prompt fnv).  Every battery from the Vision-Exp switch to
# 2026-09-23 printed "SKIP cuda-reference-gate" because nothing set this;
# rule 7 was not being exercised.  A configured dir with no readable blob is a
# FAIL in the runner, never a skip -- stage the capture or override with an
# empty value to skip deliberately (PULSAR_REF_DIR=).
PULSAR_REF_DIR ?= $(call need_path,PULSAR_REF_DIR)
# ⚠ ONE SHELL, DELIBERATELY.  Each make recipe LINE gets its own shell, so an
# `exit 0` in a guard on the first line exits only that line and make runs the
# rest anyway -- which is exactly how the first version of this target failed
# `make gates` with "cannot read reference blob /story.ref.bin" whenever
# PULSAR_REF_DIR was unset (i.e. by default).  The guard and the work must sit
# in the same shell for the skip to be a skip.
# The manual form runs the same two sub-gates the battery runs, through the
# runner, so the documented outlier depths and the skip/refuse rule have ONE
# home (tests/gates_runner.cpp).  The standalone tests/prefill_bitexact_gate
# binary still builds (LEG C of tools/tp-pair-engine-grade.sh runs it on every
# rank of a pair, and it takes --dump-kl to record a budget).
cuda-reference-gate: tests/gates_runner
	@./tests/gates_runner $(RUNNER_PRIMARY_MODEL) --prefill-baseline $(PREFILL_BASELINE) \
		--prefill-ref $(PREFILL_BASELINE_REF_SHORT) \
		--decode-baseline $(PREFILL_DECODE_BASELINE) --decode-ref $(PREFILL_DECODE_BASELINE_REF_SHORT) \
		$(RUNNER_PRIMARY_ANCHORS) --only=cuda-reference-gate-story,cuda-reference-gate-code

# Re-record the KL budgets from the CURRENT tree.  Same discipline as
# PREFILL_BASELINE_REF: do this only when a change has been GRADED CLOSER to the
# source and adopted, and say why in the commit.  Re-recording to silence a
# regression is the one thing that makes this gate worthless.
#
# A change graded UNCHANGED (L211: every moving depth under the 1e-6 noise floor
# on both sides) lands WITHOUT a re-record.  The budget stays where the last
# CLOSER change put it, so sub-floor moves accumulate against that anchor and
# the first one to cross the floor votes -- re-recording after each would reset
# the drift to zero and let it walk.
#
# ⚠ OVERRIDE ON RECORD (2026-10-13, Tyler: "I'd land the nvfp4"; ledger L292).
# The budgets were re-recorded on an AWAY grade, the one thing this note forbids,
# as an explicit, documented exception: the L292 NVFP4 exact divide (13bcbab1)
# graded story NET over the confident depths 1.54e-5 -> 1.96e-5 (2048 and 4102
# closer, 4096 and 6144 farther; every top-1 matches; code blob PASS), and the
# B300 capture was taken with kv_cache_dtype = fp8 (vllm 0.28.1rc1.dev137), so
# it cannot arbitrate FP4 rounding.  The exact divide makes the device row equal
# the host recipe (rule 7).  Next re-grade: an NVFP4-KV B300 capture (#42).  This
# is not a precedent: any other AWAY grade still means reject, not re-record.
#
# 2026-11-06 (L354), a second exception, Tyler's: the CUDA 13.5 + CUTLASS v4.8.1 toolchain graded story NET +0.78% (1.958e-5 ->
# 1.973e-5), all of it depth 4102 (2.162e-6 -> 2.316e-6), every top-1 matching, code unmoved.  "we're not gonna stay pinned
# on an old toolchain. just regrade it and reset the baseline."  The cause inside the toolchain was not split.
.PHONY: cuda-reference-gate-budget
cuda-reference-gate-budget:
	@if [ -z "$(PULSAR_REF_DIR)" ] || [ ! -f "$(PULSAR_REF_DIR)/story.ref.bin" ]; then \
		echo "REFUSING: set PULSAR_REF_DIR to the reference-capture dir"; exit 1; fi
#
# Report-only on purpose (no KL_TOL, no outlier anchors): a record run measures,
# it does not grade, so a row that would fail the grade still gets recorded.
	$(MAKE) tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
		$(PULSAR_REF_DIR)/story.ref.bin $(PULSAR_REF_DIR)/story.tokens.bin \
		--dump-kl $(KL_BUDGET_STORY)
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check-reference \
		$(PULSAR_REF_DIR)/code.ref.bin $(PULSAR_REF_DIR)/code.tokens.bin \
		--dump-kl $(KL_BUDGET_CODE)
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
	cd $(PREFILL_BASELINE_WT) && ./tests/prefill_bitexact_gate $(abspath $(BASELINE_MODEL)) \
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
# L161: a 1-row decode step and an N-row step must be the same numerics --
# the attention tier used to be chosen by row count and the battery never
# compared the two.  Same token on the same prefilled bank: classic eval,
# 1-row decode_mixed, 2-row decode_mixed (and a different batchmate); any bit
# of difference fails.  MODEL-DEPENDENT, ~2 min.
cuda-row-neutrality-gate: tests/mseq_short_ctx_probe
	./tests/mseq_short_ctx_probe $(FRONTIER_MODEL)
# L170: the same probe at ~2217 tokens of context, where the top-k selection is
# engaged and the indexed fold order must not depend on the row count.
cuda-row-neutrality-gate-deep: tests/mseq_short_ctx_probe
	./tests/mseq_short_ctx_probe $(FRONTIER_MODEL) 1100
# L175: ~10k tokens (n_comp ~2600): the 1-row indexer scorer vs the N-row
# tier and the ranking kernel past the 2048-row bucket, on logits.
cuda-row-neutrality-gate-deeper: tests/mseq_short_ctx_probe
	./tests/mseq_short_ctx_probe $(FRONTIER_MODEL) 4200
# L183: a prompt's logits do not depend on its chunking (cold, 6-row first
# chunk, two mid-size chunks, off-grid warm resume -- frontier + one decode step
# byte-identical).  The plain-weight GEMM arm's fixed reduction order is what
# makes it hold; a split-K pick fails it from layer 0.
# L317: this target is the exhaustive sweep (every schedule); the battery's runner entry runs the seeded sample
# (`sample`, PULSAR_GATE_CHUNK_SEED = the sha) and gates-dev's cuda-chunk-neutrality-gate-full the sweep.
cuda-chunk-neutrality-gate: tests/chunk_neutrality_gate
	./tests/chunk_neutrality_gate $(FRONTIER_MODEL)
# L283: an image block's logits do not depend on where its prefill chunk starts (before, at, straddling the grid,
# continued after) -- byte-identical to the one-chunk prefill, the bar text meets above.  In the runner on every
# hosted model that serves images (NEED_VISION).
cuda-vision-chunk-gate: tests/vision_chunk_gate
	./tests/vision_chunk_gate $(FRONTIER_MODEL) $(VISION_IMAGE_GOLDENS)
cuda-spec-sampling-gate: tests/spec_sampling_gate
	./tests/spec_sampling_gate $(SPEC_GATE_MODEL) $(SPEC_GATE_ARGS)
# L288 C1: the same oracle with its spec arm at 8 banks (up to 32 rows: the shared verify width, past the 16 the
# decode kernels take in one block), so the wide verify is checked by the sampled distribution.  Measurement beside the
# gate (its chi2 is informational, as the narrow one's); dump both builds (argv 6) and compare the spec arms with
# tests/spec_sampling_compare.py for the like-with-like grade.
.PHONY: cuda-spec-sampling-gate-wide
cuda-spec-sampling-gate-wide: tests/spec_sampling_gate_wide
	./tests/spec_sampling_gate_wide $(SPEC_GATE_MODEL) $(SPEC_GATE_ARGS)
# L182: teacher-forced drafter acceptance -- E[accept] = sum min(p, q) over a fixed
# context walk, no free-running trajectories, so a one-ulp drafter change is a
# 1e-4 move instead of a re-drawn sample.  A quality GAUGE for drafter-touching
# changes (run it beside the oracle); not release-blocking, so not in the runner.
spec-teacher-forced-probe: tests/spec_teacher_forced_probe
	./tests/spec_teacher_forced_probe $(SPEC_GATE_MODEL) $(SPEC_TF_POSITIONS)
SPEC_TF_POSITIONS ?= 2000

# L163: every model-dependent gate as a function in one binary.  Each gate
# source is compiled a second time with its entry renamed (GATE_ENTRY) and the
# engine calls routed to the runner's broker (PULSAR_GATE_RUNNER); the
# standalone binaries above are the same sources built without those defines.
RUNNER_GATES = payload_frontier_gate bank_residency_gate token_seam_gate \
               multiseq_decode_gate algo_stability_gate spec_sampling_gate mseq_short_ctx_probe prefill_bitexact_gate \
               chunk_neutrality_gate vision_chunk_gate \
               decode_reference_gate session_contract_gate chat_decode_smoke \
               sync_budget_gate scratch_poison_gate mtp_spec_gate bank_adopt_gate bank_isolation_gate \
               kv_accounting_gate graph_eager_gate
RUNNER_OBJS = $(RUNNER_GATES:%=tests/runner/%.o)
tests/runner/%.o: tests/%.cpp tests/gate_entry.h tests/gate_fixture.h tests/gate_util.h src/pulsar.h src/pulsar_gpu.h src/engine/pulsar_engine_internal.h src/lib/pulsar_segstore.h src/lib/pulsar_kvchain.h
	@mkdir -p tests/runner
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -DPULSAR_GATE_RUNNER -DGATE_ENTRY=gate_$*_main \
		-DPULSAR_GATE_BUILD_REF='"$(GATE_BUILD_REF)"' -c -o $@ $<
tests/gates_runner.o: tests/gates_runner.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/gates_runner.cpp
# The host libraries the runner links beside its gates (its mutants below link the same).
RUNNER_LIBS = src/lib/pulsar_help.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_kvtext.o
tests/gates_runner: tests/gates_runner.o $(RUNNER_OBJS) $(RUNNER_LIBS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# The runner receives exactly what the individual targets passed: the prefill
# baseline blob and its ref, the reference-capture dir (skips loudly when
# unset), the tolerance and the KL budgets (passed only when present).
# GATE_RUNNER_FLAGS carries the battery-invisible diagnostics (--shape prints
# each gate's prefill/step work shape); it is empty for `make gates`.
GATE_RUNNER_FLAGS ?=
# L278: the runner is a family host -- each model here runs after FRONTIER_MODEL through the same table, every
# gate whose needs it lacks skipped by name (session-contract and row-neutrality run on every family).  A new
# family adds its served model here.
# Each entry is `--model PATH` and that model's reference anchors; it must follow the primary's anchors (an option
# binds to the last --model before it).
# L317: Nemotron's LiLiCorr pack is hosted for the legs that loaded it themselves (block-spec, scratch-poison,
# bank-adopt): --gates restricts it to them.  Models are hosted one at a time (each closed before the next opens).
# L357: and graph-eager-gate, whose graphed driver steps Qwen does not take (its driver stages none: SKIPped there).
RUNNER_HOSTED_ARGS ?= --model $(LILICORR_GATE_MODEL) \
	--gates=block-spec-gate,scratch-poison-gate,bank-adopt-gate,graph-eager-gate
# Extra runner options for the primary (e.g. `--needs=fixtures`); empty = its whole table.  gates-dev calls the runner
# itself and is unaffected.
RUNNER_PRIMARY_ARGS ?=
# L357 (Tyler 2026-11-07): Qwen, the production model, is the primary and runs the whole table; DeepSeek left the
# runner (smoke only), and its mechanism gates were retired with it (the family-neutral ports above carry their intent).
RUNNER_PRIMARY_MODEL ?= $(QWEN_GATE_MODEL)
RUNNER_PRIMARY_ANCHORS ?= $(QWEN_REF_ANCHORS)
# L317: the seed of the chunk-neutrality sample is the sha the battery runs at (successive batteries rotate through
# the schedules; one sha always runs the same ones).
cuda-runner-gate: tests/gates_runner
	PULSAR_GATE_CHUNK_SEED=$(GATE_BUILD_REF) ./tests/gates_runner $(RUNNER_PRIMARY_MODEL) --prefill-baseline $(PREFILL_BASELINE) \
		--prefill-ref $(PREFILL_BASELINE_REF_SHORT) \
		--decode-baseline $(PREFILL_DECODE_BASELINE) --decode-ref $(PREFILL_DECODE_BASELINE_REF_SHORT) \
		$(RUNNER_PRIMARY_ANCHORS) $(RUNNER_PRIMARY_ARGS) $(RUNNER_HOSTED_ARGS) $(GATE_RUNNER_FLAGS)

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
#   make gates FRONTIER_MODEL=/path/to/models/<checkpoint-dir>
# The unit suite (make test) runs INSIDE the battery since 2026-09-03: it was
# the one test set a landing could skip, and L156 found a golden in it that had
# been failing since May because nobody ran it.  Same model as the frontier
# gates so one FRONTIER_MODEL= names the artifact for everything.
# seam-check used to be a prerequisite here (host-only, ~instant).  L220 moved
# it to the manifest's overlap tests: it still runs in the battery, just alongside the GPU
# tail instead of in front of pulsar_test.  A seam-check failure still fails the
# battery (it lands in the same pass/fail fold).
# 2026-10-14: --gating, every test but the informational ones (logprob-vectors: a drift dashboard whose mismatches
# can never fail the suite, at the price of its own model load).  `make test` still runs everything.
# L357: on the served Qwen pack (QWEN_GATE_MODEL), not the DeepSeek fixture: L353 no. 1 (image segment keys) escaped onto
# served Qwen while the image tests ran only on DeepSeek, which is smoke-only since 2026-11-07.  --gating runs the image
# tests (--image-chain among them), so they need no target of their own.  PULSAR_TEST_QWEN_TOKENIZER names the same
# checkpoint for the server unit tests' host-side tokenizer (src/server/server_tests.cpp reads it on its own).
unit-test-gate: pulsar_test
	PULSAR_TEST_MODEL="$(QWEN_GATE_MODEL)" PULSAR_TEST_QWEN_TOKENIZER="$(QWEN_GATE_MODEL)" ./pulsar_test --gating

# The AGENT's host-only unit suite (the DSML parser/detector over every syntax
# row, the edit anchors, and the KV store's stripped-checkpoint refusal).  It is
# in the battery because it rotted outside one: the V4 spelling rows (L218) made
# agent_dsml_close_force_min() ABORT the agent on any tool-argument value
# containing a '<', and the test that catches exactly that had been failing --
# and aborting the suite -- since those rows landed (L224).
agent-test-gate: pulsar_agent_test
	./pulsar_agent_test


# L325: the prefix-stability replay gate.  A client that changes its request only in cache-preserving ways (a tool
# loaded by ToolSearch: defer_loading + tool_reference; reminders in the last user turn; mid-conversation system
# messages; a suggestion side call; Nemotron's truncate_history_thinking) must find the previous request's rendered
# prompt in the next one up to the previous request's last cache_control breakpoint.  Claude Code 2.1.310's captured
# requests (tests/prefix-stability/) plus synthetic shapes, rendered by the server's renderer for every family
# (pulsar_test --render-cases, no model, no GPU), the first differing byte named on a failure.
.PHONY: prefix-stability-gate
prefix-stability-gate: pulsar_test
	python3 tests/prefix_stability_gate.py --pulsar-test ./pulsar_test

# The test tiers' one list is tests/manifest.toml (L357): what runs, in which tier and order, which tests overlap the
# runner (overlap = true: no GPU, no model), and every budget.  `make gates` and `make release` run it through
# tests/run.py; there is no second list here.  The model-dependent gates run inside ONE process, cuda-runner-gate (L163:
# tests/gates_runner.cpp) -- one model load per engine configuration instead of one per gate; their individual targets
# remain for iterating on one gate.  The live legs share one server per model (live-legs; L357: the hooks instance is
# the warm one's restart), and server-kv-headroom-live runs last, alone (it balloons the box's memory).
#
# Every test target is phony, declared from the manifest (the .PHONY line at the top of the file expands before this):
# a file named like a gate would otherwise satisfy make and print nothing -- the silent-PASS shape the tracked-binary
# incident documented (L178).
MANIFEST_TESTS := $(shell python3 tests/run.py --names 2>/dev/null)
.PHONY: $(MANIFEST_TESTS) cuda-vision-chunk-gate vision-tower-gate vision-span-gate vision-merge-gate vision-image-gate vision-image-sync-gate

# The numerics-critical subset, for the ITERATION loop.  `make gates` is a
# pre-merge instrument -- every release test in tests/manifest.toml, each loading the model,
# with spec_sampling alone running 2x2500 trajectories; it is the wrong tool to
# sit and watch after every edit.  (This said "17 gates" as a second copy of a
# fact that lives in the manifest and had drifted to 10; the list is the count.)
# These five are the ones that actually catch a numerics or dispatch regression:
#   cuda-prefill-gate        full-vocab logits byte-identical at 5 depths
#   cuda-chunk-neutrality-gate  the same tokens under four chunkings give byte-identical logits (L183)
#   bank-isolation-gate-qwen   an idle bank untouched by its neighbours' batched steps
#   session-contract-gate-qwen the session contract, batched rows == single (B1-B6)
#   cuda-attn-gates          attention kernel + bank isolation
# SPEC_TRAJ shortens the sampling gate's chi-square when it is included; the
# full TRAJ is required for statistical power, so quick mode leaves it out
# rather than run it under-powered and call it a pass.
# NOT a substitute for `make gates` before a merge.
GATES_QUICK_TARGETS = cuda-attn-gates cuda-prefill-gate cuda-chunk-neutrality-gate bank-isolation-gate-qwen \
                      session-contract-gate-qwen

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

# tests/gates_runner is a PREREQUISITE, not just an in-loop target: L220 starts
# the host-only gates as background sub-makes at cuda-runner-gate, and building
# the runner here (before any recipe line runs) means the whole CORE_OBJS set
# is fresh before those sub-makes start -- they then only compile their own
# test TU and link, so concurrent make processes cannot race shared objects.
# The battery's own model-load discipline, run before the first gate (L239,
# 2026-09-23): the same three checks every hand-written gate script carries.
# /tmp on the Spark is tmpfs, i.e. the same memory the engines load into --
# 17 GB of scratch left there OOMed a battery's second engine while
# MemAvailable read fine.  On a box with no GPU the battery cannot run at all,
# so the preflight only says so; it never "passes" by skipping on the Spark.
.PHONY: gates-preflight
gates-preflight:
	@if [ ! -e /dev/nvidiactl ]; then echo "gates-preflight: no GPU on this host -- the battery is a Spark job"; exit 0; fi; \
	bad=0; \
	live=$$(for p in /proc/[0-9]*; do e=$$(readlink "$$p/exe" 2>/dev/null) || continue; case "$${e##*/}" in pulsar|pulsar-server*|pulsar-bench|pulsar-eval|pulsar_test|gates_runner) echo "$${p#/proc/} $$e";; esac; done); \
	if [ -n "$$live" ]; then echo "gates-preflight: an engine process is alive (by executable):"; echo "$$live" | head -3 | sed 's/^/  /'; bad=1; fi; \
	tmp=$$(df -k --output=fstype,used /tmp 2>/dev/null | awk 'NR==2 && $$1=="tmpfs" {printf "%d", $$2/1048576}'); \
	if [ -n "$$tmp" ] && [ "$$tmp" -gt 4 ]; then \
	  echo "gates-preflight: /tmp is tmpfs (memory) and holds $${tmp} GiB -- move it off before loading engines:"; \
	  du -xsh /tmp/* 2>/dev/null | sort -rh | head -5 | sed 's/^/  /'; bad=1; fi; \
	avail=$$(awk '/MemAvailable/ {printf "%d", $$2/1048576}' /proc/meminfo); \
	if [ "$$avail" -lt 100 ]; then echo "gates-preflight: only $${avail} GiB available; the battery loads the served artifact (~87 GB)"; bad=1; fi; \
	if [ $$bad -ne 0 ]; then echo "gates-preflight: REFUSING"; exit 1; fi; \
	echo "gates-preflight: ok ($${avail} GiB available, /tmp $${tmp:-0} GiB in tmpfs)"

# The binaries of the serial GPU targets are prerequisites too (2026-10-14): they build here, in one parallel make,
# instead of one target at a time inside the timed loop (pulsar_test alone was ~40 s of unit-test-gate).
TEST_BINS = tests/gates_runner pulsar-server pulsar-eval tests/qwen_chat_gate tests/nemotron_chat_gate tests/api_golden \
       tests/sse_golden tests/spec_depth_gate \
       tests/loader_open pulsar_test pulsar_agent_test pulsar $(CUDA_REGRESSION_BINS)
.PHONY: test-bins check kernels release
test-bins: $(TEST_BINS)
# The tiers (L357, tests/manifest.toml): `make check`, `make kernels`, `make release`.
check kernels release:
	python3 tests/run.py $@
# `make gates`: the name every note and rule uses for the release battery -- the release tier, nothing else (L357)
gates: $(TEST_BINS)
	python3 tests/run.py release

# ---- gates-dev: the ITERATION tier (L220 phase 4) --------------------------
#
# `make gates` is the PRE-MERGE instrument: every release-blocking gate, one engine open per configuration (its wall
# time: tests/manifest.toml).  `make gates-dev` is the ITERATION
# tier: a fast subset chosen from what the working tree actually touches, so the
# loop between an edit and a numeric answer is one-to-two minutes.
#
# IT IS NOT A SUBSTITUTE FOR `make gates`, and it is not a release gate.  A
# green gates-dev says the lanes this diff touches still agree with themselves;
# it does not say the suite is green.  Run `make gates` before merging.
#
# Selection.  PATHS="p1 p2" is the explicit path set; by default it is the
# working tree's tracked edits plus untracked SOURCES from git.  Each path is
# classified by the lane it belongs to and mapped to a sub-gate list below; the
# runner runs them in ONE process (--only=), so the tier pays a single engine
# open.  The mapping is deliberately NARROW: the expensive general gates
# (prefill, spec-sampling, accounting, rewind) stay in `make gates` only, and the
# always-on block -- seam-check, the reap-router audit, and the host-only
# pulsar_test switches -- is cheap and lane-independent.  L317: the runner's DEV
# entries are this tier's -- the exhaustive chunk-neutrality sweep on the prefill /
# chunk / kv_state / checkpoint / session / segment-store paths, and the probes'
# deeper depths on attention and kernel-dispatch paths (the battery runs one
# depth of each and a seeded sample of the chunk schedules).
# GATES_DEV_ONLY="a b" overrides the selection outright (the runner refuses an
# unknown name and FAILS on an empty selection).
#
# ⚠ Pass CUDA_ARCH exactly as the battery does -- `make gates-dev
# CUDA_ARCH=sm_120f FRONTIER_MODEL=...`.  CUDA_ARCH is part of the CUDA flag
# stamp (`.build/cuda-flags.stamp`), so an invocation that leaves it empty
# rewrites the stamp for the empty arch and the next `make gates` rebuilds
# every .cu.  Every sub-make below carries it for the same reason.
GATES_DEV_UNIT ?= --ctxmem --spec-math --lib-utf8 --lib-think --verify-pick --fused-cost --fused-sched --ckpt-records --ckpt-host-tier --read-ahead
GATES_DEV_UNIT_SERVER ?= --server --api-sampling-flags --api-min-p-range --api-logprobs-parse --api-count-tokens
# attention kernels: their own oracle (cuda-attn-gates) + the fused mixed step,
# which is where an attention/compressor row change shows up as a logits delta.
# L317: the row-neutrality probe's depths (-deep ~2.2k, -deeper ~10k tokens) and algo-stability's -deep are gates-dev
# entries of the runner (the battery runs one depth of each): attention and kernel-dispatch paths run them.
GATES_DEV_ATTN   = session-contract-gate cuda-row-neutrality-gate-deep cuda-row-neutrality-gate-deeper cuda-vision-chunk-gate
# kernel dispatch: row-kind/batch-composition neutrality and the width sweep.
GATES_DEV_CUDA   = session-contract-gate cuda-algo-stability-gate cuda-algo-stability-gate-deep \
                   cuda-row-neutrality-gate-deep cuda-row-neutrality-gate-deeper
# engine lanes: the batched driver, the spec lane, the bank pool.
GATES_DEV_ENGINE = cuda-multiseq-gate bank-isolation-gate bank-residency-gate payload-frontier-gate
GATES_DEV_SERVER = cuda-seam-gate payload-frontier-gate
# vision: the host gates below do the real work (they need no model); the runner
# sub-gates pin the prefill lane an image span rides and (L283) its chunk-start invariance.
GATES_DEV_VISION = session-contract-gate cuda-row-neutrality-gate-deep cuda-vision-chunk-gate
# Nothing recognizable changed (or a clean tree): the conservative core -- the
# dispatch/neutrality gates that fail on the widest class of numerics breaks.
GATES_DEV_DEFAULT= session-contract-gate cuda-algo-stability-gate cuda-row-neutrality-gate-deep cuda-multiseq-gate

.PHONY: gates-dev
gates-dev:
	@t0=$$(date +%s); rc=0; \
	if [ -z "$(CUDA_ARCH)" ]; then \
	  printf 'gates-dev: CUDA_ARCH is empty -- pass the served arch (CUDA_ARCH=sm_120f)\n'; \
	  printf '  or the CUDA flag stamp flips and the next `make gates` rebuilds every .cu\n'; \
	fi; \
	$(MAKE) -j$(GATE_JOBS) --no-print-directory tests/gates_runner pulsar_test CUDA_ARCH=sm_120f || exit 1; \
	$(MAKE) --no-print-directory gates-preflight || exit 1; \
	paths='$(PATHS)'; why=''; attn=0; server=0; render=0; vision=0; exl3=0; family=0; gdn=0; router=0; blocks=0; bf16=0; minp=0; drafter=0; poison=0; resume=0; kvmem=0; chunk=0; ds4legs=0; \
	if [ -z "$$paths" ]; then \
	  paths=$$( { git diff --name-only HEAD; git ls-files --others --exclude-standard; } 2>/dev/null \
	            | grep -E '\.(c|cc|cpp|cu|cuh|h|hpp)$$' | sort -u ); \
	fi; \
	sel='$(GATES_DEV_ONLY)'; \
	if [ -z "$$sel" ]; then \
	  add() { case " $$sel " in *" $$1 "*) ;; *) sel="$$sel $$1";; esac; }; \
	  if [ -z "$$paths" ]; then \
	    why=" (no changed sources)"; \
	    for g in $(GATES_DEV_DEFAULT); do add $$g; done; \
	  else \
	    for p in $$paths; do \
	      case "$$p" in \
	        *vision*) cls=vision; vision=1 ;; \
	        src/engine/family*|src/families/*|src/engine/role_bind*|tests/nemotron_*|src/engine/*qwen*|tests/qwen_*|tests/session_contract*|src/engine/state_kinds*|src/engine/step_driver*|tests/state_kinds*|src/engine/session*|src/engine/safetensors*|src/engine/model.cpp|src/engine/weights*|src/engine/tensor_bind*|tests/loader_*|src/engine/sync_driver*|src/engine/prefill_loop*|src/engine/checkpoint*|src/engine/kv_state*) cls=engine; family=1 ;; \
	        *exl3*) cls=exl3; exl3=1 ;; \
	        *gdn*) cls=gdn; gdn=1 ;; \
	        *sconv*|*mamba2*|src/cuda/pulsar_cuda_head*|tests/head_*|*rowops*) cls=blocks; blocks=1 ;; \
	        src/cuda/*attn*|src/cuda/*attention*) cls=attn; attn=1 ;; \
	        src/cuda/*norm_kv*) cls=attn; attn=1 ;; \
	        src/cuda/*) cls=cuda ;; \
	        src/server/*) cls=server; server=1 ;; \
	        src/engine/*) cls=engine ;; \
	        *) cls=default ;; \
	      esac; \
	      why="$$why $$p($$cls)"; \
	      case "$$p" in *router*|*moe*) router=1 ;; esac; \
	      case "$$p" in *bf16_rows*|*nvfp4*|*weight_format*|src/engine/linear.cpp|src/engine/moe.cpp) bf16=1 ;; esac; \
	      case "$$p" in *sampler*|*libm*|src/engine/spec_dspark.cpp|src/engine/session_spec.cpp|src/engine/spec_mtp.cpp|src/engine/spec_block.cpp) minp=1 ;; esac; \
	      case "$$p" in *spec_block*|*blockdraft*|*candsel*|*lilicorr*|*mtp_spec*|*attn_gqa*|*rope*|src/engine/session_spec.cpp|src/families/nemotron_h/*|src/engine/family_qwen.cpp) drafter=1 ;; esac; \
	      case "$$p" in src/server/server_jobs.cpp|src/server/server_sched.cpp|src/server/kv_cache.cpp|src/lib/pulsar_kvchain*|src/lib/pulsar_segstore*|src/lib/pulsar_kvdigest*|src/lib/pulsar_kvpool*) resume=1 ;; esac; \
	      case "$$p" in src/server/*|src/lib/*_chat*|src/lib/pyjson*|src/lib/bpe_tokenizer*|src/lib/pulsar_json*|src/lib/out_parse*|src/lib/chat_render*) render=1 ;; esac; \
	      case "$$p" in src/server/cli_main.cpp|src/server/server_sched.cpp|src/engine/state_kinds*|src/engine/session_banks*|tests/server_kv_headroom_live.py) kvmem=1 ;; esac; \
	      case "$$p" in src/engine/family_deepseek*|src/engine/kv_state_ds4*|src/engine/spec_dspark*|src/engine/gpu_*|src/cuda/pulsar_cuda_csa2*|src/cuda/pulsar_cuda_dspark*|src/cuda/mmq/ds4_mmq*|src/cuda/mmq/ds4_mmid*|src/cuda/mmq/ds4_mxfp8*|src/cuda/mmq/ds4_quantize*|src/server/deepseek_*|src/server/parser_deepseek*|tests/live_legs.py) ds4legs=1 ;; esac; \
	      case "$$p" in src/cuda/pulsar_cuda_runtime.cu|src/cuda/*scratch*|src/engine/gpu_graph_*|src/engine/gpu_decode.cpp|src/engine/step_driver*|src/engine/spec_block*|src/engine/verify_curve.cpp|*scratch_poison*) poison=1 ;; esac; \
	      case "$$p" in src/engine/gpu_prefill*|src/engine/prefill_loop*|src/engine/kv_state*|src/engine/checkpoint*|src/engine/sync_driver*|src/engine/session*|src/lib/pulsar_segstore*|*chunk*) chunk=1 ;; esac; \
	      case $$cls in \
	        attn) for g in $(GATES_DEV_ATTN); do add $$g; done ;; \
	        cuda) for g in $(GATES_DEV_CUDA); do add $$g; done ;; \
	        server) for g in $(GATES_DEV_SERVER); do add $$g; done ;; \
	        engine) for g in $(GATES_DEV_ENGINE); do add $$g; done ;; \
	        vision) for g in $(GATES_DEV_VISION); do add $$g; done ;; \
	        exl3) for g in $(GATES_DEV_DEFAULT); do add $$g; done ;; \
	        gdn) for g in $(GATES_DEV_DEFAULT); do add $$g; done ;; \
	        blocks) ;; \
	        *) for g in $(GATES_DEV_DEFAULT); do add $$g; done ;; \
	      esac; \
	    done; \
	    if [ $$family -eq 1 ]; then add session-contract-gate; add cuda-row-neutrality-gate; add bank-adopt-gate; fi; \
	    if [ $$chunk -eq 1 ]; then add cuda-chunk-neutrality-gate-full; fi; \
	  fi; \
	  sel=$$(echo $$sel); \
	fi; \
	printf '\033[1m=== gates-dev: the ITERATION tier (NOT a substitute for `make gates`) ===\033[0m\n'; \
	printf '  selection driven by:%s\n' "$$why"; \
	if [ -z "$$sel" ]; then printf '  runner sub-gates: (none -- all coverage for this path set is in the targets below)\n'; \
	else printf '  runner sub-gates: %s\n' "$$(echo $$sel | tr ' ' ',')"; fi; \
	if [ $$vision -eq 1 ]; then printf '  vision host gates: selected\n'; fi; \
	if [ $$family -eq 1 ]; then printf '  family gates: selected (the family interface; the session contract on both families)\n'; fi; \
	if [ $$attn -eq 1 ]; then printf '  cuda-attn-gates: selected (attention kernels)\n'; fi; \
	if [ $$router -eq 1 ]; then printf '  router-gate + qwen-s4-gate: selected (the shared router, the rows MoE block)\n'; fi; \
	if [ $$blocks -eq 1 ]; then printf '  block gates: sconv-gate, head-gate, rowops-gate, mamba2-gate (the L295 short convolution, head options, row operations; the L296 Mamba-2 block)\n'; fi; \
	if [ $$bf16 -eq 1 ]; then printf '  bf16-rows-gate + nvfp4-rows-gate: selected (the BF16 / NVFP4 rows arms and their admission)\n'; fi; \
	if [ $$minp -eq 1 ]; then printf '  cuda-sampler-device-gate: selected (THE sampler, on the device: L302)\n'; fi; \
	if [ $$drafter -eq 1 ]; then printf '  block drafter gates: blockdraft-gate, lilicorr-head-gate, lilicorr-spec-gate, dflash2-spec-gate (L298 / L299: the LiLiCorr lane on the Nemotron artifact, the DFlash2 lane on the Qwen artifact)\n'; fi; \
	if [ $$poison -eq 1 ]; then printf '  scratch-poison-gate: selected (L307: the graphed lanes and their scratch, poisoned == unpoisoned on DeepSeek, DFlash2, LiLiCorr)\n'; fi; \
	if [ $$server -eq 1 ]; then printf '  server: chat smoke + the --server/--api unit switches\n'; fi; \
	if [ $$render -eq 1 ]; then printf '  prefix-stability-gate: selected (L325: server / chat-render paths, every family, no model)\n'; fi; \
	if [ $$ds4legs -eq 1 ]; then printf '  live-legs-deepseek: selected (L278: the DeepSeek live legs left the battery; a DeepSeek path change runs them)\n'; fi; \
	if [ $$kvmem -eq 1 ]; then printf '  server-kv-headroom-live: selected (L319: the touched-KV ceiling is the measured headroom; admission / guard / bank paths)\n'; fi; \
	if [ $$resume -eq 1 ]; then printf '  resume-deepest-live + sidecall-turns-live + kv-dup-live + fanout-prefix-live + restore-stall-live: selected (L318 / L322 / L324 / L329 / L330: routing / resume / disk-chain paths, the Open WebUI and the Claude Code side-call shapes, the fan-out duplicate census, the fan-out shared-prefix wait and the sliced disk restore on the Qwen pack)\n'; fi; \
	$(MAKE) --no-print-directory seam-check CUDA_ARCH=sm_120f || rc=1; \
	$(MAKE) --no-print-directory cuda-reap-router-audit CUDA_ARCH=sm_120f \
	      FRONTIER_MODEL="$(FRONTIER_MODEL)" || rc=1; \
	PULSAR_TEST_MODEL="$(QWEN_GATE_MODEL)" PULSAR_TEST_QWEN_TOKENIZER="$(QWEN_GATE_MODEL)" ./pulsar_test $(GATES_DEV_UNIT) || rc=1; \
	if [ $$server -eq 1 ]; then \
	  PULSAR_TEST_MODEL="$(QWEN_GATE_MODEL)" PULSAR_TEST_QWEN_TOKENIZER="$(QWEN_GATE_MODEL)" ./pulsar_test $(GATES_DEV_UNIT_SERVER) || rc=1; \
	  $(MAKE) -j$(GATE_JOBS) --no-print-directory cuda-chat-smoke-gate CUDA_ARCH=sm_120f \
	        FRONTIER_MODEL="$(FRONTIER_MODEL)" || rc=1; \
	fi; \
	if [ $$render -eq 1 ]; then \
	  $(MAKE) --no-print-directory prefix-stability-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$attn -eq 1 ]; then \
	  $(MAKE) -j$(GATE_JOBS) --no-print-directory cuda-attn-gates CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$vision -eq 1 ]; then \
	  for h in vision-layout-gate vision-pixel-gate vision-codec-gate vision-span-gate \
	           vision-visible-gate vision-placeholder-gate nemotron-vision-pixel-gate; do \
	    $(MAKE) --no-print-directory $$h CUDA_ARCH=sm_120f FRONTIER_MODEL="$(FRONTIER_MODEL)" || rc=1; \
	  done; \
	fi; \
	if [ $$router -eq 1 ]; then \
	  $(MAKE) --no-print-directory router-gate qwen-s4-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$gdn -eq 1 ]; then \
	  $(MAKE) --no-print-directory gdn-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$blocks -eq 1 ]; then \
	  $(MAKE) --no-print-directory sconv-gate head-gate rowops-gate mamba2-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$bf16 -eq 1 ]; then \
	  $(MAKE) --no-print-directory bf16-rows-gate nvfp4-rows-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$minp -eq 1 ]; then \
	  $(MAKE) --no-print-directory cuda-sampler-device-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$drafter -eq 1 ]; then \
	  $(MAKE) --no-print-directory blockdraft-gate lilicorr-head-gate lilicorr-spec-gate dflash2-spec-gate draft-refusal-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$poison -eq 1 ]; then \
	  $(MAKE) --no-print-directory scratch-poison-gate CUDA_ARCH=sm_120f FRONTIER_MODEL="$(FRONTIER_MODEL)" || rc=1; \
	fi; \
	if [ $$resume -eq 1 ]; then \
	  $(MAKE) --no-print-directory segstore-tsan || rc=1; \
	  $(MAKE) --no-print-directory preempt-wedge-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory resume-fixes-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory resume-deepest-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory sidecall-turns-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory kv-dup-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory fanout-prefix-live CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory restore-stall-live CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$kvmem -eq 1 ]; then \
	  $(MAKE) --no-print-directory server-kv-headroom-live CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$ds4legs -eq 1 ]; then \
	  $(MAKE) --no-print-directory live-legs-deepseek CUDA_ARCH=sm_120f FRONTIER_MODEL="$(FRONTIER_MODEL)" || rc=1; \
	fi; \
	if [ $$exl3 -eq 1 ]; then \
	  $(MAKE) --no-print-directory exl3-dequant-gate CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory exl3-moe-epilogue-gate exl3-dense-gate exl3-moe-prefill-gate exl3-half-gate CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ $$family -eq 1 ]; then \
	  $(MAKE) --no-print-directory state-kinds-gate CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory loader-contract-gate CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory nemotron-family-gate CUDA_ARCH=sm_120f || rc=1; \
	  $(MAKE) --no-print-directory nemotron-fwd-gate session-contract-gate-nemotron CUDA_ARCH=sm_120f || rc=1; \
	fi; \
	if [ -n "$$sel" ]; then \
	  hosted=''; if [ $$family -eq 1 ]; then hosted='$(RUNNER_HOSTED_ARGS)'; fi; \
	  PULSAR_GATE_CHUNK_SEED=$(GATE_BUILD_REF) ./tests/gates_runner "$(FRONTIER_MODEL)" --prefill-baseline $(PREFILL_BASELINE) \
	      --prefill-ref $(PREFILL_BASELINE_REF_SHORT) \
	      --decode-baseline $(PREFILL_DECODE_BASELINE) --decode-ref $(PREFILL_DECODE_BASELINE_REF_SHORT) \
	      $$hosted --only=$$(echo $$sel | tr ' ' ',') || rc=1; \
	fi; \
	printf '\n  gates-dev total: %s s\n' "$$(( $$(date +%s) - t0 ))"; \
	if [ $$rc -eq 0 ]; then printf '\nDEV TIER PASS -- the landing tier for a delta that cannot move a number; a numeric-path change gets ONE `make gates` per landing series (ENGINEERING-RULES section 10)\n'; \
	else printf '\nDEV TIER FAILED\n'; fi; \
	exit $$rc

# Ported C++ TUs (pulsar). One rule per source dir, mirroring the .c rules.
src/engine/%.o: src/engine/%.cpp src/engine/pulsar_engine_internal.h src/engine/cursor.hpp src/pulsar.h src/pulsar_gpu.h src/lib/pulsar_utf8.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
src/families/%.o: src/families/%.cpp src/engine/pulsar_engine_internal.h src/engine/role_bind.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

# L216: vision.cpp ports the checkpoint's image_processor.py, whose results are
# FLOORS of float64 expressions -- so it must evaluate them exactly as the
# reference's CPython does.  -ffast-math permits reassociation, and on aarch64 it
# takes the reference's `height * beta / patch` apart and returns one patch less
# (measured: solve(1080,1920,14,3,384) -> 574 instead of the reference's 588,
# which vision-layout-gate catches).  Strict FP semantics for this one TU.
src/engine/vision.o: src/engine/vision.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(filter-out -ffast-math,$(CXXFLAGS)) $(PULSAR_INC) -c -o $@ src/engine/vision.cpp
# L331: Nemotron 3.5 Super VL's tile rule is the processor's float64 Python (sqrt / floor / ceil / round half to even)
# and its pixels torch's float32 -- strict FP semantics for this TU too (nemotron-vision-pixel-gate grades it).
src/families/nemotron_h/nemotron_h_image.o: src/families/nemotron_h/nemotron_h_image.cpp src/families/nemotron_h/nemotron_h.h \
                                            src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(filter-out -ffast-math,$(CXXFLAGS)) $(PULSAR_INC) -c -o $@ src/families/nemotron_h/nemotron_h_image.cpp

src/tp/%.o: src/tp/%.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

# pulsar_tp_gpu.cpp is host C++ but includes <cuda_runtime.h>, so nvcc (which
# brings the CUDA include path and -lcudart) compiles it, not the generic g++
# rule above.
src/tp/pulsar_tp_gpu.o: src/tp/pulsar_tp_gpu.cpp src/tp/pulsar_tp_gpu.h
	$(NVCC) $(NVCCFLAGS) -Isrc -c -o $@ $<

src/server/%.o: src/server/%.cpp $(SERVER_HDRS) src/pulsar.h $(LIB_HDRS)
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.cpp src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.cpp src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.cpp src/pulsar.h $(LIB_HDRS) src/lib/sha1.hpp
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/vendor/%.o: src/vendor/%.cpp src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) -Wno-write-strings -c -o $@ $<

src/vendor/xgrammar/%.o: src/vendor/xgrammar/%.cc
	$(CXX) $(GRAMMAR_CXXFLAGS) -Wno-unused-parameter $(XGRAMMAR_INC) -c -o $@ $<

src/lib/pulsar_grammar.o: src/lib/pulsar_grammar.cpp src/lib/pulsar_grammar.h
	$(CXX) $(GRAMMAR_CXXFLAGS) $(XGRAMMAR_INC) -c -o $@ $<

tests/pulsar_test.o: tests/pulsar_test.cpp $(SERVER_SRCS) $(SERVER_HDRS) src/pulsar.h $(LIB_HDRS) tests/sampler_host_ref.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Itests -c -o $@ tests/pulsar_test.cpp

tests/pulsar_agent_test.o: tests/pulsar_agent_test.cpp $(AGENT_SRCS) src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/pulsar_agent_test.cpp

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ tests/cuda_long_context_smoke.cpp


tests/vision_router_gate.o: tests/vision_router_gate.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -fno-fast-math -Isrc -c -o $@ tests/vision_router_gate.cpp

tests/resume_suffix_probe.o: tests/resume_suffix_probe.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/resume_suffix_probe.cpp

tests/token_seam_gate.o: tests/token_seam_gate.cpp tests/gate_util.h tests/gate_entry.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/token_seam_gate.cpp

tests/multiseq_decode_gate.o: tests/multiseq_decode_gate.cpp tests/gate_util.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/multiseq_decode_gate.cpp

tests/mseq_rowcost_probe.o: tests/mseq_rowcost_probe.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mseq_rowcost_probe.cpp

tests/bank_carry_identity.o: tests/bank_carry_identity.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_carry_identity.cpp

tests/nt_crossover_sweep.o: tests/nt_crossover_sweep.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/nt_crossover_sweep.cpp

tests/chunk_neutrality_gate.o: tests/chunk_neutrality_gate.cpp tests/gate_entry.h src/pulsar.h src/lib/pulsar_segstore.h src/lib/pulsar_kvchain.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/chunk_neutrality_gate.cpp
tests/vision_chunk_gate.o: tests/vision_chunk_gate.cpp tests/gate_entry.h src/pulsar.h src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_chunk_gate.cpp

tests/vision_image_gate.o: tests/vision_image_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_image_gate.cpp

tests/vision_image_sync_gate.o: tests/vision_image_sync_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_image_sync_gate.cpp

tests/vision_placeholder_gate.o: tests/vision_placeholder_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_placeholder_gate.cpp

tests/vision_qwen_pixel_gate.o: tests/vision_qwen_pixel_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_qwen_pixel_gate.cpp

tests/vision_hc_gate.o: tests/vision_hc_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_hc_gate.cpp

tests/vision_visible_gate.o: tests/vision_visible_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_visible_gate.cpp


tests/state_kinds_gate.o: tests/state_kinds_gate.cpp src/engine/pulsar_engine_internal.h src/engine/state_kinds.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/state_kinds_gate.cpp

tests/vision_layout_gate.o: tests/vision_layout_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_layout_gate.cpp

tests/vision_pixel_gate.o: tests/vision_pixel_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_pixel_gate.cpp

tests/vision_codec_gate.o: tests/vision_codec_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_codec_gate.cpp

tests/vision_span_gate.o: tests/vision_span_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_span_gate.cpp

tests/vision_tower_gate.o: tests/vision_tower_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_tower_gate.cpp

tests/vision_qwen_tower_gate.o: tests/vision_qwen_tower_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_qwen_tower_gate.cpp

tests/vision_merge_gate.o: tests/vision_merge_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/vision_merge_gate.cpp

tests/algo_stability_gate.o: tests/algo_stability_gate.cpp tests/gate_util.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/algo_stability_gate.cpp

tests/mixed_zero_prefill_gate.o: tests/mixed_zero_prefill_gate.cpp tests/gate_fixture.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mixed_zero_prefill_gate.cpp

tests/fused_step_gate.o: tests/fused_step_gate.cpp tests/gate_fixture.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/fused_step_gate.cpp
tests/verify_width_probe.o: tests/verify_width_probe.cpp tests/gate_fixture.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/verify_width_probe.cpp
tests/decode_reference_gate.o: tests/decode_reference_gate.cpp tests/gate_fixture.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/decode_reference_gate.cpp

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
tests/spec_sampling_gate.o: tests/spec_sampling_gate.cpp src/pulsar.h tests/gate_util.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/spec_sampling_gate.cpp
tests/spec_sampling_gate_wide.o: tests/spec_sampling_gate.cpp src/pulsar.h tests/gate_util.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -DSAMPLED_BANKS_SPEC=8 -c -o $@ tests/spec_sampling_gate.cpp

# L302: the device sampler reproduces the host libm and the host sampler's float arithmetic bit for bit, so it is
# built WITHOUT --use_fast_math: no flush-to-zero (a probability may be subnormal and the host kept it), IEEE division,
# no contraction (its fmas are the explicit ones the host's compiled code has, pulsar_cuda_libm.cuh).
src/cuda/pulsar_cuda_sampler.o: src/cuda/pulsar_cuda_sampler.cu src/cuda/pulsar_cuda_libm.cuh \
                                src/cuda/pulsar_cuda_libm_tables.inc src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h \
                                $(CUDA_FLAG_STAMP)
	$(NVCC) $(filter-out --use_fast_math,$(NVCCFLAGS)) -ftz=false -prec-div=true -prec-sqrt=true -fmad=false \
		-Isrc -c -o $@ $<

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

# Tessera's fused window kernel (L255; vendored in tessera_routed_fused_window.cuh, VENDOR-TESSERA.md) is built
# WITHOUT --use_fast_math, as upstream builds it (-O3 -lineinfo): its SwiGLU epilogue calls expf and divides,
# and fast-math would approximate both.
src/cuda/mmq/pulsar_tessera.o: src/cuda/mmq/pulsar_tessera.cu $(MMQ_HDRS) $(CUDA_FLAG_STAMP)
	$(NVCC) $(filter-out --use_fast_math,$(NVCCFLAGS)) -std=c++17 -Isrc -Isrc/cuda/mmq -c -o $@ $<

# CUTLASS MXFP4 tensor-core expert FFN (GB10/sm_120f). Requires -arch=sm_120f (family mode) for the
# mxf4 block-scale MMA; build the whole engine with CUDA_ARCH=sm_120f so all objects match arch.
src/cuda/pulsar_mxfp4_cutlass.o: src/cuda/pulsar_mxfp4_cutlass.cu src/pulsar_gpu.h $(CUDA_FLAG_STAMP) | cutlass-check
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 20012 -diag-suppress 177 -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/pulsar_mxfp4_cutlass.cu

# L321: CUTLASS's memory-efficient attention (vendored in src/cuda/fmha/, VENDOR.md) -- the CUTLASS include path
src/cuda/pulsar_cuda_fmha.o: src/cuda/pulsar_cuda_fmha.cu src/cuda/pulsar_cuda_fmha.h $(wildcard src/cuda/fmha/*.h src/cuda/fmha/*/*.h) \
                             $(CUDA_FLAG_STAMP) | cutlass-check
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 20012 -diag-suppress 177 -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/pulsar_cuda_fmha.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# the frozen host sampler: the ENGINE's flags exactly (its bytes came from that compilation)
tests/sampler_host_ref.o: tests/sampler_host_ref.cpp tests/sampler_host_ref.h
	$(CXX) $(CXXFLAGS) -Itests -c -o $@ tests/sampler_host_ref.cpp
tests/sampler_device_gate.o: tests/sampler_device_gate.cpp tests/sampler_host_ref.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -fopenmp -Isrc -Itests -c -o $@ tests/sampler_device_gate.cpp
tests/sampler_device_gate: tests/sampler_device_gate.o tests/sampler_host_ref.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) \
                           $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -o $@ $^ $(CUDA_LDLIBS)

tests/vision_router_gate: tests/vision_router_gate.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS) $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/resume_suffix_probe: tests/resume_suffix_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/token_seam_gate: tests/token_seam_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_decode_gate: tests/multiseq_decode_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mseq_rowcost_probe: tests/mseq_rowcost_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L260: the batched spec lane, byte-reproducible, for before/after identity of a bank-switch bookkeeping change
# (not a gate: run it on both trees and diff; tests/bank_carry_identity.cpp).
tests/bank_carry_identity: tests/bank_carry_identity.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/nt_crossover_sweep: tests/nt_crossover_sweep.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L151-C stage 0: fixed-tile CUTLASS GEMM probe (pricing only, not engine code).
tests/fixed_tile_gemm_kernels.o: tests/fixed_tile_gemm_kernels.cu tests/fixed_tile_gemm_probe.h src/pulsar_gpu.h src/cuda/pulsar_cuda_mx.cuh $(CUDA_FLAG_STAMP) | cutlass-check
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 20012 -diag-suppress 177 -Isrc -Isrc/cuda -Itests $(CUTLASS_INC) -c -o $@ tests/fixed_tile_gemm_kernels.cu
tests/fixed_tile_gemm_probe.o: tests/fixed_tile_gemm_probe.cpp tests/fixed_tile_gemm_probe.h src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -Itests -c -o $@ tests/fixed_tile_gemm_probe.cpp
tests/fixed_tile_gemm_probe: tests/fixed_tile_gemm_probe.o tests/fixed_tile_gemm_kernels.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
cuda-fixed-tile-probe: tests/fixed_tile_gemm_probe
	./tests/fixed_tile_gemm_probe $(FRONTIER_MODEL)

tests/chunk_neutrality_gate: tests/chunk_neutrality_gate.o src/lib/pulsar_help.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_kvtext.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/vision_chunk_gate: tests/vision_chunk_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_image_gate: tests/vision_image_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_image_sync_gate: tests/vision_image_sync_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_placeholder_gate: tests/vision_placeholder_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_qwen_pixel_gate: tests/vision_qwen_pixel_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_hc_gate: tests/vision_hc_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_visible_gate: tests/vision_visible_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)


tests/state_kinds_gate: tests/state_kinds_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/payload_frontier_gate.o: tests/payload_frontier_gate.cpp tests/gate_entry.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/payload_frontier_gate.cpp

tests/payload_frontier_gate: tests/payload_frontier_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/prompt_end_gate.o: tests/prompt_end_gate.cpp tests/gate_entry.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/prompt_end_gate.cpp

tests/prompt_end_gate: tests/prompt_end_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_adopt_gate.o: tests/bank_adopt_gate.cpp tests/gate_entry.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/bank_adopt_gate.cpp

tests/bank_adopt_gate: tests/bank_adopt_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_residency_gate.o: tests/bank_residency_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h tests/segment_file_io.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/bank_residency_gate.cpp

tests/bank_residency_gate: tests/bank_residency_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L357: the three ports read the declared state (tests/gate_fixture.h), so they see the engine's internal headers
tests/bank_isolation_gate.o: tests/bank_isolation_gate.cpp tests/gate_entry.h tests/gate_util.h tests/gate_fixture.h \
		src/pulsar.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/bank_isolation_gate: tests/bank_isolation_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/kv_accounting_gate.o: tests/kv_accounting_gate.cpp tests/gate_entry.h tests/gate_util.h tests/gate_fixture.h \
		src/pulsar.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/kv_accounting_gate: tests/kv_accounting_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/graph_eager_gate.o: tests/graph_eager_gate.cpp tests/gate_entry.h tests/gate_util.h tests/gate_fixture.h \
		src/pulsar.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/graph_eager_gate: tests/graph_eager_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L251: end-to-end generation for the Qwen4-exp lane -- token ids in, token ids out (the family has
# no tokenizer or renderer yet, S5, so the CLI's text path refuses by design).  Decode the emitted
# ids with qwen_generate_decode.py.  Needs a real container and a prompt's tokens.bin on disk.
QWEN_GEN_MODEL  ?= $(call need_path,QWEN_GEN_MODEL)
QWEN_GEN_TOKENS ?= $(call need_path,QWEN_GEN_TOKENS)
QWEN_GEN_N      ?= 32
tests/qwen_generate.o: tests/qwen_generate.cpp
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/qwen_generate: tests/qwen_generate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L291 / L280: the teacher-forced depth walk -- windows ending at a depth, one engine load (tools/qwen/kv_grade.py
# grades its outputs).  The K/V format and the rope are the engine's (--rope-yarn F passes the option).  Needs a
# real container.
tests/qwen_kv_walk.o: tests/qwen_kv_walk.cpp
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/qwen_kv_walk: tests/qwen_kv_walk.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L280: the session mechanisms past the native 262,144 positions (resume, the disk chain, spec == plain, the
# payload) at one depth, each against one cold prefill.  Not in the battery (two prefills at the depth: ~10 min at
# 300K); needs a real container and a long token file.
tests/qwen_deep_probe.o: tests/qwen_deep_probe.cpp tests/segment_file_io.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/qwen_deep_probe: tests/qwen_deep_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/qwen_chat_smoke.o: tests/qwen_chat_smoke.cpp
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/qwen_chat_smoke: tests/qwen_chat_smoke.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/loader_open.o: tests/loader_open.cpp
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<
tests/loader_open: tests/loader_open.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L338: the f16 m16n8k16 MMA's sm_75 body (two m16n8k8) graded against the exact product and the k16 instruction --
# model-free, on an sm_80+ card (it refuses below: the k16 instruction is the comparison).  The split is not the k16
# instruction's bits (a numerics change on sm_75); both must hold 2^-21 * sum |terms| of exact.
MMA_K8_ARCH ?= sm_120f
tests/mma_k8_split_gate: tests/mma_k8_split_gate.cu src/cuda/pulsar_cuda_arch.cuh
	$(NVCC) -O3 -gencode arch=compute_$(subst sm_,,$(MMA_K8_ARCH)),code=$(MMA_K8_ARCH) -o $@ $<
.PHONY: mma-k8-split-gate
mma-k8-split-gate: tests/mma_k8_split_gate
	./tests/mma_k8_split_gate
tests/gemma4_family_gate.o: tests/gemma4_family_gate.cpp src/families/gemma4/gemma4.h src/engine/role_bind.h \
                            src/engine/pulsar_engine_internal.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/gemma4_family_gate: tests/gemma4_family_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/gemma4_jev_gate.o: tests/gemma4_jev_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/gemma4_jev_gate: tests/gemma4_jev_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/nemotron_family_gate.o: tests/nemotron_family_gate.cpp src/families/nemotron_h/nemotron_h.h src/engine/role_bind.h \
                              src/engine/pulsar_engine_internal.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/nemotron_family_gate: tests/nemotron_family_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L321: Nemotron-3.5 Parse on a container vs the HF fp32 reference (tests/nemotron_parse_gate.cpp)
tests/nemotron_parse_gate.o: tests/nemotron_parse_gate.cpp src/families/nemotron_parse/nemotron_parse.h src/pulsar.h \
                             src/engine/pulsar_engine_internal.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/nemotron_parse_gate: tests/nemotron_parse_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L321: the Nemotron-3.5 encoders on a container (tests/nemotron_enc_gate.cpp; public API only)
tests/nemotron_enc_gate.o: tests/nemotron_enc_gate.cpp src/pulsar.h src/lib/pyjson.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/nemotron_enc_gate: tests/nemotron_enc_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L296 N7: the Nemotron-H forward on the core driver, on a container (tests/nemotron_fwd_gate.cpp; public API only)
tests/nemotron_fwd_gate.o: tests/nemotron_fwd_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<
tests/nemotron_fwd_gate: tests/nemotron_fwd_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L335: a refused pulsar_tp_plan_build attaches nothing and a second call refuses again (tests/tp_plan_build_test.cpp:
# a fabricated engine and test families, no model, no GPU work).
tests/tp_plan_build_test.o: tests/tp_plan_build_test.cpp src/engine/pulsar_engine_internal.h src/engine/family.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/tp_plan_build_test: tests/tp_plan_build_test.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
.PHONY: tp-plan-build-test
tp-plan-build-test: tests/tp_plan_build_test
	./tests/tp_plan_build_test
# L296 N9: the chained-MTP drafter on a container (tests/mtp_spec_gate.cpp; reads the drafter's numbers and the state)
tests/mtp_spec_gate.o: tests/mtp_spec_gate.cpp tests/gate_entry.h tests/gate_util.h tests/gate_fixture.h src/pulsar.h \
                       src/engine/spec_mtp.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/mtp_spec_gate: tests/mtp_spec_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/draft_refusal_gate.o: tests/draft_refusal_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h src/engine/step_driver.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/draft_refusal_gate: tests/draft_refusal_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L307: the scratch-poison gate, and its mutant (one engine TU swapped for tests/scratch_poison_mutant.py's copy)
tests/scratch_poison_gate.o: tests/scratch_poison_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/scratch_poison_gate: tests/scratch_poison_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/mutant/family_qwen_s4_l307.cpp: src/engine/family_qwen_s4.cpp tests/scratch_poison_mutant.py
	@mkdir -p tests/mutant
	python3 tests/scratch_poison_mutant.py $< $@
tests/mutant/family_qwen_s4_l307.o: tests/mutant/family_qwen_s4_l307.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ $<
tests/scratch_poison_gate_mutant: tests/scratch_poison_gate.o tests/mutant/family_qwen_s4_l307.o \
                                  $(filter-out src/engine/family_qwen_s4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L317: the battery's moved legs, bitten.  tests/gates_runner linked against ONE mutated TU (tests/runner_mutants.py
# writes the copy and refuses unless its site appears once) and run with the selection that must catch it -- PASS of
# `make battery-mutants` = every mutant's target gate FAILED.  $(1) the mutant, $(2) the TU without its extension,
# $(3) the extension.
define RUNNER_MUTANT_RULES
tests/mutant/runner_$(1).$(3): $(2).$(3) tests/runner_mutants.py
	@mkdir -p tests/mutant
	python3 tests/runner_mutants.py $(1) $$< $$@
tests/mutant/runner_$(1).o: tests/mutant/runner_$(1).$(3) $$(CUDA_FLAG_STAMP)
	$(if $(filter cu,$(3)),$$(NVCC) $$(NVCCFLAGS) $$(MMQ_CPPFLAGS) -Isrc -Isrc/cuda -c -o $$@ $$<,$$(CXX) $$(CXXFLAGS) $$(PULSAR_INC) -Isrc/engine -c -o $$@ $$<)
tests/gates_runner_mutant_$(1): tests/gates_runner.o $$(RUNNER_OBJS) $$(RUNNER_LIBS) \
                               tests/mutant/runner_$(1).o $$(filter-out $(2).o,$$(CORE_OBJS))
	$$(NVCC) $$(NVCCFLAGS) -o $$@ $$^ $$(CUDA_LDLIBS)
endef
$(eval $(call RUNNER_MUTANT_RULES,chunk-resume,src/engine/sync_driver,cpp))
$(eval $(call RUNNER_MUTANT_RULES,row-sort,src/cuda/pulsar_cuda_attention,cu))
$(eval $(call RUNNER_MUTANT_RULES,row-sort-wide,src/cuda/pulsar_cuda_attention,cu))
$(eval $(call RUNNER_MUTANT_RULES,adopt-rows,src/engine/checkpoint,cpp))
$(eval $(call RUNNER_MUTANT_RULES,spec-emit,src/engine/session_spec,cpp))
# ... and the L307 scratch-poison mutant in the runner: scratch-poison-gate on the hosted Qwen
tests/gates_runner_mutant_scratch-poison: tests/gates_runner.o $(RUNNER_OBJS) $(RUNNER_LIBS) \
                                          tests/mutant/family_qwen_s4_l307.o $(filter-out src/engine/family_qwen_s4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# mutant -> the sub-gate (the moved leg) whose FAIL is the catch; the runner selection is runner_mutants.py's
BATTERY_MUTANTS = chunk-resume:cuda-chunk-neutrality-gate row-sort:cuda-row-neutrality-gate-deep \
                  row-sort:cuda-row-neutrality-gate-deeper row-sort-wide:cuda-algo-stability-gate-deep \
                  adopt-rows:bank-adopt-gate spec-emit:block-spec-gate scratch-poison:scratch-poison-gate
BATTERY_MUTANT_NAMES = $(sort $(foreach m,$(BATTERY_MUTANTS),$(firstword $(subst :, ,$(m)))))
BATTERY_MUTANTS_WORK ?= $(CURDIR)/.build/battery-mutants
.PHONY: battery-mutants
battery-mutants: $(BATTERY_MUTANT_NAMES:%=tests/gates_runner_mutant_%)
	@mkdir -p $(BATTERY_MUTANTS_WORK); rc=0; \
	for m in $(BATTERY_MUTANT_NAMES); do \
	  if [ "$$m" = scratch-poison ]; then sel=--only=scratch-poison-gate; \
	  else sel=$$(python3 tests/runner_mutants.py --list | awk -F'\t' -v m=$$m '$$1 == m {print $$3}'); fi; \
	  log=$(BATTERY_MUTANTS_WORK)/$$m.log; \
	  printf '\n\033[1m=== mutant %s: tests/gates_runner_mutant_%s %s ===\033[0m\n' "$$m" "$$m" "$$sel"; \
	  sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; \
	  PULSAR_GATE_CHUNK_SEED=$(GATE_BUILD_REF) ./tests/gates_runner_mutant_$$m $(FRONTIER_MODEL) \
	      --prefill-baseline $(PREFILL_BASELINE) --prefill-ref $(PREFILL_BASELINE_REF_SHORT) \
	      --decode-baseline $(PREFILL_DECODE_BASELINE) --decode-ref $(PREFILL_DECODE_BASELINE_REF_SHORT) \
	      $(RUNNER_HOSTED_ARGS) $$sel > $$log 2>&1; \
	  echo "  runner rc=$$? (log $$log)"; \
	  grep -E '^--- .*(PASS|FAIL)|^  SKIP ' $$log | sed 's/^/  /'; \
	done; \
	printf '\n===================== BATTERY MUTANTS =====================\n'; \
	for mg in $(BATTERY_MUTANTS); do m=$${mg%%:*}; g=$${mg#*:}; log=$(BATTERY_MUTANTS_WORK)/$$m.log; \
	  if grep -qE "^--- $$g( \[[^]]*\])?: FAIL" $$log; then \
	    printf '  CAUGHT      %-15s by %s: %s\n' "$$m" "$$g" "$$(grep -E "^--- $$g( \[[^]]*\])?: FAIL" $$log | sed 's/^--- //; s/:.*//' | tr '\n' ' ')"; \
	  else printf '  NOT CAUGHT  %-15s by %s -- the moved leg has no teeth for it (see %s)\n' "$$m" "$$g" "$$log"; rc=1; fi; \
	done; exit $$rc
tests/sync_budget_gate.o: tests/sync_budget_gate.cpp tests/gate_entry.h tests/gate_util.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/sync_budget_gate: tests/sync_budget_gate.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/session_contract_gate.o: tests/session_contract_gate.cpp tests/segment_file_io.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/session_contract_gate: tests/session_contract_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# L278 contract 6: the chat-decode smoke on the public text API -- a runner gate on every hosted model
# (chat-smoke-gate); the standalone binary is for iterating on one model.
tests/chat_decode_smoke.o: tests/chat_decode_smoke.cpp tests/gate_entry.h tests/gate_util.h
	$(CXX) $(CXXFLAGS) -Isrc -Isrc/engine -c -o $@ $<
tests/chat_decode_smoke: tests/chat_decode_smoke.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
.PHONY: qwen-generate
qwen-generate: tests/qwen_generate
	@./tests/qwen_generate $(QWEN_GEN_MODEL) $(QWEN_GEN_TOKENS) $(QWEN_GEN_N) /tmp/qwen-gen-ids.bin

tests/vision_layout_gate: tests/vision_layout_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_pixel_gate: tests/vision_pixel_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_codec_gate: tests/vision_codec_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_span_gate: tests/vision_span_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_tower_gate: tests/vision_tower_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_qwen_tower_gate: tests/vision_qwen_tower_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L268: one image question end to end (tests/image_chat_smoke.cpp), graded on the answer's content.  Manual (a
# model load of its own); the live server tier (server-live-gate) asks both families about an image too.
QWEN_IMAGE_SMOKE ?= $(QWEN_GATE_MODEL)
.PHONY: qwen-image-smoke-gate
qwen-image-smoke-gate: tests/image_chat_smoke
	python3 -c "import struct,zlib;w,h=480,360;rows=b''.join(b'\\x00'+b''.join(bytes((220,20,20)) if (x-w/2)**2+(y-h/2)**2<(0.35*h)**2 else b'\\xff\\xff\\xff' for x in range(w)) for y in range(h));c=lambda t,d:struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff);open('/tmp/qwen-image-smoke.png','wb').write(b'\\x89PNG\\r\\n\\x1a\\n'+c(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+c(b'IDAT',zlib.compress(rows,9))+c(b'IEND',b''))"
	./tests/image_chat_smoke $(QWEN_IMAGE_SMOKE) /tmp/qwen-image-smoke.png "What color is the circle in this image? Answer with one word." red
tests/image_chat_smoke.o: tests/image_chat_smoke.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/image_chat_smoke.cpp
tests/image_chat_smoke: tests/image_chat_smoke.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/vision_merge_gate: tests/vision_merge_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/algo_stability_gate: tests/algo_stability_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mixed_zero_prefill_gate: tests/mixed_zero_prefill_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/fused_step_gate: tests/fused_step_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/verify_width_probe: tests/verify_width_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/decode_reference_gate: tests/decode_reference_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/prefill_bitexact_gate: tests/prefill_bitexact_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/spec_sampling_gate: tests/spec_sampling_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/spec_sampling_gate_wide: tests/spec_sampling_gate_wide.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/mseq_short_ctx_probe.o: tests/mseq_short_ctx_probe.cpp tests/gate_util.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/mseq_short_ctx_probe.cpp
tests/spec_teacher_forced_probe.o: tests/spec_teacher_forced_probe.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/spec_teacher_forced_probe.cpp
tests/mseq_short_ctx_probe: tests/mseq_short_ctx_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/spec_teacher_forced_probe: tests/spec_teacher_forced_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L183: the chunk-mate census driver (prefill exactly N tokens as one chunk; the
# instrument is the engine's per-op dump, PULSAR_CUDA_GRAPH_DUMP_*).
tests/prefill_chunk_census_probe.o: tests/prefill_chunk_census_probe.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/prefill_chunk_census_probe.cpp
tests/prefill_chunk_census_probe: tests/prefill_chunk_census_probe.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

pulsar_test: tests/pulsar_test.o tests/sampler_host_ref.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_test.o tests/sampler_host_ref.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS) $(CUDA_LDLIBS)

# L267: the API parsers' requests dumped field by field (host-only; see tests/api_golden.cpp).
# api-golden-gate compares them with the golden recorded before L267 moved the parse/render layers.
tests/api_golden.o: tests/api_golden.cpp $(SERVER_SRCS) $(SERVER_HDRS) src/pulsar.h $(LIB_HDRS)
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/api_golden.cpp

tests/api_golden: tests/api_golden.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# L303: constrained decoding's host cost at a family's real vocabulary (mask fill, accept, a round's masks) and
# whether the template's own bytes, tokenized, stay inside the grammar.  No engine, no GPU: the checkpoint's
# tokenizer.json.  A measurement tool, not a gate.
tests/grammar_bench.o: tests/grammar_bench.cpp $(SERVER_SRCS) $(SERVER_HDRS) src/pulsar.h $(LIB_HDRS) src/lib/pulsar_grammar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/grammar_bench.cpp

tests/grammar_bench: tests/grammar_bench.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: api-golden-gate
api-golden-gate: tests/api_golden
	./tests/api_golden tests/api-golden/corpus.jsonl > tests/api-golden/out.txt
	diff -u tests/api-golden/golden.txt tests/api-golden/out.txt && echo "API-GOLDEN GATE PASS"

# L267: the DeepSeek stream projections' bytes per protocol (host-only; see tests/sse_golden.cpp).
tests/sse_golden.o: tests/sse_golden.cpp $(SERVER_SRCS) $(SERVER_HDRS) src/pulsar.h $(LIB_HDRS)
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/sse_golden.cpp

tests/sse_golden: tests/sse_golden.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o $(CORE_OBJS) $(GRAMMAR_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: sse-golden-gate
sse-golden-gate: tests/sse_golden
	./tests/sse_golden > tests/sse-golden/out.txt
	diff -u tests/sse-golden/golden.txt tests/sse-golden/out.txt && echo "SSE-GOLDEN GATE PASS"

pulsar_agent_test: tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o src/vendor/linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvtext.o src/lib/pulsar_segstore.o src/lib/pulsar_kvchain.o src/lib/pulsar_dsml.o src/vendor/linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

# TP transport-core unit test (branch tensor_parallel, slice 1).  Host-only:
# no CUDA, no sockets -- builds and runs with plain g++.  Pins the slab
# layout + identity contract the transport lift builds on.
tests/tp_core_test: tests/tp_core_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_core_test.cpp src/tp/pulsar_tp.cpp

tp-core-test: tests/tp_core_test
	./tests/tp_core_test

# TP command-free test (L335).  Host-only, built with AddressSanitizer: a command of every received frame kind
# with every malloc'd array set goes through pulsar_tp_command_free, and LeakSanitizer must find nothing (the
# pre-L335 free zeroed the command before freeing logits / spec / image / key arrays).
tests/tp_command_free_test: tests/tp_command_free_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) -fsanitize=address -fno-omit-frame-pointer $(PULSAR_INC) -o $@ tests/tp_command_free_test.cpp src/tp/pulsar_tp.cpp

tp-command-free-test: tests/tp_command_free_test
	./tests/tp_command_free_test

# TP n-way full-mesh test (slice n-way).  Host-only: forks n ranks over TCP
# loopback and drives create_mesh + allreduce_sum; asserts the all-reduce is
# the exact sum of every rank's partial, identical across ranks.  n=2 and n=3.
tests/tp_mesh_test: tests/tp_mesh_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_mesh_test.cpp src/tp/pulsar_tp.cpp

tp-mesh-test: tests/tp_mesh_test
	./tests/tp_mesh_test

# TP engine-mirror test (slice 4e).  Host-runnable but links the engine: it
# fabricates a session (no model, no weights) whose mirror id does not match
# the leader's frame, and asserts the wrappers refuse loudly, ACK the refusal,
# and that the leader reports it -- the plumbing the mesh test cannot reach and
# the single-Spark gates do not cover.  `timeout` matters: a missing ack HANGS
# the leader in wait_command_ack, and a hang is not a test result.
tests/tp_mirror_test.o: tests/tp_mirror_test.cpp src/tp/pulsar_tp.h src/engine/pulsar_engine_internal.h src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/tp_mirror_test.cpp

tests/tp_mirror_test: tests/tp_mirror_test.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tp-mirror-test: tests/tp_mirror_test
	PULSAR_TP_TIMEOUT_SEC=1 timeout 60 ./tests/tp_mirror_test

# TP transport loopback test (branch tensor_parallel, slice 3).  Host-only:
# no CUDA, no RDMA -- a forked leader/worker pair exchanges gate/batch/big
# partials and one eval<->ack lockstep round over TCP 127.0.0.1 loopback,
# built and run with plain g++.
tests/tp_transport_test: tests/tp_transport_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_transport_test.cpp src/tp/pulsar_tp.cpp

tp-transport-test: tests/tp_transport_test
	./tests/tp_transport_test

# TP NODE frame test (protocol v13): after bring-up every rank knows every
# rank's host, build and TP address, over the pair and the 3-rank mesh paths.
# Host-only: no CUDA, no RDMA.
tests/tp_node_test: tests/tp_node_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_node_test.cpp src/tp/pulsar_tp.cpp

tp-node-test: tests/tp_node_test
	./tests/tp_node_test

# TP identity / hello-robustness test (branch tensor_parallel): the real-rank
# model-mismatch matrix, the ctx-diff connect rule, the raw-peer leader
# robustness set, and the worker dial-timeout -- all with per-child alarms so
# a failure that turns into a hang FAILS instead of blocking the suite.
# Host-only: no CUDA, no RDMA.
tests/tp_identity_test: tests/tp_identity_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_identity_test.cpp src/tp/pulsar_tp.cpp

tp-identity-test: tests/tp_identity_test
	./tests/tp_identity_test

# TP wide-embd + soak test (audit F4/F8 host half): the gate/batch/big
# exchange and decode-slot reuse at n_embd 8192 (vec == 2*MSG, the RDMA
# chunked-branch boundary) and 16384 (> 2*MSG, RDMA-registration reject size),
# plus N decode tokens x 86 gates of slot-reuse soak.  Host-only: TCP loopback.
tests/tp_wide_test: tests/tp_wide_test.cpp tests/tp_sched_harness.cpp src/tp/pulsar_tp.cpp tests/tp_sched_harness.h src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Itests -o $@ tests/tp_wide_test.cpp tests/tp_sched_harness.cpp src/tp/pulsar_tp.cpp

tp-wide-test: tests/tp_wide_test
	./tests/tp_wide_test

# TP peer-fault test (branch tensor_parallel): one rank dies mid-exchange and
# the survivor must surface the failure (gate_exchange -> 0, wait_command_ack
# -> 0 + failed() latch) instead of hanging in the blocking read.  Host-only:
# TCP loopback, forked leader/worker, per-side alarm + bounded parent wait.
tests/tp_fault_test: tests/tp_fault_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_fault_test.cpp src/tp/pulsar_tp.cpp

tp-fault-test: tests/tp_fault_test
	./tests/tp_fault_test

# TP control-plane stress test (branch tensor_parallel): thousands of
# randomized interleaved session commands (create/destroy churn, sync/eval/
# rewind/invalidate, eval_batch, mixed_batch, verify+commit) against a worker
# that mirrors the session ledger from the received frames -- a foreign,
# dropped, or mis-acked command fails, and a stalled frame would be caught by
# the per-side alarm.  Host-only: TCP loopback.
tests/tp_cmd_stress_test: tests/tp_cmd_stress_test.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -o $@ tests/tp_cmd_stress_test.cpp src/tp/pulsar_tp.cpp

tp-cmd-stress-test: tests/tp_cmd_stress_test
	./tests/tp_cmd_stress_test

# TP verify/commit reference-grading test (host half of slice 4e): the pure
# decision rule (all-match vs first-mismatch stop) plus a few hundred leader
# verify + commit rounds against a worker that mirrors the grade from the
# drafts -- the grading+commit lockstep the real pair will run.  Host-only.

# TP tiny-buffer regression (audit F9): the TCP fallback's alternating
# write/read rounds must complete even when the socket buffers are clamped
# small (PULSAR_TP_TEST_TINY_BUFFERS=1 -> 32K bufs) -- the conditition under
# which the pair hosts once deadlocked (the old 2 MiB symmetric write-then-
# read filled both send buffers; 64K rounds land inside the kernel's doubled
# recv buffer, so both sides finish their write before draining reads).  Loop
# the whole transport + a wide soak under the clamp so any return to a
# symmetric-write or oversized-round shape fails the build instead of hanging
# a GPU session.  Host-only; the timeout guards a regression that hangs.
tp-tinybuf-test: tests/tp_transport_test tests/tp_wide_test
	@set -e; \
	for i in $$(seq 1 30); do \
		PULSAR_TP_TEST_TINY_BUFFERS=1 timeout 60 ./tests/tp_transport_test >/dev/null 2>&1 \
		  || { echo "tp-tinybuf: transport iteration $$i FAILED"; exit 1; }; \
	done; \
	PULSAR_TP_TEST_TINY_BUFFERS=1 PULSAR_TP_SOAK_TOKENS=4 timeout 120 ./tests/tp_wide_test >/dev/null 2>&1 \
	  || { echo "tp-tinybuf: wide soak FAILED"; exit 1; }; \
	echo "tp-tinybuf-test: ok (30x transport + wide soak under 32K clamped buffers)"

# TP GPU-slab gate probe (bring-up step 4): nvcc-built so it can run on the
# pair.  Compile-checked here (no GPU to run); run on the GB10 pair per
# docs/history/tensor-parallel-bringup.md.
tests/tp_slab_gpu_probe: tests/tp_slab_gpu_probe.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp.h src/tp/pulsar_tp_gpu.cpp src/tp/pulsar_tp_gpu.h
	$(NVCC) $(NVCCFLAGS) -Isrc -o $@ tests/tp_slab_gpu_probe.cpp src/tp/pulsar_tp.cpp src/tp/pulsar_tp_gpu.cpp $(CUDA_LDLIBS)

tp-slab-probe: tests/tp_slab_gpu_probe
	@echo "built tests/tp_slab_gpu_probe; run on the pair per docs/history/tensor-parallel-bringup.md"

# TP GPUDirect capability probe (slab-class verdict): the full dma-buf GDR
# sequence -- VMM pinned alloc, CUDA-13 GPURDMA flag, export-to-posix-fd,
# ibv_reg_dmabuf_mr.  Builds on any box with CUDA + libibverbs-dev; run on the
# pair.  NEGATIVE on GB10/driver 610 (attrs 110/116 = 0; dma-buf import EINVAL)
# -- see tests/tp_dmabuf_probe.cpp.  Recheck after any nvidia driver update.
tests/tp_dmabuf_probe: tests/tp_dmabuf_probe.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -I$(CUDA_HOME)/include -o $@ tests/tp_dmabuf_probe.cpp -L$(CUDA_HOME)/lib64/stubs -L$(CUDA_HOME)/lib64 -lcuda -l:libibverbs.so.1

tp-dmabuf-probe: tests/tp_dmabuf_probe
	@echo "built tests/tp_dmabuf_probe; run on the pair (GDR negative on GB10/610)"

test: pulsar_test seam-check
	PULSAR_TEST_QWEN_TOKENIZER="$(QWEN_GATE_MODEL)" ./pulsar_test

clean:
	rm -rf .build
	rm -rf tests/runner
	rm -f tests/gates_runner pulsar pulsar-server pulsar-bench pulsar-eval pulsar-agent pulsar_test pulsar_agent_test src/engine/*.o src/families/*/*.o src/families/*/*.d src/tp/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cuda/mmq/*.o src/cuda/mmq/test/*.o src/cli/*.o src/lib/*.o src/vendor/*.o $(XGRAMMAR_OBJS) $(XGRAMMAR_OBJS:.o=.d) tests/*.o src/engine/*.d src/agent/*.d src/server/*.d src/cuda/*.d src/cuda/mmq/*.d src/cuda/mmq/test/*.d src/cli/*.d src/lib/*.d src/vendor/*.d tests/*.d tests/vision_visible_gate tests/vision_hc_gate tests/vision_image_gate tests/vision_placeholder_gate tests/vision_image_sync_gate tests/vision_chunk_gate tests/cuda_long_context_smoke tests/moe_route_bounds_gate tests/expert_table_gate tests/multiseq_decode_gate tests/prefill_bitexact_gate tests/spec_sampling_gate tests/spec_sampling_gate_wide tests/algo_stability_gate tests/mixed_zero_prefill_gate tests/fused_step_gate tests/draft_refusal_gate tests/scratch_poison_gate tests/scratch_poison_gate_mutant tests/gates_runner_mutant_* tests/mutant/family_qwen_s4_l307.cpp tests/mutant/family_qwen_s4_l307.o tests/mutant/family_qwen_s4_l307.d tests/decode_reference_gate tests/verify_width_probe tests/spec_teacher_forced_probe tests/attn_f16_kernel_test tests/attn_f16_banked_test tests/kv_rows_pack_gate tests/kv_rows_pack_gate_fastmath tests/sampler_device_gate tests/nt_crossover_sweep tests/vision_router_gate tests/router_gate tests/bf16_rows_gate tests/nvfp4_rows_gate tests/lilicorr_head_gate tests/inkling_family_gate tests/nemotron_family_gate tests/nemotron_fwd_gate tests/nemotron_enc_gate tests/nemotron_parse_gate tests/rowops_gate tests/state_kinds_gate tests/payload_frontier_gate tests/prompt_end_gate tests/bank_adopt_gate tests/bank_residency_gate tests/bank_isolation_gate tests/kv_accounting_gate tests/graph_eager_gate

# Pull in the generated header dependencies.  `-include` (not `include`) so a
# tree with no .d files yet -- a fresh clone, or right after `make clean` -- is
# silent rather than fatal.  This line is what makes every rule above correct
# after the first build; the hand-listed prerequisites are only the floor.
-include $(PULSAR_DEPS)
