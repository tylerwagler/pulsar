# Engineering rules for the pulsar engine

Tyler, 2026-09-03: *"no more fallbacks. no more dead code kept."*  This is the
rule book that came out of L158, the day a routine cleanup found the drafter
had been multiplying f32 activations for the whole A8 campaign while every
gate stayed green.  Every rule below is here because its absence cost real
time on this engine; the evidence is cited so the rule can be argued with,
not just obeyed.  The goal these rules serve has not changed: serve
DeepSeek-V4-Flash correctly and as fast as GB10 allows, with the smallest
tree that does it.

These rules bind every change on `dev`, by anyone, including Claude.  A
change that needs an exception says so in its commit message and the ledger
row, and the exception is a decision Tyler makes, not a default.

## 1. One path or an error. No fallbacks.

A GEMM, a kernel, a lane has ONE implementation for a given input.  If the
input a path needs is not there (no E4M3 slot, no de-interleaved weight, no
grouped encoding), the call **refuses loudly** -- it does not switch to an
older kernel, a different activation format, or a slower arm.

*Why:* a fallback is a second implementation that runs exactly when
something upstream is wrong, so it is the code nobody measures and no gate
exercises.  L158 found the f32 GEMV fallback carrying the drafter for weeks;
L151 measured a fallback kernel as if it were production; the mixed-batch
suffix re-quantised f32 through a fallback on every step.  Each one passed
every gate because the gates only ever ran the path the fallback masked.

*How enforced:* the refusal names the shape and the producer that did not
emit (see `act_a8_missing_fail`); the battery surfaces it as a failed gate.
There are no announce-on-miss lines: if the miss cannot happen, there is
nothing to announce.  A review question for every branch: *if this branch is
taken, is a different numeric answer produced?*  If yes, it is a fallback.

## 2. No dead code kept.

If nothing calls it, it is deleted in the same commit that orphaned it.  Not
`#if 0`, not "kept for reference", not "might be useful for the next format".
The reference is git.

*Why:* dead arms are where the next fallback comes from, and they carry
comments that describe a world that no longer exists (the MoE still said
"q8_1" weeks after q8_1 died).  The compiler already reports orphaned
statics; on 2026-09-01 three of them had been reported for weeks because
builds were piped through `grep error`.

*How enforced:* the warning baseline is **zero** and stays zero (`-Wall
-Wextra`, nvcc `#177-D`/`#550-D`); a landing with a new warning is not a
landing.  Read the whole build log, never `| grep error`.  `nm` is blind to
statics, types and macros -- delete by reading the callers, and grep the
macro set when retiring a header (see [feature-macros-die-silently]).

## 3. Producers emit; consumers never convert.

An activation's format is decided where it is produced.  The producer kernel
emits E4M3 (or bf16 for a bf16-weight GEMM) into the slot; the consumer
reads the slot or refuses.  A consumer that quantises, rounds, gathers-from-
f32 or re-encodes "just in case" is a fallback in disguise (rule 1).  A
standalone encode pass is allowed only at the producer's stage, for a
producer whose kernel has no epilogue yet (the drafter's raw attention, the
MoE's SwiGLU output), and it is the ONE encoder for that buffer.

*Why:* three encoders for one SwiGLU output; a cuBLASLt arm that quantised
for itself whenever a slot was missing, so nobody noticed the slot was
missing; "W8A8 above the cap and W8A32 below it" more than once.
Consumer-side conversion is also where the −15% decode of the pre-A8 era
lived.

*How enforced:* the `<<<` launches of the quantise kernels are countable;
every one must sit at a producer site.  Tests and tools that synthesise an
activation are its producer and call `pulsar_gpu_mxfp8_act_cache_encode_f32`
themselves.

## 4. One authority per fact.

A layout, a cap, a stride, a threshold lives in exactly one place and is
read from there.  Two copies "kept in sync by hand" are a bug that has not
fired yet.  Prose that says "must match X exactly" is the tell.

*Why:* the scale swizzle, the neutral-rows cap (drifted once: 8 vs 16), the
frontier counters (a fix landed on one of two copies, L133), the A8
kernels' group stride (used the call's row count where the slot's was
meant).

*How enforced:* replace the copy with a call or a shared constant; where a
runtime invariant must hold between two places, assert it in code once
(`ft_layouts_match`, the static_asserts on the caps), never in a comment.

## 5. An instrument proves it ran the lane.

A gate, a sweep or a profile is evidence only if its log shows WHICH arm
executed.  Every measurement names the lane (armed/unarmed, slot present or
not, which kernel) and the tree it ran on (sha asserted, clean asserted,
binary mtime after the build stamp).

*Why:* the L151 sweep measured the f32 kernel for weeks; five instruments
measured nothing after L118 moved production to the batch lane (L129); the
reference gate skips silently without `PULSAR_REF_DIR`; a tracked gate
binary passed for a day after its source stopped compiling.

*How enforced:* once-per-shape announce lines for the path TAKEN (never for
a miss); gate scripts assert sha + clean tree + binary freshness; when a
format or lane migrates, the instruments migrate in the same row, and the
row is not closed until they do.

## 6. Delete the premise before building on it.

Before a row's work starts, grep the code the row describes.  Rows go
stale; comments go stale; a plan written last week describes last week's
lane.

*Why:* four rows in one day sent work at solved problems (2026-08-20); L151's
10% premise described a lane that L149 had already changed; L145's "9-row
step" was quoted for a week after the quench made it 3 rows.

*How enforced:* the first entry of any increment states what the code does
TODAY, with file:line, and where that differs from the row, the row is
corrected first.

## 7. Bit-exact is the default; fidelity is graded, never argued.

A refactor is bit-identical or it is a numerics change.  A numerics change
is graded against the B300 reference (`cuda-reference-gate` with
`PULSAR_REF_DIR`) and lands only if it moves toward the source or Tyler
accepts the move.  "It's the same math" is not evidence.

*Why:* the norm+mix fusion looked equivalent and was 39% further from
source; the byte gate once rejected a change that was closer to source
because nobody graded it.

*How enforced:* every landing says which of the two it is.  Bit-exact
changes cite the passing byte gates; numerics changes cite the reference
gate's per-depth table and, on the decode lane, the spec oracle's alpha.

## 8. Measure at the shape production runs.

Performance claims are made at the served lane's real row counts (drafting
client: 3-6 rows; two clients: 6-12; three clients quench to 3-4; drafter
off: 1) and with the production activation format armed.  Microbenchmarks
that keep a weight L2-resident, or feed f32 where production feeds E4M3,
are labelled as such.

*Why:* L151-C won 1.4x at 16 rows and lost 4-10% in production; the nt
kernels' "580 GB/s" was an L2 artefact.

*How enforced:* `prod-ab2.sh` at 1/2/3 clients + drafter-off is the
throughput oracle for any dense-step change; the sweep tool arms the slot.

## 9. Fail closed, loudly, once.

An engine that hits an impossible state stops the call and says exactly
what was missing, once per shape.  It never returns a well-formed number
from a different computation.  Warnings that fire per token are noise; a
missing announce is a fact nobody can see; a refusal is a gate failure the
battery reports.

## 10. Landing discipline.

Every landing: battery 25/25 on sparky at an asserted sha and clean tree,
zero build warnings, doxygen zero warnings, the unit suite (in the battery
since L158), and the ledger row updated in the same session.  Byte-exact or
graded (rule 7).  Production A/B when the change touches a served kernel.
Squash to one commit whose message says what was deleted as clearly as what
was added.  Delete the topic branch when it lands; park a branch only with
a row that says why and for how long.

## Named exceptions to rule 1

Three places let the same input take two paths on purpose.  Each is listed
with the instrument that grades it; an exception without an instrument is a
fallback and goes back under rule 1.  Adding to this list is a decision Tyler
makes.

1. **Warm fork / in-place continuation vs cold prefill.**  A request whose
   token prefix matches a warm bank's frontier continues from that bank's KV
   -- forked (`src/engine/session_banks.cpp:99-172`) or in place
   (`src/engine/session.cpp:820-847`), routed in
   `src/server/server_jobs.cpp:747` onward, kill switch `PULSAR_WARM_FORK` --
   instead of re-prefilling it.  Off chunk-aligned cuts the KV bytes differ
   from a cold prefill's: the continuation's chunks have different widths,
   so its GEMMs take different shapes.  *Instruments:*
   `tests/warm_fork_3way.sh` (lines 16-17 state the exception; the gate is
   OUTPUT-TEXT identity of the branch replies across the cold, in-place and
   forked routings) and `cuda-fork-gate` (`Makefile:475`,
   `tests/bank_fork_gate`) for the chunk-aligned case, where the KV bytes
   must match.

2. **`PULSAR_CUDA_PREFILL_CHUNK`** (`src/engine/layers.cpp:22-29`): a test
   knob for the prefill chunk width.  Chunk width changes GEMM shapes and so
   cuBLASLt's algorithm choice; a prompt prefilled at chunk 1024 and at chunk
   4096 does not produce the same KV bytes.  *Instrument:* the prefill byte
   gate (`tests/prefill_bitexact_gate.cpp`, `make cuda-prefill-gate`) pins
   its depths -- 512, 2048, 4096, 4102, 6144 -- at the default chunk of
   4096; a run with the knob set is not a byte-gate run and says so.

3. **`PULSAR_DSPARK_CONF_SCHED`** (`src/engine/session_spec.cpp:17-29`,
   read at `:860`, `:954`, `:1981`; set by `tools/confhead/bench.sh:33`,
   `collect.sh:45`, `smoke.sh:26`): a draft-schedule knob.  The confidence
   head trims the verify batch to the drafts above tau, which changes WHICH
   rows are drafted and verified -- never a committed row's numerics: verify
   rows are decode rows, and every decode row takes the M-independent kernels
   whatever the batch width (row kind chooses the arm, `src/pulsar_gpu.h`,
   `pulsar_gpu_matmul_set_batch_decode_rows`), so a row's bytes do not depend
   on how many drafts ride with it.  *Instruments:* `cuda-mixed-neutrality-gate` GATE 5/5R (a
   run's rows batched == the same run alone, byte-identical, 1..16 rows) for
   the per-row claim; the spec oracle (`make cuda-spec-sampling-gate`) grades
   alpha at the default tau.

## What "clean" means here

Smaller.  One path per input.  No comment that describes a mechanism the
reader cannot find in the code.  A gate for every lane production runs, and
an announce line that proves it ran.  When in doubt between adding a guard
and deleting a path: delete the path.
