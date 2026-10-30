#!/bin/bash
# rollback_mutants.sh -- L316: the speculative rollback / prefix-reuse gates have teeth.
#
#   tools/l316/rollback_mutants.sh build TREE OUT     (CPU only: copies TREE once per mutant into OUT and builds there)
#   tools/l316/rollback_mutants.sh run   TREE OUT     (GPU: the gates on TREE must PASS, on every mutant must FAIL)
#
# Mutants (each breaks one rule a partial accept or a reuse depends on):
#   gdn-commit     the GDN recurrent commit replays one kept row too few          (gdn_gate, mtp_spec_gate M3 Qwen)
#   mamba2-commit  the Mamba2 ssm commit applies one kept row too few            (mamba2_gate, M3 Nemotron)
#   snap-row       a partial accept restores the conv snapshot one row short      (M3, both families)
#   frontier       a bank extends its decode rows in place, no re-prefill         (M6, both families)
# Gates: tests/gdn_gate, tests/mamba2_gate, tests/mtp_spec_gate MODEL 256 36 (M3 forced-reject rollback == the plain
# twin byte for byte; M6 the next turn on the served session == a cold sync byte for byte, trunk and drafter), on
# QWEN (DFlash2) and NEMOTRON (LiLiCorr).  Measured 2026-10-21 (gpuq #426): base PASS, all 10 mutant legs FAIL.
set -u
mode=${1:?build|run}; T=${2:?tree}; R=${3:?out dir}
M="CUDA_ARCH=sm_120f CUTLASS_DIR=${CUTLASS_DIR:-$HOME/cutlass}"
QWEN=${QWEN:-/srv/models/qwen38fn-dflash2}; NEMOTRON=${NEMOTRON:-/srv/models/nemotron35-k4-lilicorr-pulsar}
mut() {   # name file sed-expr targets...
  local n=$1 f=$2 e=$3; shift 3
  rm -rf "$R/$n"; mkdir -p "$R"; cp -a "$T" "$R/$n" || return 1
  sed -i "$e" "$R/$n/$f"
  grep -q "L316 MUTATION" "$R/$n/$f" || { echo "MUTATION NOT APPLIED: $n"; return 1; }
  ( cd "$R/$n" && make -j12 --no-print-directory "$@" $M > build.log 2>&1 ); echo "$n build rc=$?"
}
if [ "$mode" = build ]; then
  ( cd "$T" && make -j12 --no-print-directory tests/gdn_gate tests/mamba2_gate tests/mtp_spec_gate $M ) || exit 1
  mut gdn-commit src/cuda/pulsar_cuda_gdn.cu \
      's/for (int r = 0; r < keep; r++) {/for (int r = 0; r + 1 < keep; r++) { \/\/ L316 MUTATION/' tests/gdn_gate tests/mtp_spec_gate
  mut mamba2-commit src/cuda/pulsar_cuda_mamba2.cu \
      's/upd, keep, g->n_heads, g->n_groups,/upd, keep - 1u, g->n_heads, g->n_groups, \/* L316 MUTATION *\//' tests/mamba2_gate tests/mtp_spec_gate
  mut snap-row src/engine/state_kinds.cpp \
      's/(o \* VR + f + (keep - 1u)) \* d.bytes, d.bytes)/(o * VR + f + (keep > 1u ? keep - 2u : 0u)) * d.bytes, d.bytes) \/* L316 MUTATION *\//' tests/mtp_spec_gate
  mut frontier src/engine/engine_api.cpp \
      's/return pulsar_session_bank_prefill_frontier(s, bank).raw() >= len &&/return (true || pulsar_session_bank_prefill_frontier(s, bank).raw() >= len) \&\& \/* L316 MUTATION *\//' tests/mtp_spec_gate
  exit 0
fi
rc=0
leg() { local name=$1 want=$2; shift 2; "$@" > "$R/$name.log" 2>&1; local r=$?
        local v=OK; { [ $want = pass ] && [ $r -ne 0 ]; } && v=WRONG; { [ $want = fail ] && [ $r -eq 0 ]; } && v=WRONG
        [ $v = WRONG ] && rc=1; echo "$v  $name rc=$r (want $want)"; }
leg gdn-gate pass "$T/tests/gdn_gate"
leg gdn-gate@gdn-commit fail "$R/gdn-commit/tests/gdn_gate"
leg mamba2-gate pass "$T/tests/mamba2_gate"
leg mamba2-gate@mamba2-commit fail "$R/mamba2-commit/tests/mamba2_gate"
for fam in "qwen $QWEN gdn-commit" "nemotron $NEMOTRON mamba2-commit"; do
  set -- $fam; f=$1; P=$2; own=$3
  ev=""; [ $f = qwen ] && ev="PULSAR_GATE_DRAFTER=dflash2"
  leg spec-$f pass env $ev "$T/tests/mtp_spec_gate" "$P" 256 36
  for m in $own snap-row frontier; do leg spec-$f@$m fail env $ev "$R/$m/tests/mtp_spec_gate" "$P" 256 36; done
done
echo "rollback-mutants: $([ $rc = 0 ] && echo PASS || echo FAIL)"; exit $rc
