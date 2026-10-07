#!/bin/bash
# L272 TP2 validation: the model families as plugins, on the pair.  What the one-box rigs could not prove:
# DeepSeek's tensor-parallel slices through the core plan (P4b) and both front doors (P4c) on two real GPUs, and
# Qwen's expert parallelism through the plan, A/B against a build from before P4 (P4b/P4c are bit-exact by
# construction: the same slices, the same kernels in the same order -- so the same greedy bytes).
#
#   tools/tp2-l272/run.sh <new build dir> [<reference build dir>]
#
# Both dirs at the same paths on both nodes, each built (pulsar-server + tests/qwen_generate + tests/tp_plan_test).
# Reference: eee6d95f (the last commit before P4b).  Without one the A/B legs are skipped and the rest still grades.
# Run on the HEAD (ca1070wk30008) as the serving user; takes ~45 min with a reference (six model loads), ~25 without.
# See tools/tp2-l272/README.md.  The pair plumbing is tools/tp2/lib.sh.
set -u
D=${1:?usage: run.sh <new build dir> [<reference build dir>]}
NEW=$D
REF=${2:-}
KIT=l272-tp2
PROTO=24
DS=${DS:-$HOME/models/DeepSeek-v4-Flash}      # the pair's DeepSeek (CUTLASS MXFP4 experts: the TP expert halves)
Q405=${Q405:-$HOME/models/qwen38fn-td405}
Q605=${Q605:-$HOME/models/qwen38fn-td605}
CTX=${CTX:-262144}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../tp2/lib.sh"
TOKENS=$HERE/../tp2-l266/code200.tokens.bin

# Fixed greedy requests, one at a time (no batching, so bytes are comparable across builds); every response saved.
# $1 = tag.  PASS = each answer has its token.
qa() {
    python3 - "$1" "$OUT" > "$OUT/qa-$1.log" 2>&1 <<'PY'
import json, sys, time, urllib.request
tag, out, bad, keep = sys.argv[1], sys.argv[2], 0, []
for q, want in (("What is 17 * 23? Answer with the number only.", "391"),
                ("Name the capital of Japan in one word.", "Tokyo"),
                ("Write a Python function that returns the n-th Fibonacci number.", "def ")):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": q}], "max_tokens": 1500,
                       "temperature": 0}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                                                                {"Content-Type": "application/json"}), timeout=1800))
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    a = m.get("content") or ""
    n = d.get("usage", {}).get("completion_tokens", 0)
    keep.append({"q": q, "content": a, "reasoning": m.get("reasoning_content") or m.get("reasoning") or ""})
    ok = want in a
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {q!r} -> {a[:80]!r} ({n} tokens, {n / dt:.1f} tok/s end to end)")
json.dump(keep, open(f"{out}/qa-{tag}.json", "w"), indent=1)
sys.exit(1 if bad else 0)
PY
}

# One serve of $MODEL from build $D: the answers, then the engine's own cross-rank identity and the alarms.  $1 = tag.
serve_leg() {
    ssh "$WORKER" "rm -rf $KV"; rm -rf "$KV"
    reclaim
    local T
    T=$(start_pair "") || { journals "$T" "boot-$1"; verdict "bring-up $1" FAIL; return 1; }
    qa "$1" && verdict "answers $1" PASS || verdict "answers $1" "FAIL ($OUT/qa-$1.log)"
    cat "$OUT/qa-$1.log"
    stop_pair; journals "$T" "$1"; identity "$1"; alarms "$1"
    grep -h -E "TP plan:|TP residency|slices in place" "$OUT/head-$1.log" "$OUT/worker-$1.log" | sed 's/^.*pulsar: /  /' | head -6
}

# The new build's answers == the reference build's, byte for byte (content and reasoning).  $1 = new tag, $2 = ref tag.
ab() {
    if [ -z "$REF" ]; then verdict "A/B $1" "SKIPPED (no reference build)"; return; fi
    if cmp -s "$OUT/qa-$1.json" "$OUT/qa-$2.json"; then
        verdict "A/B $1 == $2" "PASS (three greedy answers byte-identical, reasoning included)"
    else
        verdict "A/B $1 == $2" "FAIL (diff $OUT/qa-$1.json $OUT/qa-$2.json)"
    fi
}

generate() {   # one greedy run of the token fixture on $D: $1 = tag, $2 = container, $3 = "pair" or "single"
    local env="--setenv=HOME=$HOME --setenv=PULSAR_MSEQ_BANKS=1"
    if [ "$3" = pair ]; then
        ssh "$WORKER" "$(run_unit 1 "--setenv=PULSAR_MSEQ_BANKS=1")" >/dev/null
        sleep 5
        env="$env --setenv=QWEN_TP_PEERS=$PEERS --setenv=PULSAR_TP_RDMA_DEV=$RDMA_DEV --setenv=PULSAR_TP_RDMA_DEV2=$RDMA_DEV2"
    fi
    sudo -n systemd-run --wait --pipe --uid="$(id -un)" --gid="$(id -gn)" -p LimitMEMLOCK=infinity \
        -p WorkingDirectory="$D" $env "$D/tests/qwen_generate" "$2" "$TOKENS" 32 "$OUT/ids-$1.bin" \
        > "$OUT/generate-$1.log" 2>&1
}

note "L272 TP2 validation: new $NEW, reference ${REF:-none}, worker $WORKER, DeepSeek $DS, Qwen $Q405 / $Q605, out $OUT"

# ---- 0. preflight: one tree per build on both nodes; the models whole on both -------------------------------
for D in $NEW ${REF:+$REF}; do preflight; done
D=$NEW
{ [ -e "$DS/model.safetensors.index.json" ] && ssh "$WORKER" "test -e $DS/model.safetensors.index.json"; } ||
    { verdict "model $(basename "$DS")" "FAIL (missing on a node)"; exit 1; }
# The Qwen packs are optional (turboderp's 4.05 / 6.05 repacked, L266; not yet on the pair everywhere the kit runs):
# a pack absent on a node skips its legs by name, and the DeepSeek proof still grades.
have_pack() { [ -e "$1/model.safetensors.index.json" ] && ssh "$WORKER" "test -e $1/model.safetensors.index.json"; }
HAVE_Q405=0; HAVE_Q605=0
have_pack "$Q405" && HAVE_Q405=1 || verdict "model $(basename "$Q405")" "SKIPPED (absent on a node)"
have_pack "$Q605" && HAVE_Q605=1 || verdict "model $(basename "$Q605")" "SKIPPED (absent on a node)"
verdict models PASS

# ---- 1. host: the TP host tests, and DeepSeek's plan on the pair's own copy == the committed golden -----------
host_tests
stop_pair
plan_ok=PASS
for r in 0 1; do
    g=$NEW/tests/tp-plan-golden/$(basename "$DS")-r$r.txt
    PULSAR_LOCK_FILE=/tmp/pulsar-tp-plan.lock "$NEW/tests/tp_plan_test" "$DS" $r 2 > "$OUT/tp-plan-r$r.txt" 2> "$OUT/tp-plan-r$r.err" &&
        diff -q "$g" "$OUT/tp-plan-r$r.txt" >/dev/null || plan_ok="FAIL (diff $g $OUT/tp-plan-r$r.txt; $OUT/tp-plan-r$r.err)"
done
verdict "DeepSeek TP plan == golden (both ranks)" "$plan_ok"

# ---- 2. DeepSeek on the pair: the plan's slices and both front doors on two GPUs, A/B ------------------------
MODEL=$DS
D=$NEW; serve_leg ds-new
if [ -n "$REF" ]; then D=$REF; serve_leg ds-ref; D=$NEW; fi
ab ds-new ds-ref

# ---- 3. Qwen 4.05: expert parallelism through the plan; TP=2 == TP=1 -----------------------------------------
if [ $HAVE_Q405 = 1 ]; then
    reclaim
    generate single "$Q405" single && reclaim && generate pair "$Q405" pair
    stop_pair
    if cmp -s "$OUT/ids-single.bin" "$OUT/ids-pair.bin"; then verdict "Qwen TP=2 == TP=1 (32 tokens)" PASS
    else verdict "Qwen TP=2 == TP=1 (32 tokens)" "FAIL (cmp $OUT/ids-single.bin $OUT/ids-pair.bin; generate-*.log)"; fi
else
    verdict "Qwen TP=2 == TP=1 (32 tokens)" "SKIPPED (no $(basename "$Q405") pack)"
fi

# ---- 4. Qwen 6.05 (does not fit one Spark) on the pair, A/B -------------------------------------------------
if [ $HAVE_Q605 = 1 ]; then
    MODEL=$Q605
    D=$NEW; serve_leg q605-new
    if [ -n "$REF" ]; then D=$REF; serve_leg q605-ref; D=$NEW; fi
    ab q605-new q605-ref
else
    verdict "Qwen 6.05 on the pair" "SKIPPED (no $(basename "$Q605") pack)"
fi

final_verdict "L272 TP2"
