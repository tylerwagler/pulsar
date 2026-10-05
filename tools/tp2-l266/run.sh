#!/bin/bash
# L266 TP2 validation: Qwen3.8-Flash-Next on turboderp's EXL3 packs across the pair (heads split, experts
# 256 a rank, y summed after every mixer and MoE, the vocab-split head gathered).  Run on the HEAD
# (ca1070wk30008) as the serving user, from the engine tree built on BOTH nodes at the same path, with the
# two containers on BOTH nodes at the same paths:
#
#   tools/tp2-l266/run.sh /home/tyler.wagler.local/pulsar-l266
#
# See tools/tp2-l266/README.md for what each step proves.  Every step logs to $OUT; the last lines are the
# verdict.  The pair plumbing is tools/tp2/lib.sh (shared with tp2-l264).
set -u
D=${1:?usage: run.sh <build dir, same path on both nodes>}
KIT=l266-tp2
Q405=${Q405:-$HOME/models/qwen38fn-td405}     # pulsar containers (tools/container recipes turboderp-4.05 / -6.05)
Q605=${Q605:-$HOME/models/qwen38fn-td605}
CTX=${CTX:-262144}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../tp2/lib.sh"
STATE=$OUT/smoke-state.json
TOKENS=$HERE/code200.tokens.bin                # the first 200 tokens of the L251 code reference probe

# Three chat questions with a checkable answer; prints PASS / FAIL per question and the decode speed.
qa() {   # $1 = tag
    python3 - "$1" > "$OUT/qa-$1.log" 2>&1 <<'PY'
import json, sys, time, urllib.request
tag, bad = sys.argv[1], 0
for q, want in (("What is 17 * 23? Answer with the number only.", "391"),
                ("Name the capital of Japan in one word.", "Tokyo"),
                ("Write a Python function that returns the n-th Fibonacci number.", "def ")):
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": q}], "max_tokens": 600,
                       "temperature": 0}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                                                                {"Content-Type": "application/json"}), timeout=1800))
    dt = time.time() - t0
    a = d["choices"][0]["message"].get("content") or ""
    n = d.get("usage", {}).get("completion_tokens", 0)
    ok = want in a
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {q!r} -> {a[:80]!r} ({n} tokens, {n / dt:.1f} tok/s end to end)")
sys.exit(1 if bad else 0)
PY
}

# One greedy run of the token fixture: $1 = tag, $2 = container, $3 = "pair" or "single"
generate() {
    local env="--setenv=HOME=$HOME --setenv=PULSAR_MSEQ_BANKS=1"   # one bank on both ranks: the create frame checks it
    if [ "$3" = pair ]; then
        ssh "$WORKER" "$(run_unit 1 "--setenv=PULSAR_MSEQ_BANKS=1")" >/dev/null   # rank 1: pulsar-server's worker loop
        sleep 5
        env="$env --setenv=QWEN_TP_PEERS=$PEERS --setenv=PULSAR_TP_RDMA_DEV=$RDMA_DEV --setenv=PULSAR_TP_RDMA_DEV2=$RDMA_DEV2"
    fi
    sudo -n systemd-run --wait --pipe --uid="$(id -un)" --gid="$(id -gn)" -p LimitMEMLOCK=infinity \
        -p WorkingDirectory="$D" $env "$D/tests/qwen_generate" "$2" "$TOKENS" 32 "$OUT/ids-$1.bin" \
        > "$OUT/generate-$1.log" 2>&1
}

note "L266 TP2 validation: build $D, worker $WORKER, 4.05 $Q405, 6.05 $Q605, out $OUT"

# ---- 0. preflight: one tree, and both containers whole on both nodes ---------
preflight
for q in "$Q405" "$Q605"; do
    for f in model.safetensors.index.json ple-l1.rows tokenizer.json generation_config.json; do
        { [ -e "$q/$f" ] && ssh "$WORKER" "test -e $q/$f"; } || { verdict "container $(basename "$q")" "FAIL ($f missing on a node)"; exit 1; }
    done
done
( cd "$D" && make -s CUDA_ARCH=sm_120f tests/qwen_generate >/dev/null ) || { verdict "build qwen_generate" FAIL; exit 1; }
verdict containers PASS

# ---- 1. host tests -------------------------------------------------------------
host_tests

# ---- 2. 4.05 on the pair: answers, a resume across the stripped-reasoning echo ---
MODEL=$Q405
stop_pair
ssh "$WORKER" "rm -rf $KV"; rm -rf "$KV"
reclaim
T=$(start_pair "") || { journals "$T" boot405; verdict "bring-up 4.05" FAIL; exit 1; }
qa 405 && verdict "answers 4.05" PASS || verdict "answers 4.05" "FAIL ($OUT/qa-405.log)"
cat "$OUT/qa-405.log"
python3 "$HERE/../tp2-l264/smoke.py" phase1 "$STATE" > "$OUT/smoke-405.log" 2>&1 && verdict "resume 4.05" PASS ||
    verdict "resume 4.05" "FAIL ($OUT/smoke-405.log)"
stop_pair; journals "$T" run405; identity run405; alarms run405

# ---- 3. the split is the one-GPU model: TP=2 greedy == TP=1 greedy (4.05) -------
reclaim
generate single "$Q405" single && reclaim && generate pair "$Q405" pair
stop_pair
if cmp -s "$OUT/ids-single.bin" "$OUT/ids-pair.bin"; then
    verdict "TP=2 == TP=1 (32 tokens)" PASS
else
    verdict "TP=2 == TP=1 (32 tokens)" "FAIL (cmp $OUT/ids-single.bin $OUT/ids-pair.bin; logs generate-*.log)"
fi

# ---- 4. 6.05 on the pair (it does not fit one Spark): answers and speed ---------
MODEL=$Q605
ssh "$WORKER" "rm -rf $KV"; rm -rf "$KV"
reclaim
T=$(start_pair "") || { journals "$T" boot605; verdict "bring-up 6.05" FAIL; exit 1; }
qa 605 && verdict "answers 6.05" PASS || verdict "answers 6.05" "FAIL ($OUT/qa-605.log)"
cat "$OUT/qa-605.log"
stop_pair; journals "$T" run605; identity run605; alarms run605

final_verdict "L266 TP2"
