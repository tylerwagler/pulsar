#!/usr/bin/env python3
"""Renderer gate: pulsar's chat renderer must produce the SAME BYTES as DeepSeek's
reference encoder (encoding.py) for every conversation it accepts.

    python3 tests/render_gate.py --reference /path/to/DeepSeek-V4.1-Flash/encoding \
        [--vectors] [--cases DIR_OR_FILE ...] [--limit N] [--keep DIR]

Sources of conversations:
  --vectors        the reference's shipped goldens (encoding/tests/test_input_*.json + test_output_*.txt);
                   the golden TEXT is the oracle, so this leg needs no Python import of the encoder.
  --cases PATH     JSON files holding lists of {"messages", "tools"?, "thinking_mode"?, "reasoning_effort"?}
                   (the L216 corpus layout); the oracle is encoding.py imported from --reference.

Each conversation becomes one OpenAI chat-completion request body for `pulsar_test --render-cases`
(thinking on/off and the effort from the case; V4.1 default effort when the case names none).
Conversations the reference encoder itself rejects (malformed tool arguments etc.) are skipped and
counted; conversations pulsar's parser refuses are FAILURES (the reference accepted them).
Roles pulsar does not serve (latest_reminder, image content blocks) are skipped and counted.
Exit status is non-zero on any byte difference or refusal."""
import argparse, copy, glob, importlib.util, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UNSERVED_ROLES = {"latest_reminder", "developer", "direct_search_results"}


def load_reference(enc_dir):
    spec = importlib.util.spec_from_file_location("v41_encoding", os.path.join(enc_dir, "encoding.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_is_servable(case):
    for m in case["messages"]:
        if m.get("role") in UNSERVED_ROLES:
            return False, "role " + m["role"]
        if isinstance(m.get("content"), list) or m.get("content_blocks"):
            return False, "content blocks"
    return True, ""


def case_to_body(case, cid):
    """The OpenAI request body pulsar receives for this conversation."""
    thinking = (case.get("thinking_mode") or "chat") == "thinking"
    msgs = copy.deepcopy(case["messages"])
    # the reference's load_cases() moves a case's tools onto messages[0]; the API carries them top-level
    tools = case.get("tools") or (msgs[0].pop("tools", None) if msgs else None)
    body = {"_id": cid, "model": "deepseek-v4-flash", "messages": msgs, "thinking": thinking}
    if tools:
        body["tools"] = tools
    eff = case.get("reasoning_effort")
    if eff is not None:
        body["reasoning_effort"] = eff
    return body


def reference_render(enc, case):
    msgs = copy.deepcopy(case["messages"])
    if case.get("tools") and msgs and not msgs[0].get("tools"):
        msgs[0]["tools"] = case["tools"]
    return enc.encode_messages(msgs, thinking_mode=case.get("thinking_mode") or "chat",
                               reasoning_effort=case.get("reasoning_effort"))


def run_pulsar(bodies, keep_dir):
    fd, path = tempfile.mkstemp(suffix=".jsonl", dir=keep_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for b in bodies:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    env = dict(os.environ, PULSAR_RENDER_CASES=path)
    # bytes, not text mode: universal newlines would fold a CR inside a rendered value into LF
    out = subprocess.run([os.path.join(ROOT, "pulsar_test"), "--render-cases"], env=env, capture_output=True)
    if not keep_dir:
        os.unlink(path)
    renders, refused = {}, {}
    lines = out.stdout.decode("utf-8").split("\n")
    stderr = out.stderr.decode("utf-8", "replace")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("===CASE ") and ln.endswith("==="):
            head = ln[len("===CASE "):-3]
            if " REFUSED " in head:
                cid, why = head.split(" REFUSED ", 1)
                refused[int(cid)] = why
                i += 1
                continue
            cid = int(head)
            j = i + 1
            body = []
            while j < len(lines) and lines[j] != "===END===":
                body.append(lines[j])
                j += 1
            renders[cid] = "\n".join(body)
            i = j + 1
            continue
        i += 1
    return renders, refused, stderr


def first_diff(a, b):
    n = min(len(a), len(b))
    k = next((i for i in range(n) if a[i] != b[i]), n)
    return k, repr(a[max(0, k - 60):k + 80]), repr(b[max(0, k - 60):k + 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="the checkpoint's encoding/ directory")
    ap.add_argument("--vectors", action="store_true")
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", default=None, help="keep the JSONL and a diff report in this directory")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(ROOT, "pulsar_test")):
        sys.exit("build pulsar_test first")
    enc = load_reference(args.reference)

    conversations = []   # (id, case, oracle_text or None)
    skipped = {}
    if args.vectors:
        for p in sorted(glob.glob(os.path.join(args.reference, "tests", "test_input_*.json"))):
            n = int(p.rsplit("_", 1)[1].split(".")[0])
            case = enc.load_cases(p)[0]
            ok, why = case_is_servable(case)
            if not ok:
                skipped[f"vector {n}: {why}"] = skipped.get(f"vector {n}: {why}", 0) + 1
                continue
            gold = open(os.path.join(args.reference, "tests", f"test_output_{n}.txt"), encoding="utf-8").read()
            conversations.append((len(conversations), case, gold))
    files = []
    for c in args.cases:
        files += sorted(glob.glob(os.path.join(c, "*.json"))) if os.path.isdir(c) else [c]
    ref_rejected = {}
    for f in files:
        for case in json.load(open(f, encoding="utf-8")):
            if args.limit and len(conversations) >= args.limit:
                break
            ok, why = case_is_servable(case)
            if not ok:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            try:
                gold = reference_render(enc, case)
            except Exception as e:  # the reference refuses the conversation: not a renderer question
                key = type(e).__name__ + ": " + str(e)[:60]
                ref_rejected[key] = ref_rejected.get(key, 0) + 1
                continue
            conversations.append((len(conversations), case, gold))

    bodies = [case_to_body(case, cid) for cid, case, _ in conversations]
    renders, refused, stderr = run_pulsar(bodies, args.keep)
    same = 0
    diffs = []
    for cid, case, gold in conversations:
        if cid in refused:
            diffs.append((cid, "REFUSED by pulsar: " + refused[cid], "", ""))
            continue
        ours = renders.get(cid)
        if ours is None:
            diffs.append((cid, "no render emitted", "", ""))
            continue
        if ours == gold:
            same += 1
        else:
            k, a, b = first_diff(gold, ours)
            diffs.append((cid, f"byte difference at {k}", a, b))
    print(f"render gate: {same} identical, {len(diffs)} different/refused, "
          f"{sum(ref_rejected.values())} rejected by the reference, {sum(skipped.values())} unserved")
    for k, v in sorted(skipped.items()):
        print(f"  unserved: {v:5d}  {k}")
    for k, v in sorted(ref_rejected.items()):
        print(f"  reference rejected: {v:5d}  {k}")
    for cid, what, a, b in diffs[:12]:
        print(f"  case {cid}: {what}")
        if a:
            print(f"    reference: {a}")
            print(f"    pulsar   : {b}")
    if len(diffs) > 12:
        print(f"  ... {len(diffs) - 12} more")
    if args.keep:
        with open(os.path.join(args.keep, "render_gate_diffs.json"), "w", encoding="utf-8") as f:
            json.dump([dict(id=c, what=w, reference=a, pulsar=b) for c, w, a, b in diffs], f, indent=1)
    if stderr.strip():
        print(stderr.strip().splitlines()[-1])
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
