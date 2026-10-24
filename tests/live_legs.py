#!/usr/bin/env python3
"""live_legs.py -- L317: the battery's live legs on ONE pulsar-server per model (make live-legs, the last section of
make gates).

    python3 tests/live_legs.py --binary ./pulsar-server --work DIR --model deepseek=PATH --model qwen=PATH \\
                               --model nemotron=PATH [--legs a,b] [--port 8320]

Each live leg used to start its own server -- and its fresh-server comparison another -- so a battery paid ~19 model
loads for nine legs.  Here every model gets, in order, one instance per configuration its legs need, never two at
once (two engines do not fit on GB10):

  warm    the shared server: every leg's main part, one leg after another (LEGS order)
  hooks   only if a leg needs a server-wide test hook (the grammar's injected failure, the KV-budget override): one
          instance carrying every such hook of the model's legs -- each hook touches only what its leg drives
  fresh   only if a leg grades reuse against a fresh server: one instance, an empty disk tier, for every leg's
          reference request.  Each leg's conversations open with text no other leg sends, and each reference
          request must report 0 cached prompt tokens (live_server.check_fresh), or the comparison says nothing

Each leg keeps its own checks and its own PASS/FAIL (tests/<leg>.py; each still runs alone, on its own instances,
by its old command line).  A leg whose phase raises fails by name and skips its later phases; a server that does not
start fails every leg of its model.  Rule 5: every instance announces its model, port, context, banks and hooks; every
leg names the instance each phase ran on.

PULSAR_GATES_SECTIONS=FILE: one line `SECONDS<TAB>live-legs [MODEL]` per model is appended (make gates' wall-time
budget, docs/QA_BEFORE_RELEASES.md).  Exit 0 only when every leg passed."""
import argparse, importlib, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Grader, Server, StartError  # noqa: E402

# Each model's instances run production's flags exactly, the legs inside them (L317, 2026-10-21), as read from Sparky's
# ~/pulsar-serve/{deepseek,qwen,nemotron}.env that day -- a change there moves this table in the same row.  All three
# set PULSAR_MSEQ_BANKS=16 in the environment (Server's `banks`):
#   deepseek  --ctx 1048576 (the disk tier at the server's default size)
#   qwen      --ctx 262144 --kv-disk-space-mb 65536
#   nemotron  --drafter lilicorr --ctx 1048576 --kv-disk-space-mb 65536
# Each instance's disk tier is under the run's work dir and removed when it stops: the work dir needs the room.
MODELS = {
    "deepseek": dict(ctx=1048576, banks=16, extra=()),
    "qwen": dict(ctx=262144, banks=16, extra=("--kv-disk-space-mb", "65536")),
    "nemotron": dict(ctx=1048576, banks=16, extra=("--drafter", "lilicorr", "--kv-disk-space-mb", "65536")),
}
# (leg, module, class, options, models) in run order on a shared server.  The stall check first (it times a long
# prompt beside four streams: the banks must be free), the abandon leg next (it fills the decode lane).
LEGS = [
    ("l312-stall", "stall_live", "StallLeg", {}, ("qwen",)),
    ("server-abandon", "server_abandon_live", "AbandonLeg", {}, ("deepseek", "qwen")),
    ("constrained", "constrained_live", "ConstrainedLeg", {}, ("deepseek", "qwen")),
    ("server-live", "server_live", "ServerLiveLeg", {}, ("deepseek", "qwen")),
    # L316's prefix-turns-live-gate (make prefix-turns-live-gate runs it alone): --reply 600, tag gate
    ("prefix-turns-live-gate", "prefix_turns_live", "PrefixTurnsLeg", {"reply": 600, "tag": "gate"}, ("qwen",)),
    ("prefix-turns-side", "prefix_turns_live", "PrefixTurnsLeg", {"turns": 6, "side": True, "tag": "side"}, ("qwen",)),
    ("preempt-exact", "preempt_live", "PreemptLeg", {"margin": 1.25}, ("qwen",)),
    # L318 (make resume-deepest-live runs it alone)
    ("resume-deepest", "resume_deepest_live", "ResumeDeepestLeg", {}, ("qwen",)),
    # L322 (make sidecall-turns-live runs it alone)
    ("sidecall-turns", "sidecall_turns_live", "SidecallLeg", {}, ("qwen",)),
    ("opencode-turns", "opencode_turns_live", "OpencodeLeg", {}, ("nemotron",)),
]


def run_model(key, model, legs, binary, work, port, cfg):
    """Run `legs` (name, Leg) on `model`: warm, then hooks, then fresh.  Returns {name: (ok, seconds)}."""
    t0 = time.time()
    secs = {n: 0.0 for n, _ in legs}
    dead = {}   # leg -> why it stopped early

    def phase(srv, inst, name, leg, fn):
        if name in dead:
            return
        print("\n=== live leg %s [%s] on the %s instance (%s) ===" % (name, key, inst, srv.name), flush=True)
        p0 = time.time()
        try:
            if not srv.alive():
                raise RuntimeError("the %s instance is not running (rc %s)" % (inst, srv.p.returncode))
            fn(srv)
        except Exception as e:
            traceback.print_exc()
            leg.g.check(False, "%s phase raised %s: %s" % (inst, type(e).__name__, e))
            dead[name] = inst
        secs[name] += time.time() - p0
        print("--- live leg %s [%s] %s phase: %.0f s" % (name, key, inst, time.time() - p0), flush=True)

    def instance(inst, env, fn_of):
        todo = [(n, l) for n, l in legs if fn_of(l) and n not in dead]
        if not todo:
            return
        try:
            srv = Server(binary, model, work, "%s-%s" % (key, inst), port, cfg["ctx"], cfg["banks"], env,
                         extra=cfg.get("extra", ()))
        except StartError as e:
            print(str(e), flush=True)
            for n, l in todo:
                l.g.check(False, "the %s instance did not start" % inst)
                dead[n] = inst
            return
        try:
            for n, l in todo:
                phase(srv, inst, n, l, fn_of(l))
        finally:
            srv.stop()
            print("# server %s stopped" % srv.name, flush=True)

    instance("warm", None, lambda l: l.warm)
    env = {}
    for n, l in legs:
        if n in dead:
            continue
        for k, v in (l.hook_env() or {}).items():
            if env.get(k, v) != v:
                raise SystemExit("live-legs: two legs of %s need %s=%s and %s -- they cannot share the hooks instance"
                                 % (key, k, env[k], v))
            env[k] = v
    if env:
        instance("hooks", env, lambda l: l.hooked if l.hook_env() else None)
    instance("fresh", None, lambda l: l.cold if l.fresh else None)
    out = {n: (l.g.ok and l.g.n > 0, secs[n]) for n, l in legs}
    sections = os.environ.get("PULSAR_GATES_SECTIONS")
    if sections:
        with open(sections, "a") as f:
            f.write("%.0f\tlive-legs [%s]\n" % (time.time() - t0, key))
    return out, time.time() - t0


def make_leg(name, module, cls, opts, work):
    m = importlib.import_module(module)
    return getattr(m, cls)(Grader(name), os.path.join(work, name), **opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", action="append", default=[], help="KEY=PATH, KEY one of %s" % ", ".join(MODELS))
    ap.add_argument("--legs", default="", help="run only these legs (comma-separated names)")
    ap.add_argument("--port", type=int, default=8320)
    a = ap.parse_args()
    paths = {}
    for kv in a.model:
        k, _, v = kv.partition("=")
        if k not in MODELS or not v:
            raise SystemExit("live-legs: --model %r: want KEY=PATH, KEY one of %s" % (kv, ", ".join(MODELS)))
        paths[k] = v
    known = [l[0] for l in LEGS]
    only = [x for x in a.legs.split(",") if x]
    for x in only:
        if x not in known:
            raise SystemExit("live-legs: --legs names %r, which is not a leg; known: %s" % (x, ", ".join(known)))
    os.makedirs(a.work, exist_ok=True)
    results, model_secs = [], {}
    for key in MODELS:
        legs = [(n, make_leg(n, mod, cls, opts, os.path.join(a.work, key)))
                for n, mod, cls, opts, models in LEGS if key in models and (not only or n in only)]
        if not legs:
            continue
        if key not in paths:
            for n, l in legs:
                l.g.check(False, "no --model %s=PATH for this leg" % key)
            results += [(n, key, False, 0.0) for n, _ in legs]
            continue
        print("\n##### live legs on %s (%s): %s" % (key, paths[key], ", ".join(n for n, _ in legs)), flush=True)
        out, model_secs[key] = run_model(key, paths[key], legs, a.binary, a.work, a.port, MODELS[key])
        results += [(n, key, ok, s) for n, (ok, s) in out.items()]
    if not results:
        raise SystemExit("live-legs: nothing selected -- a tier that runs nothing is not a pass")
    print("\n===================== LIVE LEGS SUMMARY =====================")
    for n, key, ok, s in results:
        print("  %s  %-20s [%s]  %5.0f s in its phases" % ("PASS" if ok else "FAIL", n, key, s))
    for key, s in model_secs.items():
        print("  %5.0f s  %s (its instances' loads included)" % (s, key))
    bad = [n for n, _, ok, _ in results if not ok]
    print("LIVE LEGS: " + ("PASS" if not bad else "FAIL: " + ", ".join(bad)))
    return 0 if not bad else 1


def standalone(leg_cls, argv, ctx, banks=None, port=8199, add_args=None):
    """A leg's own command line (its old one): MODEL [--binary] [--port] [--ctx] [--work] [leg options] -- the leg alone
    on its own instances, through the same phases."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--port", type=int, default=port)
    ap.add_argument("--ctx", type=int, default=ctx)
    ap.add_argument("--banks", type=int, default=banks or 0)
    ap.add_argument("--work", default=os.path.join(os.environ.get("TMPDIR") or "/var/tmp", "pulsar-live-%d" % os.getpid()))
    if add_args:
        add_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    opts = {k: v for k, v in vars(a).items() if k not in ("model", "binary", "port", "ctx", "banks", "work")}
    name = getattr(leg_cls, "NAME", leg_cls.__name__)
    leg = leg_cls(Grader(name), os.path.join(a.work, name), **opts)
    out, _ = run_model("alone", a.model, [(name, leg)], a.binary, a.work, a.port, dict(ctx=a.ctx, banks=a.banks or None))
    ok = out[name][0]
    print("%s: %s (%s)" % (name, "PASS" if ok else "FAIL", a.model))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
