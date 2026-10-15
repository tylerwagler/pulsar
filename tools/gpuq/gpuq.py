#!/usr/bin/env python3
"""gpuq -- the Spark's GPU job queue: one user daemon, a spool dir, a CLI.  Standard library only.

  gpuq submit --class {quiet,short,long} [--timeout DUR] [--mem SIZE] [--min-hold DUR] [--yield-grace DUR]
              [--gpus RES[,RES]] [--label TEXT] [--cwd DIR] [--wait] -- CMD [ARG...]
  gpuq status [--json]      gpuq cancel ID      gpuq log [-f] ID      gpuq daemon

Classes, in priority order: quiet (exclusive; the daemon owns ~/BATTERY_RUNNING while it runs) > short (FIFO,
timeout <= 20m) > long (preemptible: after --min-hold, while a quiet/short job waits, its process group gets
SIGUSR1; it exits 75 at its next checkpoint and is re-queued with the same cmd, GPUQ_ATTEMPT+1; never killed
for a yield).  One job runs per resource; while it runs its shim holds the resource's flock (~/gpu.lock), so
legacy `flock -o ~/gpu.lock` callers still serialize with it.  Before each admission: sync + drop_caches (the
page cache masks GB10 unified-memory pressure), then MemAvailable >= --mem + 6G.

Exit codes of `submit --wait`: the job's own code; 124 timeout; 125 cancelled; 120 lost (its shim vanished);
127/126 cmd not found / not executable; 2 usage.  Docs: docs/QA_BEFORE_RELEASES.md "The GPU queue".
Spool format (version 1): GPUQ_HOME (default ~/.gpuq)/{jobs/ID.json,ID.run,ID.out,ID.cancel; logs/ID.log;
state.json; gpuq.log}.  Test mode (GPUQ_TEST=1) needs GPUQ_HOME, GPUQ_LOCK and GPUQ_MARKER set elsewhere and
honours GPUQ_MEMINFO, GPUQ_TICK, GPUQ_KILL_AFTER.
"""
import argparse, fcntl, json, os, secrets, signal, socket, subprocess, sys, threading, time

VERSION = 1
TEST = os.environ.get("GPUQ_TEST") == "1"
HOME = os.path.expanduser(os.environ.get("GPUQ_HOME", "~/.gpuq"))
LOCK = os.path.expanduser(os.environ.get("GPUQ_LOCK", "~/gpu.lock"))
MARKER = os.path.expanduser(os.environ.get("GPUQ_MARKER", "~/BATTERY_RUNNING"))
TICK = float(os.environ.get("GPUQ_TICK", "0.5")) if TEST else 0.5
KILL_AFTER = float(os.environ.get("GPUQ_KILL_AFTER", "30")) if TEST else 30.0
DROP = os.environ.get("GPUQ_DROP_CACHES", "1") == "1"
DROP_CMD = os.environ.get("GPUQ_DROP_CACHES_CMD", "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
HOST = socket.gethostname().split(".")[0]
# Resources this daemon owns: each has its own lock and memory pool.  v1 = one GPU per host; a v2 coordinator
# reads several hosts' spools and gang-schedules jobs naming resources on more than one host.
RESOURCES = {f"{HOST}:gpu0": {"lock": LOCK, "meminfo": os.environ.get("GPUQ_MEMINFO", "/proc/meminfo")
                              if TEST else "/proc/meminfo"}}
MARGIN = 6 << 30
PRIO = {"quiet": 0, "short": 1, "long": 2}
DEF_MEM = {"quiet": 16 << 30, "short": 8 << 30, "long": 16 << 30}
DEF_TMO = {"quiet": 7200, "short": 1200, "long": 86400}
SHORT_MAX, DEF_MIN_HOLD, DEF_GRACE = 1200, 300, 1200
EX_TEMPFAIL, RC_TIMEOUT, RC_CANCEL, RC_LOST = 75, 124, 125, 120
FINAL = ("done", "timeout", "cancelled", "lost")
JOBS, LOGS = os.path.join(HOME, "jobs"), os.path.join(HOME, "logs")

if TEST and ("GPUQ_HOME" not in os.environ or LOCK == os.path.expanduser("~/gpu.lock")
             or MARKER == os.path.expanduser("~/BATTERY_RUNNING")):
    sys.exit("gpuq: GPUQ_TEST=1 needs GPUQ_HOME, GPUQ_LOCK and GPUQ_MARKER away from the real ones")


def die(msg, rc=2):
    print(f"gpuq: {msg}", file=sys.stderr)
    sys.exit(rc)


def write_json(path, obj):   # atomic: a reader sees the old file or the new one, never half of one
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def jp(i, ext="json"):
    return os.path.join(JOBS, f"{i}.{ext}")


def ensure_dirs():   # the spool holds every submitter's environment: owner-only
    for d in (HOME, JOBS, LOGS):
        os.makedirs(d, mode=0o700, exist_ok=True)


def daemon_up(st):
    return bool(st) and alive(st.get("daemon_pid", 0)) and time.time() - st.get("heartbeat", 0) < 15


def parse_dur(s):
    s = s.strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        return float(s[:-1]) * mult[s[-1]] if s[-1] in mult else float(s)
    except (ValueError, IndexError):
        die(f"bad duration {s!r} (use 90, 90s, 20m, 2h)")


def parse_size(s):
    s = s.strip().upper().rstrip("B")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    try:
        return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)
    except (ValueError, IndexError):
        die(f"bad size {s!r} (use 8G, 512M)")


def fdur(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


def fsize(b):
    return f"{b / (1 << 30):.1f}G"


def fclock(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def mem_available(res):
    with open(RESOURCES[res]["meminfo"]) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def all_jobs():
    out = []
    for n in os.listdir(JOBS) if os.path.isdir(JOBS) else []:
        if n.endswith(".json") and n[:-5].isdigit():
            j = read_json(os.path.join(JOBS, n))
            if j:
                out.append(j)
    return sorted(out, key=lambda j: j["id"])


def order(queued):
    return sorted(queued, key=lambda j: (PRIO[j["cls"]], j["id"]))


def alive(pid, must=None):   # live (not zombie) and, when given, its cmdline contains every word of `must`
    try:
        with open(f"/proc/{pid}/stat") as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        if must:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().split(b"\0")
            return all(w.encode() in args for w in must)
        return True
    except (OSError, IndexError):
        return False


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:150]
    except OSError:
        return "(gone)"


def tagged(token):   # every live process carrying this attempt's token, daemonized ones included
    want = f"GPUQ_TOKEN={token}".encode()
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                with open(f"/proc/{p}/environ", "rb") as f:
                    if want in f.read().split(b"\0"):
                        yield int(p)
            except OSError:
                pass


def reap(token, pgid, say):   # kill whatever is left of a job: its process group and every tagged process
    left = list(tagged(token))
    if not left:
        return
    say(f"the job left {len(left)} process(es) running -- terminating: "
        + "; ".join(f"{p} {cmdline(p)}" for p in left))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid and pgid > 1:   # never 0: that is our own group
                os.killpg(pgid, sig)
        except OSError:
            pass
        for p in left:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        for _ in range(20):
            left = list(tagged(token))
            if not left:
                return
            time.sleep(0.5)
        say(f"still alive after 10 s, SIGKILL: {left}")


def lock_info(path):   # (holders, waiters) of a flock from /proc/locks, like gpu-lock-status.sh did
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return [], []
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    holders, waiters = [], []
    with open("/proc/locks") as f:
        for line in f:
            t = line.split()
            if len(t) > 6 and t[1] == "->" and t[2] == "FLOCK" and t[6] == key:
                waiters.append(int(t[5]))
            elif len(t) > 5 and t[1] == "FLOCK" and t[5] == key:
                holders.append(int(t[4]))
    return holders, waiters


# ---------------------------------------------------------------------------------------------- the shim
def shim(jid, fd):
    """Runs one attempt.  Holds the resource lock (fd, inherited from the daemon; the job gets no lock fd),
    runs the job in its own process group, enforces the timeout, reaps leftovers, writes ID.out."""
    j = read_json(jp(jid))
    att, token = j["attempt"], j["token"]
    log = open(os.path.join(LOGS, f"{jid}.log"), "ab", buffering=0)
    say = lambda m: log.write(f"gpuq[{fclock(time.time())}]: {m}\n".encode())
    say(f"job {jid} attempt {att} ({j['cls']}, timeout {fdur(j['timeout'])}) in {j['cwd']}: {j['cmd']}")
    env = dict(j["env"], GPUQ_JOB_ID=str(jid), GPUQ_CLASS=j["cls"], GPUQ_ATTEMPT=str(att), GPUQ_TOKEN=token,
               GPUQ_RESOURCES=",".join(j["resources"]))
    out = {"version": VERSION, "attempt": att, "timed_out": False}
    try:
        p = subprocess.Popen(j["cmd"], cwd=j["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, process_group=0, close_fds=True)
    except OSError as e:
        say(f"cannot start: {e}")
        out["rc"] = 127 if isinstance(e, FileNotFoundError) else 126
        write_json(jp(jid, "out"), out)
        return
    write_json(jp(jid, "run"), {"version": VERSION, "shim_pid": os.getpid(), "pgid": p.pid, "attempt": att})

    def fwd(sig, _frm):   # the shim itself told to stop: pass it on to the job
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, fwd)
    try:
        rc = p.wait(timeout=j["timeout"])
    except subprocess.TimeoutExpired:
        out["timed_out"] = True
        say(f"timeout after {fdur(j['timeout'])}: SIGTERM to process group {p.pid}")
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=KILL_AFTER)
        except subprocess.TimeoutExpired:
            say(f"still running {KILL_AFTER:.0f} s after SIGTERM: SIGKILL")
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        except OSError:
            p.wait()
        rc = RC_TIMEOUT
    reap(token, p.pid, say)
    out["rc"] = rc if rc >= 0 else 128 - rc
    say(f"job {jid} attempt {att} exit {out['rc']}")
    write_json(jp(jid, "out"), out)


# --------------------------------------------------------------------------------------------- the daemon
class LockWaiter:   # a blocking flock in a thread: the daemon joins the kernel's queue like any legacy waiter
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        self.got = threading.Event()
        threading.Thread(target=self._take, daemon=True).start()

    def _take(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.got.set()


class Daemon:
    def __init__(self):
        self.procs, self.waiters, self.said, self.dropped = {}, {}, {}, {}
        self.stop = False
        ensure_dirs()
        self.logf = open(os.path.join(HOME, "gpuq.log"), "a", buffering=1)

    def log(self, msg):
        line = f"{time.strftime('%F %T')} {msg}"
        self.logf.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    def note(self, key, msg):   # log a waiting reason once, not every tick
        if self.said.get(key) != msg:
            self.said[key] = msg
            self.log(msg)

    def save(self, j):
        write_json(jp(j["id"]), j)

    def shim_alive(self, j):
        p = self.procs.get(j.get("shim_pid"))
        if p is not None and p.poll() is not None:
            return False
        return bool(j.get("shim_pid")) and alive(j["shim_pid"], ["_shim", str(j["id"])])

    def pgid(self, j):
        return (read_json(jp(j["id"], "run")) or {}).get("pgid")

    def signal_job(self, j, sig):
        pg = self.pgid(j)
        if pg and pg > 1:
            try:
                os.killpg(pg, sig)
            except OSError:
                pass

    def recover(self):
        for j in all_jobs():
            if j["state"] != "running":
                continue
            run = read_json(jp(j["id"], "run"))
            if not j.get("shim_pid") and run:
                j["shim_pid"] = run["shim_pid"]
                self.save(j)
            if os.path.exists(jp(j["id"], "out")):
                self.log(f"recover: job {j['id']} finished while the daemon was down")
            elif self.shim_alive(j):
                self.log(f"recover: re-adopted running job {j['id']} (shim pid {j['shim_pid']})")
            else:
                self.lost(j, "recover: its shim is gone")

    def lost(self, j, why):
        self.log(f"job {j['id']} lost ({why}); killing what is left of it")
        reap(j["token"], self.pgid(j) or 0, self.log)
        self.finalize(j, "lost", RC_LOST)

    def finalize(self, j, state, rc):
        j.update(state=state, rc=rc, ended=time.time())
        self.save(j)
        if j["cls"] == "quiet" and j.get("started"):
            self.clear_marker(j)
        for ext in ("run", "out", "cancel"):
            try:
                os.unlink(jp(j["id"], ext))
            except FileNotFoundError:
                pass
        self.log(f"job {j['id']} {state} rc={rc} ({j['label']})")

    def clear_marker(self, j):
        try:
            with open(MARKER) as f:
                mine = f.read().startswith(f"gpuq job {j['id']} ")
            if mine:
                os.unlink(MARKER)
                self.log(f"removed {MARKER}")
        except FileNotFoundError:
            pass

    def finish(self, j, out):
        now = time.time()
        j.setdefault("attempts", []).append({"attempt": out["attempt"], "started": j["started"], "ended": now,
                                             "rc": out["rc"], "timed_out": out["timed_out"],
                                             "yield_requested_at": j.get("yield_requested_at")})
        if j.get("cancel_sent"):
            return self.finalize(j, "cancelled", RC_CANCEL)
        if out["timed_out"]:
            return self.finalize(j, "timeout", RC_TIMEOUT)
        if out["rc"] == EX_TEMPFAIL and j["cls"] == "long":
            for k in ("run", "out"):
                try:
                    os.unlink(jp(j["id"], k))
                except FileNotFoundError:
                    pass
            j.update(state="queued", attempt=j["attempt"] + 1, shim_pid=None, started=None,
                     yield_requested_at=None, over_grace_logged=False)
            self.save(j)
            return self.log(f"job {j['id']} yielded (exit 75) after {fdur(now - j['attempts'][-1]['started'])};"
                            f" re-queued as attempt {j['attempt']}")
        self.finalize(j, "done", out["rc"])

    def cancels(self, jobs):
        for j in jobs:
            if not os.path.exists(jp(j["id"], "cancel")):
                continue
            if j["state"] == "queued":
                self.finalize(j, "cancelled", RC_CANCEL)
            elif j["state"] == "running" and not j.get("cancel_sent") and self.pgid(j):
                j["cancel_sent"] = time.time()
                self.save(j)
                self.log(f"cancel job {j['id']}: SIGTERM to its process group")
                self.signal_job(j, signal.SIGTERM)
            elif j["state"] == "running" and j.get("cancel_sent") and not j.get("cancel_killed") and \
                    time.time() - j["cancel_sent"] > KILL_AFTER:
                j["cancel_killed"] = True
                self.save(j)
                self.log(f"cancel job {j['id']}: still running {KILL_AFTER:.0f} s after SIGTERM, SIGKILL")
                self.signal_job(j, signal.SIGKILL)
            elif j["state"] in FINAL:
                os.unlink(jp(j["id"], "cancel"))

    def watch(self, j, queued):
        out = read_json(jp(j["id"], "out"))
        if out is None and not self.shim_alive(j):
            out = read_json(jp(j["id"], "out"))   # it may have written it on the way out
            if out is None:
                return self.lost(j, f"shim pid {j.get('shim_pid')} died without an outcome")
        if out is not None:
            self.procs.pop(j.get("shim_pid"), None)
            return self.finish(j, out)
        now = time.time()
        if j["cls"] == "long" and not j.get("yield_requested_at"):
            urgent = [q for q in queued if q["cls"] != "long"]
            if urgent and now - j["started"] >= j["min_hold"] and self.pgid(j):   # pgid: the shim has started it
                j["yield_requested_at"] = now
                self.save(j)
                self.log(f"job {j['id']}: SIGUSR1 (yield) -- job {urgent[0]['id']} ({urgent[0]['cls']}) waits")
                self.signal_job(j, signal.SIGUSR1)
        yra = j.get("yield_requested_at")
        if yra and not j.get("over_grace_logged") and now - yra > j["yield_grace"]:
            j["over_grace_logged"] = True
            self.save(j)
            self.log(f"job {j['id']}: not yielded {fdur(now - yra)} after SIGUSR1 (grace "
                     f"{fdur(j['yield_grace'])}) -- still waiting, never killed for a yield")

    def schedule(self, res, queued):
        w = self.waiters.get(res)
        if not queued:
            if w and w.got.is_set():
                os.close(w.fd)
                del self.waiters[res]
            return self.note(res, f"{res}: idle")
        head = order(queued)[0]
        if os.path.exists(MARKER):   # ours is cleared when its job ends: this one is a legacy battery's
            return self.note(res, f"{res}: job {head['id']} waits: {MARKER} present (a legacy battery)")
        if w is None:
            w = self.waiters[res] = LockWaiter(RESOURCES[res]["lock"])
        if not w.got.is_set():
            h, _ = lock_info(RESOURCES[res]["lock"])
            return self.note(res, f"{res}: job {head['id']} waits for the lock; legacy holder "
                             + ", ".join(f"{p} {cmdline(p)}" for p in h if p != os.getpid()))
        key = (head["id"], head["attempt"])
        if DROP and self.dropped.get(res) != key:   # once per admission, with the lock held: nothing runs
            self.dropped[res] = key
            r = subprocess.run(["sh", "-c", DROP_CMD], capture_output=True, text=True)
            self.log(f"{res}: drop_caches before job {head['id']}: rc={r.returncode} {r.stderr.strip()[:200]}")
        avail, need = mem_available(res), head["mem"] + MARGIN
        if avail < need:
            return self.note(res, f"{res}: job {head['id']} waits for memory: MemAvailable {fsize(avail)}"
                             f" < {fsize(head['mem'])} + 6G margin (holding the lock)")
        del self.waiters[res]
        self.start(head, w.fd, avail)

    def start(self, j, fd, avail):
        if j["cls"] == "quiet":
            with open(MARKER, "w") as f:
                f.write(f"gpuq job {j['id']} pid {os.getpid()} {time.strftime('%F %T')}\n")
            self.log(f"created {MARKER}")
        j.update(state="running", started=time.time(), shim_pid=None, yield_requested_at=None,
                 token=f"gpuq-{j['id']}-{j['attempt']}-{secrets.token_hex(4)}")
        self.save(j)
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_shim", str(j["id"]), str(fd)],
                             pass_fds=(fd,), start_new_session=True, stdin=subprocess.DEVNULL)
        os.close(fd)
        self.procs[p.pid] = p
        j["shim_pid"] = p.pid
        self.save(j)
        self.said.clear()
        self.log(f"start job {j['id']} attempt {j['attempt']} ({j['cls']}, {j['label']}) on "
                 f"{','.join(j['resources'])}: MemAvailable {fsize(avail)}, shim pid {p.pid}")

    def run(self):
        me = open(os.path.join(HOME, "daemon.lock"), "a")
        try:
            fcntl.flock(me, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            die("another gpuq daemon is running", 1)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, lambda *_: setattr(self, "stop", True))
        self.log(f"daemon start pid {os.getpid()} resources {list(RESOURCES)} drop_caches={DROP}")
        self.recover()
        while not self.stop:
            try:
                self.tick()
            except Exception as e:   # one bad spool file must not take the queue down
                self.note("error", f"tick error: {e!r}")
            time.sleep(TICK)
        self.log("daemon stop (running jobs keep running; a restart re-adopts them)")

    def tick(self):
        jobs = all_jobs()
        self.cancels(jobs)
        jobs = all_jobs()
        for res in RESOURCES:
            mine = [j for j in jobs if j["resources"][0] == res]
            queued = [j for j in mine if j["state"] == "queued"]
            running = [j for j in mine if j["state"] == "running"]
            if running:
                self.watch(running[0], queued)
            else:
                self.schedule(res, queued)
        write_json(os.path.join(HOME, "state.json"), {"version": VERSION, "daemon_pid": os.getpid(),
                   "heartbeat": time.time(), "host": HOST, "resources": list(RESOURCES),
                   "waiting": {r: self.said.get(r) for r in RESOURCES}})


# -------------------------------------------------------------------------------------------------- CLI
def cmd_submit(a):
    if not a.cmd:
        die("submit needs a command after --")
    cmd = a.cmd[1:] if a.cmd[0] == "--" else a.cmd
    tmo = parse_dur(a.timeout) if a.timeout else DEF_TMO[a.cls]
    if a.cls == "short" and tmo > SHORT_MAX:
        die(f"a short job's timeout is at most 20m (got {fdur(tmo)}); use --class long")
    res = [r if ":" in r else f"{HOST}:{r}" for r in (a.gpus.split(",") if a.gpus else ["gpu0"])]
    foreign = [r for r in res if r not in RESOURCES]
    if foreign:
        die(f"not a resource of this daemon: {', '.join(foreign)} (this host has {', '.join(RESOURCES)})")
    ensure_dirs()
    with open(os.path.join(HOME, "seq"), "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        jid = int(f.read().strip() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(f"{jid}\n")
        f.flush()
        os.fsync(f.fileno())
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUQ_") or k in
           ("GPUQ_HOME", "GPUQ_LOCK", "GPUQ_MARKER", "GPUQ_TEST")}
    j = {"version": VERSION, "id": jid, "cls": a.cls, "cmd": cmd, "cwd": os.path.abspath(a.cwd or os.getcwd()),
         "env": env, "label": a.label or " ".join(cmd)[:80], "resources": sorted(set(res)), "timeout": tmo,
         "mem": parse_size(a.mem) if a.mem else DEF_MEM[a.cls],
         "min_hold": parse_dur(a.min_hold) if a.min_hold else DEF_MIN_HOLD,
         "yield_grace": parse_dur(a.yield_grace) if a.yield_grace else DEF_GRACE,
         "state": "queued", "attempt": 1, "submitted": time.time(), "submitter_pid": os.getppid(), "attempts": []}
    write_json(jp(jid), j)
    if not daemon_up(read_json(os.path.join(HOME, "state.json"))):
        print(f"gpuq: warning: the daemon is not running; job {jid} waits until it is "
              "(systemctl --user start gpuq)", file=sys.stderr)
    if not a.wait:
        print(jid)
        return 0
    print(f"gpuq: job {jid} queued ({a.cls}); waiting", file=sys.stderr)
    return wait_job(jid)


def wait_job(jid):
    """Follow the job's log to stdout until it is final; return its exit code.  INT/TERM/HUP cancel it."""
    hits = []

    def on_sig(sig, _frm):
        hits.append(sig)
        open(jp(jid, "cancel"), "a").close()
        if len(hits) > 1:
            sys.exit(128 + sig)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_sig)
    pos, path = 0, os.path.join(LOGS, f"{jid}.log")

    def pump():
        nonlocal pos
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read()
        except FileNotFoundError:
            return
        pos += len(data)
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    while True:
        j = read_json(jp(jid))
        if j is None:
            die(f"job {jid} vanished", RC_LOST)
        if j["state"] in FINAL:
            pump()
            return j["rc"]
        pump()
        time.sleep(0.2)


def snapshot():
    now, jobs = time.time(), all_jobs()
    st = read_json(os.path.join(HOME, "state.json")) or {}
    up = daemon_up(st)
    s = {"version": VERSION, "time": now, "host": HOST,
         "daemon": {"pid": st.get("daemon_pid"), "up": up, "waiting": st.get("waiting", {})},
         "battery_marker": None, "resources": {}, "anomalies": []}
    if os.path.exists(MARKER):
        with open(MARKER) as f:
            s["battery_marker"] = {"path": MARKER, "content": f.read().strip()}
    gpu = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10) if not TEST else None
        gpu = [l.split(",") for l in r.stdout.splitlines() if l.strip()] if r and r.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        pass
    for res, cfg in RESOURCES.items():
        mine = [j for j in jobs if j["resources"][0] == res]
        run = next((j for j in mine if j["state"] == "running"), None)
        holders, waiters = lock_info(cfg["lock"])
        dpid = st.get("daemon_pid")
        legacy = [] if run else [{"pid": p, "cmd": cmdline(p), "alive": alive(p)} for p in holders if p != dpid]
        for h in legacy:
            if not h["alive"]:
                s["anomalies"].append(f"ORPHANED LOCK: holder {h['pid']} is gone, an inherited fd holds {cfg['lock']}")
        r = {"lock": cfg["lock"], "mem_available": mem_available(res), "legacy_holders": legacy,
             "legacy_waiters": [{"pid": p, "cmd": cmdline(p)} for p in waiters if p != dpid],
             "running": None, "queue": [], "recent": []}
        t = now
        if run:
            el = now - (run.get("started") or now)
            r["running"] = {k: run.get(k) for k in ("id", "label", "cls", "attempt", "mem", "timeout", "resources",
                                                   "yield_requested_at", "yield_grace", "min_hold")}
            r["running"].update(pid=(read_json(jp(run["id"], "run")) or {}).get("pgid"), elapsed=el)
            t += max(0, run["timeout"] - el)
        for q in order([j for j in mine if j["state"] == "queued"]):
            r["queue"].append({"id": q["id"], "label": q["label"], "cls": q["cls"], "attempt": q["attempt"],
                               "mem": q["mem"], "timeout": q["timeout"], "resources": q["resources"],
                               "waited": now - q["submitted"], "eta_start": t})
            t += q["timeout"]
        done = sorted((j for j in mine if j["state"] in FINAL), key=lambda j: j.get("ended", 0))[-5:]
        r["recent"] = [{k: j.get(k) for k in ("id", "label", "cls", "state", "rc", "ended")} for j in done]
        s["resources"][res] = r
        tok = run and run.get("token")
        for g in gpu:
            pid = int(g[0])
            if tok and pid in tagged(tok):
                continue
            if any(h["alive"] and descends(pid, h["pid"]) for h in legacy):
                continue
            s["anomalies"].append(f"OUTSIDE LOCK: GPU process {pid} ({g[1].strip()}) {cmdline(pid)}")
    return s


def descends(pid, anc):
    for _ in range(64):
        if pid == anc:
            return True
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError):
            return False
        if pid <= 1:
            return False
    return False


def cmd_status(a):
    s = snapshot()
    if a.json:
        print(json.dumps(s, indent=1))
        return 1 if s["anomalies"] else 0
    now = s["time"]
    d = s["daemon"]
    print(f"gpuq on {s['host']}: daemon " + (f"pid {d['pid']} up" if d["up"] else "DOWN (jobs wait)"))
    m = s["battery_marker"]
    print("battery marker: " + (f"{m['path']} ({m['content']})" if m else "absent"))
    for res, r in s["resources"].items():
        print(f"{res}: MemAvailable {fsize(r['mem_available'])}   lock {r['lock']}")
        for h in r["legacy_holders"]:
            print(f"  LEGACY HOLDER  {h['pid']} {h['cmd'] if h['alive'] else '(gone)'}")
        if r["legacy_waiters"]:
            print(f"  legacy waiters on the lock: {len(r['legacy_waiters'])}")
            for w in r["legacy_waiters"]:
                print(f"    {w['pid']} {w['cmd']}")
        x = r["running"]
        if x:
            print(f"  RUNNING #{x['id']} {x['cls']} \"{x['label']}\" pid {x['pid']} attempt {x['attempt']} "
                  f"{fdur(x['elapsed'])} / {fdur(x['timeout'])} mem {fsize(x['mem'])}")
            if x["yield_requested_at"]:
                print(f"    yield requested at {fclock(x['yield_requested_at'])} "
                      f"({fdur(now - x['yield_requested_at'])} ago, grace {fdur(x['yield_grace'])})")
        w = d["waiting"].get(res)
        if not x and w and r["queue"]:
            print(f"  waiting: {w}")
        for i, q in enumerate(r["queue"], 1):
            print(f"  {i:2d}. #{q['id']} {q['cls']:5s} \"{q['label']}\" attempt {q['attempt']} waited "
                  f"{fdur(q['waited'])}, start <= {fclock(q['eta_start'])} (+{fdur(q['eta_start'] - now)}), "
                  f"timeout {fdur(q['timeout'])} mem {fsize(q['mem'])}")
        if not r["queue"]:
            print("  queue empty")
        for j in r["recent"]:
            print(f"  done #{j['id']} {j['cls']} {j['state']} rc={j['rc']} at {fclock(j['ended'] or 0)} "
                  f"\"{j['label']}\"")
    for an in s["anomalies"]:
        print(f"ANOMALY {an}")
    return 1 if s["anomalies"] else 0


def cmd_cancel(a):
    j = read_json(jp(a.id))
    if j is None:
        die(f"no job {a.id}", 1)
    if j["state"] in FINAL:
        die(f"job {a.id} is already {j['state']}", 1)
    open(jp(a.id, "cancel"), "a").close()
    print(f"gpuq: cancel requested for job {a.id} ({j['state']})")
    return 0


def cmd_log(a):
    if read_json(jp(a.id)) is None:
        die(f"no job {a.id}", 1)
    if a.follow:
        return wait_job(a.id)
    try:
        with open(os.path.join(LOGS, f"{a.id}.log"), "rb") as f:
            sys.stdout.buffer.write(f.read())
    except FileNotFoundError:
        print(f"gpuq: job {a.id} has no log yet", file=sys.stderr)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "_shim":
        return shim(int(sys.argv[2]), int(sys.argv[3]))
    ap = argparse.ArgumentParser(prog="gpuq", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--class", dest="cls", required=True, choices=list(PRIO))
    for o in ("timeout", "mem", "min-hold", "yield-grace", "label", "cwd", "gpus"):
        s.add_argument(f"--{o}")
    s.add_argument("--wait", action="store_true")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status").add_argument("--json", action="store_true")
    sub.add_parser("cancel").add_argument("id", type=int)
    lg = sub.add_parser("log")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("id", type=int)
    sub.add_parser("daemon")
    a = ap.parse_args()
    if a.op == "daemon":
        return Daemon().run()
    return {"submit": cmd_submit, "status": cmd_status, "cancel": cmd_cancel, "log": cmd_log}[a.op](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
