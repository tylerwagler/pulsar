#!/usr/bin/env python3
"""gpuq's tests: a real daemon per test in test mode (own spool, lock, marker, fake meminfo, fake
drop_caches), dummy sleep/python jobs.  Never touches the GPU, ~/gpu.lock or ~/BATTERY_RUNNING.

    GPUQ_TEST_ROOT=~/gpuq-test python3 tools/gpuq/test_gpuq.py -v
"""
import json, os, shutil, signal, subprocess, sys, time, unittest

GPUQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpuq.py")
ROOT = os.path.expanduser(os.environ.get("GPUQ_TEST_ROOT", "~/gpuq-test"))
PY = sys.executable

# a long job that checkpoints: SIGUSR1 -> exit 75 at the next 0.1 s step; records each attempt
LONG_JOB = r"""
import os, signal, sys, time
flag = []
signal.signal(signal.SIGUSR1, lambda *_: flag.append(1))
open(sys.argv[1], "a").write("long:%s\n" % os.environ["GPUQ_ATTEMPT"])
t0 = time.time()
while time.time() - t0 < float(sys.argv[2]):
    if flag and sys.argv[3] == "yield":
        sys.exit(75)
    time.sleep(0.1)
"""


def rd(path):
    with open(path) as f:
        return f.read()


def wr(path, text):
    with open(path, "w") as f:
        f.write(text)


def meminfo(path, gib):
    with open(path, "w") as f:
        f.write(f"MemTotal: {128 << 20} kB\nMemAvailable: {int(gib * (1 << 20))} kB\n")


class Q(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(ROOT, f"{self._testMethodName}-{os.getpid()}")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.lock, self.marker = os.path.join(self.dir, "gpu.lock"), os.path.join(self.dir, "BATTERY_RUNNING")
        self.mem, self.out, self.drops = (os.path.join(self.dir, n) for n in ("meminfo", "out", "drops"))
        meminfo(self.mem, 100)
        self.env = dict(os.environ, GPUQ_TEST="1", GPUQ_HOME=os.path.join(self.dir, "spool"),
                        GPUQ_LOCK=self.lock, GPUQ_MARKER=self.marker, GPUQ_MEMINFO=self.mem, GPUQ_TICK="0.1",
                        GPUQ_KILL_AFTER="2", GPUQ_DROP_CACHES_CMD=f"echo drop >> {self.drops}")
        self.daemon = None
        self.start_daemon()

    def tearDown(self):
        self.stop_daemon()
        for j in self.jobs():   # nothing a test started may outlive it (only this test's own pids)
            run = os.path.join(self.dir, "spool", "jobs", f"{j['id']}.run")
            pg = json.loads(rd(run))["pgid"] if os.path.exists(run) else None
            for kill, pid in ((os.killpg, pg), (os.kill, j.get("shim_pid") if j["state"] == "running" else None)):
                try:
                    pid and kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if os.environ.get("GPUQ_KEEP") != "1":
            shutil.rmtree(self.dir, ignore_errors=True)

    def start_daemon(self):
        self.dlog = open(os.path.join(self.dir, "daemon.err"), "a")
        self.daemon = subprocess.Popen([PY, GPUQ, "daemon"], env=self.env, stderr=self.dlog,
                                       start_new_session=True)
        time.sleep(0.3)

    def stop_daemon(self, sig=signal.SIGTERM):
        if self.daemon and self.daemon.poll() is None:
            self.daemon.send_signal(sig)
            self.daemon.wait(10)
        self.dlog.close()

    def q(self, *args, check=True):
        r = subprocess.run([PY, GPUQ, *args], env=self.env, capture_output=True, text=True, cwd=self.dir)
        if check and r.returncode not in (0, 1):
            raise AssertionError(f"gpuq {args}: rc={r.returncode} {r.stderr}")
        return r

    def submit(self, cls, *cmd, opts=()):
        return int(self.q("submit", "--class", cls, *opts, "--", *cmd).stdout.strip())

    def sh(self, cls, script, opts=()):
        return self.submit(cls, "sh", "-c", script, opts=opts)

    def job(self, i):
        with open(os.path.join(self.dir, "spool", "jobs", f"{i}.json")) as f:
            return json.load(f)

    def jobs(self):
        d = os.path.join(self.dir, "spool", "jobs")
        return [self.job(int(n[:-5])) for n in os.listdir(d) if n.endswith(".json") and n[:-5].isdigit()] \
            if os.path.isdir(d) else []

    def until(self, pred, timeout=20, what="condition"):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = pred()
            if v:
                return v
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}; daemon log:\n{self.dlogtext()}")

    def final(self, i, timeout=30):
        return self.until(lambda: self.job(i)["state"] in ("done", "timeout", "cancelled", "lost")
                          and self.job(i), timeout, f"job {i} final")

    def state(self, i):
        return self.job(i)["state"]

    def lines(self):
        return rd(self.out).split() if os.path.exists(self.out) else []

    def dlogtext(self):
        p = os.path.join(self.dir, "spool", "gpuq.log")
        return rd(p) if os.path.exists(p) else ""

    def status(self):
        return json.loads(self.q("status", "--json").stdout)["resources"]

    # ------------------------------------------------------------------------------------------ tests
    def test_fifo_short(self):
        blocker = self.sh("short", "sleep 1")
        ids = [self.sh("short", f"echo {n} >> {self.out}") for n in range(6)]
        for i in ids:
            self.final(i)
        self.assertEqual(self.lines(), [str(n) for n in range(6)])
        self.assertEqual(self.job(blocker)["rc"], 0)
        st = self.job(ids[0])
        self.assertLess(self.job(blocker)["ended"], st["attempts"][0]["started"] + 0.01)

    def test_short_timeout_cap(self):
        r = self.q("submit", "--class", "short", "--timeout", "21m", "--", "true", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("at most 20m", r.stderr)

    def test_quiet_exclusive_and_marker(self):
        a = self.sh("short", f"sleep 1; echo A >> {self.out}")
        self.until(lambda: self.state(a) == "running")
        b = self.sh("short", f"echo B >> {self.out}")
        quiet = self.sh("quiet", f"test -e {self.marker} && echo Q-marker >> {self.out}; sleep 1;"
                                 f" echo Q-end >> {self.out}")
        self.assertFalse(os.path.exists(self.marker))   # A still runs: quiet waits for it
        self.until(lambda: self.state(quiet) == "running")
        self.assertTrue(os.path.exists(self.marker))
        c = self.sh("short", f"echo C >> {self.out}")   # submitted while quiet runs: not admitted
        time.sleep(0.4)
        self.assertEqual(self.state(c), "queued")
        self.assertEqual(self.state(b), "queued")
        for i in (b, c):
            self.final(i)
        self.assertEqual(self.lines(), ["A", "Q-marker", "Q-end", "B", "C"])
        self.assertFalse(os.path.exists(self.marker))
        self.assertEqual(self.job(quiet)["rc"], 0)

    def test_foreign_marker_blocks(self):
        wr(self.marker, "legacy battery\n")
        a = self.sh("short", "true")
        time.sleep(0.5)
        self.assertEqual(self.state(a), "queued")
        os.unlink(self.marker)
        self.assertEqual(self.final(a)["rc"], 0)

    def test_long_yield_and_resume(self):
        lg = self.submit("long", PY, "-c", LONG_JOB, self.out, "3", "yield", opts=("--min-hold", "0.3"))
        self.until(lambda: os.path.exists(self.out))
        s1 = self.sh("short", f"echo s1 >> {self.out}")
        s2 = self.sh("short", f"echo s2 >> {self.out}")
        self.final(s2)
        j = self.final(lg)
        self.assertEqual(self.lines(), ["long:1", "s1", "s2", "long:2"])
        self.assertEqual(j["state"], "done")
        self.assertEqual(j["rc"], 0)
        self.assertEqual([a["rc"] for a in j["attempts"]], [75, 0])
        self.assertIsNotNone(j["attempts"][0]["yield_requested_at"])
        self.assertIn("SIGUSR1 (yield)", self.dlogtext())
        self.assertIn("re-queued as attempt 2", self.dlogtext())

    def test_min_hold_respected(self):
        lg = self.submit("long", PY, "-c", LONG_JOB, self.out, "5", "yield", opts=("--min-hold", "1.5"))
        self.until(lambda: os.path.exists(self.out))
        t0 = time.time()
        s = self.sh("short", "true")
        self.until(lambda: self.job(lg).get("yield_requested_at"), what="yield request")
        self.assertGreaterEqual(self.job(lg)["yield_requested_at"] - self.job(lg)["started"], 1.4)
        self.final(s)
        self.assertGreater(time.time() - t0, 0.5)
        self.final(lg)

    def test_yield_grace_logged_never_killed(self):
        # ignores SIGUSR1 (no checkpoint for 3 s); grace 1 s: logged once, job left to finish
        lg = self.submit("long", PY, "-c", LONG_JOB, self.out, "3", "ignore",
                         opts=("--min-hold", "0", "--yield-grace", "1"))
        self.until(lambda: os.path.exists(self.out))
        s = self.sh("short", f"echo s >> {self.out}")
        self.until(lambda: "never killed for a yield" in self.dlogtext(), what="over-grace log")
        self.assertEqual(self.state(lg), "running")
        j = self.final(lg)
        self.assertEqual((j["state"], j["rc"]), ("done", 0))
        self.final(s)
        self.assertEqual(self.lines(), ["long:1", "s"])
        self.assertEqual(self.dlogtext().count("never killed for a yield"), 1)

    def test_yield_grace_90m_and_status_shows_pending_yield(self):
        lg = self.submit("long", PY, "-c", LONG_JOB, self.out, "4", "ignore",
                         opts=("--min-hold", "0", "--yield-grace", "90m"))
        self.assertEqual(self.job(lg)["yield_grace"], 5400)
        self.until(lambda: os.path.exists(self.out))
        self.sh("short", "true")
        self.until(lambda: self.job(lg).get("yield_requested_at"), what="yield request")
        run = self.status()[next(iter(self.status()))]["running"]
        self.assertEqual(run["id"], lg)
        self.assertIsNotNone(run["yield_requested_at"])
        text = self.q("status").stdout
        self.assertIn("yield requested at", text)
        self.assertIn("grace 1:30:00", text)
        time.sleep(1.5)   # well past the SIGUSR1 and not yielded: still running, nothing logged as over grace
        self.assertEqual(self.state(lg), "running")
        self.assertNotIn("never killed for a yield", self.dlogtext())
        self.assertEqual(self.final(lg)["rc"], 0)
        self.assertIsNone(self.status()[next(iter(self.status()))]["running"])

    def test_timeout_kills_process_group_and_leftovers(self):
        pids = os.path.join(self.dir, "pids")
        i = self.sh("short", f"sleep 100 & echo $! >> {pids}; setsid sleep 100 & echo $! >> {pids}; sleep 100",
                    opts=("--timeout", "1"))
        j = self.final(i)
        self.assertEqual((j["state"], j["rc"]), ("timeout", 124))
        for p in rd(pids).split():
            self.assertFalse(os.path.exists(f"/proc/{p}") and
                             rd(f"/proc/{p}/stat").split(")")[1].split()[0] != "Z", f"pid {p} alive")
        log = self.q("log", str(i)).stdout
        self.assertIn("timeout after", log)

    def test_timeout_escalates_to_kill(self):
        i = self.sh("short", "trap '' TERM; sleep 100", opts=("--timeout", "1"))
        j = self.final(i, 15)
        self.assertEqual(j["rc"], 124)
        self.assertIn("SIGKILL", self.q("log", str(i)).stdout)

    def test_leftover_reaped_on_normal_exit(self):
        pids = os.path.join(self.dir, "pids")
        i = self.sh("short", f"setsid sleep 100 & echo $! > {pids}; exit 0")
        self.assertEqual(self.final(i)["rc"], 0)
        p = rd(pids).strip()
        self.assertFalse(os.path.exists(f"/proc/{p}"), "daemonized child survived")
        self.assertIn("left 1 process", self.q("log", str(i)).stdout)

    def test_memory_gate_and_drop_caches(self):
        meminfo(self.mem, 10)   # 8G + 6G margin needed
        i = self.sh("short", "true", opts=("--mem", "8G"))
        self.until(lambda: "waits for memory" in self.dlogtext(), what="memory wait")
        time.sleep(0.5)
        self.assertEqual(self.state(i), "queued")
        meminfo(self.mem, 14.5)
        self.final(i)
        self.assertEqual(rd(self.drops).split(), ["drop"])   # once per admission, not per tick
        k = self.sh("short", "true")
        self.final(k)
        self.assertEqual(len(rd(self.drops).split()), 2)
        self.assertIn("drop_caches before job", self.dlogtext())

    def test_legacy_holder(self):
        holder = subprocess.Popen(["flock", "-o", self.lock, "sleep", "2"])
        time.sleep(0.2)
        i = self.sh("short", f"flock -n {self.lock} true; echo $? >> {self.out}")
        time.sleep(0.5)
        self.assertEqual(self.state(i), "queued")
        res = self.status()[next(iter(self.status()))]
        self.assertEqual([h["pid"] for h in res["legacy_holders"]], [holder.pid])
        self.assertIn("LEGACY HOLDER", self.q("status").stdout)
        holder.wait()
        self.final(i)
        self.assertEqual(self.lines(), ["1"])   # the lock was held (by the shim) while the job ran
        # and a legacy flock waits while a gpuq job runs
        k = self.sh("short", "sleep 1")
        self.until(lambda: self.state(k) == "running")
        r = subprocess.run(["flock", "-n", self.lock, "true"])
        self.assertEqual(r.returncode, 1)
        self.final(k)
        self.assertEqual(subprocess.run(["flock", "-n", self.lock, "true"]).returncode, 0)

    def test_daemon_restart_readopts(self):
        i = self.sh("short", f"sleep 2; echo done >> {self.out}; exit 7")
        self.until(lambda: self.state(i) == "running")
        self.stop_daemon()
        time.sleep(0.5)
        self.assertEqual(self.state(i), "running")
        self.start_daemon()
        self.until(lambda: "re-adopted running job" in self.dlogtext(), what="re-adopt")
        j = self.final(i)
        self.assertEqual((j["state"], j["rc"]), ("done", 7))
        self.assertEqual(self.lines(), ["done"])
        k = self.sh("short", "true")   # the queue still works
        self.assertEqual(self.final(k)["rc"], 0)

    def test_daemon_restart_finished_while_down(self):
        i = self.sh("short", "sleep 0.5; exit 3")
        self.until(lambda: self.state(i) == "running")
        self.stop_daemon(signal.SIGKILL)
        time.sleep(1.5)
        self.start_daemon()
        j = self.final(i)
        self.assertEqual((j["state"], j["rc"]), ("done", 3))

    def test_daemon_restart_lost(self):
        pids = os.path.join(self.dir, "pids")
        i = self.sh("short", f"sleep 100 & echo $! > {pids}; wait")
        self.until(lambda: self.state(i) == "running" and self.job(i).get("shim_pid") and os.path.exists(pids))
        time.sleep(0.3)
        self.stop_daemon()
        os.kill(self.job(i)["shim_pid"], signal.SIGKILL)
        time.sleep(0.3)
        self.start_daemon()
        j = self.final(i)
        self.assertEqual((j["state"], j["rc"]), ("lost", 120))
        self.assertFalse(os.path.exists(f"/proc/{rd(pids).strip()}"), "lost job's process survived")

    def test_wait_exit_codes(self):
        def run(*cmd, opts=()):
            return subprocess.run([PY, GPUQ, "submit", "--class", "short", *opts, "--wait", "--", *cmd],
                                  env=self.env, capture_output=True, text=True)
        r = run("sh", "-c", "echo hello; exit 3")
        self.assertEqual(r.returncode, 3)
        self.assertIn("hello", r.stdout)
        self.assertEqual(run("true").returncode, 0)
        self.assertEqual(run("sleep", "10", opts=("--timeout", "0.5")).returncode, 124)
        self.assertEqual(run("/nonexistent/cmd").returncode, 127)
        self.assertEqual(run("sh", "-c", "kill -9 $$").returncode, 137)
        # cancel from elsewhere -> 125; and the waiter's own SIGTERM cancels its job
        w = subprocess.Popen([PY, GPUQ, "submit", "--class", "short", "--wait", "--", "sleep", "30"],
                             env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        jid = self.until(lambda: [j["id"] for j in self.jobs() if j["cmd"] == ["sleep", "30"]
                                  and j["state"] == "running"], what="waiter's job")[0]
        self.q("cancel", str(jid))
        self.assertEqual(w.wait(15), 125)
        w = subprocess.Popen([PY, GPUQ, "submit", "--class", "short", "--wait", "--", "sleep", "31"],
                             env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.until(lambda: [j for j in self.jobs() if j["cmd"] == ["sleep", "31"] and j["state"] == "running"])
        w.send_signal(signal.SIGTERM)
        self.assertEqual(w.wait(15), 125)

    def test_cancel(self):
        a = self.sh("short", "sleep 100")
        b = self.sh("short", f"echo B >> {self.out}")
        self.until(lambda: self.state(a) == "running")
        self.q("cancel", str(b))
        self.assertEqual(self.final(b)["state"], "cancelled")
        self.q("cancel", str(a))
        j = self.final(a)
        self.assertEqual((j["state"], j["rc"]), ("cancelled", 125))
        self.assertEqual(self.lines(), [])
        self.assertEqual(self.q("cancel", str(a), check=False).returncode, 1)

    def test_env_and_resources(self):
        i = self.sh("short", f"echo $GPUQ_JOB_ID $GPUQ_CLASS $GPUQ_ATTEMPT >> {self.out};"
                             f" ls -l /proc/$$/fd >> {os.path.join(self.dir, 'fds')}")
        self.final(i)
        self.assertEqual(self.lines(), [str(i), "short", "1"])
        self.assertNotIn("gpu.lock", rd(os.path.join(self.dir, "fds")))   # no lock fd for the job
        j = self.job(i)
        self.assertEqual(j["version"], 1)
        self.assertEqual(len(j["resources"]), 1)
        self.assertTrue(j["resources"][0].endswith(":gpu0"))
        r = self.q("submit", "--class", "short", "--gpus", "otherhost:gpu0", "--", "true", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("otherhost:gpu0", r.stderr)
        self.assertEqual(self.final(self.sh("short", "true", opts=("--gpus", "gpu0")))["rc"], 0)


if __name__ == "__main__":
    unittest.main()
