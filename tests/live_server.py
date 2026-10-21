"""live_server.py -- L317: the one pulsar-server instance of the live tier (tests/live_legs.py and every live leg).

A Server is a pulsar-server process on 127.0.0.1:PORT with its own disk-KV dir under the run's work dir (never /tmp:
on the Spark /tmp is the memory the engine loads into), started after a sync + drop_caches (the page cache masks
unified-memory pressure on GB10), waited for on /health, stopped with TERM then KILL.  It replaces the six copies the
live scripts each carried (rule 4: one authority).  What a leg reads of it: the URL, the log (whole, or what arrived
after a mark), /metrics.

    srv = Server(binary, model, work, "qwen-warm", port=8320, ctx=65536, banks=8, env={...})
    srv.post_json("/v1/chat/completions", body) -> (status, json)
    m = srv.mark(); ...; srv.since(m) -> the log text after the mark
    srv.stop()
"""
import json, os, shutil, subprocess, sys, time, urllib.error, urllib.request


def drop_caches():
    subprocess.run(["sync"], check=False)
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class StartError(RuntimeError):
    pass


class Server:
    def __init__(self, binary, model, work, name, port, ctx, banks=None, env=None, extra=(), start_timeout=900.0,
                 kv_disk=True):
        self.model, self.name, self.port, self.ctx, self.banks = model, name, port, ctx, banks
        self.env_extra = dict(env or {})
        self.url = "http://127.0.0.1:%d" % port
        self.dir = os.path.join(work, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        env = dict(os.environ)
        env.pop("PULSAR_MSEQ_BANKS", None)
        if banks:
            env["PULSAR_MSEQ_BANKS"] = str(banks)
        env.update(self.env_extra)
        cmd = [binary, "-m", model, "--ctx", str(ctx), "--host", "127.0.0.1", "--port", str(port)]
        cmd += ["--kv-disk-dir", os.path.join(self.dir, "kv")] if kv_disk else ["--no-kv-disk"]
        cmd += list(extra)
        drop_caches()
        t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        while True:
            if self.p.poll() is not None:
                raise StartError("%s: the server exited during startup (rc %s)\n%s" % (name, self.p.returncode,
                                                                                       self.tail()))
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2).read()
                break
            except Exception:
                pass
            if time.time() - t0 > start_timeout:
                self.stop()
                raise StartError("%s: the server did not listen within %.0f s" % (name, start_timeout))
            time.sleep(2)
        self.load_s = time.time() - t0
        # rule 5: the instance says what it is -- every leg's log names the instance it ran on
        print("# server %s up in %.0f s: %s, port %d, ctx %d, banks %s, disk KV %s%s (pid %d)" % (
            name, self.load_s, model, port, ctx, banks or "auto", "on" if kv_disk else "off",
            ", env " + " ".join("%s=%s" % kv for kv in sorted(self.env_extra.items())) if self.env_extra else "",
            self.p.pid), flush=True)

    # ---- the log ----
    def text(self):
        self.log.flush()
        return open(self.log_path, "rb").read().decode("utf-8", "replace")

    def tail(self, n=3000):
        return self.text()[-n:]

    def mark(self):
        return len(self.text())

    def since(self, mark):
        return self.text()[mark:]

    def grep(self, word):
        return [l for l in self.text().splitlines() if word in l]

    def log_size(self):
        self.log.flush()
        return os.path.getsize(self.log_path)

    # ---- HTTP ----
    def post_json(self, path, body, timeout=900):
        """(status, parsed body) -- an HTTP error is a status, not an exception"""
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(txt)
            except Exception:
                return e.code, {"error": txt}

    def metrics(self):
        out = {}
        for line in urllib.request.urlopen(self.url + "/metrics", timeout=10).read().decode().splitlines():
            if line and not line.startswith("#"):
                k, _, v = line.rpartition(" ")
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if not self.log.closed:
            self.log.close()
        shutil.rmtree(os.path.join(self.dir, "kv"), ignore_errors=True)   # the disk tier is scratch; the log stays


class Grader:
    """One leg's verdict: every check printed with the leg's name, PASS only when all held."""

    def __init__(self, leg):
        self.leg, self.ok, self.n = leg, True, 0

    def check(self, c, msg):
        self.ok &= bool(c)
        self.n += 1
        print("%s [%s] %s" % ("PASS" if c else "FAIL", self.leg, msg), flush=True)
        return bool(c)

    def note(self, msg):
        print("# [%s] %s" % (self.leg, msg), flush=True)


class Leg:
    """A live leg's phases (tests/live_legs.py runs them): warm(srv) on the model's shared server; hook_env() -> the
    server-wide test hooks its hooked(srv) part needs (None: no such part); cold(srv) on the fresh instance when
    `fresh`.  `g` is its Grader; `work` a directory of its own."""
    fresh = False
    warm = None
    hooked = None
    cold = None

    def __init__(self, g, work):
        self.g, self.work = g, work
        os.makedirs(work, exist_ok=True)

    def hook_env(self):
        return None


def cached_tokens(usage):
    """the prompt tokens a response says came from cache: OpenAI's prompt_tokens_details, Anthropic's cache_read"""
    usage = usage or {}
    if "cache_read_input_tokens" in usage:
        return usage["cache_read_input_tokens"]
    d = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if d is not None and "cached_tokens" in d:
        return d["cached_tokens"]
    return None


def check_fresh(g, what, usage):
    """rule 5 for a fresh-server comparison: the reference request reused nothing (the cold lane ran)"""
    c = cached_tokens(usage)
    return g.check(c == 0, "%s ran cold on the fresh instance: %s cached prompt tokens" % (what, c))


if __name__ == "__main__":
    sys.exit(__doc__)
