#!/usr/bin/env python3
"""encode_gpu_thread_live.py -- L374: the encoder/Parse server runs every forward of an engine on that engine's one
long-lived GPU thread, on a real pulsar-server.

    python3 tests/encode_gpu_thread_live.py CONTAINER [CONTAINER ...] --binary ./pulsar-server [--port 8374] [-n 100]

Every container is an embedder, a reranker or a parser (several: one server, -m repeated).  For each served model:

  (1) after WARM requests, each on its own connection, the next N requests -- each on its own connection again --
      all answer 200;
  (2) the server's log gains nothing over those N but their request lines: no shape announced again, no plan
      rebuilt, no refusal (the CUDA side's caches are per submitting thread; a forward on the connection's thread
      announced every shape once a request);
  (3) the process's thread count is what it was before them (the connection threads have gone, the GPU threads stay);
  (4) the device's used memory has not grown by 256 MiB (a per-thread scratch arena a request did).

A manual leg: it needs encoder / Parse containers, not on the battery's box.  Exit 0 only when every check passed."""
import argparse, base64, http.client, json, re, struct, subprocess, sys, tempfile, time, zlib, os

WARM = 3
REQUEST_LINE = re.compile(r"pulsar-server: /\S+ .* ms$")


def png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def page_png(w, h):
    """a white page with a black bar: a real PNG the parser decodes"""
    rows = b"".join(b"\0" + (b"\x00\x00\x00" if h // 4 <= y < h // 4 + 8 else b"\xff\xff\xff") * w for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            png_chunk(b"IDAT", zlib.compress(rows, 9)) + png_chunk(b"IEND", b""))


def call(port, method, path, body=None, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)   # a new connection every call
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def threads_of(pid):
    return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[17])


def gpu_used_mib():
    """the device memory in use, summed over the visible GPUs; None without nvidia-smi"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return sum(int(x) for x in out.stdout.split()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def other_lines(path):
    """the log's lines that are not a request's own line"""
    return [l for l in open(path, errors="replace").read().splitlines() if not REQUEST_LINE.search(l)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--port", type=int, default=8374)
    ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()
    fails = []

    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            fails.append(msg)
        return ok

    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    cmd = [a.binary, "--host", "127.0.0.1", "--port", str(a.port)]
    for m in a.models:
        cmd += ["-m", m]
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            if srv.poll() is not None or "listening on" in open(log.name, errors="replace").read():
                break
            time.sleep(1)
        if not check(srv.poll() is None, "the server is up"):
            print(open(log.name, errors="replace").read()[-1500:])
            return 1
        st, body = call(a.port, "GET", "/health")
        if not check(st == 200, f"/health answers (got {st})"):
            return 1
        url = "data:image/png;base64," + base64.b64encode(page_png(640, 480)).decode()
        for m in json.loads(body)["models"]:
            mid, path = m["model"], m["endpoint"]
            if m.get("parser"):
                req = {"model": mid, "image": url, "max_tokens": 16}
            elif m["encoder"] == "embed":
                req = {"model": mid, "input": ["the same request every time", "a second input"], "input_type": "query"}
            else:
                req = {"model": mid, "query": "what is asked", "documents": ["one document", "another document"]}
            for i in range(WARM):
                st, body = call(a.port, "POST", path, req)
                if not check(st == 200, f"{mid}: warm request {i} answers 200 (got {st}: {body[:200]!r})"):
                    break
            else:
                time.sleep(0.5)
                before, th0, mem0 = other_lines(log.name), threads_of(srv.pid), gpu_used_mib()
                t0 = time.time()
                bad = [(i, st) for i in range(a.n) for st in [call(a.port, "POST", path, req)[0]] if st != 200]
                dt = time.time() - t0
                time.sleep(0.5)
                after, th1, mem1 = other_lines(log.name), threads_of(srv.pid), gpu_used_mib()
                print(f"# {mid} ({path}): {a.n} requests on {a.n} connections in {dt:.1f} s, threads {th0} -> {th1}, "
                      f"device memory {mem0} -> {mem1} MiB", flush=True)
                check(not bad, f"{mid}: all {a.n} answer 200 (not 200: {bad[:8]})")
                new = after[len(before):]
                if not check(not new, f"{mid}: the log gained only request lines ({len(new)} others)"):
                    print("\n".join("    " + l for l in new[:40]))
                check(th1 == th0, f"{mid}: the thread count is unchanged ({th0} -> {th1})")
                if mem0 is None or mem1 is None:
                    print(f"N/A  {mid}: device memory (no nvidia-smi here)")
                else:
                    check(mem1 - mem0 < 256, f"{mid}: device memory grew {mem1 - mem0} MiB, under 256")
        print("# the server's log, without the request lines:")
        print("\n".join("    " + l for l in other_lines(log.name)[-60:]))
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
        os.unlink(log.name)
    print("encode-gpu-thread-live: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
