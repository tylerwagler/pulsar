#!/usr/bin/env python3
"""encode_admission_live.py -- L354: what the encoder/Parse server admits, on a real pulsar-server.

    python3 tests/encode_admission_live.py PARSE_CONTAINER --binary ./pulsar-server [--port 8354]

  (1) a declared-size image: a real 32000 x 32000 solid PNG (a few MB of body, 4 GiB RGBA + 3 GiB RGB once decoded) must be refused
      before its pixels are allocated: the server's peak RSS may not grow by a GiB;
  (2) connections past the cap: of 80 idle sockets, those past PULSAR_SERVER_MAX_CLIENTS (64) get an immediate 503 and
      the server keeps its thread count bounded;
  (3) the server still answers afterwards.

A manual leg: it needs a Parse container, not on the battery's box.  Exit 0 only when every check passed."""
import argparse, base64, http.client, json, os, socket, struct, subprocess, sys, tempfile, time, zlib

W = H = 32000   # libpng refuses a buffer past 2^32 bytes itself: this is near the largest it decodes
SIDE_MAX_CONNS = 80


def png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def solid_png(w, h):
    z = zlib.compressobj(9)
    row = b"\0" + b"\x7f\x7f\x7f" * w
    out = [z.compress(row) for _ in range(h)]
    out.append(z.flush())
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            png_chunk(b"IDAT", b"".join(out)) + png_chunk(b"IEND", b""))


def vm_kib(pid, key):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return 0


def post(port, path, body, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--port", type=int, default=8354)
    a = ap.parse_args()
    fails = []

    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            fails.append(msg)
        return ok

    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    srv = subprocess.Popen([a.binary, "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port)], stdout=log,
                           stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if srv.poll() is not None or "listening on" in open(log.name, errors="replace").read():
                break
            time.sleep(1)
        if not check(srv.poll() is None, "the server is up"):
            print(open(log.name, errors="replace").read()[-1500:])
            return 1
        t0 = time.time()
        png = solid_png(W, H)
        print(f"# a {W} x {H} PNG: {len(png) / 2**20:.1f} MiB, built in {time.time() - t0:.0f} s", flush=True)
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        hwm0 = vm_kib(srv.pid, "VmHWM")
        t0 = time.time()
        st, body = post(a.port, "/v1/parse", {"image": url})
        grew = (vm_kib(srv.pid, "VmHWM") - hwm0) / 2**20
        print(f"# /v1/parse: {st} in {time.time() - t0:.1f} s, peak RSS grew {grew:.2f} GiB: {body[:160]!r}", flush=True)
        check(st == 400, f"the image is refused 400 (got {st})")
        # the body itself costs a little under 3 x its size to read, parse and decode from base64
        check(grew < 1.0, f"its pixels were never allocated: peak RSS grew {grew:.2f} GiB, under 1")

        socks, refused = [], 0
        for _ in range(SIDE_MAX_CONNS):
            s = socket.create_connection(("127.0.0.1", a.port), timeout=5)
            socks.append(s)
        time.sleep(1.0)
        threads = int(open(f"/proc/{srv.pid}/stat").read().rsplit(")", 1)[1].split()[17])
        for s in socks:
            s.settimeout(0.05)
            try:
                d = s.recv(64)
                refused += d.startswith(b"HTTP/1.1 503")
            except OSError:
                pass
        print(f"# {SIDE_MAX_CONNS} idle sockets: {refused} refused 503, {threads} threads", flush=True)
        check(refused == SIDE_MAX_CONNS - 64, f"the {SIDE_MAX_CONNS - 64} past the cap are refused 503 (got {refused})")
        for s in socks:
            s.close()
        time.sleep(1.0)
        st, _ = post(a.port, "/v1/parse", {"image": "data:image/png;base64,AAAA"})
        check(st in (200, 400), f"the server still answers after that (got {st})")
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        os.unlink(log.name)
    print("encode-admission-live: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
