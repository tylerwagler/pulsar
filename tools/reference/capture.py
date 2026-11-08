#!/usr/bin/env python3
"""capture.py -- the reference gates' captures, on a PUBLISHED CONTAINER'S OWN WEIGHTS (L357, 2026-11-08).

    capture.py --model SNAPSHOT_DIR --out DIR [--prompts story,code] [--q-chunk 1024] [--check-dense]

One kind of reference for every family (Tyler 2026-11-07, "we need to be consistent"): Hugging Face transformers' own
module code, run layer by layer over the container's bytes, so the gate's drift is OUR arithmetic only -- not the
quantization (the container IS the model we serve).  Decisions (Tyler 2026-11-08): the reference computes in FP32
(weights and activations), and the head is the container's served head (an EXL3 head where the container has one,
never the source's tied / BF16 one).

Weights.  Every tensor is read from the container: EXL3 linears are exllamav3's own trellis / suh / svh bytes, copied
verbatim by tools/container (hf_source.py, entries.py), decoded EXACTLY in fp32 by exllamav3's codebook
(LinearEXL3.get_inner_weight_tensor, then the two 128-Hadamards and the per-channel scales applied in fp32 --
get_weight_tensor() does them in fp16, which would round twice); MXFP8 through tools/container/producers.py's own
inverse; native tensors as stored.  No pulsar CUDA code runs.  No source checkpoint is read.

Protocol (unchanged from the DeepSeek and Qwen anchors, so the gates read it as before):
  - prompts: story = tests/long_context_story_prompt.txt's system + user, re-rendered through the model's own chat
    template (add_generation_prompt, enable_thinking=False); code = tests/test-vectors/prompts/long_code_audit.txt raw.
  - depths: story 512, 2048, 4096, 4102, 6144, 16388, end-10; code 512, 2048, end.
  - one fresh forward per depth; the row is the last position's logits, fp32.
  - <p>.ref.bin: "DS4PFXG1" v2, n_depths, width, reserved, prompt_fnv (FNV-1a 64 over the first `deepest` ids),
    depths[8], build_ref[24] = "hf-<transformers version>", then n_depths x width f32 rows
    (tests/prefill_bitexact_gate.cpp reads it).  <p>.tokens.bin int32 LE.  <p>.ref.json: argmax / top-64 / entropy.
  - capture.json: what the gate checks before it grades -- the container's repo and REVISION (the runner fails when
    the cached model's revision differs), this tool's sha256, the pulsar git sha, the versions (transformers is a dev
    build: its git sha too), the decode and the dtype, and per prompt its depths, fnv and sha256s.

Families (an adapter each: embed, the layer, the final norm / mixer, the head): qwen4_exp.  Others refuse by name
until their adapter is written and checked (--check).

Runs on the GB10 in a GPU window (production stopped): it refuses while any other process holds the GPU.
"""
import argparse, hashlib, json, os, re, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAGIC, VERSION, MAX_DEPTHS, REF_LEN = b"DS4PFXG1", 2, 8, 24
EXL3_WORDS = {"exl3m_k4": 64, "exl3m_k5": 80, "exl3m_k6": 96}
DS_BOS, DS_USER, DS_ASSIST = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def refuse_if_gpu_tenants():
    """Before torch touches CUDA: a reference run beside another engine measures nothing and can OOM the box."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    bad = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode or bad:
        sys.exit("capture: REFUSING -- GPU tenants present (or nvidia-smi failed rc=%d): %s" % (r.returncode, bad))


# ----------------------------------------------------------------------------------------------- the container
class Container:
    """A published container read by its shard headers: name -> (file, byte range); pulsar.tensors (dims_ne, layout),
    pulsar.experts, pulsar.kv.  Names are the HF checkpoint's (tools/container/README.md)."""

    def __init__(self, d):
        self.d = os.path.realpath(d)
        m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})$", self.d)
        if not m:
            sys.exit("capture: %s is not an HF cache snapshot (models--ORG--NAME/snapshots/REV)" % d)
        self.repo, self.revision = "%s/%s" % (m.group(1), m.group(2)), m.group(3)
        self.config = json.load(open(os.path.join(self.d, "config.json")))
        self.loc, self.tensors, self.experts, self.kv = {}, {}, {}, {}
        shards = sorted(f for f in os.listdir(self.d) if f.endswith(".safetensors"))
        if not shards:
            sys.exit("capture: no .safetensors in %s" % self.d)
        for f in shards:
            path = os.path.join(self.d, f)
            with open(path, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                h = json.loads(fh.read(n))
            md = h.pop("__metadata__", None) or {}
            for nm, v in h.items():
                o0, o1 = v["data_offsets"]
                self.loc[nm] = (path, 8 + n + o0, o1 - o0)
            for nm, v in json.loads(md.get("pulsar.tensors", "{}")).items():
                self.tensors[nm] = (v["dims_ne"], v["layout"])
            for e in json.loads(md.get("pulsar.experts", "[]")):
                self.experts[e["gguf_name"]] = e
            for e in json.loads(md.get("pulsar.kv", "[]")):
                self.kv[e["key"]] = e["value"]
        self._fds = {}

    def raw(self, name):
        path, off, size = self.loc[name]
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY)
        b = os.pread(fd, size, off)
        assert len(b) == size, (name, len(b), size)
        return b

    def has(self, name):
        return name in self.tensors

    def sha_of_headers(self):
        """sha256 over every shard's name and header bytes: what the capture was made from, beyond the revision"""
        h = hashlib.sha256()
        for f in sorted(x for x in os.listdir(self.d) if x.endswith(".safetensors")):
            with open(os.path.join(self.d, f), "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                h.update(f.encode() + fh.read(n))
        return h.hexdigest()


# ----------------------------------------------------------------------------------------------- decoding, in fp32
class Decoder:
    def __init__(self, c, dev):
        import numpy as np
        import torch
        self.c, self.dev, self.np, self.torch = c, dev, np, torch
        self._producers = None

    def producers(self):
        if self._producers is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("pulsar_producers",
                                                          os.path.join(ROOT, "tools", "container", "producers.py"))
            self._producers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._producers)
        return self._producers

    def exl3(self, blob, k, n, words):
        """exllamav3's bytes -> the weight they encode, fp32 [out, in], exact to fp32 rounding"""
        np, torch = self.np, self.torch
        from exllamav3.modules.quant.exl3 import LinearEXL3
        from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l_, preapply_had_r_, had_k, had_n
        tb, sb = (k // 16) * (n // 16) * words * 2, (k + n) * 2
        if len(blob) != tb + sb:
            sys.exit("capture: exl3 blob %d != trellis %d + scales %d (k=%d n=%d words=%d)" % (len(blob), tb, sb, k, n, words))
        tr = torch.from_numpy(np.frombuffer(blob[:tb], dtype=np.int16).copy().reshape(k // 16, n // 16, words))
        suh = torch.from_numpy(np.frombuffer(blob[tb:tb + 2 * k], dtype=np.float16).copy())
        svh = torch.from_numpy(np.frombuffer(blob[tb + 2 * k:], dtype=np.float16).copy())
        lin = LinearEXL3(None, k, n, suh=suh.to(self.dev), svh=svh.to(self.dev), trellis=tr.to(self.dev),
                         mul1=torch.zeros(1, dtype=torch.int32, device=self.dev))   # mul1: the codebook (a flag)
        w = lin.get_inner_weight_tensor().float()          # [in, out]: the codebook's values, exact in fp16
        preapply_had_l_(w, had_k)                          # in place; computes in fp32, and w is fp32
        w *= suh.to(self.dev).float().unsqueeze(1)
        preapply_had_r_(w, had_n)
        w *= svh.to(self.dev).float().unsqueeze(0)
        return w.t().contiguous()                          # [out, in]

    def mxfp8(self, blob, k, n):
        """producers.mxfp8_lt's inverse (the producer's own unswizzle and E4M3 table) -> fp32 [out, in]"""
        np, torch, pr = self.np, self.torch, self.producers()
        kb = k // 32
        if len(blob) != n * k + pr._rup(n, 128) * pr._rup(kb, 4):
            sys.exit("capture: mxfp8 blob %d does not hold in=%d out=%d" % (len(blob), k, n))
        codes = np.frombuffer(blob[:n * k], dtype=np.uint8).reshape(n, k)
        s_exp = pr.unswizzle_sf(np.frombuffer(blob[n * k:], dtype=np.uint8), n, kb).astype(np.int32)
        val = pr.E4M3_VALUE[codes].reshape(n, kb, 32) * np.exp2(s_exp - 127)[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(val, dtype=np.float32).reshape(n, k)).to(self.dev)

    def native(self, blob, layout):
        np, torch = self.np, self.torch
        dt = {"bf16": (np.uint16, torch.bfloat16), "f16": (np.float16, None), "f32": (np.float32, None),
              "i32": (np.int32, None), "i64": (np.int64, None)}.get(layout)
        if dt is None:
            sys.exit("capture: a native layout this tool does not read: %r" % layout)
        t = torch.from_numpy(np.frombuffer(blob, dtype=dt[0]).copy())
        if dt[1] is not None:
            t = t.view(dt[1])
        return t.to(self.dev).float() if t.is_floating_point() else t.to(self.dev)

    def tensor(self, name):
        """A container tensor as fp32 on the device: [out, in] for a linear, else its stored shape (flat)"""
        dims, lay = self.c.tensors[name]
        if lay in EXL3_WORDS:
            return self.exl3(self.c.raw(name), dims[0], dims[1], EXL3_WORDS[lay])
        if lay == "mxfp8_lt":
            return self.mxfp8(self.c.raw(name), dims[0], dims[1])
        return self.native(self.c.raw(name), lay)

    def expert(self, entry, e):
        k, n = entry["dims_per_expert_ne"][0], entry["dims_per_expert_ne"][1]
        lay = entry["layout"]
        if lay not in EXL3_WORDS:
            sys.exit("capture: expert layout %r is not EXL3" % lay)
        return self.exl3(self.c.raw(entry["entry_name"].format(e=e)), k, n, EXL3_WORDS[lay])


def fill_module(mod, prefix, dec, special=None):
    """Every parameter / buffer of a meta-built module from the container, fp32 on the device, by its HF name.
    `special(local_name, meta_tensor)` may return a tensor (or None to let the container serve it)."""
    torch = dec.torch
    sd = {}
    for local, t in list(mod.state_dict(keep_vars=True).items()):
        got = special(local, t) if special else None
        if got is None:
            full = prefix + local
            if full in dec.c.experts:
                e = dec.c.experts[full]
                stack = torch.empty(t.shape, dtype=torch.float32, device=dec.dev)
                for ei in range(e["n_experts"]):
                    stack[ei] = dec.expert(e, ei).reshape(t.shape[1:])
                got = stack
            elif dec.c.has(full):
                got = dec.tensor(full)
            else:
                sys.exit("capture: %s is not in the container (and no adapter rule serves it)" % full)
        if got.numel() != t.numel():
            sys.exit("capture: %s: the container holds %d values, the module wants %s" % (prefix + local, got.numel(),
                                                                                         tuple(t.shape)))
        sd[local] = got.reshape(t.shape).to(t.dtype if not t.is_floating_point() else torch.float32)
    missing, unexpected = mod.load_state_dict(sd, strict=True, assign=True)
    for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
        if t.device.type != dec.dev.type:
            sys.exit("capture: %s%s left on %s" % (prefix, n, t.device))
    return mod.eval()


# ----------------------------------------------------------------------------------------------- the Qwen adapter
class QwenAdapter:
    """qwen4_exp (Qwen3.8-Flash-Next): hyper-connection streams (hc_count copies of the embedding), rotary over 3
    identical position rows, PLE (layer 1's n-gram rows from the container's ple-l1.rows side file, one pread a
    row), the real QSA indexer, the final mixer on the last position, the container's K5 head."""
    PFX = "model.language_model."

    def __init__(self, c, dec, a):
        import torch
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
        self.c, self.dec, self.torch, self.M = c, dec, torch, M
        cfg = Qwen4ExpConfig.from_pretrained(c.d).text_config
        cfg._attn_implementation = "sdpa"
        cfg._experts_implementation = "eager"   # the reference per-expert loop
        self.cfg = cfg
        self.rows_file = os.path.join(c.d, c.kv.get("pulsar.ple_rows.file", "ple-l1.rows"))
        self.real_indexer = M.Qwen4ExpTextQSAIndexer.forward
        self.dense = False
        self.rotary = None

    def layers(self):
        return self.cfg.num_hidden_layers

    def embed(self, seqs):
        torch = self.torch
        emb = self.dec.tensor(self.PFX + "embed_tokens.weight").view(-1, self.cfg.hidden_size)
        self.rotary = self.M.Qwen4ExpTextRotaryEmbedding(config=self.cfg).to(self.dec.dev)
        hs = [emb[ids.to(self.dec.dev)].unsqueeze(0).repeat(1, 1, self.cfg.hc_count) for ids in seqs]
        del emb
        return hs

    def _ple_buffer(self, local):
        names = {"ple.ple_embedding.layer_multipliers": "qwen4_exp.ple_layer_multipliers",
                 "ple.ple_embedding.ngram_heads_vocab_sizes": "qwen4_exp.ple_ngram_heads_vocab_sizes",
                 "ple.ple_embedding.ngram_heads_offsets": "qwen4_exp.ple_ngram_heads_offsets"}
        if local in names:
            return self.torch.tensor(self.c.kv[names[local]], dtype=self.torch.int64, device=self.dec.dev)
        return None

    def build(self, i, seqs):
        torch, M = self.torch, self.M
        with torch.device("meta"):
            layer = M.Qwen4ExpTextDecoderLayer(self.cfg, i).float()
        if layer.ple is not None:
            # the table is 320M rows: never materialized -- the rows a sequence touches come from the side file
            layer.ple.ple_embedding.ngram_embedding = DiskRows(self.rows_file, self.dec.dev, torch)
        layer = fill_module(layer, "%slayers.%d." % (self.PFX, i), self.dec, special=self._ple_buffer)
        if layer.ple is not None:
            ng = layer.ple.ple_embedding.ngram_embedding
            ng.record = []
            for ids in seqs:
                layer.ple.ple_embedding(ids.view(1, -1).to(self.dec.dev), None)
            ng.load(torch.cat([x.flatten() for x in ng.record]))
            ng.record = None
        return layer

    def run(self, layer, h, ids):
        torch, cfg, dev = self.torch, self.cfg, self.dec.dev
        T = ids.numel()
        pid = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
        pos = self.rotary(torch.empty(1, T, cfg.hidden_size, device=dev, dtype=torch.float32), pid)
        mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril().view(1, 1, T, T)
        return layer(h, position_embeddings=pos, attention_mask=mask, conv_mask=None, past_key_values=None,
                     ple_input_ids=ids.view(1, -1).to(dev))

    def logits(self, hs):
        torch, M = self.torch, self.M
        with torch.device("meta"):
            mix = M.Qwen4ExpTextGatedResidual(self.cfg, use_combine=False).float()
        mix = fill_module(mix, self.PFX + "hyper_connection_mixer.", self.dec)
        head = self.dec.tensor("lm_head.weight").view(-1, self.cfg.hidden_size)   # the container's K5 head
        rows = []
        for h in hs:
            fin = mix(h[:, -1:, :])
            rows.append((fin[0, -1].float() @ head.t()).cpu())
        return rows

    def set_dense(self, on):
        """--check-dense: at depths <= the indexer budget QSA must equal dense attention, bit for bit"""
        def all_visible(slf, hidden_states, position_embeddings, attention_mask, past_key_values):
            assert hidden_states.shape[1] <= slf.token_budget
            return self.torch.ones_like(attention_mask, dtype=self.torch.bool)
        self.M.Qwen4ExpTextQSAIndexer.forward = all_visible if on else self.real_indexer

    def dense_limit(self):
        return self.cfg.indexer_budget

    def chunk_sdpa(self, q_chunk):
        from transformers.integrations import sdpa_attention as SA
        orig = SA.sdpa_attention_forward

        def fwd(module, query, key, value, attention_mask, **kw):
            T = query.shape[2]
            if T <= q_chunk:
                return orig(module, query, key, value, attention_mask, **kw)
            outs = [orig(module, query[:, :, c0:c0 + q_chunk], key, value, attention_mask[:, :, c0:c0 + q_chunk], **kw)[0]
                    for c0 in range(0, T, q_chunk)]
            return self.torch.cat(outs, dim=1), None
        self.M.ALL_ATTENTION_FUNCTIONS["sdpa"] = fwd


class DiskRows:
    """Stands in for ple_embedding.ngram_embedding: the rows a run touches, read from the container's PENGRAM1 v2 row
    file (tools/container/ple_rows.py: 64-byte header, then n_rows x row_bytes BF16, global row order)."""

    def __new__(cls, path, dev, torch):
        class _Rows(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with open(path, "rb") as f:
                    hdr = f.read(64)
                if hdr[:8] != b"PENGRAM1" or struct.unpack("<I", hdr[8:12])[0] != 2:
                    sys.exit("capture: %s is not a PENGRAM1 v2 row file" % path)
                self.n_rows, = struct.unpack("<Q", hdr[16:24])
                self.row_bytes, self.dim = struct.unpack("<II", hdr[24:32])
                self.fd = os.open(path, os.O_RDONLY)
                self.register_buffer("weight", torch.empty(1, device=dev), persistent=False)  # HF reads .weight.device
                self.record, self.uniq, self.rows = None, None, None

            def load(self, ids):
                import numpy as np
                uniq = torch.unique(ids.flatten().cpu())
                assert int(uniq.max()) < self.n_rows, "an n-gram id past the table"
                buf = bytearray(len(uniq) * self.row_bytes)
                for j, r in enumerate(uniq.tolist()):
                    buf[j * self.row_bytes:(j + 1) * self.row_bytes] = os.pread(self.fd, self.row_bytes,
                                                                               64 + r * self.row_bytes)
                rows = torch.from_numpy(np.frombuffer(bytes(buf), dtype=np.uint16).copy()).view(torch.bfloat16)
                self.uniq, self.rows = uniq.to(dev), rows.view(len(uniq), self.dim).float().to(dev)
                log("PLE: %d distinct n-gram rows read from %s" % (len(uniq), os.path.basename(path)))

            def forward(self, ngram_ids):
                if self.record is not None:
                    self.record.append(ngram_ids.detach().cpu())
                    return torch.zeros(*ngram_ids.shape, self.dim, dtype=torch.float32, device=ngram_ids.device)
                pos = torch.searchsorted(self.uniq, ngram_ids.flatten())
                assert bool((self.uniq[pos] == ngram_ids.flatten()).all()), "an n-gram id missing from the rows read"
                return self.rows[pos].view(*ngram_ids.shape, -1)
        return _Rows()


ADAPTERS = {"qwen4_exp": QwenAdapter}


# ----------------------------------------------------------------------------------------------- prompts and output
def fnv1a64(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def story_ids(tok):
    raw = open(os.path.join(ROOT, "tests", "long_context_story_prompt.txt"), encoding="utf-8").read()
    assert raw.startswith(DS_BOS) and raw.count(DS_USER) == 1 and raw.count(DS_ASSIST) == 1, "unexpected story fixture"
    system, rest = raw[len(DS_BOS):].split(DS_USER)
    user, tail = rest.split(DS_ASSIST)
    assert tail == "</think>", repr(tail[:40])
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = dict(add_generation_prompt=True, enable_thinking=False)
    ids = tok.apply_chat_template(msgs, tokenize=True, **kw)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids), tok.apply_chat_template(msgs, tokenize=False, **kw)


def code_ids(tok):
    raw = open(os.path.join(ROOT, "tests", "test-vectors", "prompts", "long_code_audit.txt"), encoding="utf-8").read()
    return tok(raw, add_special_tokens=False)["input_ids"], raw


def write_blob(path, depths, rows, fnv, build_ref):
    hdr = MAGIC + struct.pack("<IIIIQ", VERSION, len(depths), rows[0].numel(), 0, fnv)
    hdr += struct.pack("<%dI" % MAX_DEPTHS, *(list(depths) + [0] * (MAX_DEPTHS - len(depths))))
    hdr += build_ref.encode()[:REF_LEN - 1].ljust(REF_LEN, b"\0")
    with open(path, "wb") as f:
        f.write(hdr)
        for r in rows:
            f.write(r.float().contiguous().numpy().astype("<f4").tobytes())
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sidecar(torch, tok, depths, rows):
    out = []
    for d, r in zip(depths, rows):
        p = torch.softmax(r.double(), -1)
        H = float(-(p[p > 0] * p[p > 0].log()).sum())
        v, ix = r.topk(64)
        out.append({"depth": d, "argmax_id": int(ix[0]), "argmax_text": tok.decode([int(ix[0])]),
                    "argmax_val": float(v[0]), "entropy_nats": H, "p_top1": float(p.max()),
                    "top64": [[int(a), float(b)] for a, b in zip(ix, v)]})
    return out


def versions(torch):
    import transformers
    v = {"transformers": transformers.__version__, "torch": torch.__version__, "python": sys.version.split()[0]}
    tdir = os.path.dirname(transformers.__file__)
    r = subprocess.run(["git", "-C", tdir, "rev-parse", "HEAD"], capture_output=True, text=True)
    v["transformers_git"] = r.stdout.strip() or None   # a dev build: the version string alone does not name it
    try:
        import exllamav3
        v["exllamav3"] = getattr(exllamav3, "__version__", None)
        r = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(exllamav3.__file__)), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        v["exllamav3_git"] = r.stdout.strip() or None
    except ImportError:
        v["exllamav3"] = None
    return v


def git_sha():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "-uno"], capture_output=True, text=True)
    return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "")


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the container's HF cache snapshot dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="story,code")
    ap.add_argument("--q-chunk", type=int, default=1024, help="query rows per sdpa call (fp32 scores: keep it small)")
    ap.add_argument("--check-dense", action="store_true", help="Qwen: QSA == dense at depths <= the indexer budget")
    a = ap.parse_args()
    c = Container(a.model)
    fam = c.config.get("model_type")   # the top-level type: qwen4_exp, gemma4, nemotron_h_omni (the text configs differ)
    if fam not in ADAPTERS:
        sys.exit("capture: no adapter for family %r yet (have: %s)" % (fam, ", ".join(sorted(ADAPTERS))))
    refuse_if_gpu_tenants()
    import torch
    from transformers import AutoTokenizer
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 means fp32 (TF32 would round the products to 10 bits)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    dec = Decoder(c, dev)
    ad = ADAPTERS[fam](c, dec, a)
    ad.chunk_sdpa(a.q_chunk)
    tok = AutoTokenizer.from_pretrained(c.d)
    os.makedirs(a.out, exist_ok=True)
    build_ref = ("hf-%s" % transformers.__version__)[:REF_LEN - 1]
    plan = []
    for name in a.prompts.split(","):
        ids, text = story_ids(tok) if name == "story" else code_ids(tok) if name == "code" else sys.exit(name)
        n = len(ids)
        depths = [512, 2048, 4096, 4102, 6144, 16388, n - 10] if name == "story" else [512, 2048, n]
        assert all(x < y for x, y in zip(depths, depths[1:])) and depths[-1] <= n, (depths, n)
        tb = struct.pack("<%di" % n, *ids)
        open(os.path.join(a.out, name + ".tokens.bin"), "wb").write(tb)
        open(os.path.join(a.out, name + ".rendered.txt"), "w", encoding="utf-8").write(text)
        plan.append((name, ids, depths, tb, fnv1a64(tb[:4 * depths[-1]])))
        log("%s: %d ids, depths %s" % (name, n, depths))
    # every prompt's depth sequences through one layer pass: each layer's weights are decoded once
    seqs = [torch.tensor(ids[:d], dtype=torch.long) for _, ids, depths, _, _ in plan for d in depths]

    def forward(sel):
        hs = ad.embed([seqs[j] for j in sel])
        t_all = time.time()
        for i in range(ad.layers()):
            t0 = time.time()
            layer = ad.build(i, [seqs[j] for j in sel])
            for k, j in enumerate(sel):
                hs[k] = ad.run(layer, hs[k], seqs[j])
                if not torch.isfinite(hs[k]).all():
                    sys.exit("capture: non-finite stream after layer %d" % i)
            log("layer %3d/%d %.1f s (%.0f s)" % (i + 1, ad.layers(), time.time() - t0, time.time() - t_all))
            del layer
            torch.cuda.empty_cache()
        return ad.logits(hs)

    t0 = time.time()
    rows = forward(list(range(len(seqs))))
    secs = time.time() - t0
    dense_check = None
    if a.check_dense and hasattr(ad, "set_dense"):
        flat = [d for _, _, depths, _, _ in plan for d in depths]
        small = [j for j, d in enumerate(flat) if d <= ad.dense_limit()]
        ad.set_dense(True)
        rows_d = forward(small)
        ad.set_dense(False)
        dense_check = {str(flat[j]): bool(torch.equal(rows[j], r)) for j, r in zip(small, rows_d)}
        log("QSA == dense at depths <= %d: %s" % (ad.dense_limit(), dense_check))
    tool_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    cap = {"container": {"repo": c.repo, "revision": c.revision, "headers_sha256": c.sha_of_headers(), "family": fam},
           "tool": {"path": "tools/reference/capture.py", "sha256": tool_sha, "pulsar_git": git_sha()},
           "versions": versions(torch),
           "weights": {"decode": "exllamav3 codebook (get_inner_weight_tensor) + 128-Hadamards and scales in fp32; "
                                 "mxfp8 via tools/container/producers; native as stored",
                       "dtype": "float32", "tf32": False, "head": "the container's lm_head"},
           "attention": "sdpa, query slices of %d" % a.q_chunk, "dense_check": dense_check,
           "seconds": round(secs), "host": os.uname().nodename, "date_utc": time.strftime("%FT%TZ", time.gmtime()),
           "prompts": {}}
    j = 0
    for name, ids, depths, tb, fnv in plan:
        mine = rows[j:j + len(depths)]
        j += len(depths)
        sha = write_blob(os.path.join(a.out, name + ".ref.bin"), depths, mine, fnv, build_ref)
        side = {"mode": "logits", "width": mine[0].numel(), "build_ref": build_ref, "prompt_fnv": "%#018x" % fnv,
                "rows": sidecar(torch, tok, depths, mine)}
        json.dump(side, open(os.path.join(a.out, name + ".ref.json"), "w"), indent=1)
        cap["prompts"][name] = {"n_tokens": len(ids), "depths": depths, "prompt_fnv": "%#018x" % fnv,
                                "tokens_sha256": hashlib.sha256(tb).hexdigest(), "ref_sha256": sha,
                                "render": ("apply_chat_template(system+user, add_generation_prompt=True, "
                                           "enable_thinking=False)" if name == "story" else
                                           "raw text, add_special_tokens=False"),
                                "argmax": [r["argmax_id"] for r in side["rows"]]}
        log("%s: wrote %s.ref.bin sha256 %s" % (name, name, sha))
    json.dump(cap, open(os.path.join(a.out, "capture.json"), "w"), indent=1)
    log("done in %.0f s -> %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()
