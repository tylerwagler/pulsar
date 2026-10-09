"""Layer-streamed HF `qwen4_exp` glue for Qwen3.8-Flash-Next -- L251's KL instrument, vendored (L286, L285 E9).

This is `qwen_capture.py` + the helpers of `qwen_kl_check.py` from pulsar-notes `research/l251/` (the harness that
measured A == REF bit for bit at all 48 layers), with the paths made arguments instead of constants:

  ALLOC_SRC    the BF16 checkpoint (default: the NAS HF snapshot -- sparky's old /srv copy is gone)
  ALLOC_TOK    tokenizer + config dir
  ALLOC_CALIB  calib-qwen38-v1.jsonl (466 rows)

Faithfulness choices are L251's, each asserted:
  - QSA == dense attention only for T <= indexer budget (2048): the indexer is replaced by an all-visible mask under
    that assertion.
  - PLE: the 51B n-gram table is never materialised; rows are gathered from the checkpoint's 128 shards.
  - Stream: embeddings repeated hc_count times; rotary over 3 identical text position rows.
  - Experts: the eager reference per-expert loop (batched_mm drained sparky to 1.4 GiB in an L251 test).

The held-out set (48 rows, 73,738 tokens) and the calibration set (418 rows, 592,960 tokens) are re-derived by
`split_rows` and asserted equal to `capture_meta.json` (the per-expert check's record) by the callers.
"""
import json, os, random, subprocess, sys, time

r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                   capture_output=True, text=True)
_bad = [l for l in r.stdout.splitlines() if l.strip()]
if r.returncode or _bad:
    sys.exit(f"REFUSING: GPU tenants present (or nvidia-smi failed rc={r.returncode}): {_bad}")

import torch                                                                   # noqa: E402
from safetensors import safe_open                                              # noqa: E402
from transformers import AutoTokenizer                                         # noqa: E402,F401
from transformers.models.qwen4_exp import modeling_qwen4_exp as M              # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig   # noqa: E402

SRC = os.environ.get("ALLOC_SRC", "/mnt/models/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/"
                                  "de4b8e4d43b917e7706784d8bb445c9af86a3540")
TOK = os.environ.get("ALLOC_TOK", "/mnt/models/qwen38fn-tok")
CALIB = os.environ.get("ALLOC_CALIB", "/mnt/models/qwen38-calib/calib-qwen38-v1.jsonl")
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "capture_meta.json")
DEV = torch.device("cuda:0")
PFX = "model.language_model."
N_HELDOUT, SEQ_LEN = 48, 2048
PROBE_SEEDS, PROBE_N, PROBE_T = (7, 11), 32, 1024


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def text_config():
    cfg = Qwen4ExpConfig.from_pretrained(TOK).text_config
    cfg._attn_implementation = "sdpa"
    cfg._experts_implementation = "eager"
    return cfg


class Weights:
    def __init__(self, src=SRC):
        self.src = src
        self.map = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
        self._open = {}

    def f(self, name):
        fn = self.map[name]
        if fn not in self._open:
            self._open[fn] = safe_open(f"{self.src}/{fn}", framework="pt", device="cpu")
        return self._open[fn]

    def get(self, name, device=DEV):
        return self.f(name).get_tensor(name).clone().to(device)   # host clone first: mmap view -> CUDA is slow on GB10

    def prefixed(self, prefix, skip=()):
        return {k[len(prefix):]: self.get(k) for k in self.map if k.startswith(prefix) and not any(s in k for s in skip)}


class DiskNgram(torch.nn.Module):
    """Stands in for ple_embedding.ngram_embedding: rows gathered from the checkpoint's shards."""

    def __init__(self, W, layer, n_shards):
        super().__init__()
        self.W, self.layer, self.n = W, layer, n_shards
        self.weight = torch.empty(1, device=DEV)
        self.shard_rows = None
        self.uniq = self.rows = None
        self.record = None

    def _name(self, s):
        return f"{PFX}layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"

    def load_rows(self, ids):
        uniq = torch.unique(ids.flatten().cpu())
        sl = self.W.f(self._name(0)).get_slice(self._name(0))
        self.shard_rows = sl.get_shape()[0]
        dim = sl.get_shape()[1]
        rows = torch.empty(len(uniq), dim, dtype=torch.bfloat16)
        shard_of = uniq // self.shard_rows
        t0 = time.time()
        for s in torch.unique(shard_of).tolist():
            sel = (shard_of == s).nonzero().flatten()
            assert s < self.n, (s, self.n)
            tab = self.W.f(self._name(s)).get_tensor(self._name(s))
            assert tab.shape[0] == self.shard_rows and tab.dtype == torch.bfloat16, (s, tab.shape, tab.dtype)
            rows[sel] = tab[uniq[sel] - s * self.shard_rows]
            del tab
        self.uniq, self.rows = uniq.to(DEV), rows.to(DEV)
        log(f"PLE: {len(uniq):,} distinct n-gram rows gathered from {len(torch.unique(shard_of))} shards "
            f"in {time.time() - t0:.0f}s")

    def forward(self, ngram_ids):
        if self.record is not None:
            self.record.append(ngram_ids.detach().cpu())
            return torch.zeros(*ngram_ids.shape, self.rows.shape[1] if self.rows is not None else 160,
                               dtype=torch.bfloat16, device=ngram_ids.device)
        pos = torch.searchsorted(self.uniq, ngram_ids.flatten())
        assert bool((self.uniq[pos] == ngram_ids.flatten()).all()), "n-gram id missing from the gathered rows"
        return self.rows[pos].view(*ngram_ids.shape, -1)


def build_layer(cfg, W, i):
    with torch.device("meta"):
        layer = M.Qwen4ExpTextDecoderLayer(cfg, i)
    sd = W.prefixed(f"{PFX}layers.{i}.", skip=("ngram_embedding.shard_",))
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    assert set(missing) <= {"ple.ple_embedding.ngram_embedding.weight"}, f"layer {i}: missing {missing}"
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = DiskNgram(W, i, cfg.split_ngram_parts)
    for n, t in list(layer.named_parameters()) + list(layer.named_buffers()):
        assert t.device.type == "cuda", f"layer {i}: {n} left on {t.device}"
    return layer.eval()


def all_visible_indexer(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    T = hidden_states.shape[1]
    assert T <= self.token_budget, f"QSA==dense only holds for T <= {self.token_budget} (got {T})"
    return torch.ones_like(attention_mask, dtype=torch.bool)


def positions(cfg, rotary, T):
    pid = torch.arange(T, device=DEV).view(1, 1, -1).expand(3, 1, -1)
    return rotary(torch.empty(1, T, cfg.hidden_size, device=DEV, dtype=torch.bfloat16), pid)


def causal(T):
    return torch.ones(T, T, dtype=torch.bool, device=DEV).tril().view(1, 1, T, T)


def split_rows(tok):
    """L251's split: held-out = 48 rows the probe did not use (round-robin across source classes, file order);
    calibration = every other row."""
    rows = [json.loads(l) for l in open(CALIB)]
    lens = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
    assert all(l == r["n_tokens"] for l, r in zip(lens, rows))
    probe = set()
    for seed in PROBE_SEEDS:
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        probe |= set([i for i in idx if lens[i] >= PROBE_T][:PROBE_N])
    assert len(probe) == 2 * PROBE_N, len(probe)
    by = {}
    for i, r in enumerate(rows):
        if i not in probe:
            by.setdefault(r["source"].rsplit("-", 1)[0], []).append(i)
    ho = []
    while len(ho) < N_HELDOUT:
        for c in sorted(by):
            if by[c] and len(ho) < N_HELDOUT:
                ho.append(by[c].pop(0))
    cal = [i for i in range(len(rows)) if i not in set(ho)]

    def ids(i):
        return torch.tensor(tok(rows[i]["text"], add_special_tokens=False)["input_ids"][:SEQ_LEN], dtype=torch.long)
    return [(i, ids(i)) for i in ho], [(i, ids(i)) for i in cal]


def heldout(tok):
    """The 48 held-out rows, asserted equal to the L251 record."""
    ho, cal = split_rows(tok)
    meta = json.load(open(META))
    assert [i for i, _ in ho] == meta["heldout_rows"], "held-out rows differ from L251's"
    assert [int(s.numel()) for _, s in ho] == meta["heldout_T"], "held-out lengths differ from L251's"
    assert [i for i, _ in cal] == meta["calib_rows"], "calibration rows differ from L251's"
    return ho, cal


def gather_ple(layer, seqs):
    ng = layer.ple.ple_embedding.ngram_embedding
    ng.record = []
    for (_, ids) in seqs:
        layer.ple.ple_embedding(ids.view(1, -1).to(DEV), None)
    rec, ng.record = torch.cat([x.flatten() for x in ng.record]), None
    ng.load_rows(rec)


def run_layer(cfg, rotary, layer, h, ids):
    T = ids.numel()
    return layer(h, position_embeddings=positions(cfg, rotary, T), attention_mask=causal(T), conv_mask=None,
                 past_key_values=None, ple_input_ids=ids.view(1, -1).to(DEV))


def build_mixer(cfg, W):
    with torch.device("meta"):
        mix = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    sd = W.prefixed(f"{PFX}hyper_connection_mixer.")
    mix.load_state_dict(sd, strict=True, assign=True)
    return mix.eval()


def install():
    """The two module patches every pass needs (the indexer under the T <= budget assertion)."""
    M.Qwen4ExpTextQSAIndexer.forward = all_visible_indexer
