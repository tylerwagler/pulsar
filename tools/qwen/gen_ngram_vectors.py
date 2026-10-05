#!/usr/bin/env python3
"""L251 S4: vectors for tests/qwen_ngram_test -- the source module's PLE n-gram ids and table rows.

The oracle is transformers' own Qwen4ExpTextNGramEmbedding (modeling_qwen4_exp.py), built on the meta device
(the 51B-parameter table is never materialised) with its buffers loaded from the checkpoint; its embedding
lookup is replaced by a recorder, so `forward` hands back exactly the `ngram_ids` it would look up.  Sequences:
the calibration corpus rendered through Qwen's chat template (whole texts, one sequence each, tokenized with the
checkpoint's tokenizer) plus synthetic EOS cases (EOS first, EOS runs, EOS last, a lone EOS).  Rows: a sample of
the ids' rows read from the checkpoint's own 128 shard tensors with safetensors, and the (file, byte offset) of
every part so the test can read the same bytes with the engine's gather pool.

Output (little endian):
  "QNGRAM01" u32 vocab i32 eos u64 n_rows i64 mult[3] u64 prime[16] u64 offset[16] u64 rows_per_part u32 n_parts
  n_parts x (u64 base, u16 path_len, path bytes)
  u32 n_seq; n_seq x (u32 n, i32 ids[n], u64 rows[n][16])
  u32 n_check; n_check x (u64 row, 320 bytes)

  ~/venvs/q38/bin/python gen_ngram_vectors.py OUT [--max-seq N] [--small]
"""
import argparse, json, os, random, struct, sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

SRC = "/srv/models/qwen38fn-bf16"
TOK = "/srv/models/qwen38fn-tok"
CALIB = "/srv/models/calib-qwen38-v1.jsonl"
PFX = "model.language_model.layers.1.ple.ple_embedding."


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(1)
        self.got = None

    def forward(self, ids):
        self.got = ids.detach().clone()
        return torch.zeros(*ids.shape, 160, dtype=torch.bfloat16)


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max-seq", type=int, default=0, help="corpus sequences (0 = all)")
    ap.add_argument("--small", action="store_true", help="4 short corpus sequences + the synthetic ones")
    ap.add_argument("--n-check", type=int, default=4096)
    a = ap.parse_args()

    cfg = Qwen4ExpConfig.from_pretrained(TOK).text_config
    with torch.device("meta"):
        mod = M.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=0)
    wmap = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]

    def get(name):
        with safe_open(f"{SRC}/{wmap[PFX + name]}", "pt") as f:
            return f.get_tensor(PFX + name).clone()

    mult, primes, offs = get("layer_multipliers"), get("ngram_heads_vocab_sizes"), get("ngram_heads_offsets")
    # the checkpoint's buffers are the authority; the module's own construction must agree with them
    assert mod.head_vocab_sizes == primes.tolist() and mod.head_offsets == offs.tolist(), "buffers vs construction"
    mod.layer_multipliers = mult
    mod.ngram_heads_vocab_sizes = primes
    mod.ngram_heads_offsets = offs
    rec = Recorder()
    mod.ngram_embedding = rec
    eos = mod.eos_token_id
    parts = sorted((k for k in wmap if k.startswith(PFX + "ngram_embedding.shard_")),
                   key=lambda k: int(k.split("shard_")[1].split(".")[0]))
    part_info, rows_per_part = [], None
    for k in parts:
        path = f"{SRC}/{wmap[k]}"
        base, hdr = st_header(path)
        t = hdr[k]
        assert t["dtype"] == "BF16" and t["shape"][1] == 160, t
        rows_per_part = rows_per_part or t["shape"][0]
        assert t["shape"][0] == rows_per_part or k == parts[-1]
        part_info.append((base + t["data_offsets"][0], os.path.realpath(path)))
    n_rows = rows_per_part * (len(parts) - 1) + st_header(part_info[-1][1])[1][parts[-1]]["shape"][0]
    print(f"vocab {cfg.vocab_size} eos {eos} mult {mult.tolist()} n_rows {n_rows:,} parts {len(parts)} x {rows_per_part:,}")

    tok = AutoTokenizer.from_pretrained(TOK)
    texts = [json.loads(l)["text"] for l in open(CALIB)]
    if a.small:
        texts = texts[:4]
    elif a.max_seq:
        texts = texts[:a.max_seq]
    seqs = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    if a.small:
        seqs = [s[:300] for s in seqs]
    # packed documents: corpus texts joined by EOS, so the reset is exercised mid-sequence on real text
    seqs += [seqs[i % len(seqs)][:200] + [eos] + seqs[(i + 1) % len(seqs)][:150] + [eos, eos] +
             seqs[(i + 2) % len(seqs)][:100] for i in range(3)]
    seqs += [[eos, 11, 22, 33], [5, eos, eos, 7, 8, eos, 9, 10, 11], [100, 200, eos], [eos], [eos, eos, eos, 42],
             [cfg.vocab_size - 1, 0, 1, eos, cfg.vocab_size - 1]]
    n_eos = sum(s.count(eos) for s in seqs)
    print(f"{len(seqs)} sequences, {sum(map(len, seqs)):,} tokens, {n_eos:,} EOS")

    all_rows = []
    with open(a.out, "wb") as f:
        f.write(b"QNGRAM01")
        f.write(struct.pack("<IiQ", cfg.vocab_size, eos, n_rows))
        f.write(struct.pack("<3q", *mult.tolist()))
        f.write(struct.pack("<16Q", *primes.tolist()))
        f.write(struct.pack("<16Q", *offs.tolist()))
        f.write(struct.pack("<QI", rows_per_part, len(parts)))
        for base, path in part_info:
            p = path.encode()
            f.write(struct.pack("<QH", base, len(p)) + p)
        f.write(struct.pack("<I", len(seqs)))
        for s in seqs:
            with torch.no_grad():
                mod(torch.tensor([s], dtype=torch.long), None)
            ids = rec.got[0]
            assert ids.shape == (len(s), 16), ids.shape
            assert int(ids.min()) >= 0 and int(ids.max()) < n_rows
            f.write(struct.pack("<I", len(s)))
            f.write(struct.pack(f"<{len(s)}i", *s))
            f.write(ids.to(torch.int64).numpy().astype("<u8").tobytes())
            all_rows.append(ids.flatten())
        # rows: a random sample of the ids plus the first and last row of every part
        ids = torch.cat(all_rows)
        rnd = random.Random(251)
        pick = sorted(set(rnd.sample(ids.tolist(), min(a.n_check, len(ids)))) |
                      {i * rows_per_part for i in range(len(parts))} |
                      {min((i + 1) * rows_per_part, n_rows) - 1 for i in range(len(parts))}) if a.n_check else []
        f.write(struct.pack("<I", len(pick)))
        by_part = {}
        for r in pick:
            by_part.setdefault(r // rows_per_part, []).append(r)
        for pi in sorted(by_part):
            with safe_open(f"{SRC}/{wmap[parts[pi]]}", "pt") as sf:
                sl = sf.get_slice(parts[pi])
                for r in by_part[pi]:
                    lr = r - pi * rows_per_part
                    row = sl[lr:lr + 1].contiguous()
                    f.write(struct.pack("<Q", r) + row.view(torch.int16).numpy().tobytes())
    print(f"wrote {a.out}: {os.path.getsize(a.out):,} B, {len(pick)} checked rows")


if __name__ == "__main__":
    sys.exit(main())
