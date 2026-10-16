#!/usr/bin/env python3
"""exl3_logit_ref.py -- pulsar's EXL3 Nemotron-H container graded against exllamav3's own forward of the SAME pack
(L296: the format's reference implementation, as Inkling's mini was graded at 97% top-1).

For each case: the chat prompt rendered by the checkpoint's template (HF tokenizer, enable_thinking=False -- what
`pulsar --nothink` renders; the prompt length is checked against pulsar's), followed by pulsar's greedy continuation
(its --dump-logprobs file: per step the selected id and the top-K logprobs).  exllamav3 runs ONE cache-less forward
over prompt + continuation (teacher-forced on pulsar's tokens) and at every generated step reports:

  top-1      exllamav3's argmax == pulsar's greedy pick
  |dlp|      |logprob_pulsar(pick) - logprob_exl3(pick)|
  KL_topK    sum over pulsar's top-K of p_pulsar * (lp_pulsar - lp_exl3) -- a truncated KL(pulsar || exl3) (its
             lower-bound reading, the fmtcmp estimator's convention)

The step-0 row is the FIRST-TOKEN comparison (the prompt's last position).

    python exl3_logit_ref.py --pack /srv/models/nemotron35-exl3-k4 --case NAME PROMPT_FILE DUMP_JSON [--case ...]
        [--json OUT]

Needs exllamav3 with the NemotronH Omni adapter on PYTHONPATH (~/inkling/exllamav3) and a GPU; run it under gpuq.
The adapter reads llm_config.num_hidden_layers, which the Omni config does not carry: patched here at run time to
len(layers_block_type) (the same fact), nothing written to the fork.
"""
import argparse
import json
import math
import os
import sys

import torch


def patch_adapter():
    from exllamav3.architecture import nemotronh_omni as om
    orig = om.NemotronHOmniConfig.read_cfg

    def read_cfg(self, expected_types, keys, default=None, *args):
        ks = [keys] if isinstance(keys, str) else list(keys)
        llm = self.config_dict.get("llm_config", {})
        if "num_hidden_layers" in ks and "num_hidden_layers" not in llm and "layers_block_type" in llm:
            return len(llm["layers_block_type"])
        return orig(self, expected_types, keys, default, *args)
    om.NemotronHOmniConfig.read_cfg = read_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--case", nargs=3, action="append", metavar=("NAME", "PROMPT_FILE", "DUMP_JSON"), required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    patch_adapter()
    from exllamav3 import Config, Model
    config = Config.from_directory(a.pack)
    model = Model.from_config(config)
    model.load(progressbar=False)
    vocab = config.vocab_size if hasattr(config, "vocab_size") else None

    report, all_steps = [], []
    for name, prompt_file, dump in a.case:
        text = open(prompt_file).read()
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        d = json.load(open(dump))
        if d["prompt_tokens"] != len(ids):
            sys.exit(f"{name}: pulsar rendered {d['prompt_tokens']} prompt tokens, the HF template {len(ids)} -- "
                     f"not the same prompt")
        gen = [s["selected"]["id"] for s in d["steps"]]
        seq = torch.tensor([list(ids) + gen], dtype=torch.long)
        logits = model.forward(seq, {"attn_mode": "flash_attn_nc"})
        V = vocab or logits.shape[-1]
        lp = torch.log_softmax(logits[0, :, :V].float(), dim=-1).cpu()
        P = len(ids)
        rows = []
        for i, s in enumerate(d["steps"]):
            e = lp[P - 1 + i]
            pick = s["selected"]["id"]
            top = [(t["token"]["id"], t["logprob"]) for t in s["top_logprobs"]]
            kl = sum(math.exp(l) * (l - e[t].item()) for t, l in top)
            rows.append({"step": i, "pick": pick, "exl3_argmax": int(e.argmax().item()),
                         "top1": int(e.argmax().item()) == pick,
                         "dlp": abs(top[0][1] - e[pick].item()) if top and top[0][0] == pick else
                         abs(next((l for t, l in top if t == pick), float("nan")) - e[pick].item()),
                         "kl_topk": kl})
        t1 = sum(r["top1"] for r in rows) / len(rows)
        print(f"[{name}] prompt {P} tokens, {len(rows)} steps: top-1 {100 * t1:.1f}%, first token "
              f"{'==' if rows[0]['top1'] else '!='} (pulsar {rows[0]['pick']} / exl3 {rows[0]['exl3_argmax']}), "
              f"first-token KL_topK {rows[0]['kl_topk']:.3e}, mean |dlp| "
              f"{sum(r['dlp'] for r in rows) / len(rows):.3e}, mean KL_topK {sum(r['kl_topk'] for r in rows) / len(rows):.3e}")
        print(f"    pulsar: {tok.decode(gen)!r}")
        greedy_exl3 = tok.decode([r["exl3_argmax"] for r in rows])
        print(f"    exl3 argmax (teacher-forced): {greedy_exl3!r}")
        report.append({"case": name, "prompt_tokens": P, "steps": rows, "top1": t1})
        all_steps += rows
    n = len(all_steps)
    t1 = sum(r["top1"] for r in all_steps) / n
    f1 = sum(r["top1"] for c in report for r in c["steps"][:1]) / len(report)
    print(f"[all] {len(report)} cases, {n} steps: top-1 {100 * t1:.1f}%, first-token top-1 {100 * f1:.1f}%, "
          f"mean |dlp| {sum(r['dlp'] for r in all_steps) / n:.3e}, mean KL_topK "
          f"{sum(r['kl_topk'] for r in all_steps) / n:.3e}")
    if a.json:
        json.dump({"pack": os.path.abspath(a.pack), "cases": report, "top1": t1, "first_top1": f1}, open(a.json, "w"),
                  indent=1)


if __name__ == "__main__":
    main()
