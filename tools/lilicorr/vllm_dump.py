#!/usr/bin/env python3
"""vllm_dump.py -- golden dumps of vLLM's LiLiCorr block drafter (L298): every draft round of a few greedy
generations of Nemotron 3.5 Super VL (NVFP4) drafted by /mnt/models/nemotron35-super-lilicorr, as vLLM computes it.

Runs INSIDE the vLLM image job.sh serves LiLiCorr with (vllm 0.32.0rc1.dev190+g3f1bcfd82), offline (`vllm.LLM`),
in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) and eager (no torch.compile, no CUDA graphs) so the Python hooks below
see every round.  Prefix caching is off: every prompt prefills from token 0, so the dump of a prompt holds the
drafter's whole context.  Temperature 0 (the selector's greedy walk).

One .safetensors per prompt (<out>/<prompt>.safetensors) plus <prompt>.json (meta: prompt ids, output ids, the
per-round tensor list, vLLM's attention scales).  Round r = the r-th DFlashSpeculator.propose call of the request
(prefill chunks included: a non-final chunk drafts too, and its draft is discarded).  Keys, per round `r{r:03d}.`:

  ctx_taps        [n, 5*4096] bf16  the target's tapped hidden states (layers 1,22,43,64,85), concatenated, for the
                                    n target rows of this step (prefill chunk, or verify block of 1+7)
  ctx_pos         [n] i64           their positions
  num_rejected    [1] i64           rows at the END of the step the target rejected (they never enter the context)
  fc_out          [n, 4096] bf16    fc(ctx_taps) -- the drafter's context state (pre hidden_norm)
  anchor_hidden   [4096] bf16       hidden_norm(fc_out[last accepted row]) (zero when invalid)
  anchor_valid    [1] bool
  q_ids, q_pos    [8] i64           the drafted block: the bonus token then 7 x mask (990), at its positions
  L{i}.in_hidden / in_residual      decoder layer i's inputs (layer 0: residual absent -- in_hidden is the embedding)
  L{i}.q, k, v    [8, 4096/1024/1024] bf16  attention inputs after q/k-norm and RoPE (what the cache stores for the block)
  L{i}.attn       [8, 4096] bf16    the attention output (before o_proj)
  L{i}.out_hidden / out_residual    decoder layer i's outputs
  last_hidden     [8, 4096] bf16    norm(hidden + residual) after the last layer
  cand_ids        [7, 8] i64        top-8 of the drafter's own LM head (log-softmax over the full vocab)
  cand_logp       [7, 8] f32
  tok_emb         [7, 8, 4096] bf16 the target embedding of each candidate (the head's token_embeddings)
  head_start      [8] / head_pairs [6, 8, 8]  LiLiCorrHead.score() (normalized dots, before logit_scale)
  scores          [7, 8, 8] f32     LiLiCorrHead.forward(): logit_scale * [start broadcast; pairs]
  draft           [7] i64           the chain the selector walk chose

    docker run ... vllm-image python3 vllm_dump.py --out /work/goldens [--tokens 96]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

TARGET = "/srv/models/nemotron35-super-vl-nvfp4"
DRAFT = "/mnt/models/nemotron35-super-lilicorr"

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather and a 3-day forecast for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "units": {"type": "string", "enum": ["metric", "imperial"]},
        "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights", "description": "Search flights between two airports on a date.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"},
            "passengers": {"type": "integer"}}, "required": ["origin", "destination", "date"]}}}]


def prompts(story_file, edit_file):
    """name -> (messages, tools).  code / prose / tool are ~/lilicorr/bench.py's cells byte for byte; edit quotes
    a ~165-line module; long is ~9.5k tokens of story so the drafter's 8192 sliding window is exercised."""
    out = {
        "code": [{"role": "user", "content": "Write a Python LRU cache class with O(1) get and put using a dict and a "
                  "doubly linked list. Include type hints, docstrings, and a set of unittest test cases."}],
        "prose": [{"role": "user", "content": "Explain how matrix multiplication uses a GPU in plain English, then give "
                   "a small numerical example."}],
        "tool": [{"role": "user", "content": "I'm flying from Boston to Denver on 2026-11-20 with my partner. Find "
                  "flights, and also check the weather in Denver for the next 3 days in imperial units. Call the tools."}],
    }
    with open(edit_file, encoding="utf-8") as f:
        src = f.read()
    out["edit"] = [{"role": "user", "content": "Here is a Python module:\n\n```python\n" + src + "```\n\nRename the "
                    "class `Inventory` to `Warehouse` and the method `restock_plan` to `reorder_plan` everywhere, add "
                    "type hints to every function and method signature, and reply with the complete updated module in "
                    "one code block and nothing else."}]
    with open(story_file, encoding="utf-8") as f:
        story = f.read().split("<｜User｜>", 1)[1][:38000]
    out["long"] = [{"role": "user", "content": story + "\n\nContinue the story for a few more paragraphs."}]
    return {k: (v, TOOLS if k == "tool" else None) for k, v in out.items()}


class Rec:
    active = False
    rounds = []        # list of dict name -> tensor
    cur = None
    layer = 0
    scales = {}


def cpu(t):
    return t.detach().to("cpu").contiguous().clone()


def install_hooks():
    from vllm.model_executor.models import lilicorr as lmod
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dsp
    from vllm.v1.worker.gpu.spec_decode.lilicorr import speculator as lsp

    D = dsp.DFlashSpeculator
    L = lsp.LiLiCorrSpeculator

    orig_propose = D.propose

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, *args, **kw):
        dummy = kw.get("dummy_run", False) or kw.get("is_profile", False)
        if not Rec.active or dummy:
            return orig_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                                aux_hidden_states, num_sampled, num_rejected, *args, **kw)
        assert input_batch.num_reqs == 1, "the dump runs one request at a time"
        n = input_batch.num_tokens
        Rec.cur = {"ctx_taps": cpu(torch.cat([h[:n] for h in aux_hidden_states], dim=-1)),
                   "ctx_pos": cpu(input_batch.positions[:n].long()),
                   "num_rejected": cpu(num_rejected[:1].long()),
                   "num_sampled": cpu(num_sampled[:1].long())}
        Rec.layer = 0
        out = orig_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                           num_sampled, num_rejected, *args, **kw)
        Rec.cur["fc_out"] = cpu(self.hidden_states[:n])
        Rec.cur["draft"] = cpu(out[0].long())
        Rec.rounds.append(Rec.cur)
        Rec.cur = None
        return out

    D.propose = propose

    orig_anchor = L.prepare_context_anchor

    def prepare_context_anchor(self, *a, **kw):
        orig_anchor(self, *a, **kw)
        if Rec.cur is not None:
            Rec.cur["anchor_hidden"] = cpu(self.anchor_hidden[0])
            Rec.cur["anchor_valid"] = cpu(self.anchor_valid[:1])

    L.prepare_context_anchor = prepare_context_anchor

    orig_run = D._run_model

    def _run_model(self, num_tokens, *a, **kw):
        out = orig_run(self, num_tokens, *a, **kw)
        if Rec.cur is not None:
            Rec.cur["q_ids"] = cpu(self.input_buffers.input_ids[:8].long())
            Rec.cur["q_pos"] = cpu(self.input_buffers.positions[:8].long())
            Rec.cur["last_hidden"] = cpu(out[:8])
        return out

    D._run_model = _run_model

    orig_cand = lmod.LiLiCorrForCausalLM.compute_candidates

    def compute_candidates(self, hidden_states):
        ids, logp = orig_cand(self, hidden_states)
        if Rec.cur is not None:
            Rec.cur["cand_ids"] = cpu(ids.view(7, -1).long())
            Rec.cur["cand_logp"] = cpu(logp.view(7, -1).float())
            Rec.cur["pass_hidden"] = cpu(hidden_states.view(7, -1))
        return ids, logp

    lmod.LiLiCorrForCausalLM.compute_candidates = compute_candidates

    orig_scorec = L._score_candidates

    def _score_candidates(self, candidate_ids, unary_logits, hidden_states):
        out = orig_scorec(self, candidate_ids, unary_logits, hidden_states)
        if Rec.cur is not None:
            Rec.cur["tok_emb"] = cpu(self.target_embeddings(candidate_ids)[0])
            Rec.cur["scores"] = cpu(out[0].float())
        return out

    L._score_candidates = _score_candidates

    orig_score = lmod.LiLiCorrHead.score

    def score(self, **kw):
        start, pairs = orig_score(self, **kw)
        if Rec.cur is not None:
            Rec.cur["head_start"] = cpu(start[0].float())
            Rec.cur["head_pairs"] = cpu(pairs[0].float())
        return start, pairs

    lmod.LiLiCorrHead.score = score

    orig_load = L.load_draft_model

    def load_draft_model(self, *a, **kw):
        model = orig_load(self, *a, **kw)
        for i, layer in enumerate(model.model.layers):
            def layer_pre(mod, args, kwargs, i=i):
                if Rec.cur is not None:
                    Rec.cur[f"L{i}.in_hidden"] = cpu(kwargs["hidden_states"][:8])
                    if kwargs.get("residual") is not None:
                        Rec.cur[f"L{i}.in_residual"] = cpu(kwargs["residual"][:8])

            def layer_post(mod, args, kwargs, out, i=i):
                if Rec.cur is not None:
                    Rec.cur[f"L{i}.out_hidden"] = cpu(out[0][:8])
                    Rec.cur[f"L{i}.out_residual"] = cpu(out[1][:8])

            def attn_post(mod, args, out, i=i):
                if Rec.cur is not None:
                    q, k, v = args[:3]
                    Rec.cur[f"L{i}.q"] = cpu(q[:8])
                    Rec.cur[f"L{i}.k"] = cpu(k[:8])
                    Rec.cur[f"L{i}.v"] = cpu(v[:8])
                    Rec.cur[f"L{i}.attn"] = cpu(out[:8])

            layer.register_forward_pre_hook(layer_pre, with_kwargs=True)
            layer.register_forward_hook(layer_post, with_kwargs=True)
            layer.self_attn.attn.register_forward_hook(attn_post)
            a_ = layer.self_attn.attn
            Rec.scales[f"L{i}"] = {"k_scale": float(a_._k_scale.float().item()),
                                   "v_scale": float(a_._v_scale.float().item()),
                                   "kv_cache_dtype": str(a_.kv_cache_dtype),
                                   "impl": type(a_.impl).__name__,
                                   "sliding_window": a_.sliding_window if hasattr(a_, "sliding_window") else None}
        return model

    L.load_draft_model = load_draft_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--prompts", default="code,prose,tool,edit,long")
    p.add_argument("--story", default="/work/tools/long_context_story_prompt.txt")
    p.add_argument("--edit", default="/work/tools/edit_fixture.txt")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    install_hooks()
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams

    llm = LLM(model=TARGET, trust_remote_code=True, max_model_len=32768, max_num_seqs=1,
              max_num_batched_tokens=8192, gpu_memory_utilization=0.75, block_size=64,
              load_format="fastsafetensors", mamba_backend="flashinfer", mamba_ssm_cache_dtype="float16",
              enable_mamba_cache_stochastic_rounding=True, mamba_cache_philox_rounds=5,
              enable_prefix_caching=False, enforce_eager=True, compilation_config={"mode": 0},
              speculative_config={"model": DRAFT, "num_speculative_tokens": 7, "method": "dflash"})
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, seed=1234)
    ps = prompts(a.story, a.edit)
    for name in a.prompts.split(","):
        msgs, tools = ps[name]
        Rec.rounds = []
        Rec.active = True
        res = llm.chat(msgs, sp, tools=tools, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        Rec.active = False
        r0 = res[0]
        tensors = {}
        for r, rec in enumerate(Rec.rounds):
            for k, v in rec.items():
                tensors[f"r{r:03d}.{k}"] = v
        save_file(tensors, os.path.join(a.out, f"{name}.safetensors"))
        acc = [int(rec["num_rejected"][0]) for rec in Rec.rounds]
        meta = {"prompt": name, "messages": msgs, "tools": tools, "prompt_ids": list(r0.prompt_token_ids),
                "output_ids": list(r0.outputs[0].token_ids), "output_text": r0.outputs[0].text,
                "rounds": len(Rec.rounds), "num_rejected": acc, "attn": Rec.scales,
                "keys": sorted(Rec.rounds[0].keys()) if Rec.rounds else []}
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[vllm_dump] {name}: prompt {len(r0.prompt_token_ids)} tok, {len(r0.outputs[0].token_ids)} out, "
              f"{len(Rec.rounds)} rounds -> {a.out}/{name}.safetensors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
