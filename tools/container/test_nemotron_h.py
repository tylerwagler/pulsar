#!/usr/bin/env python3
"""test_nemotron_h.py -- the Nemotron-H family in the artifact builder (nemotron_h.py, L296 N1), on a synthetic Omni
checkpoint (tiny dims, the real layout: llm_config nested, `language_model.` prefix, per-expert up / down, an MTP
module, vision tensors):

  * BF16 build: plan, emit --all twice byte-identical, verify --all (side files byte-equal), audit; GOLDEN
  * the grouping: blocks [mamba, moe, mamba, attention, moe] -> 3 pulsar layers {M,E} {M,-} {*,E}; every record's
    role / block / layer is the PULSAR layer; the conv declared [K, C]; the shared expert index 0; routed families
    up / down bf16 with entry_name + layer + role + block; the drafter's input side and out norm on mtp.0
  * pulsar.kv: llm_config verbatim, time_step_limit as its two pinned facts, mtp / vision presence; no vision
    tensor in the container
  * an EXL3 build of every dense Linear and the shared expert (dense arm), the routed experts BF16: bytes == the
    pack's [trellis | suh | svh], the pack keyed as exllamav3 writes it (no `language_model.` prefix)
  * a full EXL3 build (L296): the up-only routed experts too, from the pack (expert width 128, an EXL3 multiple):
    every expert-projection's bytes == the pack's, its family exl3m_k4; a pack keyed under the HF names refused
  * the drafter omitted whole -> no mtp shard, pulsar.mtp_present false
  * refusals by name: an unknown tensor, a missing tensor, a wrong shape, a switch (relu2 -> silu), a dt clamp
    (time_step_limit [0.001, inf]), a moe with no mixer before it, an `mlp` block, half a drafter omitted, an
    omitted main-model tensor
  * the LiLiCorr drafter (--drafter lilicorr DIR, L298) on a synthetic ModelOpt W4A16_NVFP4 checkpoint at small dims
    (the real key set and tensor list): the target's MTP omitted -> shard `drafter` after `top`, pulsar.mtp_present
    false, pulsar.drafter "lilicorr"; the exact lilicorr.* kv set; every tensor's role / block / layer / layout /
    dims_ne; an nvfp4 payload == nibbles + the E4M3 scales tiled by a direct loop of the 128x4 formula + scale_2 +
    zeros; refusals: an unknown config key (top level and dflash_config), a pinned switch, an unknown / missing tensor,
    a wrong dtype, the exclude list disagreeing both ways, the target disagreeing, --drafter with the MTP written, the
    drafter recipe without --drafter, --drafter lilicorr on another family (Qwen reads dflash2)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_qwen import bf16, write_st, exl3_parts  # noqa: E402

FAILS: list[str] = []
# sha256 over (file name, sha256(file)) of every emitted file but the side files; fixed when the family landed (L296 N1),
# moved once by L298: pulsar.kv gained `pulsar.drafter` = "" (and its pulsar.kv_arch copy) -- every tensor header and
# payload byte-identical to L296's (graded against origin/dev 8e9d4c13's builder on this checkpoint, bf16 and k4)
GOLDEN = {
    "bf16": "fb82c1c902bf57de861545a4c7cb8f0598b16cb1284af1eb4679fbcce25e9b06",
}
H, V, NQ, NKV, HD = 128, 64, 4, 2, 32
NH, HDM, N, G, CK = 8, 32, 16, 2, 4          # Mamba: heads, head dim (d_inner 256 = 2 x H), state, groups, conv
X, TOPK, FF, LAT, SFF = 4, 2, 64, 128, 128    # experts, top-k, expert width, latent, shared width
BLOCKS = ["mamba", "moe", "mamba", "attention", "moe"]
DI, CD = NH * HDM, NH * HDM + 2 * G * N
LM, BB, MTPP = "language_model.", "language_model.backbone.", "language_model.mtp."
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def config():
    return {
        "architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni", "eos_token_id": 11,
        "vision_config": {"model_type": "radio", "hidden_size": 32}, "tie_word_embeddings": False,
        "llm_config": {
            "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": H, "vocab_size": V,
            "layers_block_type": list(BLOCKS), "mtp_layers_block_type": ["attention", "moe"],
            "num_nextn_predict_layers": 1, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD,
            "mamba_num_heads": NH, "mamba_head_dim": HDM, "ssm_state_size": N, "n_groups": G, "conv_kernel": CK,
            "expand": 2, "chunk_size": 128, "n_routed_experts": X, "num_experts_per_tok": TOPK,
            "moe_intermediate_size": FF, "intermediate_size": FF, "moe_latent_size": LAT,
            "moe_shared_expert_intermediate_size": SFF, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": True, "routed_scaling_factor": 5.0, "layer_norm_epsilon": 1e-5,
            "attention_bias": False, "mlp_bias": False, "use_bias": False, "mamba_proj_bias": False,
            "use_conv_bias": True, "mlp_hidden_act": "relu2", "mamba_hidden_act": "silu", "sliding_window": None,
            "tie_word_embeddings": False, "residual_in_fp32": False, "moe_shared_expert_overlap": False,
            "time_step_limit": [0.0, {"__float__": "Infinity"}], "time_step_min": 0.001, "time_step_max": 0.1,
            "time_step_floor": 0.0001, "max_position_embeddings": 1048576, "eos_token_id": 11, "bos_token_id": 1,
            "pad_token_id": 0, "mamba_ssm_cache_dtype": "float32", "dtype": "bfloat16"},
    }


def make_hf(root, rng, cfg=None, edit=None):
    os.makedirs(root, exist_ok=True)
    json.dump(cfg or config(), open(os.path.join(root, "config.json"), "w"))
    ff = (cfg or config())["llm_config"]["moe_intermediate_size"]
    for f in SIDE[:-1]:
        open(os.path.join(root, f), "w").write(f"{{\"synthetic\": \"{f}\"}}\n")
    t = {}

    def mat(name, *shape, dt="BF16"):
        a = rng.standard_normal(shape).astype(np.float32) * 0.05
        t[name] = (dt, list(shape), bf16(a).tobytes() if dt == "BF16" else a.tobytes())

    def block(p, kind):
        mat(p + "norm.weight", H)
        m = p + "mixer."
        if kind == "mamba":
            mat(m + "in_proj.weight", DI + CD + NH, H); mat(m + "conv1d.weight", CD, 1, CK); mat(m + "conv1d.bias", CD)
            mat(m + "dt_bias", NH); mat(m + "A_log", NH); mat(m + "D", NH); mat(m + "norm.weight", DI)
            mat(m + "out_proj.weight", H, DI)
        elif kind == "attention":
            mat(m + "q_proj.weight", NQ * HD, H); mat(m + "k_proj.weight", NKV * HD, H)
            mat(m + "v_proj.weight", NKV * HD, H); mat(m + "o_proj.weight", H, NQ * HD)
        else:
            mat(m + "gate.weight", X, H); mat(m + "gate.e_score_correction_bias", X, dt="F32")
            mat(m + "fc1_latent_proj.weight", LAT, H); mat(m + "fc2_latent_proj.weight", H, LAT)
            mat(m + "shared_experts.up_proj.weight", SFF, H); mat(m + "shared_experts.down_proj.weight", H, SFF)
            for e in range(X):
                mat(m + f"experts.{e}.up_proj.weight", ff, LAT); mat(m + f"experts.{e}.down_proj.weight", LAT, ff)
    mat(BB + "embeddings.weight", V, H); mat(BB + "norm_f.weight", H); mat(LM + "lm_head.weight", V, H)
    for b, kind in enumerate(BLOCKS):
        block(f"{BB}layers.{b}.", kind)
    block(f"{MTPP}layers.0.", "attention"); block(f"{MTPP}layers.1.", "moe")
    mat(MTPP + "layers.0.enorm.weight", H); mat(MTPP + "layers.0.hnorm.weight", H)
    mat(MTPP + "layers.0.eh_proj.weight", H, 2 * H); mat(MTPP + "layers.1.final_layernorm.weight", H)
    mat("vision_model.radio_model.model.blocks.0.attn.qkv.weight", 96, 32); mat("mlp1.0.weight", 128)
    mat("vision_projector.vision_final_layernorm.weight", 32)
    if edit:
        edit(t)
    names = sorted(t)
    half = len(names) // 2
    wm = {}
    for fn, part in (("model-00001-of-00002.safetensors", names[:half]), ("model-00002-of-00002.safetensors", names[half:])):
        write_st(os.path.join(root, fn), {n: t[n] for n in part})
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    return t


DENSE = ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "fc1_latent_proj", "fc2_latent_proj",
         "shared_experts.up_proj", "shared_experts.down_proj")


def make_exl3(root, hf_t, rng, K=4, experts=False, prefixed=False):
    """The EXL3 pack of the trunk's dense Linears and shared experts -- and its routed experts when `experts` -- keyed
    as exllamav3's NemotronH Omni adapter writes them: the HF names minus `language_model.` and `.weight` (`prefixed`:
    under the HF names, which the builder must refuse)."""
    t = {}
    for name, (_dt, shape, _b) in hf_t.items():
        routed = experts and ".mixer.experts." in name
        if name.startswith(BB + "layers.") and (routed or any(name.endswith(f"mixer.{d}.weight") for d in DENSE)):
            out, inp = shape
            key = name[:-len(".weight")] if prefixed else name[len(LM):-len(".weight")]
            for s, v in exl3_parts(rng, inp, (out + 127) // 128 * 128, K).items():
                t[f"{key}.{s}"] = v
    os.makedirs(root, exist_ok=True)
    write_st(os.path.join(root, "model-00001-of-00001.safetensors"), t)
    return t


# ---------------------------------------------------------------------------
# the LiLiCorr drafter (L298): small dims, the real key set and tensor list
# ---------------------------------------------------------------------------
DL, DNQ, DNKV, DHD, DI_, DTAPS, DGRP = 5, 4, 2, 32, 128, 2, 16      # layers, heads, kv heads, head dim, mlp, conv
LL, LNH, LTOPK, LBLK, LF, LNL = 64, 2, 2, 4, 32, 2                  # lattice hidden, heads, top-k, block, factor, layers
DTL = [1, 3]                                                        # target_layer_ids (HF blocks of the 5-block target)
D_EXCLUDE = ["*attention_conv*", "*mlp_conv*", "*self_attn*", "*norm", "*lilicorr.anchor_out_head",
             "*lilicorr.factor_input_proj", "*lilicorr.feature_mlp*", "*lilicorr.in_head", "*lilicorr.out_head",
             "*qkv_proj"]


def drafter_config():
    return {
        "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "block_size": LBLK,
        "dflash_config": {"mask_token_id": 3, "target_layer_ids": list(DTL), "causal": False,
                          "projector_type": "lilicorr", "lilicorr_enabled": True, "lilicorr_candidate_topk": LTOPK,
                          "lilicorr_hidden_size": LL, "lilicorr_num_layers": LNL, "lilicorr_num_heads": LNH,
                          "lilicorr_mlp_ratio": 2.0, "lilicorr_factor_dim": LF, "lilicorr_vector_eps": 0.0001,
                          "lilicorr_logit_scale": 8.0, "conv_kernel_size": DTAPS, "conv_group_size": DGRP,
                          "use_swa": True, "swa_window_size": 64},
        "hidden_size": H, "num_hidden_layers": DL, "num_attention_heads": DNQ, "num_key_value_heads": DNKV,
        "head_dim": DHD, "intermediate_size": DI_, "hidden_act": "silu", "rms_norm_eps": 1e-05, "vocab_size": V,
        "max_position_embeddings": 4096, "initializer_range": 0.02, "attention_bias": False, "attention_dropout": 0.0,
        "rope_theta": 1000000.0, "rope_scaling": None, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "num_target_layers": len(BLOCKS), "layer_types": ["sliding_attention"] * DL, "sliding_window": 64,
        "use_sliding_window": True, "max_window_layers": 0, "has_own_lm_head": True, "draft_vocab_size": V,
        "quantization_config": {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                                "exclude_modules": list(D_EXCLUDE), "quant_method": "modelopt_fp4",
                                "ignore": list(D_EXCLUDE)},
    }


def drafter_shapes():
    """{HF name: (shape, nvfp4?)}: the LOGICAL [out, in] of a quantized Linear."""
    kp = [2 * DTAPS * H // DGRP, H]
    s = {"fc.weight": ([H, len(DTL) * H], True), "hidden_norm.weight": ([H], False), "norm.weight": ([H], False),
         "lm_head.weight": ([V, H], True)}
    for i in range(DL):
        p = f"layers.{i}."
        s.update({p + "input_layernorm.weight": ([H], False), p + "post_attention_layernorm.weight": ([H], False),
                  p + "self_attn.q_proj.weight": ([DNQ * DHD, H], False),
                  p + "self_attn.k_proj.weight": ([DNKV * DHD, H], False),
                  p + "self_attn.v_proj.weight": ([DNKV * DHD, H], False),
                  p + "self_attn.o_proj.weight": ([H, DNQ * DHD], False),
                  p + "self_attn.q_norm.weight": ([DHD], False), p + "self_attn.k_norm.weight": ([DHD], False),
                  p + "attention_conv.base_kernel": ([2, DTAPS, H], False),
                  p + "attention_conv.kernel_projection.weight": (kp, False),
                  p + "mlp_conv.base_kernel": ([2, DTAPS, H], False),
                  p + "mlp_conv.kernel_projection.weight": (kp, False),
                  p + "mlp.gate_proj.weight": ([DI_, H], True), p + "mlp.up_proj.weight": ([DI_, H], True),
                  p + "mlp.down_proj.weight": ([H, DI_], True)})
    q = "lilicorr."
    for n in ("token_proj", "pass_hidden_proj", "context_proj"):
        s[q + n + ".weight"], s[q + n + ".bias"] = ([LL, H], True), ([LL], False)
    s.update({q + "feature_norm.weight": ([5], False), q + "feature_norm.bias": ([5], False),
              q + "feature_mlp.up_proj.weight": ([LL, 5], False), q + "feature_mlp.up_proj.bias": ([LL], False),
              q + "feature_mlp.down_proj.weight": ([LL, LL], False), q + "feature_mlp.down_proj.bias": ([LL], False),
              q + "slot_embedding": ([1, 1, LBLK - 1, 1, LL], False), q + "rank_embedding": ([1, 1, 1, LTOPK, LL], False),
              q + "relative_slot_bias": ([LNH, 2 * LBLK - 1], False), q + "same_slot_bias": ([LNH], False),
              q + "output_norm.weight": ([LL], False), q + "anchor_norm.weight": ([LL], False),
              q + "factor_input_proj.weight": ([LL, 3 * LL], False), q + "factor_input_proj.bias": ([LL], False)})
    for n in ("out_head", "in_head", "anchor_out_head"):
        s[q + n + ".weight"], s[q + n + ".bias"] = ([LF, LL], False), ([LF], False)
    for i in range(LNL):
        p = f"{q}layers.{i}."
        s.update({p + "attn_norm.weight": ([LL], False), p + "attn.in_proj_weight": ([3 * LL, LL], False),
                  p + "attn.in_proj_bias": ([3 * LL], False), p + "attn.out_proj.weight": ([LL, LL], True),
                  p + "attn.out_proj.bias": ([LL], False), p + "mlp_norm.weight": ([LL], False),
                  p + "mlp.up_proj.weight": ([2 * LL, LL], True), p + "mlp.up_proj.bias": ([2 * LL], False),
                  p + "mlp.down_proj.weight": ([LL, 2 * LL], True), p + "mlp.down_proj.bias": ([LL], False)})
    return s


def make_drafter(root, rng, cfg=None, edit=None):
    """A one-file ModelOpt W4A16_NVFP4 checkpoint (config.json + model.safetensors, no index -- as shipped)."""
    os.makedirs(root, exist_ok=True)
    json.dump(cfg or drafter_config(), open(os.path.join(root, "config.json"), "w"))
    t = {}
    for n, (shape, q) in drafter_shapes().items():
        if q:
            out, inp = shape
            t[n] = ("U8", [out, inp // 2], rng.integers(0, 256, out * inp // 2, dtype=np.uint8).tobytes())
            t[n + "_scale"] = ("F8_E4M3", [out, inp // 16], rng.integers(0, 0x7F, out * inp // 16, dtype=np.uint8).tobytes())
            t[n + "_scale_2"] = ("F32", [], np.float32(rng.uniform(1e-4, 1e-2)).tobytes())
        else:
            a = rng.standard_normal(shape).astype(np.float32) * 0.05
            t[n] = ("BF16", list(shape), bf16(a).tobytes())
    if edit:
        edit(t)
    write_st(os.path.join(root, "model.safetensors"), t)
    return t


def nvfp4_ref(w, sc, s2, out, inp):
    """The nvfp4 entry by a direct loop over the engine's 128x4 tile formula (not producers.swizzle_sf)."""
    kb_n, rows_pad = inp // 16, (out + 127) // 128 * 128
    plane = bytearray(rows_pad * kb_n)
    for r in range(out):
        for kb in range(kb_n):
            plane[((r // 128) * (inp // 64) + kb // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + kb % 4] = \
                sc[r * kb_n + kb]
    return w + bytes(plane) + s2 + bytes(28)


DRAFTER_KV = (["block_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
               "head_dim", "intermediate_size", "hidden_act", "rms_norm_eps", "vocab_size", "max_position_embeddings",
               "attention_bias", "rope_theta", "tie_word_embeddings", "num_target_layers", "layer_types",
               "sliding_window", "use_sliding_window", "max_window_layers", "has_own_lm_head", "draft_vocab_size"]
              + ["dflash." + k for k in drafter_config()["dflash_config"]])


def drafter_tests(tmp, hf, t):
    import drafters as DR
    import nemotron_h as NH
    dd = os.path.join(tmp, "lilicorr")
    dt = make_drafter(dd, np.random.default_rng(298))
    rl = os.path.join(HERE, "format-maps", "nemotron_h-k4-lilicorr.json")
    rows = [
        *[{"role": r, "block": "layers", "format": "native"} for r in ("dense", "norm", "other", "shared_expert",
                                                                        "expert_up", "expert_down", "router")],
        ["language_model.backbone.norm_f.weight", "native"],
        *[{"role": r, "format": "native"} for r in ("embed", "head")],
        ["language_model.mtp.*", "omit"],
        {"role": "drafter_quantized", "format": "nvfp4"}, {"role": "drafter_native", "format": "native"}]
    rd = recipe(tmp, "lilicorr", rows)
    args = ["--hf", hf, "--recipe", rd, "--drafter", "lilicorr", dd]
    o = build_grade(tmp, "lilicorr", args, 1 + 3 + 1 + 1)
    shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
    hv, _ = header(os.path.join(o, shards[0]))
    kvl = json.loads(hv["__metadata__"]["pulsar.kv"])
    kv = {e["key"]: e["value"] for e in kvl}
    cfg = drafter_config()
    lili = [e["key"] for e in kvl if e["key"].startswith("lilicorr.")]
    check(kv.get("pulsar.drafter") == "lilicorr" and kv.get("pulsar.mtp_present") is False
          and kv.get("pulsar.source.drafter") == "lilicorr"
          and lili == ["lilicorr." + k for k in DRAFTER_KV]
          and kv["lilicorr.dflash.target_layer_ids"] == {"__array__": "u32", "n": 2, "v": DTL}
          and kv["lilicorr.dflash.causal"] is False and kv["lilicorr.hidden_act"] == "silu"
          and kv["lilicorr.dflash.lilicorr_mlp_ratio"] == 2.0 and kv["lilicorr.num_target_layers"] == len(BLOCKS)
          and kv["lilicorr.layer_types"]["v"] == ["sliding_attention"] * DL
          and not any(k.startswith("lilicorr.quantization") or k == "lilicorr.dflash_config.causal" for k in kv),
          "pulsar.kv: pulsar.drafter lilicorr, mtp_present false, exactly the written lilicorr.* keys in order "
          "(dflash_config as lilicorr.dflash.*; quantization_config consumed)")
    last, _ = header(os.path.join(o, shards[-1]))
    md = last["__metadata__"]
    decl = json.loads(md["pulsar.tensors"])
    shapes = drafter_shapes()
    want = {"drafter." + n for n in shapes}
    ok_rec = md["pulsar.shard_key"] == "drafter" and set(decl) == want and set(last) - {"__metadata__"} == want
    bad = []
    for n, (shape, q) in shapes.items():
        d = decl.get("drafter." + n, {})
        if n.startswith("lilicorr."):
            rest = n[len("lilicorr."):]
            if rest.startswith("layers."):
                _l, i, sfx = rest.split(".", 2)
                role, block, layer = DR.LILI_HEAD_LAYER[sfx][0], "lilicorr", int(i)
            else:
                role, block, layer = DR.LILI_HEAD[rest][0], "lilicorr", None
        elif n.startswith("layers."):
            _l, i, sfx = n.split(".", 2)
            role, block, layer = DR.LILI_LAYER[sfx][0], "drafter", int(i)
        else:
            role, block, layer = DR.LILI_TOP[n][0], "drafter", None
        exp = {"role": role, "block": block, "layout": "nvfp4" if q else "bf16", "dims_ne": list(reversed(shape)),
               "gguf_name": "drafter." + n, **({"layer": layer} if layer is not None else {})}
        if d != exp:
            bad.append((n, d, exp))
    if bad:
        print(f"    {bad[:2]}")
    check(ok_rec and not bad, f"shard `drafter` last: every one of {len(want)} tensors (scales folded) declared with "
                              f"its role / block / layer, nvfp4 or bf16, dims_ne [in, out]")
    check(decl["drafter.fc.weight"]["role"] == "fc" and "layer" not in decl["drafter.fc.weight"]
          and decl["drafter.norm.weight"]["role"] == "out_norm" and decl["drafter.lm_head.weight"]["role"] == "head"
          and decl["drafter.layers.4.mlp_conv.kernel_projection.weight"] == {
              "role": "mlp_conv_proj", "block": "drafter", "layer": 4, "layout": "bf16", "dims_ne": [H, 32],
              "gguf_name": "drafter.layers.4.mlp_conv.kernel_projection.weight"}
          and decl["drafter.lilicorr.layers.1.attn.in_proj_bias"]["role"] == "attn_in_bias"
          and decl["drafter.lilicorr.same_slot_bias"]["block"] == "lilicorr",
          "spot records: fc / out_norm / head (no layer), mlp_conv_proj layer 4, lattice attn_in_bias layer 1")
    pay_ok = True
    for n in ("fc.weight", "lm_head.weight", "layers.2.mlp.down_proj.weight", "lilicorr.layers.1.attn.out_proj.weight"):
        out, inp = shapes[n][0]
        exp = nvfp4_ref(dt[n][2], dt[n + "_scale"][2], dt[n + "_scale_2"][2], out, inp)
        pay_ok &= entry_bytes(o, shards[-1], "drafter." + n) == exp and len(exp) == \
            out * inp // 2 + (out + 127) // 128 * 128 * inp // 16 + 32
    check(pay_ok, "nvfp4 payload == nibbles + E4M3 scales tiled by a direct loop of the 128x4 formula (rows padded "
                  "to 128 with zero scales) + scale_2 + 28 zero bytes")
    check(all(entry_bytes(o, shards[-1], "drafter." + n) == dt[n][2] for n in
              ("norm.weight", "layers.0.attention_conv.base_kernel", "lilicorr.rank_embedding",
               "lilicorr.layers.0.attn.in_proj_weight")),
          "native drafter tensors == their source bytes")
    big = {}
    for h in shards[:-1]:
        hh, _ = header(os.path.join(o, h))
        big.update({k: v for k, v in hh.items() if k != "__metadata__"})
    check(not any(k.startswith(MTPP) for k in big), "the target's MTP module written nowhere")

    print("drafter refusals:")
    cases = []

    def cfg_case(label, f, msg):
        c = drafter_config()
        f(c)
        cases.append((label, c, None, msg))
    cfg_case("an unknown config key", lambda c: c.__setitem__("rope_type", "yarn"), "'rope_type' is neither written")
    cfg_case("an unknown dflash_config key", lambda c: c["dflash_config"].__setitem__("num_anchors", 2),
             "'num_anchors' is not one")
    cfg_case("a causal drafter", lambda c: c["dflash_config"].__setitem__("causal", True), "causal = True")
    cfg_case("another quant_algo", lambda c: c["quantization_config"].__setitem__("quant_algo", "NVFP4"),
             "quant_algo = 'NVFP4'")
    cfg_case("target_layer_ids past the target", lambda c: c["dflash_config"].__setitem__("target_layer_ids", [1, 5]),
             "target_layer_ids")
    cfg_case("num_target_layers != the target's blocks", lambda c: c.__setitem__("num_target_layers", 88),
             "num_target_layers 88")
    cfg_case("a hidden size != the target's", lambda c: c.__setitem__("hidden_size", 256), "the target's hidden")

    def drop_excl(c):
        c["quantization_config"]["exclude_modules"].remove("*self_attn*")
        c["quantization_config"]["ignore"].remove("*self_attn*")
    cfg_case("exclude list misses a BF16 Linear", drop_excl, "no exclude pattern names it")

    def add_excl(c):
        c["quantization_config"]["exclude_modules"].append("*mlp.gate_proj")
        c["quantization_config"]["ignore"].append("*mlp.gate_proj")
    cfg_case("exclude list names an NVFP4 Linear", add_excl, "excluded by '*mlp.gate_proj'")
    cfg_case("exclude_modules != ignore", lambda c: c["quantization_config"]["ignore"].pop(), "exclude_modules != ignore")
    cases += [
        ("an unknown tensor", None, lambda t: t.__setitem__("lilicorr.extra_bias", t["lilicorr.same_slot_bias"]),
         "does not map"),
        ("a missing tensor", None, lambda t: t.pop("layers.3.self_attn.k_norm.weight"), "missing from"),
        ("a missing weight_scale_2", None, lambda t: t.pop("lm_head.weight_scale_2"), "missing from"),
        ("a quantized Linear stored BF16", None,
         lambda t: t.__setitem__("layers.1.mlp.up_proj.weight", ("BF16", [DI_, H], bytes(DI_ * H * 2))),
         "disagree with the declaration"),
        ("an F16 norm", None, lambda t: t.__setitem__("norm.weight", ("F16", [H], t["norm.weight"][2])),
         "disagree with the declaration"),
    ]
    for i, (label, c, edit, msg) in enumerate(cases):
        d = os.path.join(tmp, f"bad-drafter-{i}")
        make_drafter(d, np.random.default_rng(1), cfg=c, edit=edit)
        ok, _ = run("plan", "--hf", hf, "--recipe", rd, "--drafter", "lilicorr", d, expect_fail=True, contains=msg)
        check(ok, f"{label} -> refused")
    ok, _ = run("plan", "--hf", hf, "--recipe", rd, "--drafter", "medusa", dd, expect_fail=True,
                contains="the Nemotron-H build reads ['lilicorr']")
    check(ok, "--drafter medusa -> refused by name")
    rbf = os.path.join(HERE, "format-maps", "nemotron_h-bf16.json")
    rmtp = recipe(tmp, "lilicorr-mtp", json.load(open(rbf))["rows"] + rows[-2:])
    ok, _ = run("plan", "--hf", hf, "--recipe", rmtp, "--drafter", "lilicorr", dd, expect_fail=True,
                contains="writes the MTP module too")
    check(ok, "--drafter with a recipe that writes the MTP module -> refused (one drafter)")
    ok, _ = run("plan", "--hf", hf, "--recipe", rd, expect_fail=True, contains="match no checkpoint tensor")
    check(ok, "the drafter recipe without --drafter -> refused (its drafter rows are dead)")
    ok, _ = run("plan", "--hf", hf, "--recipe", rbf, "--drafter", "lilicorr", dd, expect_fail=True,
                contains="no row of")
    check(ok, "--drafter with a recipe that has no drafter rows -> refused")
    import test_qwen
    qd = os.path.join(tmp, "qwen")
    os.makedirs(qd)
    test_qwen.make_hf(qd, np.random.default_rng(3))
    ok, _ = run("plan", "--hf", qd, "--drafter", "lilicorr", dd, expect_fail=True,
                contains="the qwen4_exp build reads ['dflash2']")
    check(ok, "--drafter lilicorr on another family (qwen4_exp, whose drafter is dflash2: L299) -> refused by name")
    r = json.load(open(rl))
    check(r["recipe"] == "nemotron_h-k4-lilicorr" and ["language_model.mtp.*", "omit"] in r["rows"]
          and {"role": "drafter_quantized", "format": "nvfp4"} in r["rows"],
          "format-maps/nemotron_h-k4-lilicorr.json: MTP omitted, drafter rows")


def run(*args, expect_fail=False, contains=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py")] + list(args), capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if contains is not None:
        ok = ok and contains in out
    if not ok:
        print("    " + "\n    ".join(out.strip().splitlines()[-4:]))
    return ok, out


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    return h, 8 + n


def entry_bytes(d, f, name):
    h, base = header(os.path.join(d, f))
    o0, o1 = h[name]["data_offsets"]
    with open(os.path.join(d, f), "rb") as g:
        g.seek(base + o0)
        return g.read(o1 - o0)


def shard_digest(d):
    h = hashlib.sha256()
    for f in sorted(x for x in os.listdir(d) if x not in SIDE):
        h.update(f.encode() + b"\0" + hashlib.sha256(open(os.path.join(d, f), "rb").read()).digest())
    return h.hexdigest()


def recipe(tmp, label, rows):
    p = os.path.join(tmp, f"recipe-{label}.json")
    json.dump({"recipe": f"test-{label}", "model_type": "nemotron_h_omni", "rows": rows}, open(p, "w"))
    return p


def build_grade(tmp, label, args, n_files):
    o1, o2 = os.path.join(tmp, f"{label}-1"), os.path.join(tmp, f"{label}-2")
    ok, out = run("plan", *args)
    check(ok, f"{label}: plan")
    if ok:
        print("    " + "\n    ".join(out.strip().splitlines()[:1]))
    ok1, _ = run("emit", *args, "--out", o1, "--all")
    ok2, _ = run("emit", *args, "--out", o2, "--all")
    files = sorted(os.listdir(o1)) if ok1 else []
    same = ok1 and ok2 and files == sorted(os.listdir(o2)) and all(
        open(os.path.join(o1, f), "rb").read() == open(os.path.join(o2, f), "rb").read() for f in files)
    check(same and len(files) == n_files + 1 + len(SIDE),
          f"{label}: emit --all twice, byte-identical ({len(files)} files: shards, index, side files)")
    ok, out = run("verify", *args, "--out", o1, "--all")
    check(ok and "failing: 0" in out and out.count("side file ") == len(SIDE) and "-> FAIL" not in out,
          f"{label}: verify --all PASS (the side files byte-equal)")
    ok, out = run("audit", "--out", o1)
    check(ok and "AUDIT PASS" in out, f"{label}: audit PASS")
    return o1


def main():
    rng = np.random.default_rng(296)
    tmp = tempfile.mkdtemp(prefix="nemotron-builder-test-")
    try:
        hf = os.path.join(tmp, "hf")
        t = make_hf(hf, rng)
        rbf = os.path.join(HERE, "format-maps", "nemotron_h-bf16.json")
        n_files = 1 + 3 + 1 + 1

        print("BF16 reference build:")
        o = build_grade(tmp, "bf16", ["--hf", hf, "--recipe", rbf], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["bf16"], f"GOLDEN bf16: {got}")
        shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
        hv, _ = header(os.path.join(o, shards[0]))
        kv = {e["key"]: e["value"] for e in json.loads(hv["__metadata__"]["pulsar.kv"])}
        check(kv["general.architecture"] == "nemotron_h" and kv["nemotron_h.layers_block_type"]["v"] == BLOCKS
              and kv["nemotron_h.mtp_layers_block_type"]["v"] == ["attention", "moe"]
              and kv["nemotron_h.time_step_limit.min"] == 0.0 and kv["nemotron_h.time_step_limit.max_inf"] is True
              and "nemotron_h.time_step_limit" not in kv and kv["nemotron_h.moe_latent_size"] == LAT
              and kv["pulsar.mtp_present"] is True and kv["pulsar.vision_present"] is False
              and kv["pulsar.drafter"] == "" and not any(k.startswith("lilicorr.") for k in kv)
              and "pulsar.source.drafter" not in kv
              and not any(k.startswith("nemotron_h.vision") for k in kv)
              and [k for k in hv if k != "__metadata__"] == [],
              "pulsar.kv: llm_config verbatim, time_step_limit as min 0 + max_inf, presence; the primary shard holds no "
              "tensor")
        allnames = set()
        decl = {}
        fams = {}
        for f in shards:
            h, _ = header(os.path.join(o, f))
            allnames |= set(h) - {"__metadata__"}
            decl.update(json.loads(h["__metadata__"]["pulsar.tensors"]))
            fams[h["__metadata__"]["pulsar.shard_key"]] = json.loads(h["__metadata__"]["pulsar.experts"])
        check(not any(n.startswith(("vision_model.", "vision_projector.", "mlp1.")) for n in allnames),
              "no vision tensor in the container (consumed by name)")
        m0, m4 = BB + "layers.0.mixer.", BB + "layers.4.mixer."
        check(decl[m0 + "conv1d.weight"]["dims_ne"] == [CK, CD] and decl[m0 + "conv1d.weight"]["role"] == "ssm_conv"
              and decl[BB + "layers.0.norm.weight"]["role"] == "ssm_pre_norm"
              and decl[BB + "layers.1.norm.weight"] == {"layout": "bf16", "dims_ne": [H], "role": "ffn_norm",
                                                        "gguf_name": BB + "layers.1.norm.weight", "block": "layers",
                                                        "layer": 0}
              and decl[BB + "layers.2.mixer.in_proj.weight"]["layer"] == 1
              and decl[BB + "layers.3.mixer.q_proj.weight"]["role"] == "attn_q"
              and decl[BB + "layers.3.norm.weight"]["role"] == "attn_norm"
              and decl[m4 + "gate.e_score_correction_bias"]["layout"] == "f32"
              and decl[m4 + "gate.e_score_correction_bias"]["role"] == "router_select_bias"
              and decl[m4 + "gate.e_score_correction_bias"]["layer"] == 2
              and decl[m4 + "fc1_latent_proj.weight"]["role"] == "latent_in"
              and decl[m4 + "shared_experts.down_proj.weight"]["index"] == 0
              and decl[m4 + "shared_experts.down_proj.weight"]["role"] == "shared_down",
              "records: role / block / PULSAR layer (blocks 0+1 -> 0, 2 -> 1, 3+4 -> 2); conv declared [K, C]; the "
              "shared expert index 0")
        f2 = fams["layers.2"]
        check(sorted(x["part"] for x in f2) == ["down_proj", "up_proj"]
              and all(x["layout"] == "bf16" and x["layer"] == 2 and x["n_experts"] == X and x["block"] == "layers"
                      and x["entry_name"] == m4 + "experts.{e}." + x["part"] + ".weight"
                      and x["role"] == "expert_" + x["part"].split("_")[0]
                      and x["dims_per_expert_ne"] == ([LAT, FF] if x["part"] == "up_proj" else [FF, LAT]) for x in f2)
              and fams["layers.1"] == [],
              "routed families: up / down bf16 [latent <-> ff], entry_name + layer + role + block; layer 1 has none")
        f_l2 = [f for f in shards if m4 + "experts.3.up_proj.weight" in header(os.path.join(o, f))[0]]
        check(bool(f_l2) and entry_bytes(o, f_l2[0], m4 + "experts.3.up_proj.weight") == t[m4 + "experts.3.up_proj.weight"][2],
              "routed expert 3 up == its HF tensor's bytes")
        md = MTPP + "layers."
        check(decl[md + "0.eh_proj.weight"]["role"] == "draft_in_proj" and decl[md + "0.eh_proj.weight"]["block"] == "mtp"
              and decl[md + "0.eh_proj.weight"]["dims_ne"] == [2 * H, H]
              and decl[md + "1.final_layernorm.weight"]["role"] == "draft_out_norm"
              and decl[md + "1.norm.weight"]["role"] == "ffn_norm" and decl[md + "1.norm.weight"]["layer"] == 0
              and header(os.path.join(o, shards[-1]))[0]["__metadata__"]["pulsar.shard_key"] == "mtp.0",
              "drafter: mtp.0 last; eh_proj draft_in_proj [2H -> H]; final_layernorm draft_out_norm; its layer 0")

        print("EXL3 dense build (dense arm K4; routed BF16):")
        ex = os.path.join(tmp, "exl3")
        ext = make_exl3(ex, t, np.random.default_rng(7))
        rk4 = recipe(tmp, "k4", [
            {"role": "dense", "block": "layers", "format": "exl3m_k4"}, {"role": "dense", "block": "mtp", "format": "native"},
            {"role": "shared_expert", "block": "layers", "format": "exl3m_k4"},
            {"role": "shared_expert", "block": "mtp", "format": "native"},
            *[{"role": r, "format": "native"} for r in ("expert_up", "expert_down", "router", "norm", "embed", "head",
                                                       "other")]])
        o4 = build_grade(tmp, "k4", ["--hf", hf, "--exl3", ex, "--recipe", rk4], n_files)
        key = BB + "layers.2.mixer.in_proj"
        f_l1 = [f for f in sorted(os.listdir(o4)) if f.endswith(".safetensors")
                and key + ".weight" in header(os.path.join(o4, f))[0]]
        d1 = json.loads(header(os.path.join(o4, f_l1[0]))[0]["__metadata__"]["pulsar.tensors"]) if f_l1 else {}
        check(bool(f_l1) and d1[key + ".weight"]["layout"] == "exl3m_k4" and d1[key + ".weight"]["dims_ne"] == [H, DI + CD + NH]
              and entry_bytes(o4, f_l1[0], key + ".weight") == b"".join(ext[f"{key[len(LM):]}.{s}"][2]
                                                                        for s in ("trellis", "suh", "svh")),
              "ssm_in exl3m_k4 at its logical width, bytes == the pack's [trellis | suh | svh]")

        print("full EXL3 build (L296: the up-only routed experts from the pack too; expert width 128):")
        c128 = config()
        c128["llm_config"]["moe_intermediate_size"] = c128["llm_config"]["intermediate_size"] = 128
        hf128 = os.path.join(tmp, "hf128")
        t128 = make_hf(hf128, np.random.default_rng(11), cfg=c128)
        ex128 = os.path.join(tmp, "exl3-128")
        ext128 = make_exl3(ex128, t128, np.random.default_rng(12), experts=True)
        rfull = recipe(tmp, "full", [
            *[{"role": r, "block": "layers", "format": "exl3m_k4"} for r in ("dense", "shared_expert", "expert_up",
                                                                              "expert_down")],
            *[{"role": r, "block": "mtp", "format": "native"} for r in ("dense", "shared_expert", "expert_up",
                                                                         "expert_down")],
            *[{"role": r, "format": "native"} for r in ("router", "norm", "embed", "head", "other")]])
        of = build_grade(tmp, "full", ["--hf", hf128, "--exl3", ex128, "--recipe", rfull], n_files)
        ek = BB + "layers.4.mixer.experts.3.up_proj"
        ff = [f for f in sorted(os.listdir(of)) if f.endswith(".safetensors") and ek + ".weight" in header(os.path.join(of, f))[0]]
        fam = [x for x in json.loads(header(os.path.join(of, ff[0]))[0]["__metadata__"]["pulsar.experts"])
               if x["part"] == "up_proj"] if ff else []
        check(bool(ff) and fam and fam[0]["layout"] == "exl3m_k4" and fam[0]["n_experts"] == X
              and entry_bytes(of, ff[0], ek + ".weight") == b"".join(ext128[f"{ek[len(LM):]}.{s}"][2]
                                                                      for s in ("trellis", "suh", "svh")),
              "routed up expert 3 exl3m_k4, bytes == the pack's [trellis | suh | svh]; its family exl3m_k4 x 4 experts")
        exp = os.path.join(tmp, "exl3-prefixed")
        make_exl3(exp, t128, np.random.default_rng(12), experts=True, prefixed=True)
        ok, _ = run("plan", "--hf", hf128, "--exl3", exp, "--recipe", rfull, expect_fail=True,
                    contains="missing from the EXL3 checkpoint")
        check(ok, "a pack keyed under the HF names (language_model.) -> refused (not the exllamav3 layout)")

        print("drafter omitted:")
        rom = recipe(tmp, "omit", [
            *[{"role": r, "block": "layers", "format": "native"} for r in ("dense", "norm", "other", "shared_expert",
                                                                            "expert_up", "expert_down", "router")],
            *[{"role": r, "block": "mtp", "format": "omit"} for r in ("dense", "norm", "shared_expert", "expert_up",
                                                                       "expert_down", "router")],
            {"name": BB + "norm_f.weight", "format": "native"},
            *[{"role": r, "format": "native"} for r in ("embed", "head")]])
        oo = os.path.join(tmp, "omit")
        ok, _ = run("emit", "--hf", hf, "--recipe", rom, "--out", oo, "--all")
        files = sorted(f for f in os.listdir(oo) if f.endswith(".safetensors")) if ok else []
        kvo = {e["key"]: e["value"] for e in json.loads(header(os.path.join(oo, files[0]))[0]["__metadata__"]
                                                         ["pulsar.kv"])} if files else {}
        check(ok and len(files) == 1 + 3 + 1 and kvo.get("pulsar.mtp_present") is False,
              "no mtp shard, pulsar.mtp_present false")
        ok, out = run("verify", "--hf", hf, "--recipe", rom, "--out", oo, "--all")
        check(ok and "failing: 0" in out, "verify PASS (drafter omitted)")

        print("LiLiCorr drafter (--drafter lilicorr, L298):")
        drafter_tests(tmp, hf, t)

        print("refusals:")
        cases = [
            ("an unknown tensor", None, lambda t: t.__setitem__(BB + "layers.0.mixer.extra.weight", t[BB + "norm_f.weight"]),
             "does not map"),
            ("a missing tensor", None, lambda t: t.pop(BB + "layers.2.mixer.D"), "missing from"),
            ("a missing expert", None, lambda t: t.pop(BB + "layers.4.mixer.experts.3.down_proj.weight"), "missing from"),
            ("a wrong shape", None, lambda t: t.__setitem__(BB + "layers.3.mixer.k_proj.weight",
                                                            t[BB + "layers.3.mixer.q_proj.weight"]),
             "disagree with the declared shape"),
        ]
        for label, key, val, msg in (("relu2 -> silu experts", "mlp_hidden_act", "silu", "assumes 'relu2'"),
                                     ("a dt clamp", "time_step_limit", [0.001, {"__float__": "Infinity"}], "[0, inf]"),
                                     ("a moe with no mixer before it", "layers_block_type",
                                      ["moe", "mamba", "moe", "attention", "moe"], "no mixer before it"),
                                     ("an mlp block", "layers_block_type", ["mamba", "mlp", "mamba", "attention", "moe"],
                                      "not a block kind"),
                                     ("a second shared expert", "n_shared_experts", 2, "assumes 1")):
            c = config()
            c["llm_config"][key] = val
            cases.append((label, c, None, msg))
        for i, (label, cfg, edit, msg) in enumerate(cases):
            d = os.path.join(tmp, f"bad-{i}")
            make_hf(d, np.random.default_rng(1), cfg=copy.deepcopy(cfg) if cfg else None, edit=edit)
            ok, _ = run("plan", "--hf", d, "--recipe", rbf, expect_fail=True, contains=msg)
            check(ok, f"{label} -> refused")
        r = json.load(open(rbf))
        hp = recipe(tmp, "half-mtp", [x for x in r["rows"] if x["role"] != "router"] + [
            {"role": "router", "block": "layers", "format": "native"}, {"role": "router", "block": "mtp", "format": "omit"}])
        ok, _ = run("plan", "--hf", hf, "--recipe", hp, expect_fail=True, contains="all of it or none")
        check(ok, "half the drafter omitted -> refused")
        rr = recipe(tmp, "omit-router", [x for x in r["rows"] if x["role"] != "router"] + [
            {"role": "router", "block": "layers", "format": "omit"}, {"role": "router", "block": "mtp", "format": "native"}])
        ok, _ = run("plan", "--hf", hf, "--recipe", rr, expect_fail=True, contains="the Nemotron-H loader binds")
        check(ok, "an omitted main-model tensor (the router) -> refused")
        print("side files:")
        bare = os.path.join(tmp, "hf-bare")
        shutil.copytree(hf, bare, ignore=shutil.ignore_patterns("tokenizer.json"))
        ok, _ = run("emit", "--hf", bare, "--recipe", rbf, "--out", os.path.join(tmp, "bare"), "--all",
                    expect_fail=True, contains="--tokenizer")
        check(ok and not os.path.exists(os.path.join(tmp, "bare")), "a checkpoint without tokenizer.json -> refused "
                                                                     "naming --tokenizer, before any shard is written")
    finally:
        shutil.rmtree(tmp)
    print(f"test_nemotron_h: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
