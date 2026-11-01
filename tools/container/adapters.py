"""adapters.py -- a LoRA adapter (and the decision head it was trained with) written into the container beside the base
model's tensors (L338: autotrust's JEV System 1 judge on Gemma 4 26B-A4B), for an engine that applies it at RUNTIME --
one base pack serves both the plain model and the adapted one (y = W x + (alpha / r) * B (A x) on every targeted
projection, the base weights never merged).

build.py --adapter KIND DIR: the family's with_adapter(hf, kind, path) returns the checkpoint view carrying the adapter
(WithAdapter, the drafter view's pattern: drafters.WithDrafter).  KIND names the adapter's declaration:

  jev   DIR is the JEV snapshot: adapter/adapter_config.json + adapter/adapter_model.safetensors (PEFT LoRA, F32) and
        head.safetensors (proj.weight F32 [slots, hidden], proj.bias F32 [slots]) beside judge_config.json and
        calibration_gold.json (the head's slots, verbalizers, soft-cap, template and per-kind temperatures).

What is checked (rule 1 -- refuse, never guess): the adapter config is a plain LoRA (no DoRA, rsLoRA, bias, fan-in-fan-out,
layer replication, alpha / rank patterns, modules to save) whose target_modules is exactly the declaration's projections;
every tensor the declaration expects is present at the shape the TARGET's declaration gives (A [r, in], B [out, r], the
model's logical widths -- the dense MLP at 2112, not the pack's stored 2176: the engine pads), nothing else is in the file;
the head is [slots, hidden] + [slots] with slots = judge_config's num_slots and hidden = the target's.

The container: shard `adapter`; names `adapter.layers.N.<the target's HF suffix>.lora_{A,B}.weight`,
`adapter.head.weight` / `.bias`; recipe role `adapter` (native F32); bind records (role <target bind role>_lora_a /
_lora_b, block layers, layer N) and (role decision_head / decision_head_bias); kv `pulsar.adapter` = KIND,
`pulsar.adapter.lora_r`, `pulsar.adapter.lora_alpha`; side files judge_config.json, calibration_gold.json and
adapter_config.json as shipped.
"""
from __future__ import annotations

import json
import os

from hf_source import HFCheckpoint
from names import Mapped

ADAPTER = "adapter."
KINDS = ("jev",)


class SafetensorsFiles(HFCheckpoint):
    """Named safetensors files in one directory read like a checkpoint (no config.json): names / shape / dtype / span /
    raw / copy_src are HFCheckpoint's over the files' headers."""

    def __init__(self, directory, files):
        self.dir = directory
        self.config, self.generation_config, self._hdr, self.weight_map = {}, {}, {}, {}
        for f in files:
            p = os.path.join(directory, f)
            if not os.path.isfile(p):
                raise SystemExit(f"{p}: missing")
            for name in self._shard_header_of(p):
                self.weight_map[name] = f


# the JEV LoRA's targets: the target family's per-layer HF suffix -> its bind role (gemma4.LAYER's)
JEV_TARGETS = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
               "self_attn.o_proj": "attn_o", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
               "mlp.down_proj": "ffn_down"}
JEV_REGEX = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
PEFT_PREFIX = "base_model.model."
# PEFT config keys and the one value the runtime implements (anything else refused); keys not listed must be absent
# or falsy (null, {}, [], false)
PEFT_PIN = {"peft_type": "LORA", "bias": "none", "use_dora": False, "use_rslora": False, "fan_in_fan_out": False,
            "lora_bias": False}
PEFT_IGNORE = ("alora_invocation_tokens", "arrow_config", "auto_mapping", "base_model_name_or_path", "corda_config",
               "ensure_weight_tying", "eva_config", "exclude_modules", "init_lora_weights", "inference_mode",
               "layers_pattern", "layers_to_transform", "loftq_config", "lora_dropout", "megatron_config",
               "megatron_core", "modules_to_save", "monteclora_config", "qalora_group_size", "revision", "task_type",
               "trainable_token_indices", "target_parameters", "layer_replication", "alpha_pattern", "rank_pattern",
               "use_qalora", "lora_ga_config", "kasa_config", "peft_version")
PEFT_FALSY = ("layer_replication", "alpha_pattern", "rank_pattern", "modules_to_save", "exclude_modules",
              "layers_to_transform", "trainable_token_indices", "target_parameters", "use_bdlora", "velora_config",
              "use_qalora", "lora_ga_config", "kasa_config", "corda_config", "eva_config", "arrow_config",
              "monteclora_config", "alora_invocation_tokens")


class Target:
    """What the adapter is checked against: the target's per-layer projection shapes ([out, in] logical), its hidden
    width and its layer count."""

    def __init__(self, hidden, n_layer, proj_shape):
        self.hidden, self.n_layer = hidden, n_layer
        self.proj_shape = proj_shape     # (layer, HF suffix) -> [out, in], or None when the layer has no such projection


class Adapter:
    """The checked adapter: its readers, {container name: Mapped}, the container name -> (reader, source name) map,
    the kv triples and the side files."""

    def __init__(self, kind, path, target: Target):
        if kind not in KINDS:
            raise SystemExit(f"--adapter {kind}: the builder reads {list(KINDS)}")
        self.kind, self.dir = kind, path
        cfg_path = os.path.join(path, "adapter", "adapter_config.json")
        cfg = json.load(open(cfg_path))
        for k, want in PEFT_PIN.items():
            if cfg.get(k, want) != want:
                raise SystemExit(f"{cfg_path}: {k} = {cfg.get(k)!r}; the runtime LoRA implements {want!r} only")
        for k in PEFT_FALSY:
            if cfg.get(k):
                raise SystemExit(f"{cfg_path}: {k} = {cfg[k]!r}; the runtime LoRA implements none")
        known = set(PEFT_PIN) | set(PEFT_IGNORE) | set(PEFT_FALSY) | {"r", "lora_alpha", "target_modules"}
        unknown = sorted(k for k in cfg if k not in known)
        if unknown:
            raise SystemExit(f"{cfg_path}: keys the declaration does not know -- refusing: {unknown}")
        if cfg.get("target_modules") != JEV_REGEX:
            raise SystemExit(f"{cfg_path}: target_modules {cfg.get('target_modules')!r}, the jev declaration's is "
                             f"{JEV_REGEX!r}")
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        if not isinstance(r, int) or r <= 0 or not isinstance(alpha, (int, float)) or alpha <= 0:
            raise SystemExit(f"{cfg_path}: r {r!r} / lora_alpha {alpha!r} are not a positive rank and scale")
        self.r, self.alpha = r, alpha
        jc_path = os.path.join(path, "judge_config.json")
        jc = json.load(open(jc_path))
        slots = jc.get("slots", {}).get("num_slots")
        if jc.get("hidden_size") != target.hidden or not isinstance(slots, int) or slots <= 0:
            raise SystemExit(f"{jc_path}: hidden_size {jc.get('hidden_size')} / num_slots {slots} disagree with the "
                             f"target's hidden {target.hidden}")
        lr = jc.get("lora", {})
        if lr.get("r") != r or lr.get("alpha") != alpha or lr.get("target_modules") != JEV_REGEX:
            raise SystemExit(f"{jc_path}: its lora block {lr} disagrees with {cfg_path}")
        if jc.get("readout", {}).get("type") != "last_token" or jc.get("softcap") is None:
            raise SystemExit(f"{jc_path}: readout {jc.get('readout')} / softcap {jc.get('softcap')}: the jev readout is "
                             f"the last token's hidden, soft-capped")
        self.lora = SafetensorsFiles(os.path.join(path, "adapter"), ["adapter_model.safetensors"])
        self.head = SafetensorsFiles(path, ["head.safetensors"])
        self.src = {}        # container name -> (reader, source name)
        want = {}            # container name -> (reader, source name, dtype, shape, Mapped)
        for il in range(target.n_layer):
            for sfx, bind in JEV_TARGETS.items():
                shp = target.proj_shape.get((il, sfx))
                if shp is None:
                    continue                       # the layer has no such projection (Gemma's global layers: no v)
                out, inp = shp
                for ab, s, rb in (("lora_A", [r, inp], "lora_a"), ("lora_B", [out, r], "lora_b")):
                    src = f"{PEFT_PREFIX}model.language_model.layers.{il}.{sfx}.{ab}.weight"
                    cname = f"{ADAPTER}layers.{il}.{sfx}.{ab}.weight"
                    rec = (("role", f"{bind}_{rb}"), ("block", "layers"), ("layer", il))
                    want[cname] = (self.lora, src, "F32", s, Mapped(cname, cname, "adapter", "adapter", il, None, None,
                                                                     False, role="adapter", bind=rec))
        for src, cname, s, bind in (("proj.weight", ADAPTER + "head.weight", [slots, target.hidden], "decision_head"),
                                    ("proj.bias", ADAPTER + "head.bias", [slots], "decision_head_bias")):
            want[cname] = (self.head, src, "F32", s, Mapped(cname, cname, "adapter", "adapter", None, None, None, False,
                                                            role="adapter", bind=(("role", bind),)))
        for reader, label in ((self.lora, "adapter_model.safetensors"), (self.head, "head.safetensors")):
            have = set(reader.names())
            expect = {s for rd, s, *_ in want.values() if rd is reader}
            extra = sorted(have - expect)
            if extra:
                raise SystemExit(f"{label}: {len(extra)} tensors the {kind} declaration does not map -- refusing: "
                                 f"{extra[:5]}")
            missing = sorted(expect - have)
            if missing:
                raise SystemExit(f"{label}: {len(missing)} tensors the {kind} declaration expects are missing -- "
                                 f"refusing: {missing[:5]}")
        wrong = [f"{s}: {rd.dtype(s)} {rd.shape(s)}, the declaration says {dt} {shp}"
                 for rd, s, dt, shp, _m in want.values() if (rd.dtype(s), rd.shape(s)) != (dt, shp)]
        if wrong:
            raise SystemExit(f"{len(wrong)} adapter tensors disagree with the target's shapes -- refusing: {wrong[:3]}")
        self.src = {c: (rd, s) for c, (rd, s, *_r) in want.items()}
        self.mapped = {c: m for c, (*_r, m) in want.items()}
        self.kvs = [("pulsar.adapter", "string", kind), ("pulsar.adapter.lora_r", "u32", r),
                    ("pulsar.adapter.lora_alpha", "f32", float(alpha)),
                    ("pulsar.source.adapter", "string", os.path.basename(os.path.normpath(path)))]
        self.side = [(n, os.path.join(path, n)) for n in ("judge_config.json", "calibration_gold.json")] + \
                    [("adapter_config.json", cfg_path)]
        for _n, p in self.side:
            if not os.path.isfile(p):
                raise SystemExit(f"{p}: missing (the {kind} adapter's side file)")


class WithAdapter:
    """The checkpoint view with the adapter's tensors beside the target's, under ADAPTER-prefixed names: what build.py
    and entries.py read (names / has / shape / dtype / span / raw / copy_src); the rest is the target view's."""

    def __init__(self, hf, adapter: Adapter):
        self.hf, self.adapter = hf, adapter
        self.dir, self.config, self.generation_config = hf.dir, hf.config, hf.generation_config
        if hasattr(hf, "decl"):
            self.decl = hf.decl
            self.mapped = {**hf.mapped, **adapter.mapped}

    def __getattr__(self, k):              # the target view's own attributes (gemma4.Source's embed, ...)
        return getattr(self.hf, k)

    def _of(self, name):
        return self.adapter.src[name] if name in self.adapter.src else (self.hf, name)

    def names(self):
        return self.hf.names() + sorted(self.adapter.mapped)

    def has(self, name):
        return name in self.adapter.src or self.hf.has(name)

    def shape(self, name):
        src, n = self._of(name)
        return src.shape(n)

    def dtype(self, name):
        src, n = self._of(name)
        return src.dtype(n)

    def span(self, name):
        src, n = self._of(name)
        return src.span(n)

    def raw(self, name):
        src, n = self._of(name)
        return src.raw(n)

    def copy_src(self, name):
        src, n = self._of(name)
        return src.copy_src(n)


def with_adapter(hf, kind, path, target: Target):
    """build.py --adapter KIND DIR: the checkpoint view carrying the adapter (refused by name unless KIND is one this
    builder reads and its config and tensors agree with the declaration and the target)."""
    return WithAdapter(hf, Adapter(kind, path, target))
