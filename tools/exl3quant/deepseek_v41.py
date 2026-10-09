"""DeepSeek-V4.1 for exllamav3: the config, the layer schedule and the quantization plan (DeepseekV41ForCausalLM).

Written as an exllamav3 architecture file (`exllamav3/architecture/deepseek_v41.py`: a Config subclass with
`arch_string`, `read_cfg` / `assert_cfg` against config.json, a SafetensorsCollection over the checkpoint) so it can
go upstream next to deepseek_v4.py; it lives in pulsar's tools/exl3quant until then.  What it holds is what V4.1
IS, read from config.json and refused when the checkpoint disagrees:

  * the trunk: 40 layers, hidden 5120, 384 routed experts (top 6, mid 2304, sqrt-softplus, route scale 1.5) plus
    one shared expert, the first `num_hash_layers` routed by token id; mHC (hc_mult 4);
  * V4.1's attention schedule, which V4's does not have: `compress_ratios` per layer is 0 (sliding window only)
    or a compression code (V4.1: 2 for layers 2-19, 1 for 20-39) and the compressed KV, index keys, top-k and
    candidate blocks are computed by SOURCE layers and read by the layers after them (`kv_source_layer_ids`,
    `index_source_layer_ids`, `candidate_source_layer_id`);
  * Engram at `engram_layer_ids` (n-gram hash rows added into the residual stream before the block): a row table
    per layer (E4M3 rows x E8M0 per 32) far too large to load -- streamed by row, never quantized;
  * the DSpark drafter (`mtp.*`, num_nextn_predict_layers stages, its OWN routed-expert count and top-k:
    dspark_n_routed_experts / dspark_num_experts_per_tok), fed by the attention inputs of
    `dspark_target_layer_ids`;
  * the vision tower (`vision.*`, the aligner, the image span embeddings): passed through.

Quantization plan (`plan()`): the routed experts of every trunk layer and every drafter stage are the EXL3
targets -- exllamav3's BlockSparseMLP keys (`<block>.ffn` with `experts.{expert_idx}.w1/w3/w2`, gate / up sharing
one Hessian, down one per expert).  Everything else passes through as the checkpoint holds it: pulsar serves
DeepSeek's dense / attention / shared-expert linears at the MX slot (MXFP8 from the FP8 source; the engine refuses
DeepSeek EXL3 dense), Engram as row files from disk, the tower as is.

What is NOT here, and is what an upstream DeepseekV41Model needs from exllamav3 beyond deepseek_v4.py (L285 U1):
DSV4Attention taking the compressed KV / index keys / top-k / candidate blocks from a source layer instead of
computing its own (and the 1/2 compression codes), an Engram module over a RowTable of E4M3+E8M0 rows (RowTable
streams Qwen's PLE table today; the row format and the n-gram hash are V4.1's), and the drafter's 128/3 expert
shape.  Until then pulsar's calibration forward is DeepSeek's own inference/model.py (v41_stream.py), and this
file supplies the config, the schedule and the plan that forward is checked against.
"""
from __future__ import annotations

from exllamav3.model.config import Config, no_default

ROUTED_PARTS = ("w1", "w3", "w2")          # gate, up, down -- DeepSeek's names; gate / up share the input Hessian


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        # No exllamav3 Model class yet (see the module docstring): the config, schedule and plan only
        super().__init__(directory, {}, **kwargs)

        t = "text_config->"
        self.assert_cfg(str, "model_type", "deepseek_v41")
        self.assert_cfg(str, "quantization_config->quant_method", "fp8")
        self.assert_cfg(str, "quantization_config->scale_fmt", "ue8m0")
        self.assert_cfg(str, "quantization_config->expert_dtype", "fp4")

        # Trunk
        self.hidden_size = self.read_cfg(int, t + "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, t + "num_hidden_layers", no_default)
        self.num_q_heads = self.read_cfg(int, t + "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, t + "num_key_value_heads", 1)
        assert self.num_kv_heads == 1, "DeepseekV41: expected shared-KV MQA (num_key_value_heads == 1)"
        self.head_dim = self.read_cfg(int, t + "head_dim", 512)
        self.qk_rope_head_dim = self.read_cfg(int, t + "qk_rope_head_dim", 64)
        self.q_lora_rank = self.read_cfg(int, t + "q_lora_rank", no_default)
        self.o_groups = self.read_cfg(int, t + "o_groups", 8)
        self.o_lora_rank = self.read_cfg(int, t + "o_lora_rank", 1024)
        self.sliding_window = self.read_cfg(int, t + "sliding_window", 128)
        self.rms_norm_eps = self.read_cfg(float, t + "rms_norm_eps", 1e-20)
        self.hc_mult = self.read_cfg(int, t + "hc_mult", 4)

        # MoE
        self.assert_cfg(str, t + "scoring_func", "sqrtsoftplus", optional = True)
        self.moe_intermediate_size = self.read_cfg(int, t + "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, t + "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, t + "num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, t + "n_shared_experts", 1)
        self.routed_scaling_factor = self.read_cfg(float, t + "routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, t + "swiglu_limit", 10.0)

        # Attention schedule: per-layer compression code, and who computes what the later layers read
        ratios = self.read_cfg(list, t + "compress_ratios", no_default)
        self.num_mtp_layers = self.read_cfg(int, t + "num_nextn_predict_layers", 0)
        assert len(ratios) == self.num_hidden_layers + self.num_mtp_layers, \
            f"compress_ratios has {len(ratios)} entries for {self.num_hidden_layers} + {self.num_mtp_layers} layers"
        self.compress_ratios = ratios[:self.num_hidden_layers]
        self.mtp_compress_ratios = ratios[self.num_hidden_layers:]
        self.kv_source_layers = self.read_cfg(list, t + "kv_source_layer_ids", no_default)
        self.index_source_layers = self.read_cfg(list, t + "index_source_layer_ids", no_default)
        self.candidate_source_layer = self.read_cfg(int, t + "candidate_source_layer_id", -1)
        for src in self.kv_source_layers + self.index_source_layers:
            assert self.compress_ratios[src] != 0, f"source layer {src} has no compression"

        # Engram
        self.engram_layers = self.read_cfg(list, t + "engram_layer_ids", [])
        self.engram_rows = self.read_cfg(list, t + "engram_num_embeddings", [])
        self.engram_dim = self.read_cfg(int, t + "engram_n_heads", 0) * self.read_cfg(int, t + "engram_head_dim", 0)
        assert len(self.engram_rows) == len(self.engram_layers)

        # DSpark drafter
        self.dspark_block_size = self.read_cfg(int, t + "dspark_block_size", 0)
        self.dspark_target_layers = self.read_cfg(list, t + "dspark_target_layer_ids", [])
        self.mtp_num_experts = self.read_cfg(int, t + "dspark_n_routed_experts", self.num_experts)
        self.mtp_num_experts_per_tok = self.read_cfg(int, t + "dspark_num_experts_per_tok", self.num_experts_per_tok)

        self.vision = self.read_cfg(dict, "vision_config", None)

    def routed_blocks(self) -> list[tuple[str, int, int]]:
        """(component, block index, routed experts) of every block-sparse MLP: the trunk's, then the drafter's."""
        return ([("layers", i, self.num_experts) for i in range(self.num_hidden_layers)]
                + [("mtp", i, self.mtp_num_experts) for i in range(self.num_mtp_layers)])

    def plan(self) -> dict:
        """{key: "exl3" | "passthrough"} for every checkpoint tensor (companion `.scale`s follow their weight).
        Refuses a checkpoint whose routed experts are not exactly what the config says, or that holds a tensor
        this plan does not account for."""
        names = set(self.stc.tensor_file_map)
        plan = {}
        for comp, idx, n in self.routed_blocks():
            for e in range(n):
                for p in ROUTED_PARTS:
                    base = f"{comp}.{idx}.ffn.experts.{e}.{p}"
                    self._check_routed(base, p)
                    plan[f"{base}.weight"] = plan[f"{base}.scale"] = "exl3"
            stray = f"{comp}.{idx}.ffn.experts.{n}.w1.weight"
            assert stray not in names, f"{stray}: more routed experts than the config's {n}"
        for name in names - set(plan):
            plan[name] = "passthrough"
        for i, layer in enumerate(self.engram_layers):
            meta = self.stc.get_tensor_meta(f"layers.{layer}.engram.embed.weight", optional = False)
            shape = next(iter(meta.values()))["shape"]
            assert shape == [self.engram_rows[i], self.engram_dim], f"layers.{layer}.engram.embed: {shape}"
        return plan

    def _check_routed(self, base: str, part: str):
        """An FP4 routed projection: int8 [out, in / 2] (two e2m1 per byte) with an E8M0 scale per 32 inputs."""
        out_f, in_f = ((self.moe_intermediate_size, self.hidden_size) if part != "w2"
                       else (self.hidden_size, self.moe_intermediate_size))
        w = self.stc.get_tensor_meta(f"{base}.weight", optional = False)[f"{base}.weight"]
        s = self.stc.get_tensor_meta(f"{base}.scale", optional = False)[f"{base}.scale"]
        assert w["shape"] == [out_f, in_f // 2] and w["n_bytes"] == out_f * in_f // 2, f"{base}.weight: {w}"
        assert s["shape"] == [out_f, in_f // 32] and s["n_bytes"] == out_f * in_f // 32, f"{base}.scale: {s}"

    def check_reference(self, args) -> None:
        """The snapshot's inference/config.json (the reference forward's ModelArgs) says the same model."""
        pairs = {
            "dim": self.hidden_size, "n_layers": self.num_hidden_layers, "moe_inter_dim": self.moe_intermediate_size,
            "n_routed_experts": self.num_experts, "n_activated_experts": self.num_experts_per_tok,
            "n_mtp_layers": self.num_mtp_layers, "dspark_n_routed_experts": self.mtp_num_experts,
            "dspark_n_activated_experts": self.mtp_num_experts_per_tok, "hc_mult": self.hc_mult,
            "kv_source_layers": list(self.kv_source_layers), "index_source_layers": list(self.index_source_layers),
            "candidate_source_layer": self.candidate_source_layer, "engram_layer_ids": list(self.engram_layers),
            "dspark_target_layer_ids": list(self.dspark_target_layers),
            "compress_ratios": list(self.compress_ratios) + list(self.mtp_compress_ratios),
        }
        bad = {k: (getattr(args, k), v) for k, v in pairs.items()
               if (list(getattr(args, k)) if isinstance(v, list) else getattr(args, k)) != v}
        assert not bad, f"inference/config.json disagrees with config.json: {bad}"
