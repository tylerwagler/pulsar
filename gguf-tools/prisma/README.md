# Format allocation for ds4 mixed quants

**The allocator is upstream PrismaQuant. It is not in this repo.**

`v5mx4-format-map.json` — the pinned map that decides all 474 tensor types in
the shipped artifact — was produced by upstream `prismaquant/allocator.py`, not
by anything here. A local `prisma_alloc.py` used to sit in this directory and
was described as "the allocator" by our own docs; it was a coarse per-layer
0/1-preset solver that **could not express the shipped map** (its presets all
contain Q2_K, which the map has none of), and it was retired on 2026-08-12 once
that was established. Git history has it.

## Where the shipped map came from

| piece | location |
|---|---|
| producing code | fork `dsv4-fixes-validation` @ `3ce4e99` — carries BOTH `ds4_engine.json` and the `CUTLASS_MXFP4` format |
| sensitivity probe | `/mnt/pve1-models/prisma-0731-run/probe.pkl` |
| quant cost table | `/mnt/pve1-models/prisma-0731-run/cost.pkl` — 33,160 Linears × 15 formats |
| solves | `/mnt/pve1-models/prisma-0731-run/solves/{base,noq2k,promo}/` |
| promotion survey | `/mnt/pve1-models/prisma-0731-run/promo_survey_0731.py` + `solves/promo_survey.log` |

Both pickles were re-validated against PrismaQuant **v0.11** on 2026-08-12 and
pass `validate_probe_payload` / `validate_cost_payload` unchanged, so the
expensive GPU stages do NOT need re-running to re-derive or re-solve.

The July invocation, from `solves/noq2k/alloc.log`:

    allocator.py --probe probe.pkl --costs cost.pkl
        --formats IQ2_XXS,CUTLASS_MXFP4,MXFP8_E4M3
        --target-profile ds4_engine --packed-role-split
        (budget 113,000,000,000 bytes)

`--packed-role-split` is what makes role asymmetry possible: it keys packed
expert groups as `(layer, gate_up|down)` DP units instead of one unit per
layer. That is why 7 of 43 routed layers are MIXED, and it also satisfies the
engine's "gate and up must share a format" rule by construction, since gate+up
is a single unit.

## The 7 mixed layers were priced, not hand-edited

The base solve left 9 layers mixed. `promo_survey_0731.py` priced what it would
cost to make each one uniform, and the two ~10x most efficient were promoted
(`L8 down` at 3.07/GB, `L7 down` at 2.32/GB); the remaining seven were left
mixed where efficiency collapsed to 0.03–0.29/GB. Final census, which matches
`weights.cpp` exactly: **9 all-CUTLASS_MXFP4, 27 all-IQ2_XXS_MMQ, 7 mixed**.

## Cross-check against v0.11 (2026-08-12)

Re-solved at the same 113 GB budget with a refreshed profile: **127 of 129
(layer, role) units identical**. The two that moved are both demotions —
`blk.7 down` and `blk.29 down`, MXFP4 -> IQ2_XXS — which were the marginal
calls in the survey. Note Δloss is NOT comparable across versions: the Fisher
normalization changed between our fork point and v0.11.

That solve is a cross-check, not a shipping candidate: it landed at 2.675 bpp
against the pinned 2.759 and the log warns the Pareto sweep bottomed out above
the menu's cheapest rung with denser allocations unexplored. A shipping re-solve
needs `--pareto-targets` covering the menu.

## ds4_engine.json

Our serving profile — what the engine can actually serve, as opposed to what the
container could carry. It is a **PrismaQuant** file: to use it, copy it into a
prismaquant checkout's `prismaquant/serving_profile_specs/`. It is vendored here
because it is ours, it is not upstream, and it went missing once already when
the worktree that held it was pruned.

Refreshed 2026-08-12 against `weights.cpp` (`tensor_is_routed_expert_type`,
`weights_tensor_type_supported`):

- **Q2_K removed** from the expert allow-list. Its dp4a kernel was deleted along
  with plain IQ2_XXS (16), IQ2_XXS_SOA (42) and FP4_E2M1 (39). The previous
  revision still offered it, so a re-solve could have produced a valid-looking
  map the engine refuses at load.
- **`supports_per_role_expert_schemes: true`** added — v0.11 hard-errors on
  `--packed-role-split` without it.

`IQ2_XXS` / `MXFP4` / `MXFP8_E4M3` are the allocator-facing identities; the
artifact stores the pre-formatted twins `IQ2_XXS_MMQ` (43), `CUTLASS_MXFP4` (40)
and `MXFP8_LT` (41). Those are pure layout permutations at identical bits, so
they are allocation-equivalent.

**Two of our patches are NOT upstream** (`ds4_engine.json` and the
`CUTLASS_MXFP4` format), which is why upstream HEAD cannot reproduce our solve
without them. They live on the fork branch above.

## The remaining tools

`measure_layer_kl.py` measures real end-to-end per-layer promotion KL on THIS
engine rather than a torch proxy, which is a thing upstream cannot do; it is
kept on that merit. Its former companions (`dump_inventory.py`,
`extract_sens.py`) fed only the retired local allocator and were deleted
2026-08-19.
