"""The EXL3 rate table -- the builder's ONE copy (the engine's is src/engine/exl3_trellis.h exl3_type_k2 and
exl3_arm_has_rate beside it): a layout name <-> the rate in half bits (k2) <-> the trellis words per 16x16 tile
(16 K, or 16 K + 8 for the half-integer rates).  Which rates a tensor may take is the ENGINE ARM its role runs
(ARMS, pinned to exl3_arm_has_rate by test_exl3_rates.py), not a per-family list (L279 step 5)."""

K2 = {"exl3m_k2": 4, "exl3m_k2h": 5, "exl3m_k3": 6, "exl3m_k4": 8, "exl3m_k5": 10, "exl3m_k6": 12,
      "exl3m_k8": 16}

# exl3_arm_has_rate, arm by arm: the k2 each kernel reads (EXL3_ARM_DOWN / PAIR / GATE_UP_FUSED / DENSE)
ARMS = {"down": (4, 5, 6, 8, 10, 12), "pair": (4, 5, 6, 8, 12), "gate_up_fused": (8, 10),
        "dense": (4, 6, 8, 10, 12, 16)}
# the arm a role's tensor runs (src/engine/weight_format.cpp pulsar_format_serves: a shared expert takes the dense
# arm; split gate / up the pair arm; a fused gate_up stack its own)
ROLE_ARM = {"expert_gate": "pair", "expert_up": "pair", "expert_down": "down", "expert_gate_up": "gate_up_fused",
            "dense": "dense", "shared_expert": "dense"}


def words_for(layout: str) -> int:
    """Trellis words per 16x16 tile of `layout`."""
    k2 = K2[layout]
    return 16 * (k2 >> 1) + (8 if k2 & 1 else 0)


LAYOUT_BY_WORDS = {words_for(layout): layout for layout in K2}


def admit(role: str, layout: str, what: str) -> None:
    """Refuse an EXL3 layout no engine arm reads for this role."""
    arm = ROLE_ARM.get(role)
    if arm is None:
        raise SystemExit(f"{what}: role {role} runs no EXL3 arm ({sorted(ROLE_ARM)} do)")
    if K2[layout] not in ARMS[arm]:
        have = [x for x, k in K2.items() if k in ARMS[arm]]
        raise SystemExit(f"{what}: {layout} -- the engine's {arm} arm (role {role}) reads {have} only "
                         "(exl3_trellis.h exl3_arm_has_rate)")
