"""The EXL3 rate table -- the builder's ONE copy (the engine's is src/engine/exl3_trellis.h
exl3_type_k2 and the arm table beside it): a layout name <-> the rate in half bits (k2) <-> the
trellis words per 16x16 tile (16 K, or 16 K + 8 for the half-integer rates).  Which rates a FAMILY's
recipe admits is a restriction of this table (policy.EXPERT_LAYOUTS for DeepSeek's expert arms,
qwen.FORMATS for Qwen's), never a second table (L272 P0 folded four)."""

K2 = {"exl3m_k2": 4, "exl3m_k2h": 5, "exl3m_k3": 6, "exl3m_k4": 8, "exl3m_k5": 10, "exl3m_k6": 12,
      "exl3m_k8": 16}

# the rates each family's EXL3 arms read (exl3_trellis.h exl3_arm_has_rate: DOWN / PAIR take 2, 2.5,
# 3 for DeepSeek; Qwen's trunk is turboderp's K4 / K6 and our K5 packs, its MTP layer K3, K8 dense)
DEEPSEEK_EXPERT = ("exl3m_k2", "exl3m_k2h", "exl3m_k3")
QWEN = ("exl3m_k4", "exl3m_k5", "exl3m_k6", "exl3m_k8")


def words_for(layout: str) -> int:
    """Trellis words per 16x16 tile of `layout`."""
    k2 = K2[layout]
    return 16 * (k2 >> 1) + (8 if k2 & 1 else 0)


LAYOUT_BY_WORDS = {words_for(layout): layout for layout in K2}


def rates(layouts) -> dict:
    """{words per tile: layout} for the layouts a recipe admits."""
    return {w: layout for w, layout in LAYOUT_BY_WORDS.items() if layout in layouts}
