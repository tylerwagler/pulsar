#!/usr/bin/env python3
"""test_exl3_rates.py -- the builder's EXL3 rate table against the engine's (L272 P4a).

The rate table lives in two languages: exl3_rates.py for the builder, and in the engine the layout-name
table (src/engine/model.cpp pulsar_layout_names) plus the rate switch (src/engine/exl3_trellis.h
exl3_type_k2).  One fact, so this pins them: every builder layout is an engine layout with the same rate
in half bits, the engine has no EXL3 layout the builder lacks, and the families' admitted rates are
rates some engine arm reads (exl3_arm_has_rate).  Plain python, no GPU, no checkpoints; in host-checks."""
import os
import re
import sys

import exl3_rates

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def main():
    fails = []
    names = dict(re.findall(r'\{"(exl3m_[a-z0-9]+)",\s*(PULSAR_TENSOR_EXL3M_[A-Z0-9]+)\}', read("src/engine/model.cpp")))
    trellis = read("src/engine/exl3_trellis.h")
    k2_of_id = {i: int(k) for i, k in re.findall(r"case (PULSAR_TENSOR_EXL3M_[A-Z0-9]+):\s*return (\d+);", trellis)}
    engine = {name: k2_of_id.get(ident) for name, ident in names.items()}
    if engine != exl3_rates.K2:
        fails.append(f"engine layouts/rates {engine} != builder K2 {exl3_rates.K2}")
    arms = re.search(r"exl3_arm_has_rate\(int arm, int k2\) \{(.*?)\n\}", trellis, re.S)
    read_rates = {int(k) for k in re.findall(r"k2 == (\d+)", arms.group(1))} if arms else set()
    if not read_rates:
        fails.append("could not read exl3_arm_has_rate's rates")
    for family, layouts in (("DEEPSEEK_EXPERT", exl3_rates.DEEPSEEK_EXPERT), ("QWEN", exl3_rates.QWEN)):
        for layout in layouts:
            if exl3_rates.K2.get(layout) not in read_rates:
                fails.append(f"{family} admits {layout} (k2 {exl3_rates.K2.get(layout)}), which no engine arm reads")
    for layout, k2 in exl3_rates.K2.items():
        if exl3_rates.words_for(layout) != 16 * (k2 >> 1) + (8 if k2 & 1 else 0):
            fails.append(f"{layout}: words per tile {exl3_rates.words_for(layout)}")
    for f in fails:
        print("FAIL", f)
    print(f"exl3 rates: {len(exl3_rates.K2)} layouts, builder == engine" if not fails else f"exl3 rates: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
