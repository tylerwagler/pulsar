#!/usr/bin/env python3
"""CPU check of capture_extra.py's image cases on a snapshot, no GPU and no weights: every case expands through the
model's own prepare_vl_inputs, the straddles straddle, the depth rows fit --max-seq-len, the PNG is deterministic.

    python test_cases.py --snapshot $SRC --kit bundle/refkit [--max-seq-len 8192]
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    a = ap.parse_args()
    inf = os.path.join(a.snapshot, "inference")
    sys.path[:0] = [HERE, inf, os.path.join(a.snapshot, "encoding")]

    class Stub(types.ModuleType):   # model.py imports its tilelang kernels at import time; nothing here runs them
        def __getattr__(self, name):
            return None
    sys.modules["kernel"] = Stub("kernel")
    import numpy as np
    import capture_extra as CE
    import image_processor as IP
    import model as M
    from transformers import AutoTokenizer

    args = M.ModelArgs(**json.load(open(os.path.join(inf, "config.json"))))
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    text = np.fromfile(os.path.join(a.kit, "story.tokens.bin"), dtype="<i4").astype(np.int64).tolist()
    image_id = getattr(args, "image_token_id", None) or tok.convert_tokens_to_ids("<｜deepseek_image｜>")
    png = CE.synth_png(640, 480, seed=4101)
    ok = png == CE.synth_png(640, 480, seed=4101)
    print(f"png {len(png)} bytes, deterministic {ok}; image token id {image_id}")
    for name, where in CE.CASES:
        ids, tt, img = CE.build_case(IP, args, tok, text, image_id, {"data": png}, where)
        span = img.types.numel()
        end = img.start + span
        depths = [end + k for k in CE.AFTER]
        good = (len(ids) == depths[-1] + 1 and depths[-1] <= a.max_seq_len
                and (where if isinstance(where, int) else img.start) == img.start
                and (isinstance(where, int) or img.start < where[1] < end)
                and (tt is None or len(tt) == len(ids)))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {name}: block [{img.start}, {end}) {span} slots, vit {img.n_vit_h}x{img.n_vit_w}, "
              f"depths {depths}, {len(ids)} ids")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
