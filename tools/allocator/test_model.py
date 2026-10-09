#!/usr/bin/env python3
"""CPU checks of the allocator's arithmetic (no torch, no GPU): curve evaluation, the knapsack, the byte unit."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MD                                                             # noqa: E402
import propose as P                                                            # noqa: E402


def test_curve():
    pts = [(-0.75, -1.0, -2.0, 0.5), (0.0, 0.0, 0.0, 0.0), (3.0, 3.0, 1.0, 5.0), (15.0, 9.0, 6.0, 12.0)]
    assert MD.curve_eval(pts, 0.0) == 0.0
    assert MD.curve_eval(pts, 3.0) == 3.0
    assert MD.curve_eval(pts, 9.0) == 6.0                       # between the s=2 and s=4 points
    assert MD.curve_eval(pts, 27.0) == 15.0                     # the last segment's slope continues
    assert abs(MD.curve_eval(pts, -1.5) - (-2.0)) < 1e-12       # the first segment's slope continues below
    assert MD.curve_eval(pts, 3.0, 3) == 5.0


def test_solve():
    # two blocks; block 0 gains from +1 unit, block 1 loses little from -1 unit: the matched-bytes optimum swaps them
    costs = {0: {(4, 4): (0.0, 0), (4, 5): (-3.0, 1), (4, 3): (5.0, -1)},
             1: {(4, 4): (0.0, 0), (4, 5): (-0.5, 1), (4, 3): (1.0, -1)}}
    c, u, ch = P.solve(costs, 0)
    assert (c, u, ch) == (-2.0, 0, [(0, (4, 5)), (1, (4, 3))]), (c, u, ch)
    c, u, ch = P.solve(costs, -1)
    assert u <= -1 and c == 1.0, (c, u, ch)


def test_bytes():
    # one down bit on one layer = the proposer's unit; gate_up's step is two units
    assert MD.expert_layer_bytes("dn", 5) - MD.expert_layer_bytes("dn", 4) == P.UNIT
    assert MD.expert_layer_bytes("gu", 5) - MD.expert_layer_bytes("gu", 4) == 2 * P.UNIT
    assert MD.expert_layer_bytes("dn", 4) == 512 * 825600       # the U-e4-d5 container's expert_bytes


if __name__ == "__main__":
    for t in (test_curve, test_solve, test_bytes):
        t()
    print("OK")
