# pulsar — rules of the road

Read `docs/ENGINEERING-RULES.md` before changing engine code. The short form:

1. **One path or an error. No fallbacks.** A missing input refuses loudly; it never selects another kernel or format.
2. **No dead code kept.** Orphaned code is deleted in the commit that orphans it. Warning baseline is zero.
3. **Producers emit; consumers never convert.** Activation format is decided where the activation is written.
4. **One authority per fact.** No "kept in sync" copies; assert invariants in code, not comments.
5. **An instrument proves it ran the lane.** Announce the path taken, assert the tree, migrate instruments with the lane.
6. **Delete the premise before building on it.** Grep what the code does today; correct the row first.
7. **Bit-exact by default; fidelity graded against the B300 reference, never argued.**
8. **Measure at the shape production runs**, with the production activation format armed.
9. **Fail closed, loudly, once.**
10. **Landing discipline:** the gate tier the paths call for, at an asserted sha, zero warnings, doxygen clean, ledger updated, one squashed commit, topic branch deleted.

Two tiers, chosen by the paths (ENGINEERING-RULES §10): `make gates` is the full battery (every release-blocking gate, ~20 min) and runs ONCE per landing series, on the last sha whose numeric paths (`src/engine`, `src/cuda`, `src/server`, `src/tp`) changed. `make gates-dev` is the iteration tier — a fast subset selected from the touched paths — and it is the landing tier for everything after that battery that cannot move a number (refusal paths, tests, tools, docs, Makefile recipes). A measurement someone is waiting on goes ahead of a battery in the GPU queue.

Process context lives in `~/Projects/pulsar-notes` (private): `OPEN-REGISTER.md` is the live index, `rows/Lnnn.md` are append-only. Check the register before proposing work.
