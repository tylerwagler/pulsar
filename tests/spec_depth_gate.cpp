/* spec_depth_gate -- the adaptive draft depth rule (src/engine/spec_depth.h), case by case.
 * The DSpark policy's numbers are spec_dspark.cpp's k_dspark_depth (min 2, max 5, UP at tail >= 0.70, the depth-5 veto
 * at >= 0.90, an 8-round cooldown after a down within 2 rounds of an up); every case below is a branch of
 * the rule as it stood inline there before L251 moved it here, so a change to the shared rule that moves
 * DSpark's schedule fails this gate.  usage: ./tests/spec_depth_gate */
#include "../src/engine/spec_depth.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); printf("  %s  ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); printf("\n"); fails += !ok_; } while (0)

int main() {
    const pulsar_spec_depth_policy ds = {2, 5, 0.70f, 5, 0.90f, 8u, 2u};
    const float hi[16] = {0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f}, lo[16] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    const float none[16] = {-1, -1, -1, -1, -1, -1};
    printf("spec-depth-gate: the shared adaptive draft depth rule\n");
    {   /* UP: full accept with a confident tail; with no head (conf < 0) too; not with an unsure tail */
        pulsar_spec_depth_state st{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &st, 3, 3, 3, hi) == 4 && st.rounds_since_up == 0, "UP on a full, confident chain");
        pulsar_spec_depth_state s2{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &s2, 3, 3, 3, none) == 4, "UP with no confidence head");
        pulsar_spec_depth_state s3{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &s3, 3, 3, 3, lo) == 3, "no UP on an unsure tail");
    }
    {   /* DOWN: 2 * commit < depth; hold otherwise */
        pulsar_spec_depth_state st{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &st, 4, 4, 1, lo) == 3 && st.climb_cooldown == 0, "DOWN when < half converts (no excursion)");
        pulsar_spec_depth_state s2{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &s2, 4, 4, 2, lo) == 4, "hold at exactly half");
    }
    {   /* v3 veto: only at depth 5, one forgiven down, the second backs off */
        pulsar_spec_depth_state st{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &st, 5, 5, 0, hi) == 5 && st.down_forgiven, "depth 5, confident tail: one down forgiven");
        CHECK(pulsar_spec_depth_next(&ds, &st, 5, 5, 0, hi) == 4 && !st.down_forgiven, "the second consecutive down backs off");
        pulsar_spec_depth_state s2{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &s2, 4, 4, 0, hi) == 3, "no veto below depth 5");
    }
    {   /* v5: a down within 2 rounds of an up starts the 8-round cooldown; UP is suppressed while it runs */
        pulsar_spec_depth_state st{false, 10, 0};
        int d = pulsar_spec_depth_next(&ds, &st, 3, 3, 3, hi);          /* up to 4, rounds_since_up 0 */
        d = pulsar_spec_depth_next(&ds, &st, (uint32_t)d, 4, 0, lo);     /* down within the excursion window */
        CHECK(d == 3 && st.climb_cooldown == 8, "a failed excursion starts the cooldown");
        int held = 0;
        for (int r = 0; r < 7; r++) held += pulsar_spec_depth_next(&ds, &st, 3, 3, 3, hi) == 3;
        CHECK(held == 7, "UP suppressed during the cooldown (%d of 7 rounds held)", held);
        CHECK(pulsar_spec_depth_next(&ds, &st, 3, 3, 3, hi) == 4, "UP again once it runs out");
    }
    {   /* bounds */
        pulsar_spec_depth_state st{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &st, 5, 5, 5, hi) == 5, "clamped at max 5");
        pulsar_spec_depth_state s2{false, 10, 0};
        CHECK(pulsar_spec_depth_next(&ds, &s2, 2, 2, 0, lo) == 2, "clamped at min 2");
    }
    printf("SPEC-DEPTH GATE %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
