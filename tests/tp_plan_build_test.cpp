/* tp_plan_build_test.cpp -- L335: pulsar_tp_plan_build attaches a plan only when every slice validated.
 *
 * Host-only, no model, no GPU: a zeroed engine (as tests/tp_mirror_test.cpp fabricates one) carries a two-tensor
 * directory and a test family whose tp_slices declares them.  Three families:
 *
 *   refused slice  -- a gather-able OUT slice of tensor 0, then an IN slice of a bf16 tensor (no operation serves
 *                     it): the build refuses; no plan is attached to the engine, the model or the drafter, tensor 0
 *                     is not marked unstaged, and a SECOND call refuses again (before L335 the plan was attached
 *                     before validation, so the second call found it and returned true);
 *   refused declare -- tp_slices itself fails: the same, nothing attached, the second call refuses again;
 *   valid           -- the OUT slice alone: attached everywhere, tensor 0 unstaged and its slice resolvable, and a
 *                     second call returns true without declaring again.
 *
 * Exit: 0 on success, 1 on any mismatch. */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Mismatches seen so far.
static int g_failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            g_failures++;                                             \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            fprintf(stderr, __VA_ARGS__);                             \
            fprintf(stderr, "\n");                                    \
        }                                                             \
    } while (0)

/// tp_slices calls so far (a build that reuses an attached plan declares nothing).
static int g_declares = 0;

/// Declare tensor 0's OUT rows [0, 64) -- a gather for a rows reader, which unstages the stored tensor.
/// @param e    the engine
/// @param plan the plan
/// @return as pulsar_tp_plan_add1
static bool declare_out(pulsar_engine *e, pulsar_tp_plan *plan) {
    return pulsar_tp_plan_add1(plan, &e->model, &e->model.tensors[0], PULSAR_TP_AXIS_OUT, 0, 64);
}

/// The valid family's tp_slices: tensor 0's OUT slice alone.
/// @param e    the engine
/// @param plan the plan
/// @return true when declared
static bool slices_valid(pulsar_engine *e, pulsar_tp_plan *plan) {
    g_declares++;
    return declare_out(e, plan);
}

/// The refused-slice family's tp_slices: the valid slice, then an IN slice of a bf16 tensor (no operation).
/// @param e    the engine
/// @param plan the plan
/// @return true when declared (the build, not the declaration, refuses)
static bool slices_bad_slice(pulsar_engine *e, pulsar_tp_plan *plan) {
    g_declares++;
    return declare_out(e, plan) &&
           pulsar_tp_plan_add1(plan, &e->model, &e->model.tensors[1], PULSAR_TP_AXIS_IN, 0, 64);
}

/// The refused-declare family's tp_slices: declares the valid slice, then fails.
/// @param e    the engine
/// @param plan the plan
/// @return false
static bool slices_bad_declare(pulsar_engine *e, pulsar_tp_plan *plan) {
    g_declares++;
    (void)declare_out(e, plan);
    return false;
}

/// A bf16 [64 x 128] tensor named `name`.
/// @param name the tensor's name
/// @return the tensor
static pulsar_tensor bf16_tensor(const char *name) {
    pulsar_tensor t;
    memset(&t, 0, sizeof(t));
    t.name.ptr = name;
    t.name.len = strlen(name);
    t.ndim = 2;
    t.dim[0] = 64;
    t.dim[1] = 128;
    t.type = PULSAR_TENSOR_BF16;
    t.elements = 64u * 128u;
    t.bytes = t.elements * 2u;
    return t;
}

/// A zeroed engine over `tensors` (two entries) run by `family`.
/// @param family  the family
/// @param tensors the tensor directory
/// @return the engine (release with free_engine)
static pulsar_engine *fabricate_engine(const pulsar_family *family, pulsar_tensor *tensors) {
    pulsar_engine *e = (pulsar_engine *)calloc(1, sizeof(*e));
    if (!e) {
        fprintf(stderr, "tp_plan_build_test: out of memory\n");
        exit(1);
    }
    e->family = family;
    e->model.tensors = tensors;
    e->model.n_tensors = 2;
    e->model.tp_rank = 0;
    e->model.tp_n_ranks = 2;
    return e;
}

/// Free the engine's plan, its unstaged marks and the engine.
/// @param e the engine
static void free_engine(pulsar_engine *e) {
    pulsar_tp_plan_free(e);
    free(e->model.tp_unstaged);
    free(e);
}

/// A refused build, twice: nothing attached, nothing unstaged, and the second call refuses again.
/// @param family the refusing family
/// @param what   the case's name for messages
/// @param tensors the tensor directory
static void refused_twice(const pulsar_family *family, const char *what, pulsar_tensor *tensors) {
    pulsar_engine *e = fabricate_engine(family, tensors);
    g_declares = 0;
    for (int call = 1; call <= 2; call++) {
        CHECK(!pulsar_tp_plan_build(e), "%s: call %d built", what, call);
        CHECK(!e->tp_plan && !e->model.tp_plan && !e->dspark_model.tp_plan, "%s: call %d left a plan attached", what,
              call);
        CHECK(!e->model.tp_unstaged, "%s: call %d marked tensors unstaged", what, call);
    }
    CHECK(g_declares == 2, "%s: the family declared %d times over two calls, want 2", what, g_declares);
    free_engine(e);
}

int main(void) {
    pulsar_tensor tensors[2] = {bf16_tensor("t.out"), bf16_tensor("t.in")};

    pulsar_family bad_slice = PULSAR_FAMILY_QWEN4_EXP;
    bad_slice.tp_slices = slices_bad_slice;
    bad_slice.act_kind = PULSAR_ACT_KIND_ROWS;
    refused_twice(&bad_slice, "refused slice", tensors);

    pulsar_family bad_declare = bad_slice;
    bad_declare.tp_slices = slices_bad_declare;
    refused_twice(&bad_declare, "refused declare", tensors);

    pulsar_family valid = bad_slice;
    valid.tp_slices = slices_valid;
    pulsar_engine *e = fabricate_engine(&valid, tensors);
    g_declares = 0;
    CHECK(pulsar_tp_plan_build(e), "valid: the build refused");
    CHECK(e->tp_plan && e->model.tp_plan == e->tp_plan && e->dspark_model.tp_plan == e->tp_plan,
          "valid: the plan is not attached to the engine, the model and the drafter");
    CHECK(e->model.tp_unstaged && e->model.tp_unstaged[0] == 1 && e->model.tp_unstaged[1] == 0,
          "valid: the unstaged marks are not {1, 0}");
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t lo = 1, hi = 0;
    CHECK(pulsar_tp_slice_of(&e->model, &tensors[0], &op, &lo, &hi, NULL) && op == PULSAR_TP_OP_GATHER && lo == 0 &&
              hi == 64,
          "valid: tensor 0's slice is not the gather of [0, 64)");
    const pulsar_tp_plan *first = e->tp_plan;
    CHECK(pulsar_tp_plan_build(e) && e->tp_plan == first, "valid: the second call did not keep the plan");
    CHECK(g_declares == 1, "valid: the family declared %d times, want 1", g_declares);
    free_engine(e);

    if (g_failures) {
        fprintf(stderr, "tp_plan_build_test: FAILED (%d)\n", g_failures);
        return 1;
    }
    printf("tp_plan_build_test: ok (a refused build attaches nothing and refuses again; a valid one attaches once)\n");
    return 0;
}
