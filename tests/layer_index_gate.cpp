/* layer_index_gate.cpp -- L436: the layer index spaces (src/lib/pulsar_units.h: TrunkLayer, LayerSlot, DraftLayer,
 * SrcBlock, KindOrd, VisLayer, TpSlot) and the crossings between them, graded on the host.
 *
 *   layer_index_gate [MODEL | MODEL@DRAFTER]...
 *
 * 1. Compile time (the static_asserts below): no layer index converts to another, to a unit of the L333 / L334
 *    series, or to / from a bare integer; none adds, scales or orders across spaces; each is the uint32 it replaced.
 * 2. Model-free: every crossing computes what the open-coded expression it replaced did, over every slot a plan can
 *    hold -- the state layout's (pulsar_layer_slot_of, pulsar_drafter_slot, pulsar_slot_trunk_layer), the driver's
 *    (pulsar_tp_slot_of), the families' block -> layer functions, DeepSeek's drafter crossings.
 * 3. On each MODEL, opened inspect-only (headers only: no GPU, no weight byte staged; with @DRAFTER the open selects
 *    that drafter, as --drafter does): the same, on the family's REAL shape, and the ASSUMPTIONS the crossings rest
 *    on -- the ones a new family's shape (or a new drafter beside an old one) could break:
 *      a. the declared state's trunk slots are the plan's layers, one each, in order (n_trunk == n_layer): what
 *         pulsar_layer_slot_of's "same number" means;
 *      b. the slots past the trunk are ONE drafter's, in its own order -- a block drafter's n_layer slots or the
 *         chained-MTP lane's modules, never both: what pulsar_drafter_slot's "n_trunk + l" means (spec_block's
 *         ring_of and anchor, spec_mtp's parked rows, Qwen's MTP slot, Nemotron-H's module slot all read through it);
 *      c. a block drafter's taps land on trunk layers through the family's own block -> layer function, and
 *         Nemotron-H's table is the plan's grouping (a mixer block and the MoE block after it are one layer);
 *      d. Qwen's slot-indexed weights array holds the MTP layer at the first drafter slot, and its PLE layer is a
 *         trunk layer whose slot alone carries the PLE state;
 *      e. a family with tensor parallelism sizes its slab for every slot that exchanges (pulsar_tp_slot_of,
 *         gpu_graph_tp_slot, gpu_graph_drafter_tp_slot stay inside tp_shape's slots);
 *      f. DeepSeek: a served artifact's block is its own source layer; a layer's KV / index source is itself or an
 *         earlier layer; the DSpark drafter's three layers have a trunk layer of the same number to borrow.
 *
 * Exit 0 = every check passed.  The published containers are the Makefile's (layer-index-gate). */
#include "engine/pulsar_engine_internal.h"
#include "engine/spec_block.h"
#include "engine/spec_mtp.h"
#include "engine/state_kinds.h"
#include "engine/step_driver.h"
#include "families/gemma4/gemma4.h"
#include "families/nemotron_h/nemotron_h.h"
#include "families/nemotron_parse/nemotron_parse.h"
#include "families/qwen/family_qwen.h"

#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <type_traits>
#include <utility>

/* ---- 1. compile time ----------------------------------------------------------------------------------------------- */

/** True when `A + B` compiles. */
template <class A, class B, class = void>
constexpr bool l436_adds = false;
/** `A + B` compiles. */
template <class A, class B>
constexpr bool l436_adds<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>> = true;
/** True when `A * B` compiles. */
template <class A, class B, class = void>
constexpr bool l436_scales = false;
/** `A * B` compiles. */
template <class A, class B>
constexpr bool l436_scales<A, B, std::void_t<decltype(std::declval<A>() * std::declval<B>())>> = true;
/** True when `A < B` compiles. */
template <class A, class B, class = void>
constexpr bool l436_orders = false;
/** `A < B` compiles. */
template <class A, class B>
constexpr bool l436_orders<A, B, std::void_t<decltype(std::declval<A>() < std::declval<B>())>> = true;
/** True when `A == B` compiles. */
template <class A, class B, class = void>
constexpr bool l436_equates = false;
/** `A == B` compiles. */
template <class A, class B>
constexpr bool l436_equates<A, B, std::void_t<decltype(std::declval<A>() == std::declval<B>())>> = true;
/** True when `A = B` compiles. */
template <class A, class B, class = void>
constexpr bool l436_assigns = false;
/** `A = B` compiles. */
template <class A, class B>
constexpr bool l436_assigns<A, B, std::void_t<decltype(std::declval<A>() = std::declval<B>())>> = true;
/** True when `A[B]` compiles (a subscript with the index itself, not its raw()). */
template <class A, class B, class = void>
constexpr bool l436_subscripts = false;
/** `A[B]` compiles. */
template <class A, class B>
constexpr bool l436_subscripts<A, B, std::void_t<decltype(std::declval<A>()[std::declval<B>()])>> = true;

/** Neither of two types converts to, is constructible from, equates with, orders against or assigns to the other. */
template <class A, class B>
constexpr bool l436_apart = !std::is_convertible_v<A, B> && !std::is_convertible_v<B, A> &&
                            !std::is_constructible_v<A, B> && !std::is_constructible_v<B, A> && !l436_equates<A, B> &&
                            !l436_orders<A, B> && !l436_assigns<A &, B> && !l436_assigns<B &, A> && !l436_adds<A, B>;
/** A is apart from every one of Bs. */
template <class A, class... Bs>
constexpr bool l436_apart_from_all = (l436_apart<A, Bs> && ...);

/* the seven spaces, pairwise */
static_assert(l436_apart_from_all<TrunkLayer, LayerSlot, DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot>,
              "a trunk layer is no other layer index");
static_assert(l436_apart_from_all<LayerSlot, DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot>,
              "a layer slot is no other layer index");
static_assert(l436_apart_from_all<DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot>,
              "a drafter's layer is no other layer index (L236#2: it once indexed the trunk's tables)");
static_assert(l436_apart_from_all<SrcBlock, KindOrd, VisLayer, TpSlot> && l436_apart_from_all<KindOrd, VisLayer, TpSlot> &&
                  l436_apart<VisLayer, TpSlot>,
              "a source block, a kind ordinal, a tower block and an exchange slot are four spaces");
/* against the units and ids of L333 / L334 */
static_assert(l436_apart_from_all<TrunkLayer, KvRows, KvBytes, TokCount, PromptToks, TokPos, StepRow, StepRows, HeadRow,
                                  CompRow, CompRows, TokPerRow, TokId, BankId, TensorBytes, TextBytes>,
              "a trunk layer is no unit, position, row or id of the series");
static_assert(l436_apart_from_all<LayerSlot, KvRows, KvBytes, TokCount, PromptToks, TokPos, StepRow, StepRows, HeadRow,
                                  CompRow, CompRows, TokPerRow, TokId, BankId, TensorBytes, TextBytes>,
              "a layer slot is no unit, position, row or id of the series (pulsar_state_pool(ss, slot, id, bank))");
static_assert(l436_apart_from_all<DraftLayer, TokCount, TokPos, StepRow, HeadRow, CompRow, TokId, BankId> &&
                  l436_apart_from_all<SrcBlock, TokCount, TokPos, StepRow, CompRow, TokId, BankId> &&
                  l436_apart_from_all<KindOrd, TokCount, TokPos, StepRow, HeadRow, CompRow, TokId, BankId> &&
                  l436_apart_from_all<VisLayer, TokCount, TokPos, StepRow, TokId, BankId> &&
                  l436_apart_from_all<TpSlot, TokCount, TokPos, StepRow, TokId, BankId>,
              "the other layer indices are no unit, position, row or id either");
/* against bare integers: explicit in, .raw() out, nothing implicit */
static_assert(!std::is_convertible_v<uint32_t, TrunkLayer> && !std::is_convertible_v<TrunkLayer, uint32_t> &&
                  !std::is_convertible_v<int, LayerSlot> && !std::is_convertible_v<LayerSlot, size_t> &&
                  !std::is_convertible_v<uint32_t, DraftLayer> && !std::is_convertible_v<SrcBlock, uint32_t> &&
                  !std::is_convertible_v<uint32_t, KindOrd> && !std::is_convertible_v<VisLayer, int> &&
                  !std::is_convertible_v<uint32_t, TpSlot> && !std::is_convertible_v<TpSlot, uint64_t> &&
                  !std::is_convertible_v<TrunkLayer, bool>,
              "a layer index converts to and from no integer implicitly");
static_assert(std::is_constructible_v<TrunkLayer, uint32_t> && std::is_constructible_v<LayerSlot, unsigned> &&   /* L445: no int */
                  std::is_constructible_v<KindOrd, uint8_t> && !std::is_constructible_v<TrunkLayer, bool> &&
                  !std::is_constructible_v<LayerSlot, double> && !std::is_constructible_v<DraftLayer, float>,
              "a layer index enters from an integer, explicitly; bool and floating point are refused");
static_assert(!l436_equates<TrunkLayer, uint32_t> && !l436_equates<int, LayerSlot> && !l436_orders<TrunkLayer, uint32_t> &&
                  !l436_orders<uint32_t, DraftLayer> && !l436_assigns<TrunkLayer &, uint32_t> &&
                  !l436_assigns<uint32_t &, LayerSlot>,
              "a layer index compares with and assigns from no bare integer (below(n) asks a count by name)");
/* an index, not a quantity */
static_assert(!l436_adds<TrunkLayer, TrunkLayer> && !l436_adds<TrunkLayer, uint32_t> && !l436_adds<uint32_t, LayerSlot> &&
                  !l436_scales<TrunkLayer, uint32_t> && !l436_scales<uint32_t, DraftLayer> &&
                  !l436_scales<LayerSlot, LayerSlot>,
              "no sum of two layers, no layer plus a number, no scaled layer");
static_assert(!l436_subscripts<const uint32_t *, TrunkLayer> && !l436_subscripts<const float *, LayerSlot> &&
                  !l436_subscripts<pulsar_layer_kind *, DraftLayer>,
              "an array is subscripted with .raw(), in the index's own table -- never with the index of another space");
static_assert(l436_equates<TrunkLayer, TrunkLayer> && l436_orders<LayerSlot, LayerSlot> &&
                  l436_assigns<DraftLayer &, DraftLayer>,
              "within a space: equality, ordering, assignment");
/* the representation: the uint32 it replaced (arrays of it are uint32 arrays; a zeroed struct holds index 0) */
static_assert(std::is_same_v<decltype(TrunkLayer{}.raw()), uint32_t> && std::is_same_v<decltype(KindOrd{}.raw()), uint32_t> &&
                  sizeof(TrunkLayer[7]) == sizeof(uint32_t[7]) && alignof(LayerSlot) == alignof(uint32_t) &&
                  std::is_trivially_copyable_v<SrcBlock> && std::is_standard_layout_v<TpSlot> && std::is_trivial_v<VisLayer>,
              "a layer index is the uint32 it replaced");
/* the operations compute what the integer expressions did */
static_assert(TrunkLayer{5u}.raw() == 5u && TrunkLayer{5u}.next().raw() == 6u && TrunkLayer{5u}.below(6u) &&
                  !TrunkLayer{5u}.below(5u) && TrunkLayer{5u}.last_of(6u) && !TrunkLayer{5u}.last_of(7u) &&
                  TrunkLayer{3u} < TrunkLayer{4u} && TrunkLayer{4u} <= TrunkLayer{4u} && TrunkLayer{4u} != TrunkLayer{5u},
              "below is <, last_of is + 1 ==, next is + 1");
static_assert(TrunkLayer::none().raw() == UINT32_MAX && TrunkLayer::none().is_none() && !TrunkLayer{0u}.is_none() &&
                  !TrunkLayer::none().below(UINT32_MAX) && TrunkLayer{UINT32_MAX}.is_none() &&
                  TrunkLayer::from_raw_unchecked(-1) == TrunkLayer::none() && TrunkLayer::none() == TrunkLayer::none(),
              "none() is all ones -- the UINT32_MAX / PULSAR_NO_LAYER the fields carried -- and is below nothing");
/* the typed signatures the series' hazards sat in */
static_assert(std::is_same_v<decltype(pulsar_layer_plan_check), const char *(const pulsar_layer_plan *, TrunkLayer *)> &&
                  std::is_same_v<pulsar_block_fn, bool (*)(const pulsar_driver_step *, TrunkLayer)> &&
                  std::is_same_v<decltype(pulsar_state_plan_add_slot), LayerSlot(pulsar_state_plan *, uint32_t)> &&
                  std::is_same_v<decltype(pulsar_layer_slot_of), LayerSlot(TrunkLayer)> &&
                  std::is_same_v<decltype(pulsar_drafter_slot), LayerSlot(const pulsar_state_plan *, DraftLayer)> &&
                  std::is_same_v<decltype(pulsar_slot_trunk_layer), TrunkLayer(const pulsar_state_plan *, LayerSlot)> &&
                  std::is_same_v<decltype(pulsar_tp_slot_of), TpSlot(LayerSlot)> &&
                  std::is_same_v<decltype(pulsar_layer_source), SrcBlock(TrunkLayer)> &&
                  std::is_same_v<decltype(pulsar_layer_n_expert), uint32_t(TrunkLayer)> &&
                  std::is_same_v<decltype(gpu_graph_drafter_ffn_layer), TrunkLayer(DraftLayer)> &&
                  std::is_same_v<decltype(pulsar_block_drafter::tap_block), SrcBlock[PULSAR_BLOCK_MAX_TAPS]> &&
                  std::is_same_v<decltype(pulsar_block_drafter::tap_layer), TrunkLayer[PULSAR_BLOCK_MAX_TAPS]> &&
                  std::is_same_v<decltype(pulsar_mtp_step::module), DraftLayer> &&
                  std::is_same_v<decltype(pulsar_layer_attn::kv_source), TrunkLayer> &&
                  std::is_same_v<decltype(pulsar_qwen_shape::ple_layer), TrunkLayer>,
              "the layer signatures are typed: a slot, a drafter's layer or a source block no longer passes for a "
              "trunk layer");

/* ---- the indexed tables: each takes its own space's index and nothing else ------------------------------------- */
/** Table type `A` subscripts with `Own` and with none of the other spaces' indices, nor a bare integer. */
template <class A, class Own, class... Others>
constexpr bool l436_only = l436_subscripts<A &, Own> && (!l436_subscripts<A &, Others> && ...) &&
                           !l436_subscripts<A &, uint32_t> && !l436_subscripts<A &, int> && !l436_subscripts<A &, size_t>;

static_assert(l436_only<decltype(pulsar_layer_plan::kind), TrunkLayer, LayerSlot, DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot> &&
                  l436_only<decltype(pulsar_layer_plan::desc), TrunkLayer, LayerSlot, DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot>,
              "the plan's tables take a trunk layer only");
static_assert(l436_only<decltype(pulsar_state_plan::slot), LayerSlot, TrunkLayer, DraftLayer, SrcBlock, KindOrd, VisLayer, TpSlot> &&
                  l436_only<decltype(pulsar_state_set::t), LayerSlot, TrunkLayer, DraftLayer, KindOrd> &&
                  l436_only<decltype(pulsar_state_set::pool), LayerSlot, TrunkLayer, DraftLayer, KindOrd> &&
                  l436_only<decltype(pulsar_state_set::ord), LayerSlot, TrunkLayer, DraftLayer, KindOrd>,
              "the declared state's tables take a layer slot only: a trunk layer crosses by pulsar_layer_slot_of, a "
              "drafter's by pulsar_drafter_slot");
static_assert(l436_only<decltype(pulsar_qwen_weights::layer), LayerSlot, TrunkLayer, DraftLayer, KindOrd> &&
                  l436_only<decltype(pulsar_nemotron_h_model::layer), TrunkLayer, LayerSlot, DraftLayer, SrcBlock, KindOrd> &&
                  l436_only<decltype(pulsar_gemma4_model::layer), TrunkLayer, LayerSlot, DraftLayer, SrcBlock> &&
                  l436_only<decltype(pulsar_weights::layer), TrunkLayer, LayerSlot, DraftLayer, SrcBlock> &&
                  l436_only<decltype(pulsar_dspark_weights::layer), DraftLayer, TrunkLayer, LayerSlot> &&
                  l436_only<decltype(pulsar_block_drafter::layer), DraftLayer, TrunkLayer, LayerSlot, VisLayer>,
              "per-layer weights: Qwen's by the slot (its MTP layer sits past the trunk's), the other trunks' by the "
              "trunk layer, a drafter's by the drafter's own layer (L236#2)");
static_assert(l436_only<decltype(pulsar_nemotron_h_shape::block_layer), SrcBlock, TrunkLayer, LayerSlot, KindOrd> &&
                  l436_only<decltype(pulsar_nemotron_h_vision::vit), VisLayer, TrunkLayer, LayerSlot, DraftLayer> &&
                  l436_only<decltype(pulsar_nemotron_parse_model::vit), VisLayer, TrunkLayer> &&
                  l436_only<decltype(pulsar_nemotron_parse_model::layer), TrunkLayer, VisLayer> &&
                  l436_only<decltype(pulsar_vision_weights::block), VisLayer, TrunkLayer, DraftLayer>,
              "Nemotron-H's block table takes a source block; a tower's blocks a tower layer (Parse holds a tower and "
              "a decoder side by side)");
static_assert(l436_only<decltype(pulsar_gpu_graph::layer_raw_cache), TrunkLayer, DraftLayer, LayerSlot> &&
                  l436_only<decltype(pulsar_gpu_graph::dspark_raw_cache), DraftLayer, TrunkLayer, LayerSlot> &&
                  l436_only<decltype(pulsar_gpu_graph::dspark_n_raw), DraftLayer, TrunkLayer> &&
                  l436_only<decltype(pulsar_bank_slabs::comp), TrunkLayer, DraftLayer, LayerSlot> &&
                  l436_only<decltype(pulsar_bank_slabs::dspark_raw), DraftLayer, TrunkLayer> &&
                  l436_only<std::remove_extent_t<decltype(pulsar_gpu_graph::ms_n_comp)>, TrunkLayer, DraftLayer> &&
                  l436_only<std::remove_extent_t<decltype(pulsar_gpu_graph::ms_dspark_n_raw)>, DraftLayer, TrunkLayer>,
              "DeepSeek's graph: the trunk's pools by the trunk layer, the DSpark rings by the drafter's");
static_assert(l436_only<pulsar_indexed_view<KindOrd, TrunkLayer>, TrunkLayer, KindOrd, LayerSlot, DraftLayer> &&
                  l436_only<pulsar_indexed_view<int, KindOrd>, KindOrd, TrunkLayer, LayerSlot> &&
                  l436_only<pulsar_indexed_view<const int, VisLayer>, VisLayer, TrunkLayer>,
              "a view takes its own space only (Nemotron-H: layer -> attention ordinal, then ordinal -> call)");
/* the tables are the arrays they replaced: a struct holding one is laid out as it was */
static_assert(sizeof(pulsar_layer_plan::kind) == sizeof(pulsar_layer_kind[PULSAR_FAMILY_MAX_LAYER]) &&
                  sizeof(pulsar_layer_plan::desc) == sizeof(pulsar_layer_desc[PULSAR_FAMILY_MAX_LAYER]) &&
                  sizeof(pulsar_state_plan::slot) == sizeof(uint32_t[PULSAR_STATE_SLOTS_MAX]) &&
                  sizeof(pulsar_state_set::t) == sizeof(void *[PULSAR_STATE_SLOTS_MAX][PULSAR_STATE_DECLS_MAX]) &&
                  sizeof(pulsar_state_set::ord) == PULSAR_STATE_SLOTS_MAX * PULSAR_STATE_DECLS_MAX &&
                  sizeof(pulsar_nemotron_h_shape::block_layer) == PULSAR_NEMOTRON_H_MAX_BLOCKS &&
                  sizeof(pulsar_gpu_graph::ms_n_comp) == sizeof(uint32_t[PULSAR_MSEQ_MAX][PULSAR_MAX_LAYER]) &&
                  sizeof(pulsar_gpu_graph::dspark_n_raw) == sizeof(uint32_t[3]) &&
                  sizeof(pulsar_bank_slabs::comp) == sizeof(void *[PULSAR_MAX_LAYER][PULSAR_MSEQ_MAX]) &&
                  std::is_standard_layout_v<pulsar_layer_plan> && std::is_trivial_v<pulsar_layer_plan> &&
                  std::is_trivially_copyable_v<pulsar_nemotron_h_shape> && std::is_trivially_copyable_v<pulsar_state_plan>,
              "an indexed table is laid out as the array it replaced (memset / calloc / struct copy as before)");

/* ---- 2 / 3. run time ---------------------------------------------------------------------------------------------- */

/// checks failed so far
static int n_fail;

/** One check's verdict, printed.  @param ok the verdict  @param fmt what was checked (printf) */
__attribute__((format(printf, 2, 3))) static void check(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) n_fail++;
}

/** The crossings against the open-coded expressions they replaced, on every plan shape the tables can hold. */
static void model_free(void) {
    fprintf(stderr, "layer-index gate: the crossings, model-free\n");
    bool slot_ok = true, draft_ok = true, back_ok = true, tp_ok = true, fam_ok = true, ds4_ok = true;
    for (uint32_t n_trunk = 1; n_trunk <= PULSAR_FAMILY_MAX_LAYER; n_trunk++) {
        pulsar_state_plan p{};
        p.n_trunk = n_trunk;
        p.n_slots = n_trunk;
        pulsar_layer_plan lp{};
        lp.n_layer = n_trunk;
        for (uint32_t il = 0; il < n_trunk; il++) {
            const LayerSlot s = pulsar_layer_slot_of(TrunkLayer{il});
            slot_ok = slot_ok && s.raw() == il && pulsar_slot_is_trunk(&p, s);          /* was: the slot IS il */
            back_ok = back_ok && pulsar_slot_trunk_layer(&p, s) == TrunkLayer{il};
            tp_ok = tp_ok && pulsar_tp_slot_of(s).raw() == il && gpu_graph_tp_slot(TrunkLayer{il}).raw() == il;
            fam_ok = fam_ok && pulsar_qwen_block_layer(SrcBlock{il}) == TrunkLayer{il} &&   /* was: tap_layer = tap_block */
                     pulsar_gemma4_block_layer(SrcBlock{il}) == TrunkLayer{il};
        }
        for (uint32_t l = 0; n_trunk + l < PULSAR_STATE_SLOTS_MAX; l++) {
            const LayerSlot s = pulsar_drafter_slot(&p, DraftLayer{l});
            draft_ok = draft_ok && s.raw() == n_trunk + l && !pulsar_slot_is_trunk(&p, s);   /* was: n_trunk + l */
            tp_ok = tp_ok && pulsar_tp_slot_of(s).raw() == n_trunk + l;
        }
        fam_ok = fam_ok && pulsar_qwen_mtp_slot(&lp).raw() == n_trunk &&                      /* was: plan.n_layer */
                 pulsar_qwen_mtp_slot(&lp) == pulsar_drafter_slot(&p, DraftLayer{0u});
    }
    for (uint32_t li = 0; li < 3u; li++)   /* was: the bare li handed to the trunk's FFN and its exchange */
        ds4_ok = ds4_ok && gpu_graph_drafter_ffn_layer(DraftLayer{li}) == TrunkLayer{li} &&
                 gpu_graph_drafter_tp_slot(DraftLayer{li}) == gpu_graph_tp_slot(TrunkLayer{li});
    check(slot_ok, "pulsar_layer_slot_of(il) is slot il, a trunk slot, for every plan of 1 .. %u layers",
          PULSAR_FAMILY_MAX_LAYER);
    check(back_ok, "pulsar_slot_trunk_layer undoes it");
    check(draft_ok, "pulsar_drafter_slot(plan, l) is slot n_trunk + l, a drafter's, up to the slot table's %u",
          (unsigned)PULSAR_STATE_SLOTS_MAX);
    check(tp_ok, "pulsar_tp_slot_of(slot) / gpu_graph_tp_slot(il) are the slot's / the layer's number");
    check(fam_ok, "Qwen's and Gemma 4's block -> layer are the same number; Qwen's MTP slot is n_layer = the first "
                  "drafter slot");
    check(ds4_ok, "DeepSeek's drafter layer li crosses to trunk layer li and exchanges in its slot");
    /* an appended slot is the next one, in order: what the drafters' "n_trunk + l" rests on */
    pulsar_state_plan q{};
    q.n_trunk = q.n_slots = 5u;
    bool app_ok = true;
    for (uint32_t l = 0; l < 4u; l++)
        app_ok = app_ok && pulsar_state_plan_add_slot(&q, 0u) == pulsar_drafter_slot(&q, DraftLayer{l});
    check(app_ok && q.n_slots == 9u, "pulsar_state_plan_add_slot appends drafter slot l at pulsar_drafter_slot(plan, l)");
    /* Nemotron-H's table against the grouping it encodes, on a synthetic interleave: mixer, mixer+moe, mixer, mixer+moe */
    pulsar_nemotron_h_shape s{};
    const uint8_t layer_of[6] = {0, 1, 1, 2, 3, 3}, sub_of[6] = {0, 0, 1, 0, 0, 1};
    bool nh_ok = true;
    for (uint32_t b = 0; b < 6u; b++) {
        s.block_layer[SrcBlock{b}] = layer_of[b];
        s.block_sub[SrcBlock{b}] = sub_of[b];
    }
    for (uint32_t b = 0; b < 6u; b++)
        nh_ok = nh_ok && pulsar_nemotron_h_block_layer(&s, SrcBlock{b}) == TrunkLayer{layer_of[b]} &&
                pulsar_nemotron_h_block_sub(&s, SrcBlock{b}) == sub_of[b];
    check(nh_ok, "Nemotron-H's block -> layer reads the shape's table (a block is not its layer: block 5 is layer 3)");
}

/** L436 piece 2, the fail-closed checks: a second drafter's slots are refused at the plan (the load's
 *  pulsar_state_plan_slots_fit says so; the create's plan check refuses it too), and a drafter's slot has no verify capture
 *  (fatal, graded in a child process). */
static void refusal_checks(void) {
    fprintf(stderr, "layer-index gate: the fail-closed checks (the refusal lines below are expected)\n");
    pulsar_block_drafter d{};
    d.n_layer = 3u;
    d.n_head = d.n_kv = 4u;
    d.head_dim = 64u;
    d.hidden = 256u;
    d.window = 128u;
    d.head_kind = PULSAR_BLOCK_HEAD_PLAIN;
    pulsar_engine *e = (pulsar_engine *)calloc(1, sizeof(pulsar_engine));
    e->block_drafter = &d;
    /* a trunk of 6 layers and the block drafter alone: its 3 slots follow the trunk's */
    pulsar_state_plan a{};
    a.name = "layer-index-gate";
    a.n_trunk = a.n_slots = 6u;
    pulsar_block_drafter_plan(e, &a);
    check(a.drafter_conflict == 0 && a.n_slots == 9u && pulsar_state_plan_slots_fit(&a, "test") &&
              (a.slot[pulsar_drafter_slot(&a, DraftLayer{2u})] >> d.decl_kv & 1u),
          "a block drafter alone takes slots n_trunk .. n_trunk + 2 and the plan fits");
    /* the same trunk with an MTP-style drafter slot already appended: the block drafter is refused, nothing declared */
    pulsar_state_plan b{};
    b.name = "layer-index-gate";
    b.n_trunk = b.n_slots = 6u;
    (void)pulsar_state_plan_add_slot(&b, 0u);
    const uint32_t n_decl = b.n_decl;
    pulsar_block_drafter_plan(e, &b);
    check(b.drafter_conflict == 1u && b.n_slots == 7u && b.n_decl == n_decl,
          "a block drafter after another drafter's slot is refused at the plan: no slot, no decl added");
    check(!pulsar_state_plan_slots_fit(&b, "test"), "and the load's verdict refuses the model");
    free(e);
    /* a drafter's slot has no verify capture: the offset is fatal, never the first trunk slot's */
    pulsar_state_set ss{};
    ss.plan = &a;
    check(pulsar_state_capture_offset(&ss, LayerSlot{5u}, 0u) == SlabOff{}, "a trunk slot's capture offset is read (ordinal 0 here)");
    fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        (void)pulsar_state_capture_offset(&ss, pulsar_drafter_slot(&a, DraftLayer{0u}), 0u);
        _exit(0);
    }
    int st = 0;
    const bool waited = pid > 0 && waitpid(pid, &st, 0) == pid;
    check(waited && WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "a drafter slot's capture offset is fatal (SIGABRT), not slot 0's rows");
}

/** The engine's declared state against the plan: assumptions a and b, and the conversions on the real shape.
 *  @param e the opened engine
 *  @return the drafter slots the state plan holds past the trunk's */
static uint32_t state_checks(const pulsar_engine *e) {
    const pulsar_layer_plan *lp = &e->plan;
    const pulsar_state_plan *sp = &e->state_plan;
    if (sp->n_slots == 0 && sp->n_trunk == 0) {
        check(e->family->id == PULSAR_FAMILY_ID_DEEPSEEK4 || !e->family->driver,
              "no declared state: a family the step driver does not run (its KV state is its own)");
        return 0;
    }
    check(sp->n_trunk == lp->n_layer, "a. the declared state's trunk slots are the plan's layers: n_trunk %u == n_layer %u",
          sp->n_trunk, lp->n_layer);
    check(sp->n_slots >= sp->n_trunk && sp->n_slots <= PULSAR_STATE_SLOTS_MAX && sp->slot_overflow == 0,
          "   the slots fit the table (%u of %u)", sp->n_slots, (unsigned)PULSAR_STATE_SLOTS_MAX);
    bool ok = true;
    for (TrunkLayer il{0u}; il.below(lp->n_layer); ++il) {
        const LayerSlot s = pulsar_layer_slot_of(il);
        ok = ok && s.raw() == il.raw() && pulsar_slot_is_trunk(sp, s) && pulsar_slot_trunk_layer(sp, s) == il;
    }
    check(ok, "   every layer's slot is its number and crosses back to it");
    const uint32_t n_draft = sp->n_slots - sp->n_trunk;
    ok = true;
    for (DraftLayer l{0u}; l.below(n_draft); ++l) {
        const LayerSlot s = pulsar_drafter_slot(sp, l);
        ok = ok && s.raw() == sp->n_trunk + l.raw() && s.below(sp->n_slots) && !pulsar_slot_is_trunk(sp, s);
    }
    check(ok, "b. the %u drafter slot(s) are pulsar_drafter_slot(plan, 0 .. %u), past the trunk's", n_draft, n_draft);
    return n_draft;
}

/** A block drafter's place in the state plan and its taps (assumptions b and c).
 *  @param e       the opened engine
 *  @param n_draft the drafter slots the plan holds
 *  @param tap_of  the family's block -> layer crossing, open-coded here as the reference */
template <class TapOf>
static void block_drafter_checks(const pulsar_engine *e, uint32_t n_draft, TapOf tap_of) {
    const pulsar_block_drafter *d = e->block_drafter;
    const pulsar_state_plan *sp = &e->state_plan;
    check(n_draft == d->n_layer, "b. the drafter slots are the block drafter's %u layers and no other drafter's (%u)",
          d->n_layer, n_draft);
    bool ok = n_draft == d->n_layer;
    for (DraftLayer l{0u}; ok && l.below(d->n_layer); ++l) {
        const uint32_t decl = (d->full_layers >> l.raw() & 1u) ? d->decl_kv_full : d->decl_kv;
        ok = decl < sp->n_decl && (sp->slot[pulsar_drafter_slot(sp, l)] >> decl & 1u);
    }
    check(ok, "   drafter layer l's context ring is declared on pulsar_drafter_slot(plan, l)");
    if (d->decl_anchor != UINT32_MAX)
        check(sp->slot[pulsar_drafter_slot(sp, DraftLayer{0u})] >> d->decl_anchor & 1u,
              "   the anchor row is declared on the drafter's first slot");
    ok = d->n_taps > 0;
    for (uint32_t j = 0; ok && j < d->n_taps; j++) {
        const uint32_t want = tap_of(d->tap_block[j].raw());
        ok = d->tap_layer[j].below(e->plan.n_layer) && d->tap_layer[j].raw() == want;
        if (!ok) fprintf(stderr, "       tap %u: block %u -> layer %u, the reference says %u of %u\n", j, d->tap_block[j].raw(),
                         d->tap_layer[j].raw(), want, e->plan.n_layer);
    }
    check(ok, "c. the %u tap(s) land on trunk layers through the family's block -> layer", d->n_taps);
}

/** Qwen: assumptions b - e on the loaded shape.  @param e the engine  @param n_draft drafter slots */
static void qwen_checks(const pulsar_engine *e, uint32_t n_draft) {
    const pulsar_qwen_weights *w = pulsar_qwen_of(e)->weights;
    const pulsar_qwen_shape *s = &g_qwen_shape;
    const pulsar_state_plan *sp = &e->state_plan;
    check(s->ple_layer.below(e->plan.n_layer), "d. the PLE layer %u is a trunk layer", s->ple_layer.raw());
    bool ok = true;
    for (LayerSlot k{0u}; k.below(sp->n_slots); ++k)
        ok = ok && ((sp->slot[k] >> PULSAR_QWEN_STATE_PLE_CONV & 1u) != 0) == (k == pulsar_layer_slot_of(s->ple_layer));
    check(ok, "   and its slot alone carries the PLE state (the MTP layer's slot never equals it)");
    ok = true;
    for (TrunkLayer il{0u}; il.below(e->plan.n_layer); ++il)   /* was: w->layer[il] */
        ok = ok && &pulsar_qwen_layer_w(w, pulsar_layer_slot_of(il)) == &w->layer.data()[il.raw()];
    check(ok && &pulsar_qwen_layer_w(w, pulsar_qwen_mtp_slot(&e->plan)) == &w->layer.data()[e->plan.n_layer],
          "d. the weights array is indexed by the slot: trunk layers, then the MTP layer at n_layer");
    if (w->mtp.present) {
        check(n_draft == 1u && !e->block_drafter &&
                  pulsar_qwen_mtp_slot(&e->plan) == pulsar_drafter_slot(sp, DraftLayer{0u}),
              "b. the MTP layer's slot is the plan's one drafter slot");
    } else {
        check(!e->block_drafter || !w->mtp.present, "b. the block drafter is selected and the MTP layer is not");
    }
    uint32_t tl = 0, te = 0, tv = 0;
    e->family->tp_shape(e, &tl, &te, &tv);
    ok = tl >= e->plan.n_layer;
    for (TrunkLayer il{0u}; il.below(e->plan.n_layer); ++il) ok = ok && pulsar_tp_slot_of(pulsar_layer_slot_of(il)).below(tl);
    if (w->mtp.present) ok = ok && pulsar_tp_slot_of(pulsar_qwen_mtp_slot(&e->plan)).below(tl);
    check(ok, "e. every exchanging slot is inside the slab's %u", tl);
    if (e->block_drafter) block_drafter_checks(e, n_draft, [](uint32_t b) { return b; });
}

/** Nemotron-H: assumptions b, c on the loaded shape.  @param e the engine  @param n_draft drafter slots */
static void nemotron_h_checks(const pulsar_engine *e, uint32_t n_draft) {
    const pulsar_nemotron_h_model *w = pulsar_nemotron_h_of(e);
    const pulsar_nemotron_h_shape *s = &w->shape;
    const pulsar_layer_plan *lp = &e->plan;
    /* the grouping re-derived from the plan: a layer is a mixer block, then its MoE block when it has one */
    uint32_t b = 0, n_attn = 0, n_kv = 0;
    bool ok = true;
    for (TrunkLayer il{0u}; il.below(lp->n_layer); ++il) {
        const bool moe = lp->desc[il].ffn == PULSAR_FFN_MOE;
        ok = ok && b < s->n_blocks && pulsar_nemotron_h_block_layer(s, SrcBlock{b}) == il &&
             pulsar_nemotron_h_block_sub(s, SrcBlock{b}) == PULSAR_BLOCK_TAP_MIXER;
        b++;
        if (moe) {
            ok = ok && b < s->n_blocks && pulsar_nemotron_h_block_layer(s, SrcBlock{b}) == il &&
                 pulsar_nemotron_h_block_sub(s, SrcBlock{b}) == PULSAR_BLOCK_TAP_FFN;
            b++;
        }
        n_attn += lp->desc[il].mixer == PULSAR_MIXER_ATTN_FULL;
        n_kv += e->state_plan.slot[pulsar_layer_slot_of(il)] >> PULSAR_NEMOTRON_H_STATE_KV & 1u;
    }
    check(ok && b == s->n_blocks, "c. the block -> layer table is the plan's grouping: %u blocks, %u layers", s->n_blocks,
          lp->n_layer);
    check(s->n_blocks > lp->n_layer, "   and a block's number is not its layer's (the last block %u is layer %u)",
          s->n_blocks - 1u, pulsar_nemotron_h_block_layer(s, SrcBlock{s->n_blocks - 1u}).raw());
    check(n_kv == n_attn && n_attn > 0 && n_attn < lp->n_layer,
          "   the attention layers are a strict subset (%u of %u): their calls and KV pools are compacted over the kind",
          n_attn, lp->n_layer);
    if (w->drafting) {
        check(n_draft == 1u && !e->block_drafter, "b. the MTP module's slot is the plan's one drafter slot");
        check(e->state_plan.slot[pulsar_drafter_slot(&e->state_plan, DraftLayer{0u})] >>
                      PULSAR_NEMOTRON_H_STATE_MTP_PEND & 1u,
              "   and carries the parked row");
    }
    if (e->block_drafter) {
        check(!w->drafting, "b. the block drafter is selected and the MTP module is not");
        /* the reference: count the layers opened up to block b, from the plan alone */
        block_drafter_checks(e, n_draft, [&](uint32_t blk) {
            uint32_t at = 0;
            for (uint32_t il = 0; il < lp->n_layer; il++) {
                const uint32_t n = 1u + (lp->desc[TrunkLayer{il}].ffn == PULSAR_FFN_MOE ? 1u : 0u);
                if (blk < at + n) return il;
                at += n;
            }
            return UINT32_MAX;
        });
    }
}

/** Gemma 4: the kinds on the loaded shape, and the drafter.  @param e the engine  @param n_draft drafter slots */
static void gemma4_checks(const pulsar_engine *e, uint32_t n_draft) {
    const pulsar_gemma4_shape *s = &pulsar_gemma4_of(e)->shape;
    uint32_t n_full = 0;
    bool ok = true;
    for (TrunkLayer il{0u}; il.below(e->plan.n_layer); ++il) {
        const bool full = pulsar_gemma4_is_full(s, il);
        n_full += full;
        ok = ok && full == ((s->full_layers >> il.raw() & 1u) != 0) &&                        /* was: the open-coded bit */
             full == (e->plan.desc[il].mixer == PULSAR_MIXER_ATTN_FULL) &&
             e->state_plan.slot[pulsar_layer_slot_of(il)] ==
                 1u << (full ? PULSAR_GEMMA4_STATE_KV_FULL : PULSAR_GEMMA4_STATE_KV_WIN);
    }
    check(ok && n_full > 0 && n_full < e->plan.n_layer,
          "the global layers (%u of %u) are the plan's full-attention layers, each slot its kind's KV", n_full,
          e->plan.n_layer);
    if (e->block_drafter) block_drafter_checks(e, n_draft, [](uint32_t b) { return b; });
    else check(n_draft == 0, "b. no drafter selected: no drafter slot");
}

/** DeepSeek: assumptions e, f on the loaded profile.  @param e the engine */
static void deepseek_checks(const pulsar_engine *e) {
    bool src_ok = true, ks_ok = true;
    for (TrunkLayer il{0u}; il.below(PULSAR_N_LAYER); ++il) {
        src_ok = src_ok && pulsar_layer_source(il).raw() == il.raw();
        const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
        ks_ok = ks_ok && (a->ratio == 0 ? a->kv_source.is_none() && a->index_source.is_none()
                                        : a->kv_source <= il && a->index_source <= il &&
                                              gpu_graph_kv_source(il) == a->kv_source);
    }
    check(e->plan.n_layer == PULSAR_N_LAYER, "the plan holds the profile's %u layers", (unsigned)PULSAR_N_LAYER);
    check(src_ok, "f. a served artifact's block is its own source layer (no fixture map)");
    check(ks_ok, "f. a layer's KV / index source is itself or an earlier layer (none() on a window-only layer)");
    uint32_t tl = 0, te = 0, tv = 0;
    e->family->tp_shape(e, &tl, &te, &tv);
    bool tp_ok = tl == PULSAR_N_LAYER;
    for (TrunkLayer il{0u}; il.below(PULSAR_N_LAYER); ++il) tp_ok = tp_ok && gpu_graph_tp_slot(il).below(tl);
    for (DraftLayer li{0u}; li.below(3u); ++li)
        tp_ok = tp_ok && gpu_graph_drafter_tp_slot(li).below(tl) && gpu_graph_drafter_ffn_layer(li).below(PULSAR_N_LAYER);
    check(tp_ok, "e/f. every trunk layer's and the drafter's three borrowed exchange slots are inside the slab's %u", tl);
}

/** Open one container inspect-only and run its family's checks.
 *  @param arg MODEL or MODEL@DRAFTER */
static void model_checks(const char *arg) {
    std::string path = arg, drafter;
    const size_t at = path.rfind('@');
    if (at != std::string::npos && path.find('/', at) == std::string::npos) {
        drafter = path.substr(at + 1);
        path.resize(at);
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path.c_str();
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = true;
    if (!drafter.empty()) opt.drafter = drafter.c_str();
    pulsar_engine *e = NULL;
    fprintf(stderr, "layer-index gate: open %s%s%s\n", path.c_str(), drafter.empty() ? "" : " --drafter ", drafter.c_str());
    const int rc = pulsar_engine_open(&e, &opt);
    check(rc == 0 && e, "the container opens (--inspect)");
    if (rc != 0 || !e) return;
    fprintf(stderr, "  family %s: %u layers, %u state slots (%u trunk)%s\n", e->family->arch, e->plan.n_layer,
            e->state_plan.n_slots, e->state_plan.n_trunk, e->block_drafter ? ", a block drafter" : "");
    TrunkLayer bad{0u};
    check(pulsar_layer_plan_check(&e->plan, &bad) == NULL && bad.is_none(), "the plan holds, and reports none() for no layer");
    const uint32_t n_draft = state_checks(e);
    switch (e->family->id) {
    case PULSAR_FAMILY_ID_QWEN4_EXP:  qwen_checks(e, n_draft); break;
    case PULSAR_FAMILY_ID_NEMOTRON_H: nemotron_h_checks(e, n_draft); break;
    case PULSAR_FAMILY_ID_GEMMA4:     gemma4_checks(e, n_draft); break;
    case PULSAR_FAMILY_ID_DEEPSEEK4:  deepseek_checks(e); break;
    default:
        /* an encoder, the page parser, or a family added since: it has no drafter and no kind-compacted storage this
         * gate knows of -- so it may have no drafter slot, and a new family that grows one must add its checks here */
        check(n_draft == 0 && !e->block_drafter,
              "b. a family with no checks of its own here declares no drafter slot (add them when it does)");
        break;
    }
    pulsar_engine_close(e);
}

/** @param argc the argument count  @param argv the containers (MODEL or MODEL@DRAFTER)
 *  @return 0 = every check passed */
int main(int argc, char **argv) {
    /* the opens take the engine's instance lock (PULSAR_LOCK_FILE): one of this run's own unless the caller names
     * one, so the gate neither waits on nor refuses beside another gate's open on the same box */
    char lock[64] = "/tmp/pulsar-layer-index-XXXXXX";
    bool own_lock = false;
    if (!getenv("PULSAR_LOCK_FILE")) {
        const int fd = mkstemp(lock);
        own_lock = fd >= 0;
        if (own_lock) {
            close(fd);
            setenv("PULSAR_LOCK_FILE", lock, 1);
        }
    }
    model_free();
    refusal_checks();
    for (int i = 1; i < argc; i++) model_checks(argv[i]);
    if (own_lock) unlink(lock);
    fprintf(stderr, "LAYER INDEX GATE: %s (%d container(s), %d failure(s))\n", n_fail ? "FAIL" : "PASS", argc - 1, n_fail);
    return n_fail ? 1 : 0;
}
