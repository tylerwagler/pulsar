/** @file role_bind.h
 *  @brief The role-driven binder: a family binds its container's tensors by bind records, not by names. */
/* role_bind.h -- the role-driven binder (L295 E1): a family binds its container's tensors by the builder's BIND
 * RECORDS, never by parsing names.
 *
 * The builder (tools/container) writes each tensor's identity into its pulsar.tensors / pulsar.experts record as
 * `role` (a family-neutral word for what the tensor IS: attn_q, ffn_norm, expert_gate, head, ...), `block` (the
 * stack it sits in: `layers`, `mtp`; absent at the top level), `layer` and `index` (the shared expert j, the tower's
 * linear i).  The safetensors reader carries them onto the directory (pulsar_tensor::bind_*).  A family declares
 * what its model holds -- for each (role, block, layer, index), the logical dims and the formats its readers take --
 * and this binder answers with the tensor or a refusal, then proves the declaration and the container agree BOTH
 * ways: every declared tensor is present, and every directory entry was declared (no tensor rides along unread).
 *
 * The same pass prices the load: each bound tensor's bytes are charged to its role, and the close asserts the sum
 * is every byte of the directory -- so what the core stages (accelerator_cache_model_tensors stages the whole
 * directory) and what admission charges (pulsar_engine_weights_resident_bytes) are exactly the declared roles.
 *
 * Mechanics only, the same for every family; the declaration (which roles a layer kind holds, their dims, their
 * formats) is the family's.  Refusals follow the loader's one policy (family.cpp): said where found, counted with
 * pulsar_load_refuse, the stage finishes so one load reports every problem, the family fails at its next stage
 * boundary.  Nothing here is named after a family; DeepSeek and Qwen still bind by name and can move onto it.
 *
 * Include after pulsar_engine_internal.h. */
#ifndef PULSAR_ROLE_BIND_H
#define PULSAR_ROLE_BIND_H

/** Which tensor: a role at the top level (block NULL, layer and index -1) or inside a block. */
typedef struct {
    const char *role;   ///< the builder's role word
    const char *block;  ///< `layers`, `mtp`, ...; NULL = top level
    int32_t layer;      ///< the block's layer; -1 = none
    int32_t index;      ///< the role's index (shared expert j, tower linear i); -1 = none
} pulsar_role_key;

/** The formats a role's reader takes: plain stored types read as they are, and/or -- for a linear -- whatever the
 *  format registry (weight_format.cpp) serves for `wrole` at the activations `acts`.  A tensor in neither is refused
 *  by name, with the formats that would serve. */
typedef struct {
    uint32_t plain;              ///< PULSAR_ROLE_PLAIN(PULSAR_TENSOR_*) bits
    bool linear;                 ///< also admit through the format registry
    pulsar_weight_role wrole;    ///< the registry's role (when linear)
    uint32_t acts;               ///< PULSAR_ACTS(...) mask (when linear)
} pulsar_role_admit;
/** The pulsar_role_admit::plain bit of stored tensor type `type` (a PULSAR_TENSOR_* value, < 32). */
#define PULSAR_ROLE_PLAIN(type) (1u << (type))

/** What the bound tensors cost: the binder's census at its close. */
typedef struct {
    uint64_t tensors;            ///< tensors bound
    uint64_t bytes;              ///< their stored bytes (== directory_bytes after a clean close)
    uint64_t directory_bytes;    ///< every directory entry's stored bytes
    uint32_t roles;              ///< distinct roles bound
    uint64_t unloaded;           ///< L298: entries declared unloaded (pulsar_role_binder_unload)
    uint64_t unloaded_bytes;     ///< their stored bytes: never staged, never priced
} pulsar_role_census;

/** The role binder (opaque; role_bind.cpp): opened by pulsar_role_binder_open / _open_blocks, freed by its close. */
typedef struct pulsar_role_binder pulsar_role_binder;

/** Index `m`'s directory by bind record.  Refuses (said, counted) every entry with no role, and every pair of
 *  entries with one key.  `owner` names the family in every message.  Never NULL. */
/// @param m     the model whose directory is indexed; borrowed, must outlive the binder
/// @param owner the family's name for messages; borrowed
/// @return the binder (heap; pulsar_role_binder_close frees it)
pulsar_role_binder *pulsar_role_binder_open(const pulsar_model *m, const char *owner);

/** L299: a binder over the entries whose bind block is one of `blocks` (NULL-terminated) only -- a family that
 *  binds the rest of its container by name (Qwen) binds a role-recorded part of it, its block drafter, here.  An entry
 *  outside the blocks is not this binder's: never refused (with or without a role), never priced; the close's census
 *  is the blocks'.  Never NULL. */
/// @param m      the model; borrowed
/// @param owner  the family's name for messages; borrowed
/// @param blocks NULL-terminated block names in scope; read at every open-time test only
/// @return the binder (heap; pulsar_role_binder_close frees it)
pulsar_role_binder *pulsar_role_binder_open_blocks(const pulsar_model *m, const char *owner, const char *const *blocks);

/** L299: the tensor at `k` when the container holds one (nothing bound, nothing said): a declaration's optional
 *  tensor, whose dims the caller then binds it at. */
/// @param b the binder
/// @param k the key
/// @return the directory entry (the model's), or NULL when the container has none
const pulsar_tensor *pulsar_role_binder_find(const pulsar_role_binder *b, pulsar_role_key k);

/** Bind the tensor at `k`: present, its dims [d0, d1, d2][:nd] (ne order: innermost first), its format admitted by
 *  `a`.  It is charged to its role.  NULL = refused (said, counted); a key bound twice is refused too. */
/// @param b  the binder
/// @param k  the key
/// @param a  the formats admitted
/// @param nd the tensor's dim count (<= 3 here)
/// @param d0 dim 0 (innermost)
/// @param d1 dim 1
/// @param d2 dim 2
/// @return the bound tensor (the model's directory entry), or NULL
pulsar_tensor *pulsar_role_bind(pulsar_role_binder *b, pulsar_role_key k, const pulsar_role_admit &a, uint32_t nd,
                                uint64_t d0, uint64_t d1 = 0, uint64_t d2 = 0);
/** The same for a tensor of nd <= PULSAR_MAX_DIMS dims (ne order) -- e.g. a checkpoint's [1, 1, n, 1, w] table. */
/// @param b    the binder
/// @param k    the key
/// @param a    the formats admitted
/// @param nd   the tensor's dim count, <= PULSAR_MAX_DIMS
/// @param dims [nd] the dims, innermost first
/// @return the bound tensor (the model's directory entry), or NULL
pulsar_tensor *pulsar_role_bind_n(pulsar_role_binder *b, pulsar_role_key k, const pulsar_role_admit &a, uint32_t nd,
                                  const uint64_t *dims);

/** L298: every entry of block `block` is DECLARED and NOT LOADED -- the open did not select what it serves (a drafter
 *  the engine does not run): none is bound, staged (pulsar_model::unloaded) or priced, and the close accounts for its
 *  bytes as unloaded, not as undeclared extras.  Returns the entries so marked. */
/// @param b     the binder
/// @param block the block's name
/// @return entries newly marked unloaded (bound ones are left alone)
uint64_t pulsar_role_binder_unload(pulsar_role_binder *b, const char *block);

/** Close the binding: every entry not bound is refused (said, counted) -- the container holds a tensor the family
 *  did not declare.  Prints the price by role, fills `out`, frees `b`.  false = something was refused. */
/// @param b   the binder; freed, invalid after the call
/// @param out receives the census when non-NULL
/// @return true when every in-scope entry is bound or unloaded and every byte priced
bool pulsar_role_binder_close(pulsar_role_binder *b, pulsar_role_census *out);

/** The key of a top-level role, of a role in a block's layer, and of an indexed one. */
/// @param role  the role word
/// @param index the role's index, -1 = none
/// @return the key (top level: block NULL, layer -1)
static inline pulsar_role_key pulsar_role_top(const char *role, int32_t index = -1) {
    return pulsar_role_key{role, NULL, -1, index};
}
/** The key of a role in block `block`'s layer `layer` (optionally indexed).
 *  @param role  the role word
 *  @param block the block's name
 *  @param layer the layer, -1 = none
 *  @param index the role's index, -1 = none
 *  @return the key */
static inline pulsar_role_key pulsar_role_in(const char *role, const char *block, int32_t layer, int32_t index = -1) {
    return pulsar_role_key{role, block, layer, index};
}

#endif /* PULSAR_ROLE_BIND_H */
