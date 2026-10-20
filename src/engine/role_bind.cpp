/* role_bind.cpp -- the role-driven binder (L295 E1).  The contract is role_bind.h. */
#include "pulsar_engine_internal.h"
#include "role_bind.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct pulsar_role_binder {
    const pulsar_model *m;
    const char *owner;
    std::unordered_map<std::string, uint64_t> by_key;   ///< key -> directory index
    std::vector<uint8_t> bound;                          ///< per directory entry: 1 bound, 2 unloaded, 3 not in scope
    std::map<std::string, std::pair<uint64_t, uint64_t>> price;   ///< role -> (tensors, bytes), printed in role order
};

static std::string role_key_str(const char *role, uint64_t role_len, const char *block, uint64_t block_len,
                                bool has_layer, int32_t layer, bool has_index, int32_t index) {
    std::string k(role, role_len);
    k += '\x1f';
    if (block) k.append(block, block_len);
    k += '\x1f' + (has_layer ? std::to_string(layer) : std::string()) + '\x1f' +
         (has_index ? std::to_string(index) : std::string());
    return k;
}

static std::string key_of(const pulsar_role_key &k) {
    return role_key_str(k.role, strlen(k.role), k.block, k.block ? strlen(k.block) : 0, k.layer >= 0, k.layer,
                        k.index >= 0, k.index);
}

static std::string key_of(const pulsar_tensor &t) {
    return role_key_str(t.bind_role.ptr, t.bind_role.len, t.bind_block.ptr, t.bind_block.len, t.has_bind_layer,
                        t.bind_layer, t.has_bind_index, t.bind_index);
}

/* "role r, block b, layer l, index i" -- the parts present */
static std::string describe(const char *role, const char *block, int32_t layer, int32_t index) {
    std::string s = std::string("role ") + role;
    if (block) s += std::string(", block ") + block;
    if (layer >= 0) s += ", layer " + std::to_string(layer);
    if (index >= 0) s += ", index " + std::to_string(index);
    return s;
}

static std::string describe(const pulsar_tensor &t) {
    const std::string role(t.bind_role.ptr ? t.bind_role.ptr : "", t.bind_role.len);
    const std::string block(t.bind_block.ptr ? t.bind_block.ptr : "", t.bind_block.len);
    return describe(role.c_str(), t.bind_block.len ? block.c_str() : NULL, t.has_bind_layer ? t.bind_layer : -1,
                    t.has_bind_index ? t.bind_index : -1);
}

/* the binder over the entries `in_scope` admits (the whole directory, or a set of blocks) */
template <class Scope> static pulsar_role_binder *binder_open(const pulsar_model *m, const char *owner, Scope in_scope) {
    pulsar_role_binder *b = new pulsar_role_binder();
    b->m = m;
    b->owner = owner;
    b->bound.assign(m->n_tensors, 0);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor &t = m->tensors[i];
        if (!in_scope(t)) {
            b->bound[i] = 3;
            continue;
        }
        if (t.bind_role.len == 0) {
            fprintf(stderr, "pulsar: %s: tensor %.*s carries no bind role -- a role-bound container records one for "
                            "every tensor (tools/container); refusing\n", owner, (int)t.name.len, t.name.ptr);
            pulsar_load_refuse();
            continue;
        }
        const auto ins = b->by_key.emplace(key_of(t), i);
        if (!ins.second) {
            const pulsar_tensor &o = m->tensors[ins.first->second];
            fprintf(stderr, "pulsar: %s: tensors %.*s and %.*s carry one bind record (%s) -- refusing\n", owner,
                    (int)o.name.len, o.name.ptr, (int)t.name.len, t.name.ptr, describe(t).c_str());
            pulsar_load_refuse();
        }
    }
    return b;
}

pulsar_role_binder *pulsar_role_binder_open(const pulsar_model *m, const char *owner) {
    return binder_open(m, owner, [](const pulsar_tensor &) { return true; });
}

pulsar_role_binder *pulsar_role_binder_open_blocks(const pulsar_model *m, const char *owner, const char *const *blocks) {
    return binder_open(m, owner, [blocks](const pulsar_tensor &t) {
        for (const char *const *bl = blocks; *bl; bl++)
            if (t.bind_block.len == strlen(*bl) && memcmp(t.bind_block.ptr, *bl, t.bind_block.len) == 0) return true;
        return false;
    });
}

const pulsar_tensor *pulsar_role_binder_find(const pulsar_role_binder *b, pulsar_role_key k) {
    const auto it = b->by_key.find(key_of(k));
    return it == b->by_key.end() ? NULL : &b->m->tensors[it->second];
}

static bool role_admit(const pulsar_role_binder *b, const pulsar_tensor *t, const pulsar_role_admit &a) {
    if (t->type < 32u && (a.plain >> t->type & 1u)) return true;
    if (a.linear) return pulsar_tensor_admit_role(t, b->owner, a.wrole, a.acts);
    std::string want;
    for (uint32_t type = 0; type < 32u; type++)
        if (a.plain >> type & 1u) want += std::string(want.empty() ? "" : " or ") + tensor_type_name(type);
    return pulsar_tensor_admit(t, b->owner, false, want.c_str());
}

pulsar_tensor *pulsar_role_bind(pulsar_role_binder *b, pulsar_role_key k, const pulsar_role_admit &a, uint32_t nd,
                                uint64_t d0, uint64_t d1, uint64_t d2) {
    const uint64_t dims[3] = {d0, d1, d2};
    return pulsar_role_bind_n(b, k, a, nd, dims);
}

pulsar_tensor *pulsar_role_bind_n(pulsar_role_binder *b, pulsar_role_key k, const pulsar_role_admit &a, uint32_t nd,
                                  const uint64_t *dims) {
    const auto it = b->by_key.find(key_of(k));
    if (it == b->by_key.end()) {
        fprintf(stderr, "pulsar: %s: the container has no tensor for %s -- refusing\n", b->owner,
                describe(k.role, k.block, k.layer, k.index).c_str());
        pulsar_load_refuse();
        return NULL;
    }
    pulsar_tensor *t = &b->m->tensors[it->second];
    if (b->bound[it->second]) {
        fprintf(stderr, "pulsar: %s: %s is declared twice by the family -- refusing\n", b->owner,
                describe(k.role, k.block, k.layer, k.index).c_str());
        pulsar_load_refuse();
        return NULL;
    }
    b->bound[it->second] = 1;   /* declared: whatever its verdict, it is not an undeclared extra at the close */
    bool ok = nd <= PULSAR_MAX_DIMS && t->ndim == nd;
    for (uint32_t i = 0; ok && i < nd; i++) ok = t->dim[i] == dims[i];
    if (!ok) {
        fprintf(stderr, "pulsar: %s: tensor %.*s has ne [", b->owner, (int)t->name.len, t->name.ptr);
        for (uint32_t i = 0; i < t->ndim; i++) fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)t->dim[i]);
        fprintf(stderr, "], want [");
        for (uint32_t i = 0; i < nd && i < PULSAR_MAX_DIMS; i++)
            fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)dims[i]);
        fprintf(stderr, "]\n");
    }
    ok = role_admit(b, t, a) && ok;
    if (!ok) {
        pulsar_load_refuse();
        return NULL;
    }
    auto &p = b->price[k.role];
    p.first++;
    p.second += t->bytes;
    return t;
}

uint64_t pulsar_role_binder_unload(pulsar_role_binder *b, const char *block) {
    pulsar_model *m = const_cast<pulsar_model *>(b->m);   /* the residency mark is the model's (staging reads it) */
    const size_t bl = strlen(block);
    uint64_t n = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor &t = m->tensors[i];
        if (t.bind_block.len != bl || memcmp(t.bind_block.ptr, block, bl) != 0 || b->bound[i]) continue;
        if (!m->unloaded) m->unloaded = (uint8_t *)xcalloc((size_t)m->n_tensors, 1);
        m->unloaded[i] = 1;
        b->bound[i] = 2;   /* declared, unloaded */
        n++;
    }
    return n;
}

bool pulsar_role_binder_close(pulsar_role_binder *b, pulsar_role_census *out) {
    const pulsar_model *m = b->m;
    pulsar_role_census c = {0, 0, 0, 0, 0, 0};
    uint32_t extra = 0;
    uint64_t in_scope = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor &t = m->tensors[i];
        if (b->bound[i] == 3) continue;   /* not this binder's (pulsar_role_binder_open_blocks) */
        in_scope++;
        c.directory_bytes += t.bytes;
        if (b->bound[i] == 2) {
            c.unloaded++;
            c.unloaded_bytes += t.bytes;
            continue;
        }
        if (b->bound[i] || t.bind_role.len == 0) continue;   /* a roleless entry was refused at the open */
        fprintf(stderr, "pulsar: %s: tensor %.*s (%s) is in the container but the family declares no such tensor -- "
                        "refusing\n", b->owner, (int)t.name.len, t.name.ptr, describe(t).c_str());
        pulsar_load_refuse();
        extra++;
    }
    fprintf(stderr, "pulsar: %s: tensors bound by role:", b->owner);
    for (const auto &kv : b->price) {
        c.tensors += kv.second.first;
        c.bytes += kv.second.second;
        c.roles++;
        fprintf(stderr, " %s %llu (%.1f MiB)", kv.first.c_str(), (unsigned long long)kv.second.first,
                (double)kv.second.second / 1048576.0);
    }
    fprintf(stderr, "\n");
    const bool whole = extra == 0 && c.tensors + c.unloaded == in_scope &&
                       c.bytes + c.unloaded_bytes == c.directory_bytes;
    fprintf(stderr, "pulsar: %s: %llu of %llu tensors bound in %u roles, %.3f GiB priced of the directory's %.3f GiB "
                    "(%llu tensors, %.3f GiB declared unloaded)%s\n",
            b->owner, (unsigned long long)c.tensors, (unsigned long long)in_scope, c.roles,
            (double)c.bytes / 1073741824.0, (double)c.directory_bytes / 1073741824.0, (unsigned long long)c.unloaded,
            (double)c.unloaded_bytes / 1073741824.0, whole ? " (every staged byte is a declared role's)" : " -- refusing");
    if (out) *out = c;
    delete b;
    return whole;
}
