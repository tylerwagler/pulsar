/** @file
 *  The disk KV payload digest, v2 four lanes (pulsar_kvdigest.h). */
#include "pulsar_kvdigest.h"

#include <string.h>

namespace {
/** Lane i's starting value: v1's offset basis stepped by the golden ratio, so equal words in different lanes fold
 *  differently.  @param i the lane  @return its seed */
constexpr uint64_t lane_seed(uint32_t i) {
    return UINT64_C(1469598103934665603) + (uint64_t)i * UINT64_C(0x9E3779B97F4A7C15);
}
/** Fold one 64-bit word into a lane (v1's mix).  @param h the lane  @param w the word  @return the new lane */
inline uint64_t mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= UINT64_C(0x9E3779B97F4A7C15);
    return (h << 31) | (h >> 33);
}
/** Fold one whole word into the lane its index names.  @param d the digest  @param w the word */
inline void fold_word(pulsar_kvdigest *d, uint64_t w) {
    uint64_t &h = d->h[d->words % PULSAR_KVDIGEST_LANES];
    h = mix(h, w);
    d->words++;
}
}  // namespace

void pulsar_kvdigest_init(pulsar_kvdigest *d) {
    for (uint32_t i = 0; i < PULSAR_KVDIGEST_LANES; i++) d->h[i] = lane_seed(i);
    d->words = 0;
    d->tail = 0;
    d->ntail = 0;
}

void pulsar_kvdigest_update(pulsar_kvdigest *d, const void *ptr, uint64_t bytes) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (d->ntail != 0 && bytes != 0) {
        d->tail |= (uint64_t)(*p++) << (8u * d->ntail);
        d->ntail++;
        bytes--;
        if (d->ntail == 8u) {
            fold_word(d, d->tail);
            d->tail = 0;
            d->ntail = 0;
        }
    }
    while (bytes >= 8u && d->words % PULSAR_KVDIGEST_LANES != 0u) {   /* up to a lane-0 word */
        uint64_t w;
        memcpy(&w, p, 8);
        fold_word(d, w);
        p += 8;
        bytes -= 8;
    }
    /* the four lanes side by side: four independent multiply chains per 32 bytes */
    uint64_t h0 = d->h[0], h1 = d->h[1], h2 = d->h[2], h3 = d->h[3];
    uint64_t n4 = 0;
    while (bytes >= 32u) {
        uint64_t w[4];
        memcpy(w, p, 32);
        h0 = mix(h0, w[0]);
        h1 = mix(h1, w[1]);
        h2 = mix(h2, w[2]);
        h3 = mix(h3, w[3]);
        p += 32;
        bytes -= 32;
        n4++;
    }
    d->h[0] = h0;
    d->h[1] = h1;
    d->h[2] = h2;
    d->h[3] = h3;
    d->words += 4u * n4;
    while (bytes >= 8u) {
        uint64_t w;
        memcpy(&w, p, 8);
        fold_word(d, w);
        p += 8;
        bytes -= 8;
    }
    while (bytes != 0) {
        d->tail |= (uint64_t)(*p++) << (8u * d->ntail);
        d->ntail++;
        bytes--;
    }
}

uint64_t pulsar_kvdigest_final(const pulsar_kvdigest *d) {
    pulsar_kvdigest c = *d;
    if (c.ntail != 0) fold_word(&c, c.tail);
    uint64_t r = c.h[0];
    for (uint32_t i = 1; i < PULSAR_KVDIGEST_LANES; i++) r = mix(r, c.h[i]);
    return mix(r, d->words * 8u + d->ntail);   /* the byte count: a trailing zero byte is not the shorter payload */
}

uint64_t pulsar_kvdigest_of(const void *ptr, uint64_t bytes) {
    pulsar_kvdigest d;
    pulsar_kvdigest_init(&d);
    pulsar_kvdigest_update(&d, ptr, bytes);
    return pulsar_kvdigest_final(&d);
}

void pulsar_kvdigest_seal(uint8_t *payload, uint64_t bytes) {
    const uint64_t body = bytes - PULSAR_KVDIGEST_BYTES;
    const uint64_t dg = pulsar_kvdigest_of(payload, body);
    for (uint32_t i = 0; i < PULSAR_KVDIGEST_BYTES; i++) payload[body + i] = (uint8_t)(dg >> (8u * i));
}

bool pulsar_kvdigest_check(const uint8_t *payload, uint64_t bytes) {
    if (bytes < PULSAR_KVDIGEST_BYTES) return false;
    const uint64_t body = bytes - PULSAR_KVDIGEST_BYTES;
    uint64_t stored = 0;
    for (uint32_t i = 0; i < PULSAR_KVDIGEST_BYTES; i++) stored |= (uint64_t)payload[body + i] << (8u * i);
    return stored == pulsar_kvdigest_of(payload, body);
}
