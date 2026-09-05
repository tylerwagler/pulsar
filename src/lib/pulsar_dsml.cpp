#include "pulsar_dsml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



const pulsar_dsml_syntax pulsar_dsml_syntaxes[PULSAR_DSML_SYNTAXES] = {
    {
        PULSAR_TOOL_CALLS_START, PULSAR_TOOL_CALLS_END,
        PULSAR_INVOKE_START, PULSAR_INVOKE_END,
        PULSAR_PARAM_START, PULSAR_PARAM_END,
    },
    {
        PULSAR_TOOL_CALLS_START_SHORT, PULSAR_TOOL_CALLS_END_SHORT,
        PULSAR_INVOKE_START_SHORT, PULSAR_INVOKE_END_SHORT,
        PULSAR_PARAM_START_SHORT, PULSAR_PARAM_END_SHORT,
    },
    {
        "<tool_calls>", "</tool_calls>",
        "<invoke", "</invoke>",
        "<parameter", "</parameter>",
    },
};



/* The entity table: one row per entity the decoder knows.  `encode` marks
 * the rows the attribute encoder writes; the rest are decode-only. */
static const struct {
    const char *entity;
    char ch;
    bool encode;
} dsml_entities[] = {
    {"&amp;", '&', true},
    {"&lt;", '<', true},
    {"&gt;", '>', true},
    {"&quot;", '"', true},
    {"&apos;", '\'', false},
};



/* Minimal growable byte builder; malloc-backed, aborts on OOM like the
 * server's and agent's own x-allocators (an out-of-memory tool name is not a
 * condition either program recovers from). */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} dsml_buf;

static void dsml_buf_reserve(dsml_buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + extra + 1) cap *= 2;
    char *p = (char *)realloc(b->ptr, cap);
    if (!p) {
        fputs("pulsar_dsml: out of memory\n", stderr);
        abort();
    }
    b->ptr = p;
    b->cap = cap;
}

static void dsml_buf_append(dsml_buf *b, const char *s, size_t n) {
    dsml_buf_reserve(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static char *dsml_buf_take(dsml_buf *b) {
    dsml_buf_reserve(b, 0);   /* an empty result is still a NUL-terminated allocation */
    b->ptr[b->len] = '\0';
    return b->ptr;
}



char *pulsar_dsml_unescape(const char *s) {
    dsml_buf b = {0};
    for (s = s ? s : ""; *s; s++) {
        if (*s != '&') {
            dsml_buf_append(&b, s, 1);
            continue;
        }
        size_t i;
        for (i = 0; i < sizeof dsml_entities / sizeof dsml_entities[0]; i++) {
            const size_t n = strlen(dsml_entities[i].entity);
            if (!strncmp(s, dsml_entities[i].entity, n)) {
                dsml_buf_append(&b, &dsml_entities[i].ch, 1);
                s += n - 1;
                break;
            }
        }
        if (i == sizeof dsml_entities / sizeof dsml_entities[0]) dsml_buf_append(&b, s, 1);
    }
    return dsml_buf_take(&b);
}



char *pulsar_dsml_escape_attr(const char *s) {
    dsml_buf b = {0};
    for (s = s ? s : ""; *s; s++) {
        size_t i;
        for (i = 0; i < sizeof dsml_entities / sizeof dsml_entities[0]; i++) {
            if (dsml_entities[i].encode && *s == dsml_entities[i].ch) {
                dsml_buf_append(&b, dsml_entities[i].entity, strlen(dsml_entities[i].entity));
                break;
            }
        }
        if (i == sizeof dsml_entities / sizeof dsml_entities[0]) dsml_buf_append(&b, s, 1);
    }
    return dsml_buf_take(&b);
}



char *pulsar_dsml_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    size_t n = 0;
    while (p[n] && p[n] != '"') n++;
    if (!p[n]) return NULL;   /* unterminated attribute */
    dsml_buf raw = {0};
    dsml_buf_append(&raw, p, n);
    char *raw_s = dsml_buf_take(&raw);
    char *decoded = pulsar_dsml_unescape(raw_s);
    free(raw_s);
    return decoded;
}
