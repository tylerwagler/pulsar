#ifndef PULSAR_DSML_H
#define PULSAR_DSML_H

#include <stdbool.h>
#include <stddef.h>

/* DSML -- the tool-call markup DeepSeek V4 samples -- has ONE authority for
 * its spelling in this tree: this header.  The server's prompt renderer
 * writes these literals, the server's parser and both stream projections
 * match them, and the agent's parser and detector match them; none of them
 * carries a copy (L184).  Adding a syntax means adding a row to
 * pulsar_dsml_syntaxes; every consumer loops over the table. */

#define PULSAR_DSML "｜DSML｜"
#define PULSAR_DSML_SHORT "DSML｜"
#define PULSAR_TOOL_CALLS_START "<" PULSAR_DSML "tool_calls>"
#define PULSAR_TOOL_CALLS_END "</" PULSAR_DSML "tool_calls>"
#define PULSAR_INVOKE_START "<" PULSAR_DSML "invoke"
#define PULSAR_INVOKE_END "</" PULSAR_DSML "invoke>"
#define PULSAR_PARAM_START "<" PULSAR_DSML "parameter"
#define PULSAR_PARAM_END "</" PULSAR_DSML "parameter>"
#define PULSAR_TOOL_CALLS_START_SHORT "<" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_TOOL_CALLS_END_SHORT "</" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_INVOKE_START_SHORT "<" PULSAR_DSML_SHORT "invoke"
#define PULSAR_INVOKE_END_SHORT "</" PULSAR_DSML_SHORT "invoke>"
#define PULSAR_PARAM_START_SHORT "<" PULSAR_DSML_SHORT "parameter"
#define PULSAR_PARAM_END_SHORT "</" PULSAR_DSML_SHORT "parameter>"

/** The six marker literals of one DSML spelling. */
typedef struct {
    const char *tool_calls_start;  ///< opens the tool-calls block
    const char *tool_calls_end;    ///< closes it
    const char *invoke_start;      ///< opens one invocation (attributes follow, then '>')
    const char *invoke_end;        ///< closes it
    const char *param_start;       ///< opens a parameter (attributes follow, then '>')
    const char *param_end;         ///< closes it
} pulsar_dsml_syntax;

/** Rows: [0] canonical "<｜DSML｜...", [1] the model's frequent
 * first-bar-omitted "<DSML｜...", [2] plain XML "<tool_calls>".  Row 0 is the
 * spelling the renderer WRITES; rows 1-2 are spellings the model has been
 * observed to sample.  Order is the parser's preference when a text carries
 * more than one. */
#define PULSAR_DSML_SYNTAXES 3
extern const pulsar_dsml_syntax pulsar_dsml_syntaxes[PULSAR_DSML_SYNTAXES];

/** The renderer-side spelling (row 0). */
#define PULSAR_DSML_CANONICAL (&pulsar_dsml_syntaxes[0])

/* ---- the one entity encode/decode pair ----------------------------------
 *
 * The renderer escapes attribute values (tool and parameter names) so a name
 * can carry '&', '<', '>' or '"'; the parser undoes exactly those entities
 * in attributes AND in string parameter values (the model is told to write
 * the closing parameter tag inside a value as "&lt;/｜DSML｜parameter>").
 * "&apos;" is decode-only: the renderer never writes it, the model does. */

/** Malloc'd copy of `s` with every entity the decoder knows replaced by its
 * character; an '&' that starts no known entity is kept.  Never NULL
 * (NULL input reads as ""). */
char *pulsar_dsml_unescape(const char *s);

/** Malloc'd copy of `s` with '&', '<', '>' and '"' as entities -- the
 * attribute-value encoding pulsar_dsml_unescape reverses.  Never NULL. */
char *pulsar_dsml_escape_attr(const char *s);

/** The value of attribute `name` in `tag` (an opening tag's bytes), decoded
 * with pulsar_dsml_unescape; malloc'd, NULL when the attribute is absent or
 * unterminated. */
char *pulsar_dsml_attr(const char *tag, const char *name);

#endif
