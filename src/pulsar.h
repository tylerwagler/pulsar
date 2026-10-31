/** @file
 *  The public engine API: model load, sessions, tokens, chat rendering, sampling and the
 *  speculative lanes -- the boundary the server, CLIs and agent build on. */
#ifndef PULSAR_H
#define PULSAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "lib/pulsar_units.h"   /* KvRows / KvBytes: the KV accounting's units (L333) */


/* Public engine boundary.
 *
 * The CLI and server should treat pulsar_engine as the loaded model and
 * pulsar_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * pulsar_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

/** The compute backend an engine runs on; CUDA is the only one this build has. */
typedef enum {
    PULSAR_BACKEND_CUDA,
} pulsar_backend;

/** Thinking mode and reasoning effort are ONE value (L218, V4.1): 0 is
 * thinking off; 1..100 is thinking on at that effort budget, the integer the
 * V4.1 encoder renders as "Reasoning Effort: N (range 1-100, ...)" before
 * the system message.  The three names the reference encoder accepts are
 * presets on the same axis -- low 50, high 75, max 100 -- and "high" is the
 * reference default.  There is no "thinking on, no prefix" mode any more:
 * V4.1 always renders the effort line when thinking. */
typedef int pulsar_think_mode;
enum {
    PULSAR_THINK_NONE    = 0,
    PULSAR_THINK_LOW     = 50,
    PULSAR_THINK_HIGH    = 75,
    PULSAR_THINK_MAX     = 100,
    PULSAR_THINK_DEFAULT = PULSAR_THINK_HIGH,
    PULSAR_THINK_EFFORT_MIN = 1,
    PULSAR_THINK_EFFORT_MAX = 100,
};

/** A pulsar_log line's kind: picks its colour on a terminal (DEFAULT prints uncoloured). */
typedef enum {
    PULSAR_LOG_DEFAULT,
    PULSAR_LOG_PREFILL,
    PULSAR_LOG_GENERATION,
    PULSAR_LOG_KVCACHE,
    PULSAR_LOG_TOOL,
    PULSAR_LOG_WARNING,
    PULSAR_LOG_TIMING,
    PULSAR_LOG_OK,
    PULSAR_LOG_ERROR,
} pulsar_log_type;

/** The tokenizer's image placeholder: the ONE text a renderer writes where an
 * image belongs.  `pulsar_expand_image_placeholders()` replaces every
 * occurrence with that image's sentinel block; its id is resolved from the
 * vocab by string (never hard-coded), so an artifact without the token simply
 * cannot serve images. */
#define PULSAR_IMAGE_PLACEHOLDER "<｜deepseek_image｜>"

/** One image to place in a prompt.  `start_pos` is the token index of the
 * image BLOCK's first slot -- the reference's `ImageInput.start`, i.e. the
 * length of the prompt at the moment the block was appended.  The renderer
 * writes the block's ids as `vocab_size + role` (out-of-vocab sentinels, per the
 * reference), so they are ids the tokenizer never produces and the embedder
 * zero-masks; the block begins with a compressor pad, and its IMAGE_START
 * sentinel therefore sits a few slots in.  `bytes`/`len` are the encoded image
 * file (PNG or JPEG). */
typedef struct {
    /** The encoded image file (PNG or JPEG); borrowed, the caller keeps it alive for the call. */
    const uint8_t *bytes;
    /** Bytes at `bytes`. */
    size_t         len;
    /** Token index of the image block's first slot in the prompt (the length of the prompt when the block was appended). */
    int            start_pos;
} pulsar_image_ref;

/** Growable token vector. Owns `v`; free with pulsar_tokens_free().
 *
 * `len` is the token count, `cap` the allocated slots. Many APIs accept a
 * borrowed view built by pointing `v` at someone else's buffer and setting
 * len == cap -- see the prefix helpers -- in which case it must NOT be freed. */
typedef struct {
    int *v;    ///< token ids, `cap` slots allocated, `len` in use
    int len;   ///< number of valid tokens
    int cap;   ///< allocated capacity in tokens
} pulsar_tokens;

/** One scored candidate token, as returned by the top-k/logprob helpers. */
typedef struct {
    int id;         ///< token id
    float logit;    ///< raw pre-softmax score
    float logprob;  ///< natural log of the softmax probability
} pulsar_token_score;

/** Sampling temperature a request gets when it sets none (the server's and the agent's default). */
#define PULSAR_DEFAULT_TEMPERATURE 1.0f
/** Nucleus mass a request gets when it sets none: 1 = off. */
#define PULSAR_DEFAULT_TOP_P 1.0f
/** Min-p floor a request gets when it sets none (relative to the best token's probability). */
#define PULSAR_DEFAULT_MIN_P 0.05f
/** The prefill chunk an engine uses when pulsar_engine_options::prefill_chunk
 * is 0 (and PULSAR_CUDA_PREFILL_CHUNK is unset): the production grid (L183). */
#define PULSAR_PREFILL_CHUNK_DEFAULT 4096u

/** A loaded model (opaque): weights, tokenizer and family; opened by pulsar_engine_open, freed by pulsar_engine_close. */
typedef struct pulsar_engine pulsar_engine;
/** One mutable inference timeline on an engine (opaque): its KV cache (a bank pool), logits and token history;
 *  pulsar_session_create / pulsar_session_free. */
typedef struct pulsar_session pulsar_session;
/** A model family's implementation table (opaque here; src/engine/family.h), chosen at open from the artifact. */
typedef struct pulsar_family pulsar_family;

/** The model family the loaded artifact belongs to (L251): chosen once at
 * pulsar_engine_open from the artifact's `general.architecture`, never
 * switched.  Values are stable. */
typedef enum {
    PULSAR_FAMILY_ID_DEEPSEEK4 = 0,  ///< DeepSeek V4 Flash (0731), Vision-Exp, V4.1 Flash
    PULSAR_FAMILY_ID_QWEN4_EXP = 1,  ///< Qwen3.8-Flash-Next (HF model_type qwen4_exp)
    PULSAR_FAMILY_ID_INKLING   = 2,  ///< Thinking Machines' Inkling (HF model_type inkling_mm_model; src/families/inkling)
    PULSAR_FAMILY_ID_NEMOTRON_H = 3, ///< NVIDIA Nemotron-H (Nemotron 3.5 Super VL; HF nemotron_h_omni; src/families/nemotron_h)
    PULSAR_FAMILY_ID_NEMOTRON_ENC = 4, ///< NVIDIA Nemotron-3.5 Embed / Rerank encoders (HF mistral3; src/families/nemotron_enc)
    PULSAR_FAMILY_ID_NEMOTRON_PARSE = 5, ///< NVIDIA Nemotron-3.5 Parse (HF nemotron_parse; src/families/nemotron_parse)
} pulsar_family_id;

/** The conversation format a front end must render for the loaded model: the
 * chat template, the reasoning-effort spelling, and the tool-call syntax all
 * follow it.  One fact with one authority (the family); a renderer that has
 * no arm for a format refuses it. */
typedef enum {
    PULSAR_CHAT_DS4_V4  = 0,  ///< DeepSeek V4 (0731) template
    PULSAR_CHAT_DS4_V41 = 1,  ///< DeepSeek V4.1 / Vision-Exp template
    PULSAR_CHAT_QWEN    = 2,  ///< Qwen chat template (L251 S5)
    PULSAR_CHAT_INKLING = 3,  ///< Inkling-Small native renderer (L294 E6: src/lib/inkling_chat)
    PULSAR_CHAT_NEMOTRON_H = 4,  ///< Nemotron-H ChatML + think-tag renderer (L296 N8: src/lib/nemotron_chat)
    PULSAR_CHAT_NONE    = 5,  ///< no conversation: an encoder family (L321) renders its inputs through pulsar_engine_encode_render
} pulsar_chat_format;

/** The speculative drafter a family's artifacts carry. */
typedef enum {
    PULSAR_DRAFTER_NONE   = 0,
    PULSAR_DRAFTER_DSPARK = 1,  ///< DeepSeek's DSpark (merged `dspark.*` tensors)
    PULSAR_DRAFTER_MTP    = 2,  ///< the artifact's multi-token-prediction module(s) (Qwen's `mtp.*`, Nemotron-H's MTP module)
    PULSAR_DRAFTER_BLOCK  = 3,  ///< a block drafter the container holds (L298 LiLiCorr, L299 DFlash2: the DFlash family)
} pulsar_drafter_kind;

/** Progress callback: `event` names the phase ("prefill_chunk" from pulsar_session_set_progress,
 *  "prefill_display" from the display hook), `current` of `total` tokens done; `ud` is the caller's pointer. */
typedef void (*pulsar_session_progress_fn)(void *ud, const char *event, int current, int total);
/** Cancellation callback, polled at safe chunk boundaries of pulsar_session_sync: return true to stop the sync
 *  there (it then returns PULSAR_SESSION_SYNC_INTERRUPTED). */
typedef bool (*pulsar_session_cancel_fn)(void *ud);

/** pulsar_session_sync's return when the cancel callback stopped it at a chunk boundary (the checkpoint is a valid
 *  prefix; re-issue the sync to resume). */
#define PULSAR_SESSION_SYNC_INTERRUPTED 2

/** Options for pulsar_engine_open(). Zero-initialise, then set what you need:
 * every field's 0/NULL is a working default except `model_path`. */
typedef struct {
    const char *model_path;   ///< GGUF path; required
    /** Drafter is auto-enabled when the main GGUF contains dspark.* tensors;
     * this opts out (memory saving for sampled-only workloads). */
    bool dspark_disable;
    /** L298 (--drafter NAME): which of the drafters the container holds runs -- "mtp" or its block drafter's kind
     *  ("lilicorr" on Nemotron-H, L299 "dflash2" on Qwen3.8-Flash-Next) -- and so which
     *  is loaded at all (the other is declared unloaded: never staged, its memory the KV pool's).  NULL = the one the
     *  container holds; of two, the block drafter on Qwen (L299: dflash2 by default), a choice required on Nemotron-H;
     *  a name the container does not hold refuses the open by name. */
    const char *drafter;
    /** L280 (--rope-yarn F): YaRN rope scaling at factor F over the container's trained context, for a container
     *  whose rope is native (Qwen's ship `rope_type: default`; the model card's recipe is F 4 = 1M).  0 = the
     *  container's own rope.  A container that already declares YaRN takes only its own factor, and a family whose
     *  rope has no YaRN refuses the option by name -- it never reaches a model silently. */
    double rope_yarn_factor;
    /** "FILE:PREFIX" — swap routed-expert tensors whose name starts with
     * PREFIX for the same-named tensors in FILE (a donor GGUF). Measurement
     * aid for per-layer quant-format KL probes (the prisma allocator, archived at tag archive/gguf-tooling-2026-09-24). */
    const char *expert_overlay;
    pulsar_backend backend;      ///< CPU or CUDA; CUDA is the served path
    uint32_t prefill_chunk;      ///< tokens per prefill chunk; 0 = engine default (PULSAR_PREFILL_CHUNK_DEFAULT for prompts longer than it, else the whole prompt)
    int spec_depth;              ///< the drafter's starting draft depth (--spec-depth); 0 = PULSAR_SPEC_DEPTH, else the drafter's
    float spec_tau;              ///< the draft stop threshold (--spec-tau); 0 = PULSAR_SPEC_TAU, else the drafter's; < 0 = no stop
    const char *directional_steering_file;  ///< steering-vector file, or NULL
    float directional_steering_attn;        ///< steering scale on the attention stream
    float directional_steering_ffn;         ///< steering scale on the FFN stream
    bool inspect_only;           ///< load and report, then stop: no session/graph allocation
    /** Tensor parallelism (docs/tensor-parallel-port.md), from 1 GPU up to n.
     * Two config styles:
     *  - n-way: tp_rank in [0,tp_nranks) + tp_peers (ordered "host:port,.." for all
     *    n ranks) + tp_port.  Full-mesh; all ranks are symmetric.
     *  - legacy 2-rank: tp_role (1 = leader listens on tp_port, 2 = worker dials
     *    tp_peer:tp_port), tp_peers NULL.  tp_rank/nranks are derived (leader=0).
     * A configured group wires EVERY TP lane (expert halves, vocab split, the
     * session mirror); there is no per-lane arm to select (L239 retired
     * `--tp-arm`, which had selected nothing since slice 4c). */
    int tp_role;
    /** Legacy 2-rank worker: the leader's host to dial (with tp_port); unused when tp_peers is set. */
    const char *tp_peer;
    /** This rank's control/listen port (the legacy leader listens on it, the legacy worker dials it); <= 0 = 5588. */
    int tp_port;
    /** This rank's index in the group, [0, tp_nranks); -1 = unset. */
    int tp_rank;        /* this rank's index in the group; -1 = unset */
    /** Group size; 0 = unset (legacy -> 2). */
    int tp_nranks;      /* group size; 0 = unset (legacy -> 2) */
    /** n-way: ordered "host:port,..." list for all n ranks; NULL = the legacy 2-rank style. */
    const char *tp_peers;   /* ordered "host:port,..." list for all n ranks */
    /** This build's id (git describe), carried to every TP peer in the NODE
     *  frame so each rank can report the whole group's builds.  NULL = not
     *  stamped (the CLI); the transport sends it empty. */
    const char *build_id;
    /** L264 S4e: a WORKER rank's own copies of the disk KV cache's segments
     *  (`<tp_kv_dir>/<key>.tpseg`; KV is replicated per rank, so each rank keeps
     *  its own).  NULL on a rank serving no disk cache: the segment frames are
     *  then refused there as misses. */
    const char *tp_kv_dir;
} pulsar_engine_options;

/** GPU byte breakdown of one bank's context buffers at a given context size,
 * in the stored KV row formats (packed NVFP4 attention rows, MXFP4 indexer
 * rows).
 *
 * Produced by pulsar_context_memory_estimate(). Display only: for the number
 * ADMISSION CONTROL must use, call pulsar_engine_session_cost_bytes(), which
 * is the allocator run dry (prefill working set, drafter state, every bank). */
typedef struct {
    uint64_t total_bytes;       ///< raw_bytes + comp_index_bytes + scratch_bytes
    uint64_t raw_bytes;         ///< persistent raw (SWA) KV ring, every layer
    uint64_t comp_index_bytes;  ///< persistent compressed attention + indexer caches, every compressing layer
    uint64_t scratch_bytes;     ///< the ctx-scaled indexer_scores working buffer
    uint32_t prefill_cap;       ///< max tokens in one prefill chunk
    uint32_t raw_cap;           ///< raw ring capacity, in rows
    uint32_t comp_cap;          ///< deepest compressed pool capacity, in rows
} pulsar_context_memory;

/** Serialized session state (KV + host bookkeeping) as an owned byte blob.
 *
 * Produced by the snapshot API and consumed by the restore API; the format is
 * versioned and refuses payloads written by an incompatible build. */
typedef struct {
    uint8_t *ptr;   ///< owned buffer, `cap` bytes allocated
    uint64_t len;   ///< bytes actually used
    uint64_t cap;   ///< allocated capacity in bytes
} pulsar_session_snapshot;

/** Open one engine: a group of one (pulsar_engine_open_group).
 *  @param out receives the engine, NULL on failure
 *  @param opt the options (model_path required)
 *  @return 0, or 1 on a refusal / load failure (said on stderr) */
int pulsar_engine_open(pulsar_engine **out, const pulsar_engine_options *opt);
/** L321: open n engines in ONE process, sharing its GPU (one instance, one CUDA context): out[i] is opts[i]'s model.
 *  Every engine of a process is opened in one group -- an open while engines are live is refused -- and a group
 *  refuses tensor parallelism, inspect-only opens and a device that reads weights through host registration (each
 *  member's weights are staged into device memory).  Close the members with pulsar_engine_close in any order; the
 *  GPU state goes with the last.  0, or 1 with every out[i] NULL (said on stderr). */
/** @param out receives n engines, out[i] for opts[i]
 *  @param opts n option sets
 *  @param n the group size (>= 1)
 *  @return 0, or 1 with every out[i] NULL */
int pulsar_engine_open_group(pulsar_engine **out, const pulsar_engine_options *opts, uint32_t n);
/** Tensor parallelism (slice 4e, L238): is this engine a WORKER rank of a TP
 * group?  A worker drives no sessions of its own -- its driver must call
 * pulsar_tp_worker_run right after open and exit when it returns; every
 * pulsar_session_* operation on a worker is refused by name. */
/** @param e the engine
 *  @return true on a worker rank (rank != 0) of a TP group */
bool pulsar_engine_is_tp_worker(const pulsar_engine *e);
/** Any rank of a TP group (leader included).  Generation on such an engine
 * rides the session lane, whose operations the group mirrors (slice 4e). */
/** @param e the engine
 *  @return true when e belongs to a TP group */
bool pulsar_engine_is_tp(const pulsar_engine *e);
/** The engine's TP transport, or NULL when it is not tensor-parallel.  For
 * read-only operator views (pulsar-server's /health tp block) that read the
 * transport's bring-up records; owned by the engine, valid until close. */
/** @param e the engine
 *  @return the transport (engine-owned), or NULL */
struct pulsar_tp *pulsar_engine_tp(const pulsar_engine *e);
/** The worker receive loop: applies the leader's frames (create, sync, eval,
 * batched and mixed decode, rewind, invalidate, rng state) to a session
 * registry keyed by the create ordinal until the leader sends STOP or the
 * transport fails.  Returns 0 on a clean stop, 1 on a failure (err filled). */
/** @param e a worker-rank engine
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0 on a clean STOP, 1 on a failure */
int pulsar_tp_worker_run(pulsar_engine *e, char *err, size_t errlen);
/** Close an engine and free everything it owns (the GPU state goes with a group's last member).
 *  @param e the engine; NULL is a no-op */
void pulsar_engine_close(pulsar_engine *e);
/** Print the loaded model's summary (name, architecture, container, shape) to stdout.
 *  @param e the engine */
void pulsar_engine_summary(pulsar_engine *e);
/** Tokenizer table length. NOT the logits row width — see
 * pulsar_engine_logits_width, and never size a logits buffer from this. */
/** @param e the engine
 *  @return the tokenizer's table length; 0 without an engine or a tokenizer (refused by name) */
int pulsar_engine_vocab_size(pulsar_engine *e);
/** Row width (in floats) of every logits buffer the engine writes: the shape
 * profile's n_vocab. Size logits buffers with THIS — it is the stride
 * pulsar_session_decode_multiseq writes its rows at, and the loader does not
 * check it against pulsar_engine_vocab_size. */
/** @param e the engine
 *  @return floats per logits row */
int pulsar_engine_logits_width(const pulsar_engine *e);
/** The loaded model's name, as the family reads it from the artifact.
 *  @param e the engine
 *  @return a string the engine owns (valid until close) */
const char *pulsar_engine_model_name(pulsar_engine *e);
/** The id the API serves the loaded model as -- the family's (L284: was a family switch in the server). */
/** @param e the engine
 *  @return the model id string (static or engine-owned) */
const char *pulsar_engine_served_model_id(const pulsar_engine *e);
/** Does the loaded model use the V4.1 conversation TEMPLATE?  The server's chat
 * renderer needs the fact: V4 (0731) and V4.1 were trained on different
 * templates -- the DSML tag spelling, the tools-prompt text, where the tool
 * schemas sit, how user-side messages merge, whether a mid-conversation system
 * message is an in-place System token -- so a prompt primed with the other
 * family's template is a different prompt (L218 s123: a 0731 artifact primed
 * with V4.1's tags answered with a DSML invoke and no block opener at all).
 * Follows the loaded profile (pulsar_shape::variant); an engine-less caller
 * reads as V4.1, the compile-time default profile. */
/** @param e the engine, or NULL
 *  @return true for the V4.1 template */
bool pulsar_engine_chat_v41(const pulsar_engine *e);
/** The loaded model's family (L251).  An engine-less caller reads DeepSeek. */
/** @param e the engine, or NULL
 *  @return the family id */
pulsar_family_id pulsar_engine_family(const pulsar_engine *e);
/** The loaded model's conversation format (the family's answer; for DeepSeek
 * it follows the shape profile exactly like pulsar_engine_chat_v41). */
/** @param e the engine
 *  @return the format the front end must render; PULSAR_CHAT_NONE (said once) for NULL -- no engine has no
 *          family, and a caller that renders without one names its format itself */
pulsar_chat_format pulsar_engine_chat_format(const pulsar_engine *e);
/** Printable family name ("DeepSeek V4", "Qwen4-exp"). */
/** @param e the engine
 *  @return a static string; "(no engine)" for NULL (never a family's name) */
const char *pulsar_engine_family_name(const pulsar_engine *e);
/** Does the loaded family carry a tokenizer and chat renderer in this build?
 * A front end that takes text refuses an engine without one at startup; the
 * tokenizer entries themselves refuse by name (once per entry) and return their
 * failure value if reached anyway. */
/** @param e the engine
 *  @return true when text entry points are usable */
bool pulsar_engine_has_tokenizer(const pulsar_engine *e);

/* ---- encoders (L321) ----------------------------------------------------------------------------------------------
 * An ENCODER family's model maps an input to one vector or one score instead of a next token: no session, no
 * sampler, no chat.  Its inputs are the family's own renders (its template: a query or passage prefix, a query /
 * document pair), tokenized by its tokenizer, and one call runs a batch of them through one bidirectional forward.
 * A decoder family declares no encoder: the entries below refuse by name (PULSAR_ENCODER_NONE, 0). */

/** What an encoder returns per input. */
typedef enum {
    PULSAR_ENCODER_NONE   = 0,  ///< a decoder family: no encode entry applies
    PULSAR_ENCODER_EMBED  = 1,  ///< an embedding: pulsar_engine_encode_dim floats, L2-normalised
    PULSAR_ENCODER_RERANK = 2,  ///< a relevance score (the head's raw logit; higher is more relevant): one float
} pulsar_encoder_kind;

/** Which render of the family's template an input takes. */
typedef enum {
    PULSAR_ENCODE_QUERY   = 0,  ///< an embedder's query (a search string)
    PULSAR_ENCODE_PASSAGE = 1,  ///< an embedder's passage (a document chunk)
    PULSAR_ENCODE_PAIR    = 2,  ///< a reranker's (query, document) pair
} pulsar_encode_role;

/** The loaded family's encoder kind (NONE on a decoder or no engine). */
/** @param e the engine, or NULL
 *  @return the encoder kind */
pulsar_encoder_kind pulsar_engine_encoder(const pulsar_engine *e);
/** Floats per encoded input (an embedding's width, or 1 for a score); 0 on a decoder. */
/** @param e the engine
 *  @return floats per input */
uint32_t pulsar_engine_encode_dim(const pulsar_engine *e);
/** The most tokens one input may have (its render included): the positions the model serves exactly; 0 on a
 *  decoder.  A longer input is refused, never cut -- truncation is the caller's choice. */
/** @param e the engine
 *  @return the per-input token cap */
uint32_t pulsar_engine_encode_max_tokens(const pulsar_engine *e);
/** Render one input in the family's template and tokenize it, appending to `out`: `text` the query or passage
 *  (QUERY / PASSAGE), or the query with `doc` the document (PAIR).  Client text never forms a special token.
 *  false + `err` on a role the encoder does not take or text that is not UTF-8. */
/** @param e the encoder engine
 *  @param role which render
 *  @param text the query or passage (UTF-8)
 *  @param doc PAIR: the document; else unused
 *  @param out the tokens are appended here
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return true on success */
bool pulsar_engine_encode_render(pulsar_engine *e, pulsar_encode_role role, const char *text, const char *doc,
                                 pulsar_tokens *out, char *err, size_t errlen);
/** Encode n rendered inputs: out [n][pulsar_engine_encode_dim] floats.  Each input's result is the same bytes
 *  whatever else rides in the call (the forward is row-independent outside attention, and attention stays inside
 *  its input).  A group's encoders and parsers take turns on the GPU a forward at a time (they share one scratch).  0, or -1 + `err` (an empty or over-long input, a decoder
 *  family, a device failure). */
/** @param e the encoder engine
 *  @param inputs n rendered inputs (pulsar_engine_encode_render)
 *  @param n input count
 *  @param out receives n x pulsar_engine_encode_dim floats
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0 or -1 */
int pulsar_engine_encode(pulsar_engine *e, const pulsar_tokens *inputs, uint32_t n, float *out, char *err,
                         size_t errlen);

/* ---- parsers (L321) -------------------------------------------------------------------------------------------------
 * A PARSER family's model reads a page image and writes its text (Nemotron Parse: markdown with classes and boxes in
 * the model's own markup): no session, no chat -- one call, one page, generated greedily by the model's own recipe.
 * Calls are serialised on the engine; between its GPU phases (the vision, the prefill, each decode step) the group's
 * other engines run theirs, so an encode waits a step, not a page.  A family without a parser refuses by name. */
/** Options for one pulsar_engine_parse_page call; NULL options = the model's defaults. */
typedef struct {
    const char *prompt;        ///< the task prompt (the model's added tokens, back to back); NULL = the model's default
    uint32_t max_new_tokens;   ///< 0 = the model's default
} pulsar_parse_options;

/** What one pulsar_engine_parse_page call did: the page's size, token counts and per-phase wall times. */
typedef struct {
    uint32_t image_w, image_h;     ///< the image as decoded
    uint32_t prefix_rows;          ///< the visual prefix the page became
    uint32_t prompt_tokens, generated_tokens;
    bool finished;                 ///< the model ended the page (its eos), not a cap
    double vision_ms, prefill_ms, decode_ms;
} pulsar_parse_stats;
/** @var pulsar_parse_stats::image_w
 *  Decoded image width, pixels (image_h its height). */
/** @var pulsar_parse_stats::prompt_tokens
 *  Tokens of the task prompt (the visual prefix not counted). */
/** @var pulsar_parse_stats::generated_tokens
 *  Tokens the model generated, its eos included when reached. */
/** @var pulsar_parse_stats::vision_ms
 *  Wall time of the vision encoder, ms. */
/** @var pulsar_parse_stats::prefill_ms
 *  Wall time of the prompt prefill, ms. */
/** @var pulsar_parse_stats::decode_ms
 *  Wall time of the greedy decode loop, ms. */

/** Does the loaded family parse pages? */
/** @param e the engine
 *  @return true for a parser family */
bool pulsar_engine_is_parser(const pulsar_engine *e);
/** Parse one page (`image`: PNG or JPEG bytes).  `ids` (appended): the whole sequence, prompt and generated, eos
 *  included when reached; `*text`: its decode (malloc'd; the model's tokenizer, special tokens kept); `stats` may be
 *  NULL.  0; -2 + `err`: the request refused (an image the model does not take, a prompt that is not its tokens);
 *  -1 + `err`: the engine failed (a device failure). */
/** @param e the parser engine
 *  @param image the encoded page (PNG or JPEG), borrowed
 *  @param len bytes at image
 *  @param opt options; NULL = the model's defaults
 *  @param ids the sequence is appended here
 *  @param text receives the malloc'd decode; caller frees
 *  @param stats receives the call's stats; may be NULL
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, -2 (refused) or -1 (engine failure) */
int pulsar_engine_parse_page(pulsar_engine *e, const uint8_t *image, size_t len, const pulsar_parse_options *opt,
                             pulsar_tokens *ids, char **text, pulsar_parse_stats *stats, char *err, size_t errlen);

/** DSpark speculative-decode counters for the server /metrics endpoint. All
 * cumulative/monotonic since engine open.
 *
 * accepted_per_pos[i] counts how often draft position i was accepted -- drafts
 * being a prefix chain, that is how often the round's commit ran past i. Its
 * denominator is verified_per_pos[i] (rounds where position i was actually
 * VERIFIED, i.e. the confidence trim kept it), NOT num_drafts: under the L107
 * adaptive depth the depth moves per round, so dividing by num_drafts
 * understates every deep position by the fraction of rounds that never reached
 * it. rate[i] = accepted_per_pos[i] / verified_per_pos[i].
 *
 * The conditional P(accept i | accept 0..i-1) = accepted_per_pos[i] /
 * accepted_per_pos[i-1] is exactly that either way -- the prefix property makes
 * its denominator implicit -- but it is confounded by the trim's selection,
 * which is stricter with depth (position i is verified only when the drafter
 * cleared tau at EVERY shallower position). A non-monotone conditional series
 * is therefore a selection artifact, not evidence about drafter quality. */
typedef struct {
    uint64_t accepted_tokens;       ///< accepted draft tokens
    uint64_t draft_tokens;          ///< proposed/verified draft tokens
    uint64_t num_drafts;            ///< draft rounds (verify steps carrying drafts)
    uint64_t gen_tokens;            ///< tokens emitted by the spec loop
    uint64_t copy_probes;           ///< L308: copy chains drafted as a stale copy estimate's probe while it stood under tau
    uint64_t accepted_per_pos[16];  ///< accepted count per draft position
    uint64_t verified_per_pos[16];  ///< count of rounds verifying position i; rate[i] = accepted_per_pos[i]/verified_per_pos[i]
    int      max_draft;             ///< the deepest draft position a round can verify (the start depth, or the adaptive controller's ceiling)
    bool     has_drafter;           ///< a drafter is loaded behind the round API (the counters above can move)
} pulsar_spec_metrics;
/** Copy the engine's speculative-decode counters (zeroed without an engine).
 *  @param e the engine, or NULL
 *  @param out receives the counters */
void pulsar_engine_spec_metrics(pulsar_engine *e, pulsar_spec_metrics *out);
/** Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
/** @param e the engine
 *  @return the shape id */
int pulsar_engine_model_id(pulsar_engine *e);
/** L280: the identity a disk-KV segment store is opened with (pulsar_segstore_identity): the model id, the routed
 *  experts' format and the rope the loaded positions run (the family's rope_id -- Qwen under YaRN).  The one
 *  answer every store opener asks; a store of another identity is not indexed. */
/** @param e the engine
 *  @return the store identity */
uint32_t pulsar_engine_kv_identity(pulsar_engine *e);
/** True when the loaded model is expert-pruned (REAP compact: some layers carry
 * fewer physical routed experts than the architecture's n_expert). Fidelity
 * tests against full-model reference vectors are meaningless when true. */
/** @param e the engine
 *  @return true for an expert-pruned model */
bool pulsar_engine_is_pruned(pulsar_engine *e);
/** @param backend the backend
 *  @return its name ("cuda"; "unknown" for any other value), static */
const char *pulsar_backend_name(pulsar_backend backend);
/** @param mode a think mode
 *  @return true when thinking is on (mode != PULSAR_THINK_NONE) */
bool pulsar_think_mode_enabled(pulsar_think_mode mode);
/** True for 0 and for 1..100; anything else is not a mode. */
/** @param mode the value to check
 *  @return true for a mode */
bool pulsar_think_mode_valid(pulsar_think_mode mode);
/** "none", the preset name ("low"/"high"/"max") or the decimal effort. */
/** @param mode a valid mode
 *  @return the name (static storage) */
const char *pulsar_think_mode_name(pulsar_think_mode mode);
/** The reasoning-effort line rendered before the system message when
 * thinking is on ("" for NONE), byte-identical to the V4.1 reference encoder:
 * "Reasoning Effort: N (range 1-100, the higher the value, the more thorough
 * the reasoning)\n\n".  Static storage; valid for the process lifetime. */
/** @param mode the mode
 *  @return the line, static */
const char *pulsar_think_effort_prefix(pulsar_think_mode mode);
/** The effort line as the LOADED MODEL's encoder spells it.  V4.1 renders the
 * numeric line for every effort (pulsar_think_effort_prefix); the V4 (0731)
 * encoder has three levels -- low adds nothing, high and max carry their own
 * texts (encoding_dsv4.py REASONING_EFFORT_PROMPTS) -- and no numeric line.
 * Rendering V4.1's line to a 0731 model put 25 foreign tokens in front of
 * every prompt (L239).  A V4 request with an effort the V4 encoder cannot
 * spell is refused by the API before it reaches here; this dies on one. */
/** @param mode the mode
 *  @param v41 the loaded model uses the V4.1 template (pulsar_engine_chat_v41)
 *  @return the line, static */
const char *pulsar_think_effort_prefix_family(pulsar_think_mode mode, bool v41);
/** Whether the V4 (0731) encoder can spell `mode` at all: none, low, high, max. */
/** @param mode the mode
 *  @return true when the V4 encoder spells it */
bool pulsar_think_effort_v4_valid(pulsar_think_mode mode);
/** Does the LOADED family render thinking at `mode`?  V4.1: every effort in [1, 100]; V4 (0731):
 * none, low, high, max (pulsar_think_effort_v4_valid); Qwen: on at the template's default
 * (PULSAR_THINK_DEFAULT) or off, no effort level.  `why` (optional) receives the family's rule when
 * it does not.  Front ends validate an explicit effort here, after open (L272 B10). */
/** @param e the engine
 *  @param mode the requested mode
 *  @param why receives the family's rule on false; may be NULL
 *  @param n bytes at why
 *  @return true when the family renders mode */
bool pulsar_engine_think_mode_supported(const pulsar_engine *e, pulsar_think_mode mode, char *why, size_t n);
/** The loaded model's DEFAULT thinking effort: V4.1's reference default is high
 * (PULSAR_THINK_DEFAULT); the V4 (0731) encoder's is low, which renders no
 * effort line; Qwen's is thinking on at the template's default effort.  Every front end that did not receive an explicit effort
 * resolves it here, after the engine is open (L239). */
/** @param e the engine
 *  @return the default mode */
pulsar_think_mode pulsar_engine_think_default(const pulsar_engine *e);
/** If `s` starts with SOME effort line, the length of that line (so a caller
 * that strips it does not have to know the effort), else 0. */
/** @param s NUL-terminated text
 *  @return the effort line's length in bytes, or 0 */
size_t pulsar_think_effort_prefix_len(const char *s);
/** One bank's context buffers at ctx_size, priced by the engine's own KV
 * sizing (the managed-KV policy's numbers), so this and the engine's
 * "context buffers" boot line agree.  Uses the active model shape and
 * per-layer compress ratios selected by pulsar_engine_open(); call after
 * opening the GGUF.  prefill_chunk 0 selects the engine default.  Refuses a
 * backend without a graph.
 * WARNING: this covers ONLY the persistent KV rows plus the indexer_scores
 * scratch -- NOT the full per-session graph (prefill batch buffers, drafter
 * state, ...).  For admission accounting use pulsar_engine_session_cost_bytes,
 * which prices the whole session; pricing sessions with this estimate
 * under-admitted by ~10x and hard-locked the GB10 (2026-07-13). */
/** @param backend the backend (refused without a graph)
 *  @param ctx_size the context, tokens
 *  @param prefill_chunk tokens per prefill chunk; 0 = the engine default
 *  @return the byte breakdown */
pulsar_context_memory pulsar_context_memory_estimate(
        pulsar_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
/** GPU bytes pulsar_session_create(e, ctx_size) allocates: persistent KV
 * caches, the prefill working set (scaled by the engine's prefill chunk), the
 * drafter state when a drafter is loaded.  It is the allocation code itself
 * run dry, so it equals pulsar_session_resident_bytes of the session it
 * prices; this is the number admission control must use.  Returns 0 if no
 * session could be created (no graph backend / weights not loaded). */
/** @param e the engine
 *  @param ctx_size the context, tokens
 *  @return GPU bytes, or 0 */
uint64_t pulsar_engine_session_cost_bytes(pulsar_engine *e, int ctx_size);
/** Same, priced for an EXPLICIT bank-pool size (Tier-2 auto-sizing): the server
 * evaluates the (banks, ctx) fit table before committing PULSAR_MSEQ_BANKS. n_banks
 * >= 1; 1 is the classic single-session cost. */
/** @param e the engine
 *  @param ctx_size the context per bank, tokens
 *  @param n_banks the pool size (>= 1)
 *  @return GPU bytes, or 0 */
uint64_t pulsar_engine_session_cost_bytes_banked(pulsar_engine *e, int ctx_size,
                                              int n_banks);
/** Tier-2 overcommit (task #55): demand-paged comp+index bytes for ONE bank at a
 * context — the physical-on-touch VA the overcommit auto-size reserves but does
 * NOT charge at admission (only the eager floor is charged). 0 if no session
 * could be priced. */
/** @param e the engine
 *  @param ctx_size the context, tokens
 *  @return demand-paged bytes for one bank, or 0 */
KvBytes pulsar_engine_demand_paged_bytes_per_bank(pulsar_engine *e, int ctx_size);
/** The bank-pool size the next session will be created with, and whether the
 * operator pinned it (PULSAR_MSEQ_BANKS).  A server that derives the size
 * itself SETS it here -- the engine owns the number; nothing else parses the
 * variable and nothing writes the environment to talk to the engine (L159). */
/** @param pinned_by_env receives nonzero when PULSAR_MSEQ_BANKS pinned the size; may be NULL
 *  @return the bank-pool size */
uint32_t pulsar_engine_bank_pool(int *pinned_by_env);
/** Set the bank-pool size the next pulsar_session_create uses (pulsar_engine_bank_pool reads it back).
 *  @param n_banks the pool size */
void     pulsar_engine_set_bank_pool(uint32_t n_banks);
/** Tier-2 overcommit (task #55): EXACT touched (physically resident) demand-paged
 * KV bytes across the pool, summed from the per-bank compressor frontier
 * (deterministic; no MemAvailable). The eviction guard triggers on this; the
 * accounting-exactness gate proves it matches the physical cudaMemGetInfo delta.
 * For the current bank the live frontier is used; idle banks use their captured
 * frontier — capture the current bank first if you need it fully current. */
/** @param s the session
 *  @return the pool's touched demand-paged KV bytes */
KvBytes pulsar_session_touched_kv_bytes(const pulsar_session *s);
/** Tier-2 task #55 increment 2b — per-bank physical evict/restore for the proactive
 * eviction guard. free_physical: DIRECT cudaFree of one idle bank's split comp/index
 * (reclaims physical on GB10); caller must have persisted the bank's KV to DISK
 * first (host RAM reclaims nothing on unified memory; the server writes the bank's
 * history as segments) and repointed away from it.
 * alloc_physical: reallocate that bank's comp/index (VA; physical on touch) + rebuild
 * the base-pointer table; the caller then reloads its KV (the server from its
 * segment chain).  A family pool (Qwen, L284 #3) frees and re-backs its bank's own
 * demand-paged KV + index tensors the same way (its always-resident state stays).
 * free_physical returns false ONLY when it refuses (the live/installed bank, a bank
 * out of range) and then frees nothing.
 * is_evicted: whether a bank's physical is currently freed. bank_touched_kv_bytes:
 * one bank's exact resident comp/index KV from its frontier (guard Δ + victim pick). */
/** @param s the session
 *  @param bank the idle bank to evict
 *  @return false when refused (nothing freed) */
bool pulsar_session_bank_free_physical(pulsar_session *s, uint32_t bank);
/** Re-back an evicted bank's comp/index KV (see the eviction note above); the caller reloads its KV after.
 *  @param s the session
 *  @param bank the bank
 *  @return true on success (on a TP group: every rank agreed) */
bool pulsar_session_bank_alloc_physical(pulsar_session *s, uint32_t bank);
/** Whether a bank's physical KV is currently freed (pulsar_session_bank_free_physical).
 *  @param s the session
 *  @param bank the bank
 *  @return true when evicted; false without a bank pool */
bool pulsar_session_bank_is_evicted(const pulsar_session *s, uint32_t bank);
/** One bank's exact resident comp/index KV from its frontier (the guard's delta and victim pick).
 *  @param s the session
 *  @param bank the bank
 *  @return bytes; 0 without a bank pool */
KvBytes pulsar_session_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank);
/** Conservative per-bank comp/index growth over `q` token rows (the guard's Delta term over a decode quantum, a
 * prompt's KV, a prefill chunk's): the ONE conversion of rows to bytes, the family's own price (family.h
 * growth_bytes) -- each pool rounds its own rows, so there is no scalar bytes-per-row (pulsar_units.h). */
/** @param s the session
 *  @param q token rows
 *  @return the growth bound, bytes */
KvBytes pulsar_session_quantum_growth_bytes_per_bank(pulsar_session *s, KvRows q);
/** GPU bytes the session's create allocated (allocator delta measured across
 * pulsar_session_create).  Equal to pulsar_engine_session_cost_bytes at the
 * same context and pool size -- the server asserts it after each create. */
/** @param s the session
 *  @return GPU bytes the create allocated */
uint64_t pulsar_session_resident_bytes(const pulsar_session *s);
/** Resident (mmap'd, read-only, shared) weight footprint in bytes: the main
 * GGUF plus an external drafter and expert overlay when mapped separately.
 * This competes with per-session KV for the unified-memory budget. */
/** @param e the engine, or NULL (0)
 *  @return mapped weight bytes */
uint64_t pulsar_engine_weights_resident_bytes(pulsar_engine *e);
/** Whether `fp` is a terminal (pulsar_log colours only there).
 *  @param fp an open stream
 *  @return true when its descriptor is a tty */
bool pulsar_log_is_tty(FILE *fp);
/** printf to `fp`, coloured by `type` when fp is a terminal (PULSAR_LOG_DEFAULT never is).
 *  @param fp the stream
 *  @param type the line's kind
 *  @param fmt printf format, then its arguments */
void pulsar_log(FILE *fp, pulsar_log_type type, const char *fmt, ...);
/** Collect an importance matrix (per-channel activation statistics for quantization) over a prompt dataset and
 *  write it to `output_path`; the family must support it (refused by name otherwise).
 *  @param e the engine
 *  @param dataset_path a text file of prompts, each after a `===== DS4_IMATRIX_PROMPT` marker line (no marker: the
 *         whole file is one prompt)
 *  @param output_path where the matrix is written
 *  @param ctx_size the session context, tokens
 *  @param max_prompts prompts to run at most; <= 0 = all
 *  @param max_tokens tokens to prefill in total at most (the last prompt is cut); <= 0 = no cap
 *  @return 0, or 1 on a failure */
int pulsar_engine_collect_imatrix(pulsar_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
/** Print each token's id and text to stdout (the family's tokenizer).
 *  @param e the engine (a family with a tokenizer)
 *  @param tokens the tokens */
void pulsar_engine_dump_tokens(pulsar_engine *e, const pulsar_tokens *tokens);
/** Open `model_path`'s vocabulary alone (no engine), tokenize `text` as rendered chat and print the tokens.
 *  @param model_path the model container
 *  @param text the text (NULL = empty)
 *  @param fp the stream; NULL = stdout
 *  @return 0, or 1 when the model does not open */
int pulsar_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);

/** Append one token, growing `v` as needed.
 *  @param tv an owned vector (not a borrowed view)
 *  @param token the id */
void pulsar_tokens_push(pulsar_tokens *tv, int token);
/** Free `v` and zero the vector (reusable empty afterwards).
 *  @param tv an owned vector */
void pulsar_tokens_free(pulsar_tokens *tv);
/** Make `dst` a copy of `src` (dst's previous contents are replaced; its buffer is reused or grown).
 *  @param dst an owned vector
 *  @param src the source */
void pulsar_tokens_copy(pulsar_tokens *dst, const pulsar_tokens *src);
/** Whether `prefix` is a prefix of `tokens`, by id.
 *  @param tokens the vector
 *  @param prefix the candidate prefix
 *  @return true when it is (an empty prefix always is) */
bool pulsar_tokens_starts_with(const pulsar_tokens *tokens, const pulsar_tokens *prefix);

/** Tokenize plain text (no special-token matching), appending to `out`.
 *  @param e the engine
 *  @param text NUL-terminated UTF-8
 *  @param out the tokens are appended here
 *  @return false, `out` untouched, when the family has no tokenizer (said once) */
bool pulsar_tokenize_text(pulsar_engine *e, const char *text, pulsar_tokens *out);
/** Tokenize a rendered chat prompt: its special-token spellings become control tokens.  Appends to `out`.
 *  @param e the engine
 *  @param text the rendered prompt
 *  @param out the tokens are appended here
 *  @return false, `out` untouched, when the family has no tokenizer (said once) */
bool pulsar_tokenize_rendered_chat(pulsar_engine *e, const char *text, pulsar_tokens *out);
/** A half-open [lo, hi) byte range of CLIENT-supplied text in a rendered prompt.
 * The renderer produces these; the tokeniser treats them as plain text. */
typedef struct {
    uint32_t lo;   ///< first byte
    uint32_t hi;   ///< one past the last byte
} pulsar_text_span;

/** As pulsar_tokenize_rendered_chat, but every byte inside one of the `n_spans`
 * half-open [lo, hi) ranges is tokenised as PLAIN TEXT: no special-token matching
 * happens there, so a spelling like `<|Assistant|>` or `｜DSML｜` that a client
 * typed into a message becomes ordinary text instead of a control token.  The
 * renderer records those ranges for exactly the bytes it copied from client data;
 * an empty span list is byte-identical to the un-spanned entry. */
/** @param e the engine
 *  @param text the rendered prompt
 *  @param spans client-data byte ranges of text
 *  @param n_spans their count
 *  @param out the tokens are appended here
 *  @return false, `out` untouched, when the family has no tokenizer (said once) */
bool pulsar_tokenize_rendered_chat_spans(pulsar_engine *e, const char *text,
                                         const pulsar_text_span *spans, uint32_t n_spans,
                                         pulsar_tokens *out);

/** The client-data ranges of `text[skip .. skip+len)` (a suffix slice of the
 * same string the ranges were recorded against), rebased to the slice and
 * clipped to it.  Returns a malloc'd array (free() it) or NULL when no range
 * survives; `*n_out` is set to the count.  Used wherever a continuation appends
 * a TAIL of a rendered prompt: the tail's client bytes must stay plain text
 * there too. */
/** @param spans the recorded ranges
 *  @param n_spans their count
 *  @param skip the slice's first byte
 *  @param len the slice's length, bytes
 *  @param n_out receives the surviving count
 *  @return the malloc'd ranges, or NULL */
pulsar_text_span *pulsar_text_spans_slice(const pulsar_text_span *spans, uint32_t n_spans,
                                          size_t skip, size_t len, uint32_t *n_out);

/** ONE authority for the tool-result body's escape rule (L185): emit `s` with
 * every literal `</tool_result>` rewritten so the body cannot close its own
 * wrapper early -- the `<` becomes `&lt;` and the rest of the sentinel is kept
 * (`&lt;/tool_result>`).  `emit` is called once per segment with a pointer into
 * `s` or into the static entity; there is no allocation and no copy.  Returns
 * the total bytes emitted.  Every producer of a tool-result body uses this --
 * the server's renderer and the engine's token-level twin -- so the rule exists
 * once. */
/** @param s the NUL-terminated body
 *  @param emit called per segment with (ud, bytes, n)
 *  @param ud passed to emit
 *  @return bytes emitted */
size_t pulsar_tool_result_escape(const char *s,
                                 void (*emit)(void *ud, const char *bytes, size_t n),
                                 void *ud);
/** One message of a chat turn (pulsar_chat_append_turn). */
typedef struct {
    const char *role;     ///< "user", "tool" (a tool result), or "system" (a note: the head's region before any
                          ///< turn, a mid-conversation system note after one)
    const char *content;  ///< client text (NULL = empty)
    /** `content` is the front end's own control text, written in the loaded family's markers (the agent's
     *  DSML tool prompt): DeepSeek tokenizes it as rendered chat; a family whose markers it does not spell
     *  renders it as text. */
    bool trusted;
} pulsar_chat_message;

/** The family's chat front, turn by turn (L284 P14): the CLI REPL and the agent build a live transcript with
 * these on every family, each piece rendered WHOLE by the family's own renderer (DeepSeek's marker template;
 * Qwen's qwen_chat, HF's template byte for byte) -- head + turns + sampled assistant turns is the
 * conversation's full render, so a transcript's KV continues across turns.  The think mode is the family's
 * (pulsar_engine_think_mode_supported); any other ends the process, never a neighbouring effort.  Each returns
 * false, `tokens` untouched, when the family has no chat front (said once per entry).
 *
 * pulsar_chat_open: the HEAD, everything before the first turn -- DeepSeek's BOS and lead-in, Qwen's system
 * block with its effort line -- carrying `trusted` (pulsar_chat_message::trusted; NULL = none) then the
 * client `system` text (NULL = none).  Appended to `tokens`, which a caller starts empty. */
/** @param e the engine
 *  @param tokens the transcript (start empty); the head is appended
 *  @param trusted front-end control text, or NULL
 *  @param system client system text, or NULL
 *  @param think_mode the family's think mode
 *  @return false when refused (no chat front) */
bool pulsar_chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
                      pulsar_think_mode think_mode);
/** One TURN appended to a transcript that ends with its head or a closed assistant turn: the messages, then,
 * with `generation_prompt`, the assistant turn opened for `think_mode`. */
/** @param e the engine
 *  @param tokens the transcript, appended to
 *  @param msgs the turn's messages
 *  @param n their count
 *  @param generation_prompt also open the assistant turn
 *  @param think_mode the mode the assistant turn opens for
 *  @return false when refused (no chat front) */
bool pulsar_chat_append_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n,
                             bool generation_prompt, pulsar_think_mode think_mode);
/** Close the sampled assistant turn the transcript ends with (sampled up to, not including, its stop token):
 * what the turn's end is in the family's template (DeepSeek: EOS; Qwen: "<|im_end|>\n"). */
/** @param e the engine
 *  @param tokens the transcript, appended to
 *  @return false when refused (no chat front) */
bool pulsar_chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens);
/** The one-shot prompt: pulsar_chat_open(system) + a turn of one user message with the generation prompt. */
/** @param e the engine
 *  @param system the system text, or NULL
 *  @param prompt the user message
 *  @param think_mode the think mode
 *  @param out the prompt's tokens are appended here
 *  @return false when refused (no chat front: nothing appended) */
bool pulsar_encode_chat_prompt(
        pulsar_engine *e,
        const char *system,
        const char *prompt,
        pulsar_think_mode think_mode,
        pulsar_tokens *out);

/** A token's text as bytes (byte-level BPE decoded; a special token's literal spelling).
 *  @param e the engine
 *  @param token the id; an out-of-range id gives ""
 *  @param len receives the byte count (0 when refused); may be NULL
 *  @return a malloc'd NUL-terminated copy (caller frees); NULL when the family has no tokenizer (said once) */
char *pulsar_token_text(pulsar_engine *e, int token, size_t *len);
/** The family's end-of-sequence id (one id; the whole stop set is pulsar_token_is_stop's).
 *  @param e the engine
 *  @return the id; -1 when the family has no tokenizer (said once) */
int pulsar_token_eos(pulsar_engine *e);
/** L251: whether `token` ends generation -- the family's whole stop set (Qwen: <|im_end|> and
 * <|endoftext|>, generation_config.json); DeepSeek: its one eos id. */
/** @param e the engine
 *  @param token the id
 *  @return true when token is in the stop set; true for every token when the family has no tokenizer (said once:
 *          a generation ends at once, fail closed) */
bool pulsar_token_is_stop(pulsar_engine *e, int token);
/** The chat template's TURN MARKERS as token sequences (L272 B5): the tokens that open a user turn
 * and an assistant turn in a rendered prompt.  DeepSeek names each with one special token
 * (\<｜User｜\>, \<｜Assistant｜\>); Qwen's template spells a turn as the added token \<|im_start|\> followed
 * by the role word, so a marker is two tokens there.  The server's prefix anchors (the sys-prefix
 * cold store, the last-turn routing anchor) scan a prompt with pulsar_turn_marker_at.  false, with
 * `out` zeroed (no marker matches anywhere), when the loaded tokenizer does not spell them -- said
 * once, never a marker that matches at the wrong place. */
typedef struct {
    int user[2];       ///< the user-turn marker, n_user tokens
    /** Tokens in `user` (1 or 2; 0 when the tokenizer does not spell the marker). */
    int n_user;
    int assistant[2];  ///< the assistant-turn marker, n_assistant tokens
    /** Tokens in `assistant` (1 or 2; 0 when the tokenizer does not spell the marker). */
    int n_assistant;
} pulsar_turn_markers;
/** Fill `out` with the loaded tokenizer's turn markers (pulsar_turn_markers above).
 *  @param e the engine
 *  @param out receives the markers; zeroed when they are not spelled
 *  @return true when both markers are spelled; false when the family has no tokenizer (said once) */
bool pulsar_chat_turn_markers(pulsar_engine *e, pulsar_turn_markers *out);
/** The role whose marker begins at v[i] of v[0..n): 1 = user, 2 = assistant, 0 = none.  A marker
 * whose tokens run past n does not match.  Pure host code. */
/** @param m the markers (NULL: 0)
 *  @param v the tokens
 *  @param n their count
 *  @param i the index to test
 *  @return 1, 2 or 0 */
int pulsar_turn_marker_at(const pulsar_turn_markers *m, const int *v, int n, int i);
/** The ONE token that closes a reasoning block -- "</think>" as the loaded tokenizer spells it
 * (DeepSeek's special, Qwen's added token); -1 when it has none.  A front end that forces the
 * close feeds this id, never the text re-tokenized (L272 B11: BPE pieces on Qwen). */
/** @param e the engine
 *  @return the id, or -1 (also when the family has no tokenizer: said once) */
int pulsar_token_think_close(pulsar_engine *e);

/** Create a session on `e` with a `ctx_size`-token context (the bank pool sized by pulsar_engine_bank_pool).
 *  Refused on a TP worker rank and for a context past the model's trained positions.
 *  @param out receives the session, NULL on failure
 *  @param e the engine
 *  @param ctx_size the context per bank, tokens
 *  @return 0, or nonzero on a refusal / failure (said on stderr) */
int pulsar_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size);
/** Free a session and its GPU state (on a TP leader: settles pending acks and tells the workers).
 *  @param s the session; NULL is a no-op */
void pulsar_session_free(pulsar_session *s);
/** Install the durable progress callback: called with "prefill_chunk" at each prefill chunk boundary (a
 *  checkpoint boundary).
 *  @param s the session
 *  @param fn the callback; NULL = none
 *  @param ud passed to fn */
void pulsar_session_set_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud);
/** UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
/** @param s the session
 *  @param fn the callback; NULL = none
 *  @param ud passed to fn */
void pulsar_session_set_display_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud);
/** Optional cooperative cancellation.  pulsar_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns PULSAR_SESSION_SYNC_INTERRUPTED when it stops. */
/** @param s the session
 *  @param fn the callback; NULL = never cancel
 *  @param ud passed to fn */
void pulsar_session_set_cancel(pulsar_session *s, pulsar_session_cancel_fn fn, void *ud);
/** L329 (b2): license the next sync's read-ahead: `prompt` is the request a SERVER is prefilling (the whole of it,
 *  past a cold-store or shared-prefix cut, so the chunk after the cut is read ahead too).  Borrowed: set right before
 *  the sync, cleared (NULL) right after.  NULL -- the default, and every sync no server licensed (the CLI, the gates,
 *  imatrix, tool syncs) -- reads nothing ahead.  A hint only (the read-ahead warms the page cache, the gather reads
 *  the bytes): a prompt that differs reads rows nobody gathers. */
/** @param s the session
 *  @param prompt the request being prefilled (borrowed), or NULL to clear */
void pulsar_session_set_prefill_ahead(pulsar_session *s, const pulsar_tokens *prompt);
/** L329 (b2): a prefill of `bank` has rows up to `run_end` of `prompt` in flight (a fused step's prompt run, a
 *  sync's chunk): read ahead the rows of the prompt's next chunk -- [run_end, run_end + the prefill chunk) -- on the
 *  disk table's background readers, so that chunk's gather finds them in the page cache.  The family's prefill
 *  must read a disk-resident row table (Qwen's PLE); 0 otherwise.  One record per bank keeps the depth to one chunk
 *  ahead: rows already queued for the same tokens are not queued again, a call that adds less than half a chunk
 *  (short of the prompt's end) waits for the next, and a bank whose prompt changed drops its old rows.  Returns the
 *  tokens whose rows were queued.  Bounded: the pool holds at most PULSAR_ENGRAM_AHEAD_PENDING_MAX unread rows (an
 *  arm past it is refused, counted), and nothing is queued while MemAvailable is under the read-ahead's headroom
 *  (skipped, counted).  A licensed sync's walk calls it per chunk itself (pulsar_session_set_prefill_ahead; and
 *  pulsar_session_invalidate drops the bank's rows); a server that prefills by other steps -- the fused lane --
 *  calls it per step.  Only a server serving `prompt` calls it. */
/** @param s the session
 *  @param bank the bank being prefilled
 *  @param prompt its prompt
 *  @param run_end the end of the rows in flight, tokens
 *  @return tokens whose rows were queued */
uint32_t pulsar_session_prompt_read_ahead(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, int run_end);
/** L329 (b2): the read-ahead's counters since the engine opened (the disk table's background class). */
typedef struct {
    uint64_t issued;          ///< read-aheads queued
    uint64_t rows_queued;     ///< rows they asked for
    uint64_t rows_read;       ///< rows read into the page cache
    uint64_t rows_failed;     ///< rows whose read failed (the later gather reports its own)
    uint64_t rows_cancelled;  ///< rows dropped before a reader took them (a bank's prompt changed or was invalidated)
    uint64_t rows_refused;    ///< rows not queued: the pool already held its cap of unread rows (bounded bytes)
    uint64_t skipped;         ///< read-aheads not queued: MemAvailable below the read-ahead's headroom (memory pressure)
    /** Bytes read into the page cache by read-ahead rows. */
    uint64_t bytes_read;
    uint64_t read_ns;         ///< reader wall time on read-ahead rows, summed over the readers
    uint64_t rows_pending;    ///< rows queued now, not yet taken
    uint32_t in_flight;       ///< read-ahead rows being read now
} pulsar_read_ahead_stats;
/** false = the loaded family reads no disk row table during prefill (nothing is read ahead); *out zeroed. */
/** @param e the engine
 *  @param out receives the counters
 *  @return true when the family reads ahead */
bool pulsar_engine_read_ahead_stats(pulsar_engine *e, pulsar_read_ahead_stats *out);

/** Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, the prompt is evaluated from the bank's latest
 * CHUNK-GRID snapshot at or below the checkpoint (a multiple of the engine's
 * prefill chunk; L183/L194: a resume is a cold prefill from that boundary, so
 * its logits are the cold prefill's byte for byte -- the chunk boundaries, row
 * counts and row offsets every kernel sees are the same); otherwise the
 * backend state is refilled from scratch. */
/** @param s the session
 *  @param prompt the full prompt
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, PULSAR_SESSION_SYNC_INTERRUPTED when the cancel callback stopped it, or 1 on a failure */
int pulsar_session_sync(pulsar_session *s, const pulsar_tokens *prompt, char *err, size_t errlen);
/** pulsar_session_sync() with images.  `n_images == 0` is the text-only path and
 * is exactly what pulsar_session_sync() calls.
 *
 * The reference merges an image only on its start_pos == 0 pass; here an image
 * request resumes from the session's KV like any sync (never from inside an
 * image block), and each block is merged by the prefill chunk that owns it,
 * wherever that chunk starts -- the block's rows are the one-pass prefill's,
 * byte for byte (L283, cuda-vision-chunk-gate).  Every span must fit inside a
 * single prefill chunk -- a request that would split one is refused loudly
 * rather than prefilled with sentinels whose embeddings were never merged. */
/** @param s the session
 *  @param prompt the expanded prompt (pulsar_expand_image_placeholders)
 *  @param images the images, start_pos set
 *  @param n_images their count (0 = text only)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return as pulsar_session_sync */
int pulsar_session_sync_mm(pulsar_session *s, const pulsar_tokens *prompt,
                           const pulsar_image_ref *images, int n_images,
                           char *err, size_t errlen);
/** The renderer's half of prepare_vl_inputs(), and the ONLY producer of the
 * sentinel blocks the engine's mm prefill consumes: walk `prompt` from `from`
 * and replace every placeholder token with that image's block (the family's
 * geometry); [0, from) is copied unchanged -- 0 for a rendered prompt, the live
 * length for a continuation whose held history already carries its blocks (a
 * family's placeholder may be its block token: Qwen's <|image_pad|>),
 * in request order, setting `images[i].start_pos` to the block's first slot
 * (the token count at that moment; see pulsar_image_ref).
 *
 * `out` must be zero-initialized and is owned by the caller afterwards (it
 * receives the expanded prompt); `images[i].bytes/len` are read for the decode.
 * Refuses, with `err` set, when the placeholder and image counts disagree, when
 * an image cannot be decoded or is not one the tower accepts, when the artifact
 * carries no vision tower, or when the vocab has no placeholder token.  Text
 * with no image passes through unchanged, but callers must NOT route a
 * text-only request through here: pulsar_session_sync() is its path.
 *
 * The engine's sync API is BLOCK-based, so this call must happen before it: a
 * placeholder id reaching pulsar_session_sync_mm() is a caller bug. */
/** @param e the engine
 *  @param prompt the rendered prompt
 *  @param from first token to walk ([0, from) is copied)
 *  @param images the images in placeholder order; start_pos is written
 *  @param n_images their count
 *  @param out receives the expanded prompt (freed on a refusal)
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return 1 on success, 0 on a refusal (note: not the 0-is-success convention) */
int pulsar_expand_image_placeholders(pulsar_engine *e, const pulsar_tokens *prompt, int from,
                                     pulsar_image_ref *images, int n_images,
                                     pulsar_tokens *out, char *err, size_t errlen);
/** The image BLOCKS already present in tokens [0, len), by the engine's family's geometry: writes each block's
 *  first slot (the start_pos an expanded image carries) to `starts`, in order, up to `cap`.  Returns how many blocks
 *  there are (which may exceed `cap`), or -1 when a sentinel id there belongs to no well-formed block.  The server
 *  uses it to place a live continuation's held images (L261). */
/** @param e the engine
 *  @param tokens the history
 *  @param len tokens of it to scan (<= tokens->len)
 *  @param starts receives block starts; may be NULL with cap 0
 *  @param cap slots at starts
 *  @return the block count, or -1 */
int pulsar_image_block_starts(pulsar_engine *e, const pulsar_tokens *tokens, int len, int *starts, int cap);
/** The bytes a token history was rendered from: each token's text, and each image block ONCE as the text of the
 *  placeholder token its expansion replaced (the family's geometry and placeholder_id -- DeepSeek's
 *  `<｜deepseek_image｜>`, Qwen's one `<|image_pad|>`), so a disk chain over an image is found by the request text that
 *  brings it; which image it is, the segment's records say (L281).  malloc'd; NULL when a sentinel id belongs to no
 *  well-formed block. */
/** @param e the engine
 *  @param tokens the history
 *  @param out_len receives the byte count; may be NULL
 *  @return the malloc'd text (caller frees), or NULL */
char *pulsar_history_text(pulsar_engine *e, const pulsar_tokens *tokens, size_t *out_len);
/** L281: the hash an image's bytes are known by in the KV's image records (what a disk chain's segment carries and a
 *  restore compares; never 0). */
/** @param img the image (its bytes are hashed)
 *  @return the content hash, nonzero */
uint64_t pulsar_image_hash(const pulsar_image_ref *img);
/** L281: the content hashes of the image blocks the session's KV holds, in order, up to `cap`; returns how many it
 *  holds. */
/** @param s the session
 *  @param hashes receives up to cap hashes
 *  @param cap slots at hashes
 *  @return the blocks held (may exceed cap); 0 without a session */
int pulsar_session_image_hashes(pulsar_session *s, uint64_t *hashes, int cap);
/** L281: how much of the session's history a disk chain may hold -- all of it, or up to the first image block whose
 *  image the session does not record: the position it ends at.  none() when its image blocks are malformed. */
/** @param s the session
 *  @return the persistable end; 0 without a valid checkpoint */
TokPos pulsar_session_persist_end(pulsar_session *s);
/** L281: the image records a stored segment carries (`fp` at its payload, `bytes` long), read without loading it:
 *  their content hashes, in order, up to `cap` (*n_out of them; a v2 segment carries none).  0 on success. */
/** @param fp the stream at the segment's payload (it is read forward)
 *  @param bytes the segment's size
 *  @param hashes receives up to cap hashes
 *  @param cap slots at hashes
 *  @param n_out receives the count written
 *  @return 0, or 1 on a malformed or other-version segment */
int pulsar_session_segment_images(FILE *fp, uint64_t bytes, uint64_t *hashes, uint32_t cap, uint32_t *n_out);
/** Where the last pulsar_session_sync started evaluating: the grid snapshot
 * position it resumed from, 0 when it prefilled from the start, none() when the
 * call did not resume (nothing to evaluate, or a checkpoint that was not a
 * prefix).  An instrument for the chunk-neutrality gate (L194). */
/** @param s the session
 *  @return the resume position, 0, or none() */
TokPos pulsar_session_resume_origin(pulsar_session *s);
/** The tokens the installed bank's history and `prompt` share from position 0 (by id). */
/** @param s the session
 *  @param prompt the prompt
 *  @return the shared length, tokens */
TokCount pulsar_session_common_prefix(pulsar_session *s, const pulsar_tokens *prompt);

/** L115 — THE prefix-reuse authority.  Every "can this state serve this
 * prompt, and from which cut?" decision (routing, the worker's cache
 * resolver, session sync) asks this and nothing else.
 *
 * The comparison is over BYTES, not token ids: generated text freezes the
 * boundaries the model SAMPLED into the live history, while the client's
 * echo re-tokenizes the same bytes CANONICALLY (live `))`+`**` vs echoed
 * `))**`).  Ids are one rendering of the history; the bytes are the
 * history.  Control/special tokens (empty text) match by id only, so role
 * markers can never byte-alias into content.
 *
 * TWO CURRENCIES, deliberately both returned: across a seam the two sides
 * spend a different NUMBER of tokens on the same bytes.  `live_cut` is the
 * live/bank side (KV rows, cut points, rewind targets); `prompt_cut` is the
 * request side (accounting, cache_read_tokens, prefill bounds).  Mixing
 * them is the hazard this struct exists to make impossible — a live-side
 * count used to index the request array can run past its end — so each is
 * its own type (L334, pulsar_units.h: TokCount, PromptToks), and the one
 * conversion between them is pulsar_prefix_match_stitched_len. */
typedef struct {
    TokCount   live_cut;    ///< live/bank-side token count: KV rows, cut points, rewind targets
    PromptToks prompt_cut;  ///< request-side token count for the SAME bytes: accounting, prefill bounds
    bool seamed;            ///< token ids disagreed inside the covered region (same bytes, different split)
} pulsar_prefix_match;

/** L334: the length, in live-side tokens, of the prompt a sync runs for a request of `prompt_len` tokens -- the
 *  stitch (live[0, live_cut) + prompt[prompt_cut, prompt_len)): past the cut the stitched prompt carries the
 *  request's own tokens, so a prompt-side distance there is a live-side one.  The conversion from prompt to live
 *  side past the cut; the other way, below it, is pulsar_tokens_prompt_held_with.
 *  @param m the match the stitch is cut at
 *  @param prompt_len the request prompt's length */
/** @return the stitched prompt's length, live-side tokens */
static inline TokCount pulsar_prefix_match_stitched_len(const pulsar_prefix_match *m, PromptToks prompt_len) {
    return m->live_cut + TokCount{(prompt_len - m->prompt_cut).raw()};
}

/** A token's text for the seam walks: a malloc'd copy the caller frees (NULL = none), its length in *len
 *  (pulsar_token_text's contract; 0 bytes = a control/special token, matched by id only). */
typedef char *(*pulsar_token_text_fn)(void *ud, int token, size_t *len);

/** The byte-level prefix match (above) of `live` against `prompt`, with the engine's token texts.
 *  @param e the engine; NULL gives an empty match
 *  @param live the live history's tokens
 *  @param live_len their count
 *  @param prompt the request's tokens
 *  @param prompt_len their count
 *  @param out receives the match */
void pulsar_tokens_prefix_match(pulsar_engine *e,
                                const int *live, int live_len,
                                const int *prompt, int prompt_len,
                                pulsar_prefix_match *out);
/** pulsar_tokens_prefix_match over any vocabulary's text (`text`, `ud`): the walk itself, which the engine's entry
 *  runs with its tokenizer's text. */
/** @param text the vocabulary's token text
 *  @param ud passed to text
 *  @param live the live history's tokens
 *  @param live_len their count
 *  @param prompt the request's tokens
 *  @param prompt_len their count
 *  @param out receives the match */
void pulsar_tokens_prefix_match_with(pulsar_token_text_fn text, void *ud,
                                     const int *live, int live_len,
                                     const int *prompt, int prompt_len,
                                     pulsar_prefix_match *out);
/** L334: the prompt tokens the live history up to `at` holds -- the conversion from live to prompt side below the
 *  cut, the one authority for "how many of the request's tokens does a state at live position `at` serve".  The
 *  request's tokens whose bytes lie wholly inside live[0, at) and equal them: at a boundary both spellings share,
 *  exactly the prompt-side position of `at` (the token-seam match of live[0, at), its prompt_cut); inside a
 *  re-spelled stretch, that shared boundary below `at` plus the prompt tokens of the stretch that end at or before
 *  `at`'s byte (a prompt token the state holds only part of is not served).  Never more than the full match's
 *  prompt_cut for any `at` up to its live_cut.
 *  @param text the vocabulary's token text
 *  @param ud its context
 *  @param live the live history
 *  @param at the live position (<= the history's length)
 *  @param prompt the request's tokens
 *  @param prompt_len their count */
/** @return the prompt tokens held */
PromptToks pulsar_tokens_prompt_held_with(pulsar_token_text_fn text, void *ud, const int *live, TokPos at,
                                          const int *prompt, int prompt_len);
/** pulsar_tokens_prefix_match of the INSTALLED bank's history against `prompt`.
 *  @param s the session; NULL gives an empty match
 *  @param prompt the request
 *  @param out receives the match */
void pulsar_session_prefix_match(pulsar_session *s, const pulsar_tokens *prompt,
                                 pulsar_prefix_match *out);
/** pulsar_tokens_prefix_match of bank `bank`'s history (installed or saved) against `prompt`.
 *  @param s the session (a bank pool family; else an empty match)
 *  @param bank the bank
 *  @param prompt the request
 *  @param out receives the match */
void pulsar_session_bank_prefix_match(pulsar_session *s, uint32_t bank,
                                      const pulsar_tokens *prompt,
                                      pulsar_prefix_match *out);
/** Greedy token over the session's live logits: the first finite value seeds,
 * the lowest id wins a tie. @return the id, or -1 when no logit is finite. */
/** @param s the session */
int pulsar_session_argmax(pulsar_session *s);
/** pulsar_session_argmax with `excluded_id` removed from the row. @return the id, or -1. */
/** @param s the session
 *  @param excluded_id the id to skip */
int pulsar_session_argmax_excluding(pulsar_session *s, int excluded_id);
/** Returned by the draws (pulsar_session_draw, _draw_rows, _sample) when the row is DEGENERATE (no finite logit,
 * a probability mass that is not a positive finite number, a non-finite knob): no token is drawn and the caller
 * must fail its request (L188).  Never a valid id. */
#define PULSAR_SAMPLE_REFUSED (-1)
/** L302: one draw (THE sampler, on the device: src/cuda/pulsar_cuda_sampler.cu).  `rng` is the request's sampler
 * stream (seed it with the request's seed): the draw reads the key of the stream's next token and steps the stream
 * past it -- greedy too, so the stream's index stays the count of tokens drawn.  The key is a function of (seed,
 * index) only and the speculative lanes draw each token at the same key, so a seeded request is the same text
 * whichever lane decodes which of its tokens (L288 sdet).  `greedy` draws the argmax whatever the temperature (a
 * tool call's structure, L296); `mask` (NULL = none; ceil(logits width / 32) words, bit i set = token i allowed)
 * makes every other token -inf before every transform (L303's structured outputs, L272's tool names). */
typedef struct pulsar_row_draw {
    /** Softmax temperature; <= 0 draws the argmax.  Not finite (and not greedy): the draw refuses. */
    float temperature;
    /** Keep the k most likely tokens; <= 0 = off, clamped to 1024. */
    int top_k;
    /** Nucleus mass to keep; outside (0, 1] = 1 (off). */
    float top_p;
    /** Drop tokens under min_p times the best token's probability; < 0 = 0 (off). */
    float min_p;
    /** The request's sampler stream (pulsar_sample_key index); read and stepped past the drawn token. Borrowed. */
    uint64_t *rng;
    /** Draw the argmax whatever the temperature (a tool call's structure, L296). */
    bool greedy;
    /** Allowed-token bitmask, ceil(logits width / 32) words, bit i set = token i allowed; NULL = none.  Borrowed. */
    const uint32_t *mask;
} pulsar_row_draw;
/** A draw from the session's live logits. @return the token, or PULSAR_SAMPLE_REFUSED. */
/** @param s the session
 *  @param d the draw's knobs (its rng is stepped) */
int pulsar_session_draw(pulsar_session *s, const pulsar_row_draw *d);
/** Draws from rows rows[0 .. n) of the last batched step's headed rows -- decode_multiseq /
 * decode_mixed / decode_fused's logits row k, which stay on the device -- one per row: tokens[k], or
 * PULSAR_SAMPLE_REFUSED for a degenerate row (that request fails; the others stand).  @return 0, or 1 when nothing
 * was drawn (bad arguments, a device failure). */
/** @param s the session
 *  @param rows the headed rows to draw from
 *  @param d n draws' knobs, d[k] for rows[k]
 *  @param n the count
 *  @param tokens receives n tokens */
int pulsar_session_draw_rows(pulsar_session *s, const uint32_t *rows, const pulsar_row_draw *d, uint32_t n,
                             int *tokens);
/** The last batched step's headed row `row` becomes `bank`'s next-token row (a mixed step's finished prompt run: the
 * bank's restore installs it).  On the device.  @return 0, or 1. */
/** @param s the session
 *  @param row the headed row
 *  @param bank the bank it becomes the next-token row of */
int pulsar_session_step_row_to_bank(pulsar_session *s, uint32_t row, uint32_t bank);
/** pulsar_session_draw with no mask and no greedy override. @return the token, or -1. */
/** @param s the session
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param rng the request's stream (stepped) */
int pulsar_session_sample(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
/** The k best tokens of the live logits, best first (on a TP leader, after settling the pending ack).
 *  @param s the session
 *  @param out receives k entries (id -1 / -inf past the finite ones)
 *  @param k entries wanted, clamped to 20
 *  @return the entries that are real (0: none, or a failure) */
int pulsar_session_top_logprobs(pulsar_session *s, pulsar_token_score *out, int k);
/** Tensor-parallel leader: read and check the pair's pending logits-identity
 * acks now (pulsar_session_eval pipelines them one step; logits readers and
 * pulsar_session_free settle on their own).  A driver that ends generation on
 * a token decision (EOS) calls this to learn whether the step that drew it
 * agreed across ranks.  @return 0 on success (also for untensored sessions),
 * 1 with `err` filled on a mismatch or a dead pair. */
/** @param s the session
 *  @param err receives the mismatch
 *  @param errlen bytes at err */
int pulsar_session_settle(pulsar_session *s, char *err, size_t errlen);
/** One token's score in the live logits.
 *  @param s the session
 *  @param token the id
 *  @param out receives the score
 *  @return 1 when written; 0 for an out-of-range id, a row with no finite logit, or a failure */
int pulsar_session_token_logprob(pulsar_session *s, int token, pulsar_token_score *out);
/** The logprobs of row `row` of the last batched step's headed rows (pulsar_session_draw_rows' rows): the scored
 * token's (*have_chosen 1, or 0 when the row has no finite logit or token is not an id) and the top k (k <= 20;
 * *n_top as pulsar_session_top_logprobs returns).  @return 0, or 1 on bad arguments / a device failure. */
/** @param s the session
 *  @param row the headed row
 *  @param token the scored token
 *  @param chosen receives its score
 *  @param have_chosen receives 1 when chosen is valid
 *  @param top receives k entries
 *  @param k entries wanted
 *  @param n_top receives the real entries */
int pulsar_session_step_row_logprobs(pulsar_session *s, uint32_t row, int token, pulsar_token_score *chosen,
                                     int *have_chosen, pulsar_token_score *top, int k, int *n_top);
/** Copy the live logits row to the host.
 *  @param s the session
 *  @param out receives pulsar_engine_logits_width floats
 *  @param cap floats at out
 *  @return floats copied (the width), or 0 when cap is short or nothing is live */
int pulsar_session_copy_logits(pulsar_session *s, float *out, int cap);
/** Overwrite the live logits row (mirrored on a TP group).
 *  @param s the session
 *  @param logits the row
 *  @param n its length; must equal pulsar_engine_logits_width
 *  @return 0, or 1 on a refusal */
int pulsar_session_set_logits(pulsar_session *s, const float *logits, int n);
/** Classic single-token decode: evaluate `token` at the installed bank's frontier and leave its next-token
 *  logits live.  Refused after a multiseq step until a sync re-establishes the state.
 *  @param s the session
 *  @param token the id
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or nonzero on a failure */
int pulsar_session_eval(pulsar_session *s, int token, char *err, size_t errlen);
/** Tier-2 batched multi-session decode (bank pool, PULSAR_MSEQ_BANKS >= 2): one
 * decode request per co-scheduled session, all advanced one token by ONE
 * weight sweep.  reqs[k] = {TRUE bank id, absolute position of `token`,
 * current input token}.
 *
 * logits (out): row k is bank[k]'s next-token distribution, rows strided by
 * the engine's logits row width (the same width pulsar_session_copy_logits
 * returns); logits_cap is the TOTAL float capacity of the buffer and is
 * rejected when it cannot hold n rows.  Sampling stays per-session on the
 * host with each session's own sampler/rng state.
 *
 * DETERMINISM: a session's row is reproducible at a FIXED batch composition,
 * and co-scheduling additional sessions does not perturb the existing rows
 * (per-row math is batch-width invariant for n >= 2).  The batched sweep is
 * NOT bit-identical to classic single-token decode, however — it is a
 * different kernel path — so a session's greedy token stream can diverge
 * from the same prompt decoded alone through pulsar_session_eval once a
 * near-tie tips.  That is the two-mode contract (plan §2.4), not a bug.
 *
 * Requirements: each bank prefilled to exactly `pos` through the classic
 * single-bank path with its per-bank counters captured, one request per
 * bank, no position-0 rows, and NO speculation co-scheduled (n >= 2 is
 * plain decode by contract).
 *
 * `logits` receives n rows of pulsar_engine_logits_width(engine) floats; row k
 * corresponds to reqs[k] (i.e. bank reqs[k].bank). `logits_cap` is the
 * buffer's capacity IN FLOATS and must be at least n * that width — size it
 * with pulsar_engine_logits_width, never pulsar_engine_vocab_size.
 *
 * On success the session's single-bank bookkeeping is INVALIDATED -- the
 * checkpoint no longer describes any one bank, and a step that touched OTHER
 * banks leaves this session's per-bank carry needing re-establishment. The
 * caller owns per-bank histories and must re-establish state explicitly.
 * (This used to say the scalar frontier counters hold a cross-bank SUPERSET;
 * stage 1b deleted those scalars, so that particular shape is now
 * unrepresentable. The invalidation and the guard below are unchanged.)
 * Until it does, the single-session entries FAIL LOUD rather than corrupt KV:
 * pulsar_session_eval, pulsar_session_generate_speculative and
 * pulsar_session_eval_speculative_block all return an error. A pulsar_session_sync
 * (which rebuilds from zero) re-establishes state and clears the condition.
 *
 * The session's own s->logits is NOT updated by a multiseq step (the step has
 * no single "the session's" row — every row belongs to a bank). Accordingly
 * pulsar_session_argmax(s) / pulsar_session_token_logprob(s, ...) keep returning
 * the pre-step classic distribution and must not be used to interpret a
 * multiseq result: sample from the `logits` rows returned here instead.
 *
 * Returns 0 on success; 1 on a recoverable rejection (bad args/contract; no
 * state mutated); -1 on a fatal mid-sweep failure (tear the session down). */
/** One row of a batched decode step: decode `token` for `bank` at `pos`.
 *
 * Rows for the same bank must be contiguous and consecutive in `pos`, and every
 * batched bank's compressor frontier must be position-true on entry -- the
 * driver rejects the step otherwise rather than corrupting KV. */
typedef struct {
    uint32_t bank;   ///< TRUE bank id in the session's pool
    TokPos   pos;    ///< absolute position of `token` (the bank's committed length)
    int      token;  ///< input token id decoded at `pos`
} pulsar_multiseq_req;
/** One batched decode step (the Tier-2 contract above the request struct).
 *  @param s the session
 *  @param reqs n rows, one per bank
 *  @param n the row count
 *  @param logits receives n rows of pulsar_engine_logits_width floats
 *  @param logits_cap the TOTAL float capacity of logits (all rows, not one row)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, 1 (rejected, nothing mutated) or -1 (fatal; tear the session down) */
int pulsar_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs,
                                uint32_t n, float *logits, int logits_cap,
                                char *err, size_t errlen);
/* plan-34 phase-2: mixed prefill+decode batched step. SAME contract, kernel path,
 * and post-step invalidation as pulsar_session_decode_multiseq, but the per-row
 * descriptor scratch is HEAP-allocated (up to the session's prefill_cap rows)
 * rather than the fixed PULSAR_MSEQ_MAX stack — the row representation the fused
 * mixed-batch step grows into. INCREMENT 1 is a byte-identical refactor: for a
 * decode-only batch (1 row per bank, n_rows == bank count) it produces the SAME
 * per-row logits as pulsar_session_decode_multiseq. n_rows must be <= prefill_cap;
 * the underlying kernel still bounds the live row count to PULSAR_MSEQ_MAX this
 * increment (later increments lift it). Returns 0 / 1 / -1 as decode_multiseq. */
/* On success `logits` receives ONE row per per-bank RUN (the LAST row of each
 * run — plan-34 inc 3), in run order (ascending first-appearance of the bank);
 * *out_n_rows (may be NULL) is set to that count (== the number of distinct
 * banks). For a decode-only batch (1 row per bank) that is n_rows rows in bank
 * order — unchanged. For a K-row prefill run it is a single row (the K-th token's
 * logits). Size `logits` for the run count, not n_rows. */
/* max_head_runs (plan-34 inc 5, LEVER 1): emit head logits for only the FIRST
 * max_head_runs runs (0 = ALL runs = inc-3/4 behavior). A fused mixed step whose
 * trailing prefill run is on an INTERMEDIATE chunk passes n_dec (the decode banks
 * only): the prefill run's intermediate logits are never consumed, so skipping them
 * is unobservable AND lets the head take the single-block identity path (no
 * two-block gather/resync, no wasted K-row prefill head GEMM). *out_n_rows is then
 * the emitted count (== min(n_runs, max_head_runs), with 0 meaning n_runs). */
/** max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS (plan-34 inc 6): emit logits for
 * EVERY row of the batch, logits row k == batch row k, *out_n_rows == n_rows.
 * This is the batched-speculative-verify contract: a run of K draft rows needs
 * all K logit rows for the accept walk, not just the run's last. The head takes
 * the identity path over the whole batch (no gather). Requires n_rows <= the
 * spec-logits row capacity (16); rejected (return 1) otherwise. */
#define PULSAR_MSEQ_HEAD_ALL_ROWS 0xffffffffu
/** The mixed prefill + decode step (the plan-34 notes above).
 *  @param s the session
 *  @param reqs n_rows rows, each bank's rows contiguous
 *  @param n_rows the row count
 *  @param logits receives the headed rows
 *  @param logits_cap the TOTAL float capacity of logits (all rows, not one row)
 *  @param out_n_rows receives the rows written; may be NULL
 *  @param max_head_runs runs to head (0 = all; PULSAR_MSEQ_HEAD_ALL_ROWS = every row)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, 1 or -1 as pulsar_session_decode_multiseq */
int pulsar_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs,
                             uint32_t n_rows, float *logits, int logits_cap,
                             uint32_t *out_n_rows, uint32_t max_head_runs,
                             char *err, size_t errlen);
/** L260 fusion: a FUSED step -- the batched speculative lane's verify rows and
 * queued prompts' prefill chunks in ONE forward.  Rows [0, n_dec) are decode /
 * verify rows: decode row kind, every row headed in place (logits rows
 * [0, n_dec), or the armed argmax / compact readbacks, exactly as
 * PULSAR_MSEQ_HEAD_ALL_ROWS).  Rows [n_dec, n_rows) are n_pf prefill runs, one
 * bank each, in order; a run may start at position 0 (a fresh bank's first
 * chunk; invalidate it first).  head_last[r] != 0 heads prefill run r's LAST row
 * into the next logits row after the decode rows, in run order -- the prompt's
 * first-token distribution on its final chunk.  Every prefill run's drafter
 * anchors fill its bank's prompt ring, as a classic prefill chunk does.
 * *out_n_rows = n_dec + the headed runs.  n_dec + headed runs <=
 * pulsar_engine_fused_heads_max (the family's cap). */
/** The most prompt runs one fused step carries (the shape's head_last bound). */
#define PULSAR_FUSED_PF_MAX 16u
/** The row layout of one pulsar_session_decode_fused step (see the L260 fusion note above). */
typedef struct {
    /** Decode / verify rows, rows [0, n_dec) of the step; every one is headed. */
    uint32_t n_dec;
    /** Prefill runs after the decode rows, one bank each, in order; <= PULSAR_FUSED_PF_MAX. */
    uint32_t n_pf;
    /** Per prefill run r: nonzero heads run r's last row into the next logits row after the decode rows. */
    uint8_t  head_last[PULSAR_FUSED_PF_MAX];   /* indexed by prefill run, n_pf <= PULSAR_FUSED_PF_MAX */
} pulsar_fused_shape;
/** The L260 fused step (the note above pulsar_fused_shape); refused without pulsar_engine_has_fused_step.
 *  @param s the session
 *  @param reqs n_rows rows: decode rows, then the prefill runs
 *  @param n_rows the row count
 *  @param shape the decode / prefill split and the head flags
 *  @param logits receives the headed rows
 *  @param logits_cap the TOTAL float capacity of logits (all rows, not one row)
 *  @param out_n_rows receives n_dec + the headed runs
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or nonzero on a failure */
int pulsar_session_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs,
                                uint32_t n_rows, const pulsar_fused_shape *shape,
                                float *logits, int logits_cap, uint32_t *out_n_rows,
                                char *err, size_t errlen);
/** Tier-2 unified bank model (server-facing).  A bank-pooled session's graph
 * hosts up to N co-scheduled conversations as banks; each server slot maps to a
 * bank id.  The pool size is chosen at session create (PULSAR_MSEQ_BANKS today);
 * pulsar_session_bank_count reports it (1 when not pooled).
 *
 * A conversation lives in ONE bank for its lifetime.  To do classic/spec work
 * on bank b the caller repoints the device views AND restores b's host carry;
 * pulsar_session_bank_restore does both (+ clears the multiseq poison cheaply, no
 * re-prefill).  pulsar_session_bank_save snapshots the live bank before switching
 * away.  Typical server flow:
 *   prefill:  pulsar_session_bank_repoint(s,b); pulsar_session_invalidate(s);
 *             pulsar_session_sync(s, prompt, ...); pulsar_session_bank_save(s,b);
 *   decode 1: pulsar_session_bank_state_restore(s,b);
 *             pulsar_session_generate_speculative(...);
 *             pulsar_session_bank_state_save(s,b);
 *   decode N: pulsar_session_decode_multiseq(s, reqs, N, ...) over the live banks
 *             (each captured via bank_state_save first); sample per row on host.
 * repoint/save/restore must run only between fully synchronized forwards. */
/** @param s the session
 *  @return the pool size (1 when not pooled; 0 for a family without banks) */
int  pulsar_session_bank_count(pulsar_session *s);
/** Point the device views at `bank` (a family without a separate repoint restores it instead).
 *  @param s the session
 *  @param bank the bank
 *  @return 0, or 1 on a refusal */
int  pulsar_session_bank_repoint(pulsar_session *s, uint32_t bank);
/** Save the live bank's host carry as `bank`'s (mirrored on a TP group).
 *  @param s the session
 *  @param bank the bank the live state belongs to */
void pulsar_session_bank_state_save(pulsar_session *s, uint32_t bank);
/** Install `bank`: repoint its device views and restore its host carry (clears the multiseq poison).
 *  @param s the session
 *  @param bank the bank
 *  @return true on success (on a TP group: every rank agreed) */
bool pulsar_session_bank_state_restore(pulsar_session *s, uint32_t bank);

/** plan-34 inc 6: batched speculative rounds. One ROUND verifies one bank's
 * pending drafts against a forward the CALLER runs -- the fused classic loop
 * and the server's batched lane share every line of round logic; only the
 * forward differs (classic verify vs one shared decode_mixed ALL_ROWS step
 * covering every decode bank's rows).
 *
 * Flow per bank, under bank_state_restore(bank):
 *   first = pulsar_session_spec_next_base(...);        // carry or fresh draw
 *   pulsar_session_spec_round_begin(...);              // guards, snapshot, push
 *   pulsar_spec_round_fill_reqs(...);                  // rows for decode_mixed
 * then ONE pulsar_session_decode_mixed(ALL_ROWS) over all banks' rows, then
 * per bank under bank_state_restore(bank):
 *   pulsar_session_spec_round_end(...);                // walk, state, redraft
 *   pulsar_session_bank_state_save(bank);              // persist trimmed truth
 * A round that was begun but whose forward failed MUST be aborted
 * (pulsar_session_spec_round_abort) or the frontier snapshot leaks and the
 * checkpoint keeps unverified rows. */
typedef struct pulsar_spec_round pulsar_spec_round;
/** @return a new, empty round (caller-owned; free with pulsar_spec_round_free) */
pulsar_spec_round *pulsar_spec_round_new(void);
/** @param r the round; NULL is a no-op */
void pulsar_spec_round_free(pulsar_spec_round *r);
/** Upper bound on the rows the next round on the live bank will contribute to
 * a shared forward (1 base + pending drafts, before begin's trims). Check the
 * row budget against this BEFORE pulsar_session_spec_next_base: the base draw
 * consumes the carry, and a rejection-residual carry must be emitted exactly
 * (the acceptance proof depends on it), not discarded and redrawn. */
/** @param s the session
 *  @return 1 + the live bank's pending drafts (at most 17) */
uint32_t pulsar_session_spec_next_rows_max(const pulsar_session *s);
/** The position the round began at (its base token's position).  After
 * pulsar_session_spec_round_end the bank's frontier must equal this plus the
 * accepted count it returned (L155); the server checks exactly that. */
/** @param r the round
 *  @return the base position, or -1 without a round */
int pulsar_spec_round_saved_len(const pulsar_spec_round *r);
/** L112: the adaptive draft controller's current depth for a bank (live state
 * for the live bank, saved carry otherwise; bank 0 in bankless sessions).
 * 0 = no drafter / nothing valid. Pure host read for metrics/monitoring. */
/** @param s the session
 *  @param bank the bank
 *  @return the depth, or 0 */
int pulsar_session_bank_spec_depth(pulsar_session *s, uint32_t bank);
/** L049 adaptive per-bank K: bank `bank`'s pending drafts from the SAVED carry
 * -- no bank switch, valid at sweep start where every bank was saved at the
 * previous round's end.  Writes the base token they continue (*base: the
 * round uses them only when its base is this one), up to 16 draft tokens to
 * tok[] and their per-position acceptance confidences to conf[], and returns
 * the pending count (0 = no saved state or no pendings; the bank still gets
 * its base row).  The allocation these feed is an UPPER bound on K --
 * round_begin's own validity trims still apply.  L309: the tokens are what the
 * server's grammar gate prunes a chain by before the pick. */
/** @param s the session
 *  @param bank the bank
 *  @param base receives the base token (-1 with none)
 *  @param tok receives the drafts
 *  @param conf receives their confidences
 *  @return the pending count */
uint32_t pulsar_session_bank_pendings(const pulsar_session *s, uint32_t bank, int32_t *base, int32_t tok[16],
                                      float conf[16]);
/** The carry-or-sample base draw (the head of generate_speculative). Never
 * returns EOS handling -- the caller checks and short-circuits like
 * generate_speculative does.  A fresh draw reads the stream's next key and
 * does not step it: the round that emits the base steps past it. */
/** @param s the session
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param rng the request's stream (read, not stepped)
 *  @return the base token, or PULSAR_SAMPLE_REFUSED */
int pulsar_session_spec_next_base(pulsar_session *s, float temperature,
                               int top_k, float top_p, float min_p,
                               uint64_t *rng);
/** Begin a round on the installed bank: guard and trim the pendings, snapshot the frontier, push the base and
 *  drafts onto the checkpoint (the flow above).
 *  @param s the session
 *  @param r the round
 *  @param first_token the base token (pulsar_session_spec_next_base)
 *  @param max_tokens tokens left in the request
 *  @param accepted_cap the caller's accepted[] capacity
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or -1 (snapshot failure; the session is poisoned) */
int pulsar_session_spec_round_begin(pulsar_session *s, pulsar_spec_round *r,
                                 int first_token, int max_tokens, int accepted_cap,
                                 float temperature, int top_k, float top_p,
                                 float min_p, char *err, size_t errlen);
/** Rows a begun round will occupy (1 + surviving drafts). Check the caller's
 * buffer space against this BEFORE fill_reqs -- fill_reqs writes this many
 * entries unconditionally. */
/** @param r a begun round
 *  @return its rows */
uint32_t pulsar_spec_round_n_rows(const pulsar_spec_round *r);
/** Rows this round contributes to the shared forward: n_batch reqs (the
 * round's base token at the pre-round frontier, then the pending drafts) --
 * every verify's rows, the single lane's included.  Returns the row count
 * written. */
/** @param r a begun round
 *  @param bank the bank the rows name
 *  @param out receives pulsar_spec_round_n_rows entries
 *  @return rows written */
uint32_t pulsar_spec_round_fill_reqs(const pulsar_spec_round *r, uint32_t bank, pulsar_multiseq_req *out);
/** Finish the round against the last shared forward's rows: this round's rows start at the step's headed row
 * `row0` (they stay on the device, L302).  Emits into accepted[] and returns the count, or -1 (session poisoned).
 * The walk -- on the device -- draws each row's token at its own key in `rng` (the stream at the round's base) and
 * steps the stream past every token it emits (L288 sdet). */
/** @param s the session
 *  @param r the begun round
 *  @param first_token its base token
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param rng the request's stream at the round's base (stepped)
 *  @param row0 the round's first headed row in the step
 *  @param accepted receives the emitted tokens
 *  @param accepted_cap slots at accepted
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return tokens emitted, or -1 */
int pulsar_session_spec_round_end(pulsar_session *s, pulsar_spec_round *r,
                               int first_token,
                               float temperature, int top_k, float top_p,
                               float min_p, uint64_t *rng,
                               uint32_t row0,
                               int *accepted, int accepted_cap,
                               char *err, size_t errlen);
/** Gate-facing (L182): pulsar_session_spec_round_end with the acceptance walk
 * replaced by TEACHER FORCING.  Draft i is accepted iff it equals truth[i]; the
 * carry is the first true token the drafts did not cover, so `truth` must hold
 * K + 1 ids (K = the round's pending drafts).  No draw is made and rng is
 * untouched; controllers, metrics, the trim and the deferred redraft run as in
 * production, so the next redraft is conditioned on the corpus and the
 * drafter's proposal becomes a deterministic function of the context.  The
 * teacher-forced acceptance probe (tests/spec_teacher_forced_probe.cpp) reads
 * that proposal against the target's row at fixed positions. */
/** @param s the session
 *  @param r the begun round
 *  @param first_token its base token
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param rng untouched
 *  @param row0 the round's first headed row
 *  @param truth K + 1 true ids
 *  @param accepted receives the emitted tokens
 *  @param accepted_cap slots at accepted
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return tokens emitted, or -1 */
int pulsar_session_spec_round_end_forced(pulsar_session *s, pulsar_spec_round *r,
                                      int first_token,
                                      float temperature, int top_k, float top_p,
                                      float min_p, uint64_t *rng,
                                      uint32_t row0,
                                      const int32_t *truth,
                                      int *accepted, int accepted_cap,
                                      char *err, size_t errlen);
/** Abort a begun round whose forward failed: drop its frontier snapshot and its unverified rows.
 *  @param s the session
 *  @param r the round */
void pulsar_session_spec_round_abort(pulsar_session *s, pulsar_spec_round *r);
/** Arm (n_rows > 0) or disarm (0) the drafter anchor capture + Stage-B comp
 * saves for the NEXT batched forward -- the shared verify step saves every
 * row so each bank's round_end can consume its slice. Mirrors what the fused
 * loop does around its own forward. */
/** @param s the session
 *  @param n_rows rows of the next forward; 0 disarms */
void pulsar_session_spec_arm_capture(pulsar_session *s, uint32_t n_rows);
/** L150: the batched lane's round_end DEFERS drafting; after every bank's
 * round_end (and bank_state_save) call this once with the live rounds, their
 * banks and each bank's rng (read, never stepped: a sampled draft is drawn at
 * the key of the token it proposes): it runs ONE drafter pass over all requesting banks
 * and leaves each bank's draft in its round. Never switches banks. Returns 0
 * (banks that could not be served keep no pendings and take a plain step next),
 * -1 with err on a device failure. Then, per bank, under the server's own bank
 * switch and before its bank_state_save, pulsar_session_spec_redraft_commit
 * stamps that bank's shadow with its draft. */
/** @param s the session
 *  @param rounds n live rounds
 *  @param banks their banks
 *  @param rngs their banks' streams (read only)
 *  @param n the count
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0 or -1 */
int pulsar_session_spec_redraft_batch(pulsar_session *s, pulsar_spec_round **rounds,
                                      const uint32_t *banks, uint64_t **rngs, int n,
                                      char *err, size_t errlen);
/** Stamp the installed bank's shadow with the draft pulsar_session_spec_redraft_batch left in `r`.
 *  @param s the session
 *  @param r the bank's round */
void pulsar_session_spec_redraft_commit(pulsar_session *s, pulsar_spec_round *r);
/** L260: the batched lane's per-bank bookkeeping, ONE call per phase.  Each
 * step names a bank; the call runs, bank after bank in order, exactly the
 * sequence the lane ran per bank -- bank_state_restore, the phase, then
 * bank_state_save -- so the banks' state and tokens are those of the per-bank
 * calls.  On a pair it is ONE mirrored frame with ONE verdict (a fingerprint
 * of every step's outcome) where the per-bank calls paid 2-3 verdict round
 * trips per bank.  The caller saves its live bank first: step 0's restore
 * overwrites whatever is installed.  Every step whose restore succeeded is
 * saved, whatever its status; afterwards the last such step's bank is
 * installed, clean. */
typedef enum {
    PULSAR_SPEC_STEP_OK = 0,
    PULSAR_SPEC_STEP_RESTORE_FAILED = 1, /* the bank is not installed; nothing ran */
    PULSAR_SPEC_STEP_LENGTH = 2,         /* assemble: the bank's frontier is at ctx */
    PULSAR_SPEC_STEP_SKIPPED = 3,        /* assemble: over the row budget; sat out untouched */
    PULSAR_SPEC_STEP_EOS = 4,            /* assemble: the base token is EOS -- emit it, never eval it */
    PULSAR_SPEC_STEP_FAILED = 5,         /* the phase refused; err says why */
} pulsar_spec_step_status;
/** One bank's step in a batched spec-lane phase (pulsar_session_spec_assemble_batch, _round_end_batch,
 *  _redraft_commit_batch): the caller fills the inputs, the phase writes the outputs. */
typedef struct {
    /* in */
    /** The bank this step restores, runs and saves. */
    uint32_t bank;
    /** The bank's spec round (caller-owned; the phases fill and consume it). */
    pulsar_spec_round *round;
    float temperature, top_p, min_p;
    /** Sampler top-k (the pulsar_row_draw knobs; temperature / top_p / min_p beside it). */
    int top_k;
    /** The request's sampler stream (assemble: the base draw; round_end: the walk).  Borrowed. */
    uint64_t *rng;           /* assemble (base draw), round_end (walk) */
    /** assemble: tokens left in the request. */
    int max_tokens;          /* assemble: tokens left in the request */
    /** assemble: drafts allocated under overflow (the verify pick); -1 = no allocation. */
    int k_alloc;             /* assemble: drafts allocated under overflow; -1 = no allocation */
    /** assemble: round_begin's cap; round_end: capacity of `accepted`, tokens. */
    int accepted_cap;        /* assemble: round_begin's cap; round_end: accepted[] capacity */
    /** round_end: receives this bank's accepted tokens (caller-owned, accepted_cap slots). */
    int *accepted;           /* round_end: this bank's accepted tokens */
    /* round_end (L296): bit i set = verify row i -- the token after the base and drafts 0..i-1; row K is a full
     * accept's carry -- draws greedy whatever `temperature` says.  The caller's sampling authority evaluated at
     * each position of the round (the server's tool-call override: a call's structure greedy, its payload sampled),
     * so a round that crosses between the two draws every token as one-token-a-step decoding would.  The carry and
     * the next chain record the carry row's params.  0 = `temperature` at every row. */
    /** round_end: bit i set = verify row i draws greedy; 0 = `temperature` at every row (see above). */
    uint32_t greedy_rows;
    /* L303: token masks, the device sampler's mask rows (pulsar_row_draw.mask; pulsar_engine_logits_width wide); NULL = unconstrained,
     * today's bytes.  assemble: row 0 masks a FRESH base draw (a carry was drawn under the previous round's row
     * mask).  round_end: row i masks verify row i (the token after the base and drafts 0..i-1; row K is a full
     * accept's carry), so each needs the request's grammar advanced through the base and drafts 0..i-1;
     * `mask_rows` > the row the walk stops at (a row past a draft the mask refuses is never read).  A masked row
     * is drawn through the mask exactly as a plain lane draws it -- spec == plain.  Not on a pair:
     * the masks do not cross it (the step fails by name). */
    /** Token mask rows, [mask_rows][ceil(pulsar_engine_logits_width / 32)] words; NULL = unconstrained (see above).  Borrowed. */
    const uint32_t *masks;
    /** Rows in `masks`. */
    uint32_t mask_rows;
    /* assemble out, round_end in */
    /** The round's base token, drawn by assemble; -1 when the step drew none. */
    int first_token;
    /** The bank's first row in the shared forward. */
    uint32_t row0;           /* the bank's first row in the shared forward */
    /** Rows the bank contributed to the shared forward (base + drafts). */
    uint32_t n_rows;         /* base + drafts it contributed */
    /* out */
    /** The step's outcome, a pulsar_spec_step_status. */
    int status;              /* pulsar_spec_step_status */
    /** round_end: tokens written to `accepted`. */
    int n_accepted;          /* round_end */
    /** round_end: the round's base position (L155). */
    TokPos pos_before;       /* round_end: the round's base position (L155) */
    /** round_end: the bank's frontier after the round. */
    TokPos pos_after;        /* round_end: the bank's frontier after the round */
    /** Why the step failed (NUL-terminated); empty when it did not. */
    char err[160];
} pulsar_spec_step;
/** @var pulsar_spec_step::temperature
 *  Sampler temperature for the base draw and the walk (pulsar_row_draw rules). */
/** @var pulsar_spec_step::top_p
 *  Sampler nucleus mass (pulsar_row_draw rules). */
/** @var pulsar_spec_step::min_p
 *  Sampler min-p floor (pulsar_row_draw rules). */
/** Per step: restore; LENGTH at ctx; SKIPPED when 1 + K (the allocation, or
 * the pendings) would pass `row_budget` -- checked BEFORE the base draw, so a
 * residual carry survives; the base draw (EOS short-circuits); round_begin;
 * the round's rows into reqs[row0..row0+n_rows) (reqs holds row_budget
 * entries).  *n_rows_out = rows written.  0, or -1 when the pair refused or
 * diverged (every step then counts as failed). */
/** @param s the session
 *  @param steps n steps
 *  @param n the count
 *  @param row_budget slots at reqs
 *  @param reqs receives the rows
 *  @param n_rows_out receives the rows written
 *  @return 0 or -1 */
int pulsar_session_spec_assemble_batch(pulsar_session *s, pulsar_spec_step *steps, int n,
                                       uint32_t row_budget,
                                       pulsar_multiseq_req *reqs, uint32_t *n_rows_out);
/** Every step's walk over the last shared forward's rows in one device launch, then per step: restore, round_end
 * (the step's row0 / first_token from assemble), save.  0 or -1 as above. */
/** @param s the session
 *  @param steps n assembled steps
 *  @param n the count
 *  @return 0 or -1 */
int pulsar_session_spec_round_end_batch(pulsar_session *s, pulsar_spec_step *steps, int n);
/** Per step: restore, pulsar_session_spec_redraft_commit, save.  0 or -1. */
/** @param s the session
 *  @param steps n steps
 *  @param n the count
 *  @return 0 or -1 */
int pulsar_session_spec_redraft_commit_batch(pulsar_session *s, pulsar_spec_step *steps, int n);
/** Gate-facing: read a batched redraft's result out of the round before it is
 * committed (ids[0] = the base, ids[1..n_draft] = the drafts; conf per draft
 * position; keep = the confidence-trimmed pending count). 1 if a result is
 * present, else 0. */
/** @param r the round
 *  @param ids receives the base and drafts; may be NULL
 *  @param conf receives the confidences; may be NULL
 *  @param n_draft receives the draft count; may be NULL
 *  @param keep receives the kept count; may be NULL
 *  @param sampled receives 1 for sampled drafts; may be NULL
 *  @return 1 or 0 */
int pulsar_session_spec_redraft_peek(const pulsar_spec_round *r, int32_t ids[17], float conf[16],
                                     uint32_t *n_draft, uint32_t *keep, int *sampled);
/** L264/L265: the resume grid of the loaded model (tokens): a resume starts at a multiple of it
 *  and a prefill leaves its checkpoints on it -- the server plans its fused chunks on it. */
/** @param s the session
 *  @return the grid, tokens */
uint32_t pulsar_session_resume_grid(const pulsar_session *s);
/** L284: true when a prompt cut ANYWHERE is the cold prefill's bytes, so a sync continues a history prefilled whole
 *  wherever it ends (Qwen); false when only a cut on the resume grid is (DeepSeek), so a sync continues only from a
 *  grid point and resumes a history that ends off it from the checkpoint below. */
/** @param s the session
 *  @return true for a split-invariant family */
bool pulsar_session_split_invariant(const pulsar_session *s);

/** Per-bank frontier readers for a bank-pooled session: the committed length,
 * token history, and common-prefix-with-prompt of ONE bank, correct even when
 * that bank is not the currently-installed one (the live bank reads the live
 * checkpoint; others read their saved host carry).  pulsar_session_pos/_tokens/
 * _common_prefix describe only the live bank, so server routing and /metrics
 * must use these for non-active banks.  Pure host reads (no CUDA); the caller
 * must keep idle banks' carries current (bank_state_save at job end). Return
 * a 0 count / NULL for an out-of-range or never-populated bank.
 *
 * L334: bank_pos is none() for a bank that holds no history -- never populated, or its view invalidated (the two
 * are one engine state: no valid checkpoint) -- and position 0 for one whose valid history is empty.  A bank outside
 * the pool is a caller's bug: refused by name (logged once) with none(). */
/** @param s the session
 *  @param bank the bank
 *  @return its committed length, or none() */
TokPos pulsar_session_bank_pos(pulsar_session *s, uint32_t bank);
/** L264: the end of the last prompt the bank PREFILLED (its generated tokens lie
 *  past it).  A request that carries all of it is that conversation's next turn. */
/** @param s the session
 *  @param bank the bank
 *  @return the prefill frontier */
TokPos pulsar_session_bank_prefill_frontier(pulsar_session *s, uint32_t bank);
/** Bank `bank`'s committed token history (the per-bank readers above).
 *  @param s the session
 *  @param bank the bank
 *  @return the session-owned history, or NULL (out of range, never populated, no bank pool) */
const pulsar_tokens *pulsar_session_bank_tokens(pulsar_session *s, uint32_t bank);
/** The tokens bank `bank`'s history and `prompt` share from position 0, by id.
 *  @param s the session
 *  @param bank the bank
 *  @param prompt the prompt
 *  @return the shared length (0 for a bad bank) */
TokCount pulsar_session_bank_common_prefix(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt);
/** True when `bank`'s compressor state is stale: a rewind into a group the
 * verify saves and the projection ring do not cover (on the pair: a 1-token ghost
 * rewind from a ratio-128 group boundary).  Such a bank can only be extended from
 * a group boundary, so a continuation must take the classic sync, whose resume
 * starts at a resume-grid point (pulsar_session_resume_grid), not a fused round at the bank's frontier.
 * Pure host read; false for an out-of-range bank. */
/** @param s the session
 *  @param bank the bank
 *  @return true when stale */
bool pulsar_session_bank_comp_stale(pulsar_session *s, uint32_t bank);
/** Tier-2: reconcile the host checkpoint after a run of pulsar_session_decode_multiseq
 * steps advanced the LIVE (just bank_state_restore'd) bank's device KV frontier
 * without touching the host token history. Append the tokens multiseq committed,
 * in order, so pulsar_session_pos/_tokens/_common_prefix and any subsequent classic
 * eval/sync/store see the true frontier. Pure host bookkeeping — no eval, no
 * CUDA. The device per-bank counters are already correct (the multiseq driver
 * maintained them and bank_state_restore installed them); this only catches the
 * host checkpoint up to them. */
/** @param s the session
 *  @param toks the committed tokens, in order
 *  @param n their count */
void pulsar_session_note_committed_tokens(pulsar_session *s, const int *toks, int n);
/** L260 fusion: record a prompt chunk that a fused step (pulsar_session_decode_fused)
 * prefilled for the INSTALLED bank -- what a classic sync of the same tokens leaves
 * behind: the tokens join the checkpoint, the prefill frontier moves to its new
 * end (L195: the rows were prefilled, so a rewind can rebuild from them), and on
 * the chunk that finishes the prompt the session's next-token logits become its
 * headed last row.  `head` names that row: the head-th headed run of the LAST
 * fused step (0 = the first run with head_last set), read from the logits block
 * that step returned, which must not have been reused since; -1 on an
 * intermediate chunk.  An index rather than a pointer so a pair's worker can
 * name the same row of its own step's block.  0 or 1. */
/** @param s the session
 *  @param toks the chunk's tokens
 *  @param n their count
 *  @param head the headed run's index, or -1
 *  @return 0, or 1 on a refusal */
int pulsar_session_note_prefilled(pulsar_session *s, const int *toks, int n, int head);
/** One speculative round (or a plain step when the drafter is quenched or absent); returns the tokens emitted.
 *  A round -- here, in the round API and in the batched phases -- stops at the family's whole stop set
 *  (pulsar_token_is_stop): nothing is committed past a stop (L284: the callers' single eos id is gone). */
/** @param s the session
 *  @param temperature as pulsar_row_draw
 *  @param top_k as pulsar_row_draw
 *  @param top_p as pulsar_row_draw
 *  @param min_p as pulsar_row_draw
 *  @param rng the request's stream (stepped per token)
 *  @param max_tokens tokens left in the request
 *  @param accepted receives the emitted tokens
 *  @param accepted_cap slots at accepted
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return tokens emitted (0 on bad arguments), or -1 */
int pulsar_session_generate_speculative(pulsar_session *s, float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen);
/** Greedy speculative block: evaluate `first_token`, then one draft/verify round (a plain eval without a drafter
 *  or when quenched); the caller supplies the base token.
 *  @param s the session
 *  @param first_token the token to evaluate first
 *  @param max_tokens tokens left in the request
 *  @param accepted receives the emitted tokens
 *  @param accepted_cap slots at accepted
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return tokens emitted (0 on bad arguments), or -1 */
int pulsar_session_eval_speculative_block(pulsar_session *s, int first_token,
                                        int max_tokens,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
/** Forget the installed bank's history: no valid checkpoint, stale logits, no image records, no pending drafts or
 *  read-ahead; the next sync prefills from scratch (mirrored on a TP group).
 *  @param s the session */
void pulsar_session_invalidate(pulsar_session *s);
/** Trim the committed history back to pos, keeping the KV below it live. For
 * dropping trailing tokens the client never saw (mid-block speculative stop);
 * rewound rows are overwritten by the next prefill/eval. */
/** @param s the session
 *  @param pos the new frontier (<= the current one) */
void pulsar_session_rewind(pulsar_session *s, TokPos pos);
/** L264 grid checkpoints.  A prefill captures the installed bank's state at each
 * grid point it ends a chunk on (every multiple of 128 a chunk boundary lands on,
 * and the last one below the prompt's end); restoring one puts the bank back at
 * that position so the next sync evaluates only the tokens after it -- the cold
 * prefill's computation, byte for byte.  best: the deepest checkpoint <= limit
 * the installed bank holds, position 0 when none (no checkpoint sits at 0, and resuming at 0 is prefilling
 * everything: 0 is the answer, not none()).  restore: 0 on success; 1 when the bank
 * holds none at G (nothing changed), or on a tensor-parallel engine, which does
 * not mirror the restore. */
/** @param s the session
 *  @param limit the deepest position allowed
 *  @return the checkpoint position, or 0 */
TokPos pulsar_session_checkpoint_best(pulsar_session *s, TokPos limit);
/** L334: where a bank's state serves a request from, in both currencies: `at` where the sync's computed rows start
 *  on the prompt it runs (the stitch: the live side -- a checkpoint, a record, or the bank's frontier), `served` how
 *  many of the REQUEST's tokens that state holds (pulsar_tokens_prompt_held_with of `at`: the prompt side, what a
 *  cache read counts and what depths are ranked by).  They differ only where a re-spelled stretch lies below `at`. */
typedef struct {
    TokPos at;           ///< the sync starts computing here, on the stitched prompt
    PromptToks served;   ///< the request's tokens the state at `at` holds
} pulsar_bank_resume;
/** L266: where a sync of `prompt` on `bank` starts its prefill -- the prompt tokens the bank's state serves,
 *  and (L325) the router's score for the bank, so the route ranks banks by the rule their syncs resume by,
 *  every family by one rule (L284; DeepSeek answered -1 before and its callers trusted the byte match).  The
 *  sync runs on the token-seam stitch (pulsar_session_bank_prefix_match): an extension of a history the bank
 *  continues where it stands (prefilled whole, at a cut the model's prefill reproduces, its state live) serves
 *  the whole match; anything else serves its deepest grid checkpoint within the match, the prefill-only history
 *  and the stitched prompt's length - 1, else nothing (0).  L334: both currencies (pulsar_bank_resume); the route's
 *  score and the cache read are `served`. */
/** @param s the session
 *  @param bank the bank
 *  @param prompt the request
 *  @return where the sync starts and what it serves */
pulsar_bank_resume pulsar_session_bank_resume_at(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt);
/** Restore the installed bank to its grid checkpoint at `G` (the L264 notes above).
 *  @param s the session
 *  @param G the checkpoint position (> 0)
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return 0, or 1 (no checkpoint at G, a family that restores only inside sync, a TP engine) */
int pulsar_session_restore_checkpoint(pulsar_session *s, TokPos G, char *err, size_t errlen);
/** L313: bank-independent checkpoints -- the INSTALLED bank adopts ANOTHER bank's published state for `prompt`: the
 *  deepest of `src`'s grid checkpoints and records the prompt carries (the resume rule's terms, over src's prefill-only
 *  history), copied in -- src's pool rows below T, then its checkpoint -- so the installed bank stands at T with src's
 *  tokens to T as its view (and, at a prompt end that IS the prompt, its logits); the sync that follows prefills the
 *  rest, the cold prefill's bytes.  src may be busy (its rows below T never change while its history reaches T).
 *  L322: "carries" is the token-seam match (pulsar_tokens_prefix_match, the route's and the own resume's), T a
 *  boundary both spellings share; the sync re-stitches the prompt onto src's tokens there.
 *  A text history only.  @return the outcome (pulsar_bank_adopt_result). */
typedef enum {
    PULSAR_ADOPT_ADOPTED = 0,   /* the installed bank stands at `at` (> 0) on src's tokens */
    PULSAR_ADOPT_NONE = 1,      /* src serves nothing for the prompt (or holds an image block below it): nothing moved */
    PULSAR_ADOPT_REFUSED = 2,   /* refused, `err` says why: no session or prompt, a tensor-parallel engine, not another
                                 * bank (nothing moved), or a copy that failed (the installed bank then describes no
                                 * position and must be synced from nothing) */
} pulsar_adopt_status;
/** L334: pulsar_session_bank_adopt's outcome and its position, apart. */
typedef struct {
    pulsar_adopt_status status;   ///< how the adopt ended
    TokPos at;                    ///< ADOPTED: T; NONE: position 0; REFUSED: none()
} pulsar_bank_adopt_result;
/** The INSTALLED bank adopts bank `src`'s published state for `prompt` (the L313 note above pulsar_adopt_status).
 *  @param s the session
 *  @param src the bank to copy from (not the installed one)
 *  @param prompt the request
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return the outcome and position */
pulsar_bank_adopt_result pulsar_session_bank_adopt(pulsar_session *s, uint32_t src, const pulsar_tokens *prompt,
                                                   char *err, size_t errlen);
/** L313: the T pulsar_session_bank_adopt would adopt from `src` for `prompt` (a host scan, no device work; 0 for the
 *  installed bank, none, or a tensor-parallel engine).  L334: `at` = T, a position on src's history (the live side),
 *  `served` the request's tokens src's history to T holds, the depth an adopt is ranked by. */
/** @param s the session
 *  @param src the bank
 *  @param prompt the request
 *  @return T and what it serves */
pulsar_bank_resume pulsar_session_bank_adopt_score(pulsar_session *s, uint32_t src, const pulsar_tokens *prompt);
/** L334: the request's tokens bank `bank`'s history up to `at` holds (pulsar_tokens_prompt_held_with over it; 0 for an
 *  out-of-range or never-populated bank) -- after an adopt, what its T serves. */
/** @param s the session
 *  @param bank the bank
 *  @param prompt the request
 *  @param at the live position
 *  @return the prompt tokens held */
PromptToks pulsar_session_bank_prompt_held(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, TokPos at);
/** The installed bank's position: the length of its committed history, where its next token goes. */
/** @param s the session
 *  @return the position */
TokPos pulsar_session_pos(pulsar_session *s);
/** The context the session holds: every position is below it. */
/** @param s the session
 *  @return the context, tokens */
TokCount pulsar_session_ctx(pulsar_session *s);
/** The widest prefill chunk, tokens (0 without a session). */
/** @param s the session
 *  @return the chunk width, tokens */
TokCount pulsar_session_prefill_cap(pulsar_session *s);
/** Multi-session serving: prefill quantum policy. A server that time-slices
 * prefill interrupts pulsar_session_sync() at chunk boundaries (via the cancel
 * callback) and re-issues the sync to resume. Returns the minimum remaining
 * suffix (target minus checkpoint, in tokens) that must still be pending for
 * an interruption to reproduce the uninterrupted chunk boundaries — hence
 * bit-identical KV state — or 0 when interrupting this session is never
 * exact and sync must run to completion. */
/** @param s the session
 *  @return the minimum suffix, tokens, or 0 */
uint32_t pulsar_session_prefill_quantum_min_suffix(const pulsar_session *s);
/** The routed experts' quantization width, bits (the family's answer; part of the KV store identity).
 *  @param e the engine, or NULL
 *  @return bits, 0 without an engine */
int pulsar_engine_routed_quant_bits(pulsar_engine *e);
/** Does speculation run on this engine: a drafter is loaded behind the round API AND the loaded family
 *  provides the verify hooks (L272 P1).  The server's spec-batched lane and the classic
 *  pulsar_session_generate_speculative key on this; which drafter it is, pulsar_engine_drafter says. */
/** @param e the engine
 *  @return true when speculation runs */
bool pulsar_engine_has_spec_rounds(const pulsar_engine *e);
/** Does the engine run the fused step (pulsar_session_decode_fused: prompt chunks riding a decode round, each
 *  recorded in its bank's history by pulsar_session_note_prefilled)?  The family's decode_fused op is present
 *  (DeepSeek; Qwen since L284 #2).  pulsar_session_decode_fused refuses on its negation (L282: the server once
 *  armed fusion on "a bank pool exists", and two concurrent Qwen requests failed "no fused step"). */
/** @param e the engine
 *  @return true when the fused step runs */
bool pulsar_engine_has_fused_step(const pulsar_engine *e);
/** The most logits rows one fused step heads: its verify rows plus its headed prompt runs (the family's cap; 0
 *  without a fused step).  L284: the ONE authority for the rows a speculative verify step carries -- the server's
 *  spec lane takes up to this many decoders (each keeps its base row; past it they ride the plain batched lane)
 *  and its allocator rations the drafts within this less the round's finishing chunks. */
/** @param e the engine
 *  @return the row cap, or 0 */
uint32_t pulsar_engine_fused_heads_max(const pulsar_engine *e);
/** Does pulsar_session_decode_mixed carry prefill runs beside its decode rows (the plain batched lane's mixed
 *  quantum)?  A family without it refuses such a step by name; its prompts ride only the fused step. */
/** @param e the engine
 *  @return true when the mixed step carries prefill runs */
bool pulsar_engine_has_mixed_prefill(const pulsar_engine *e);
/** Can the loaded family rewind a live session to an earlier position (pulsar_session_rewind)?  A
 *  recurrent family cannot; a caller with committed tokens the client never saw invalidates instead. */
/** @param e the engine
 *  @return true when the family rewinds */
bool pulsar_engine_can_rewind(const pulsar_engine *e);
/** Does the loaded family save and load whole-session snapshots and disk-KV payloads
 * (pulsar_session_save_snapshot / _load_snapshot)?  A caller that needs a session back at a known state
 * without one re-syncs the tokens instead. */
/** @param e the engine
 *  @return true when snapshots and payloads are supported */
bool pulsar_engine_has_snapshots(const pulsar_engine *e);
/** Does the OPENED artifact serve images (its family has an image front and the artifact a bound, layout-validated
 *  vision tower)?  An image request on an engine without one is refused by name. */
/** @param e the engine
 *  @return true when images are served */
bool pulsar_engine_has_vision(const pulsar_engine *e);
/** The speculative drafter the OPENED engine actually carries -- the family's answer
 *  (pulsar_family::drafter): DSPARK when DeepSeek's dspark.* drafter loaded, MTP when a Qwen artifact
 *  carries the mtp.* layer (the sidecar shard; served by the family's own
 *  pulsar_session_generate_speculative), NONE otherwise. */
/** @param e the engine
 *  @return the drafter kind */
pulsar_drafter_kind pulsar_engine_drafter(pulsar_engine *e);
/** The depth a fresh session drafts at (--spec-depth, else the drafter's), 0 = no drafter. */
/** @param e the engine
 *  @return the depth, or 0 */
int pulsar_engine_spec_depth(pulsar_engine *e);
/** The two batched decode lanes a server prices against each other (L284 lane cost): the plain lane steps every
 *  decoder one token a forward; the spec lane runs one speculative round a forward (base + verify rows). */
typedef enum { PULSAR_LANE_PLAIN = 0, PULSAR_LANE_SPEC = 1, PULSAR_LANE_COUNT = 2 } pulsar_decode_lane;
/** L263 / L284: a decode lane's step cost, MEASURED -- one fit structure for both lanes.  The spec lane's is the
 * one authority the yield quench and the server's overflow K-allocator price rows against; both lanes' are what
 * the server's lane choice weighs (server_sched.cpp lane_price_pick).  The server reports every decode step it
 * drives (the rows the forward carried, the step's wall time; a spec round's redraft included) and the engine
 * keeps, per lane, one exponentially weighted least-squares fit step_ms = flat + row * rows.  Nothing is compiled
 * in: a kernel landing, a different quantization, a second rank or a deeper context moves the fit, not a table.
 * `valid` is false until the fit has evidence (enough steps over spread row counts, positive terms); until then
 * the quench stays disarmed and the allocator admits by the row cap alone -- no number, no decision.  The terms
 * are held in microseconds so a TP group's ranks compute the same guard from the same integers (the leader's spec
 * fit crosses the wire with each round's end).  mean_rows / mean_ms are the fit's weighted centre: the cost AT the
 * centre needs no slope (pulsar_lane_cost_ms). */
typedef struct {
    int32_t flat_us;    ///< the step's fixed cost (drafting, launch, the base row), us
    int32_t row_us;     ///< the marginal row, us
    uint32_t n;         ///< steps observed
    bool valid;         ///< the terms may be used
    float mean_rows;    ///< weighted mean rows of the observed steps (0: none)
    float mean_ms;      ///< weighted mean wall of the observed steps
} pulsar_lane_cost;
/** One decode step's observation on `lane`: `rows` the forward carried, `ms` its wall. */
/** @param e the engine
 *  @param lane the lane
 *  @param rows rows the forward carried
 *  @param ms the step's wall, ms */
void pulsar_engine_lane_cost_observe(pulsar_engine *e, pulsar_decode_lane lane, uint32_t rows, double ms);
/** A lane's current step-cost fit (above).
 *  @param e the engine
 *  @param lane the lane
 *  @return the fit; all zero (invalid) without an engine or for a bad lane */
pulsar_lane_cost pulsar_engine_lane_cost(const pulsar_engine *e, pulsar_decode_lane lane);
/** The lane's predicted step wall at `rows`, ms: flat + row * rows when the fit is valid; else the observed mean
 *  when `rows` sits at the fit's centre (within half a row, 16+ steps: the slope does not enter); else 0 -- no
 *  number. */
/** @param c the fit
 *  @param rows the step's rows
 *  @return ms, or 0 */
double pulsar_lane_cost_ms(const pulsar_lane_cost *c, double rows);
/** L301: the spec lane's verify cost curve (src/engine/verify_curve.cpp has the rule and its rationale).  A round
 *  verifies n_seqs base rows and the T drafts that maximise (n_seqs + the top-T survivals) / round_ms(n_seqs + T),
 *  round_ms(n, r) = base[r] + slope * r + host(n) + draft(n): base the per-row-count verify forward PROFILED at
 *  startup and cached on disk, slope the online robust correction of the forward against it, host and draft the
 *  round's measured wall outside the forward.
 *
 *  pulsar_session_verify_curve_open installs the curve: loaded from `cache_dir` (a file named by the key: the
 *  container's digest, the drafter, the widest step, the session's bank pool and context, `build_id`) or profiled
 *  on the session's INSTALLED bank at its frontier -- every r-row verify step from the widest down, each one put
 *  back as a round's abort does -- and written there.  The installed bank must hold a prefix and room for the widest
 *  step.  On a tensor-parallel group (L315) the leader opens it: a profile runs on every rank in step
 *  (pulsar_session_verify_profile), the key names the group's width, and only the leader holds the curve.  0, or -1
 *  with `err` (no batched speculative lane: no drafter, no bank pool; or the profile failed) -- the engine then has
 *  no curve and the row budget alone admits drafts.  L327: the prompt chunk cost fit (pulsar_engine_prefill_ms) is
 *  kept beside it, under the same key: read back when the curve is (a curve profiled again starts a fresh fit),
 *  written as it moves. */
/** @param s the session
 *  @param cache_dir the cache directory
 *  @param build_id this build's id (part of the key)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0 or -1 */
int pulsar_session_verify_curve_open(pulsar_session *s, const char *cache_dir, const char *build_id, char *err,
                                     size_t errlen);
/** One decode round of the batched spec lane (the server reports every round that carried no prompt rows):
 *  `n_seqs` banks, `rows` verify rows, the forward's wall, the rest of the round outside the forward and the
 *  drafting (`host_ms`), and the drafting (`draft_ms`). */
/** @param e the engine
 *  @param n_seqs banks in the round
 *  @param rows verify rows
 *  @param forward_ms the forward's wall, ms
 *  @param host_ms the rest outside forward and drafting, ms
 *  @param draft_ms the drafting, ms */
void pulsar_engine_verify_curve_observe(pulsar_engine *e, uint32_t n_seqs, uint32_t rows, double forward_ms,
                                        double host_ms, double draft_ms);
/** The verify pick: of m drafts with survivals `surv` sorted DESCENDING, how many a round of n_seqs base rows
 *  verifies within cap_rows rows (the best ratio on the curve; no curve: as many as the rows hold).  The SERVER'S
 *  pick: on a tensor-parallel group it runs on the leader, on the leader's curve (L315), and its choice reaches the
 *  workers as each step's k_alloc. */
/** @param e the engine
 *  @param n_seqs base rows
 *  @param surv m survivals, descending
 *  @param m the draft count
 *  @param cap_rows the row cap
 *  @return drafts to verify */
uint32_t pulsar_engine_verify_pick(const pulsar_engine *e, uint32_t n_seqs, const float *surv, uint32_t m,
                                   uint32_t cap_rows);
/** L312: a spec round's predicted wall on the verify curve, round_ms(n_seqs, rows) (rows clamped to the widest
 *  profiled step), and into *outside_ms its part outside the forward (host + drafting); 0 for both with no curve. */
/** @param e the engine
 *  @param n_seqs base rows
 *  @param rows verify rows
 *  @param outside_ms receives the part outside the forward
 *  @return the predicted wall, ms */
double pulsar_engine_verify_round_ms(const pulsar_engine *e, uint32_t n_seqs, uint32_t rows, double *outside_ms);
/** The curve as /metrics shows it (valid false: none). */
typedef struct {
    /** A curve exists (profiled or loaded); false = none, the other fields zero. */
    bool valid;
    bool cached;                                  ///< loaded from the disk cache (else profiled by this process)
    uint32_t rows_max;                            ///< the widest verify step
    float base_ms[33];                            ///< [r]: the profiled r-row verify forward, ms (1..rows_max)
    bool slope_valid;                             ///< the online correction has a number
    float slope_ms;                               ///< its residual per row (applied as max(0, slope))
    float host_ms;                                ///< the round's recent wall outside the forward and drafting
    float draft_ms;                               ///< the round's recent drafting wall
    uint64_t observed;                            ///< rounds observed
    uint64_t key;                                 ///< the cache key
} pulsar_verify_curve_view;
/** Copy the verify curve for /metrics.
 *  @param e the engine, or NULL
 *  @param out receives the view; zeroed (valid false) without a curve */
void pulsar_engine_verify_curve_view(const pulsar_engine *e, pulsar_verify_curve_view *out);
/** L301: the drafters' confidence calibration, per draft position (0..15): index 0 the engine's drafter (its name:
 *  pulsar_engine_drafter_name), 1 copy drafts.  Per draft the walk reached (accepted, or the one that ended the
 *  round): its confidence, its exact acceptance label under the coupled draws (greedy: the row's argmax is the
 *  draft; sampled: the target's probability of the draft), whether it was accepted.  The sums give mean confidence
 *  vs mean label (bias), the Brier score (sq_err / n) and the mean absolute calibration error; the ten confidence
 *  bins give the ECE (sum_b |bin_conf - bin_label| / sum_b bin_n). */
typedef struct {
    /** [drafter][position]: drafts observed (those with a confidence). */
    uint64_t n[2][16];
    double conf[2][16], label[2][16], hit[2][16], abs_err[2][16], sq_err[2][16];
    /** [drafter][bin]: drafts whose confidence fell in bin b = floor(10 * conf) (1.0 in bin 9). */
    uint64_t bin_n[2][10];
    double bin_conf[2][10], bin_label[2][10];
} pulsar_spec_calibration;
/** @var pulsar_spec_calibration::conf
 *  [drafter][position]: sum of the drafts' confidences (clamped to 1). */
/** @var pulsar_spec_calibration::label
 *  [drafter][position]: sum of the drafts' acceptance labels. */
/** @var pulsar_spec_calibration::hit
 *  [drafter][position]: drafts accepted (the row drew the draft), as a sum of 1.0s. */
/** @var pulsar_spec_calibration::abs_err
 *  [drafter][position]: sum of |conf - label|. */
/** @var pulsar_spec_calibration::sq_err
 *  [drafter][position]: sum of (conf - label)^2 (Brier score = sq_err / n). */
/** @var pulsar_spec_calibration::bin_conf
 *  [drafter][bin]: sum of the bin's confidences. */
/** @var pulsar_spec_calibration::bin_label
 *  [drafter][bin]: sum of the bin's acceptance labels. */
/** Copy the drafters' calibration accumulators (zeroed without an engine).
 *  @param e the engine, or NULL
 *  @param out receives them */
void pulsar_engine_spec_calibration(const pulsar_engine *e, pulsar_spec_calibration *out);
/** L304 / L327: what a prompt chunk costs (src/engine/verify_curve.cpp has the model): ONE fit over every chunk the
 *  engine runs, solo or fused, chunk_ms = fixed[kind] + alpha * q + beta * q (c + q / 2) for q rows over c already
 *  computed.  The solo chunks are the engine's own (its prefill loop); the server reports each fused step here with
 *  its verify rows, whose forward on the verify curve (L301) is taken off the step's wall -- so a fused chunk's terms
 *  are what its prompt adds to the round (no curve: no fused observation).  pulsar_engine_fused_prompt_ms: what
 *  `n_runs` such runs add to a round's verify forward into *ms, false (and 0) until the fused terms are measured. */
/** @param e the engine
 *  @param n_runs prompt runs in the fused step
 *  @param q per run: its rows
 *  @param c per run: rows already computed before it
 *  @param verify_rows the step's verify rows
 *  @param forward_ms the step's forward wall, ms */
void pulsar_engine_fused_cost_observe(pulsar_engine *e, uint32_t n_runs, const uint32_t *q, const uint32_t *c,
                                      uint32_t verify_rows, double forward_ms);
/** What n_runs prompt runs add to a fused round's forward (the L304 / L327 fit above).
 *  @param e the engine
 *  @param n_runs the runs
 *  @param q per run: its rows
 *  @param c per run: rows already computed
 *  @param ms receives the cost, ms (0 when unmeasured)
 *  @return false until the fused terms are measured */
bool pulsar_engine_fused_prompt_ms(const pulsar_engine *e, uint32_t n_runs, const uint32_t *q, const uint32_t *c,
                                   double *ms);
/** L310 / L327: the same fit priced as a solo prefill of q tokens over the c already computed, on the prefill grid of
 *  `chunk` rows (a prefill's chunks end on its absolute multiples): each chunk pays the solo fixed term, the rows
 *  alpha * q + beta * q (c + q / 2).  false (and *ms 0) until the solo terms are measured -- the one prefill cost
 *  model, never a second (the disk tier's restore-vs-recompute reads it). */
/** @param e the engine
 *  @param q tokens to prefill
 *  @param c tokens already computed
 *  @param chunk the prefill grid, rows
 *  @param ms receives the cost, ms
 *  @return false until measured */
bool pulsar_engine_prefill_ms(const pulsar_engine *e, uint32_t q, uint32_t c, uint32_t chunk, double *ms);
/** The loaded drafter's name ("" with none). */
/** @param e the engine
 *  @return the name, static or engine-owned */
const char *pulsar_engine_drafter_name(const pulsar_engine *e);
/** The installed bank's committed history (its checkpoint).
 *  @param s the session
 *  @return the session-owned tokens (valid until the session next changes), or NULL */
const pulsar_tokens *pulsar_session_tokens(pulsar_session *s);

/** Session payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the serialized state: the kv-state payload,
 * every family's (L284).  Built from a segment's parts, read and written by the same
 * helpers: the header (the segment's layout digest extended by every pool and trailing
 * pool, the frontier slot's size and the logits width; the token count, the prefill
 * frontier, the resume grid point, the logits width), the tokens, the image section
 * (count, records), the session's draft-depth controller, the frontier's logits, the
 * resume checkpoint's slot (when there is one), the FRONTIER's slot (the state model's frontier walk at the token count), every
 * pool's and trailing pool's rows to the frontier, and the trailing digest.  A restore
 * installs the frontier exactly: the session decodes on as the saved one would, and a
 * sync that does not extend it resumes from the resume checkpoint as the saved one
 * would.
 * v2 (L284): DeepSeek's payload is this one -- its own graph format ("DSV4", v3..v15)
 * is retired and refused by name; the frontier slot is the frontier walk's
 * (DeepSeek: the grid slot plus its coff-1 lanes) and a whole-row pool carries no open
 * row.  Disk segments are unchanged.
 * v3 (L284): the draft-depth controller (depth and its counters, 4 u32) after the image
 * section, so a restored session speculates as the saved one would; v2 refuses on the
 * version.
 * v4 (L292): the 0731 NVFP4 KV rows are quantized with IEEE divides (attnpack): same layout,
 * rows written before differ in a few codes, so a resumed v3 payload would not equal a
 * fresh prefill -- v3 refuses on the version and the bank re-prefills.
 * v5 (L302): the copy drafter's measured acceptance joins the speculation record (5 u32), so a
 * restored session prices its copy rounds as the saved one does; v4 refuses on the version.
 * v6 (L308): the copy acceptance's age (tokens since its last verdict) joins it (6 u32), so a restored session
 * re-probes a stale copy estimate when the saved one would; v5 refuses on the version. */
#define PULSAR_SESSION_KV_PAYLOAD_MAGIC UINT32_C(0x3150564b) /* "KVP1" */
/** The kv-state payload's format version (the v2..v6 history above); a payload of another version is refused. */
#define PULSAR_SESSION_KV_PAYLOAD_VERSION UINT32_C(6)

/** @param s the session (a family with payloads; else 0)
 *  @return the serialized payload's size, bytes */
uint64_t pulsar_session_payload_bytes(pulsar_session *s);
/** Write the session's kv-state payload at fp's position.
 *  @param s the session
 *  @param fp an open stream
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or 1 */
int pulsar_session_save_payload(pulsar_session *s, FILE *fp, char *err, size_t errlen);
/** Read a kv-state payload of `payload_bytes` from fp and install it (refused on another version or layout).
 *  @param s the session
 *  @param fp the stream at the payload
 *  @param payload_bytes its size
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return 0, or 1 */
int pulsar_session_load_payload(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
/** Serialize the session's state into an in-memory snapshot.
 *  @param s the session
 *  @param snap receives the blob (free with pulsar_session_snapshot_free)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or 1 */
int pulsar_session_save_snapshot(pulsar_session *s, pulsar_session_snapshot *snap, char *err, size_t errlen);

/** L264 S4: a disk SEGMENT -- the tokens [G_prev, G), every append-only pool's
 * rows for those positions (DeepSeek: every kv source's compressed and index-K
 * rows), and -- when the installed bank holds it -- the grid checkpoint at G
 * (G_prev and G are grid points, G_prev may be 0, and no image block lies across
 * either).  A chain of segments from position 0 rebuilds a bank exactly at its
 * last G, which is how the server's disk cache persists a conversation: each new
 * prefill adds the segments past the chain's last one, so a write costs the new
 * tokens, not the whole history.
 *   save: the installed bank must hold the history and its rows to G; the state
 *         at G is written iff the bank holds its checkpoint there (L323).
 *   load: the installed bank must stand at G_prev (a root, G_prev 0, resets it).
 *         L323: only `last` reads the segment's state and restores it (refused
 *         when the segment carries none), so the bank stands LIVE at G and the
 *         next sync evaluates from there; an earlier segment's rows are read and
 *         its state skipped -- the bank keeps the one checkpoint at the end. */
#define PULSAR_SESSION_SEGMENT_MAGIC UINT32_C(0x31474553) /* "SEG1" */
/* v2 (L265): the header names the state layout by a digest (the model's state ops) instead of
 * DeepSeek's strides; v1 segments are refused and the chain rewrites itself. */
/** The disk segment format's version (history beside it); a segment of another version is refused. */
#define PULSAR_SESSION_SEGMENT_VERSION UINT32_C(6)   /* v3 (L281): the span's image block records; v4 (L268): with each block's 2D grid; v5 (L292): NVFP4 KV rows from IEEE divides -- v4 rows differ in a few codes, so they are refused and rewritten; v6 (L323): the state is optional and last, behind its own digest (header word 8 flags it), so a load reads rows + one state */
/** The bytes pulsar_session_save_segment writes for [G_prev, G) (0: not a segment span), and in *state_bytes (may be
 *  NULL) the trailing part a load of a non-last segment skips: the state at G and its digest (0 = rows only). */
/** @param s the session
 *  @param G_prev the span's start (a grid point)
 *  @param G its end
 *  @param state_bytes receives the state part; may be NULL
 *  @return bytes, or 0 */
uint64_t pulsar_session_segment_bytes(pulsar_session *s, TokPos G_prev, TokPos G, uint64_t *state_bytes);
/** `key`: the segment's store key (pulsar_segstore_child_key).  Off a TP group
 *  it is unused (NULL is fine); on one it names every worker's own copy, and
 *  save and load are mirrored verdicts whose disagreement is a miss on every
 *  rank (L264 S4e).  Refused inside a mirrored sync. */
/** @param s the session
 *  @param fp an open stream
 *  @param G_prev the span's start
 *  @param G its end
 *  @param key the store key (NULL off a TP group)
 *  @param err receives the failure
 *  @param errlen bytes at err
 *  @return 0, or nonzero */
int pulsar_session_save_segment(pulsar_session *s, FILE *fp, TokPos G_prev, TokPos G, const char *key,
                                char *err, size_t errlen);
/** Load one disk segment onto the installed bank (the load rule in the L264 S4 note above).
 *  @param s the session
 *  @param fp the stream at the segment
 *  @param bytes its size
 *  @param last the chain's last segment: its state is read and the bank stands live at G
 *  @param G_out receives G (0 on a refusal); may be NULL
 *  @param key the store key (NULL off a TP group)
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return 0, or nonzero */
int pulsar_session_load_segment(pulsar_session *s, FILE *fp, uint64_t bytes, bool last,
                                TokPos *G_out, const char *key, char *err, size_t errlen);
/** A store removed segment `key`: the workers delete their copies (void; a
 *  no-op off a TP leader). */
/** @param e the engine
 *  @param key the removed segment's key */
void pulsar_engine_segment_dropped(pulsar_engine *e, const char *key);
/** Bring-up: the leader's whole key set (n_keys x 40 hex); each worker removes
 *  the copies it does not name.  0 when sent or nothing to do. */
/** @param e the engine
 *  @param keys n_keys keys of 40 hex chars, back to back
 *  @param n_keys the count
 *  @return 0, or nonzero on a failure */
int pulsar_engine_segment_reconcile(pulsar_engine *e, const char *keys, int n_keys);
/** True while a TP leader runs a mirrored sync (no other mirrored operation may
 *  ship -- e.g. a disk store from the prefill's progress callback). */
/** @param s the session
 *  @return true inside a mirrored sync */
bool pulsar_session_in_mirrored_sync(const pulsar_session *s);
/** Install a snapshot made by pulsar_session_save_snapshot (refused when written by an incompatible build).
 *  @param s the session
 *  @param snap the blob
 *  @param err receives the refusal
 *  @param errlen bytes at err
 *  @return 0, or 1 */
int pulsar_session_load_snapshot(pulsar_session *s, const pulsar_session_snapshot *snap, char *err, size_t errlen);
/** Free a snapshot's buffer and zero it.
 *  @param snap the snapshot; NULL is a no-op */
void pulsar_session_snapshot_free(pulsar_session_snapshot *snap);


#endif
