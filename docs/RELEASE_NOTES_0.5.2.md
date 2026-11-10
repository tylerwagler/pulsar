# Pulsar 0.5.2 release notes

0.5.2 is 0.5.1 with the findings of a full code review fixed (L369 - L375).  Two of them changed model output on
Gemma 4; the rest are refusals where there was undefined behaviour, a crash or a silent wrong answer, and hardening of
the agent, the servers and the disk KV store.  The full list is `git log v0.5.1..v0.5.2`.

## Output changes

- **Gemma 4: window attention read an unwritten partial on short prompt steps** (L370).  A prompt step of 2 to 63
  rows past depth 1024 whose rows straddled a 256-position boundary launched one chunk too few on every window layer
  (the bounded split grid counted one row's chunks, not a tile's), and the rows at or past the boundary folded
  whatever the workspace held.  Short follow-up turns and short judge questions hit it.  Fixed; Qwen, Nemotron 3.5 and
  the encoders never took the path.
- **Gemma 4: a verify row's attention arm no longer depends on what rides in its step** (L370).  In a step that mixed
  a speculative round's verify rows with another conversation's prompt rows, the verify rows took the prompt rows'
  arm, so their bytes differed from the same rows verified alone.  The attention is now one call a row kind.
- **Stored KV is refused once** (payload v8, segment v9; no format change).  Chains written by the faulty attention
  cannot be told from sound ones, so every family's disk KV store starts empty on the first run of 0.5.2.

## Fixed: crashes and wrong answers

- **Server crash on a failed bank install** (L372).  A request whose bank could not be installed before decoding began
  (or whose output parser could not be built) reached the finish step with no parser and took the server down.  Every
  failure before decode now answers the client and frees the slot.
- **A bank left without memory after a failed readmission** kept receiving its conversation's requests, each failing
  in its first forward.  It is now re-backed empty.
- **The judge (`/v1/systemone`) ran with a chat request's prefill hooks armed** when that request was parked
  mid-prompt: the judge's progress could persist its own history under the chat request's name, and a gone chat client
  cancelled the judge.  A hook now lasts one sync.
- **A request re-sent onto a bank already at its prompt's end** was never stepped while another stream decoded.
- **The live logits row could be read before anything wrote it** (a fresh session, a bank never saved): a well-formed
  score from another bank's row.  Every reader now refuses.
- **The sampler drew from rows holding NaN or +Inf logits** as if those ids were masked.  Such a row is refused by
  name; -inf stays the mask.
- **Turing (sm_75): bf16 attention operands above the fp16 range became inf silently.**  They refuse the step.
- **EXL3 MoE: a step that failed between a fork and its join left the lane pending**; later steps joined the stale
  fork, or every fork refused for the life of the process.  A failed step abandons its lanes.
- **Vision: two unlocked grid caches** could hand one image another's grid under concurrent image requests.
- **A grow of scratch memory inside a captured step** invalidated the capture and failed that step shape for good.
  The capture is now broken off at the grow and retaken at the new size.
- **Model loading: the O_DIRECT staging read could run past its pinned buffer** on a filesystem with a large block
  size; a second model mapping could be handed to kernels as if registered.

## Fixed: the disk KV store

- A disk filled by other data no longer drains the store: eviction runs only while it can close the gap.
- A segment that cannot be read is told apart from a full RAM pool (no wait on the writer), survives a transient
  error, and is dropped after three persistent failures in a row.
- A chain's only restore point is no longer evicted while its rows stay.
- The eviction log says what a snapshot did: `snapshot=none (read-only)`, `(dead prefix)`, `(mirrored sync)`.

## Fixed: pulsar-agent

- **The workspace boundary held for existing paths only.**  `write` through a dangling symlink created the file
  outside; the edit preflight read a path before checking it.  A symlink that does not resolve is refused, `..` is
  tested a component, and every tool checks before it reads.
- **One invalid UTF-8 byte in a tool result dropped the turn** on Qwen, Nemotron and Gemma, and the model then
  re-called the tool without end.  Tool results are repaired to valid UTF-8, the bash caps cut on character
  boundaries, and a tokenizer refusal fails the turn.
- Tool-call text reaches the terminal through the sanitiser (escape sequences in arguments did not).
- Bash jobs: one teardown on every exit and on SIGTERM / SIGHUP; a job's pipe is drained and its timeout applied
  whether or not the model asks; at most 64 tracked jobs.
- `write` and `edit` are atomic (temp file, fsync, rename).
- Tool results are never dropped when a compaction fails; the open-turn state is read off the transcript; a
  compaction tail starts at a user turn.
- Arguments of the wrong type are tool errors (`timeout_sec: "10s"` used to mean an hour); `timeout_sec <= 0` is
  refused.

## Fixed: the encoder and Parse server

- Each served engine runs its forwards on one long-lived GPU thread.  Before, every connection rebuilt every GEMM
  plan, re-announced it and leaked its descriptors.

## Upgrade notes

- **Every disk KV store is wiped by the first start** (above).
- **HTTP statuses.**  A request body that is not valid UTF-8 is a 400 naming the JSON path (it was a 500 at prefill).
  An image refused because the shared decode budget is in use is a 503 with `Retry-After` (it was a 400).  The judge
  answers 503 with `Retry-After` when its prefill would cross the memory floor or the KV budget (it always ran).
- **The judge bank is releasable**: under memory pressure, after the idle chat banks, with no judge request queued.
  The next judge request prefills again.
- `pulsar:requests_finished_total` and the request histograms no longer count failed or abandoned turns.
- **Speculation under concurrency.**  The quench guard priced every batched round as one stream; it now sees the
  round's stream count, so with two or more decoders a request whose drafts do not pay is quenched sooner.  Tokens are
  the target's either way.
- `PULSAR_SPEC_TAU` / `PULSAR_SPEC_DEPTH`: an unparsable value refuses the load; `PULSAR_SPEC_TAU=0` disables the stop,
  as `--spec-tau 0` does (it read as unset).
- pulsar-agent refuses to start without a terminal unless `--non-interactive` is given, and refuses a `-sys` text
  that is not UTF-8.
- DeepSeek (the fixture): its tool-result turns are marked as Qwen's and Nemotron's are, so routing anchors no
  longer take a tool result for the user's last turn; its tokenizer's merge is O(n log n), token-identical.

## API (`src/pulsar.h`)

- The tokenizer and chat entries report a refusal instead of returning true over nothing:
  `pulsar_tokenize_text`, `pulsar_tokenize_rendered_chat`, `pulsar_tokenize_rendered_chat_spans` (which gains
  `err, errlen`), `pulsar_chat_open`, `pulsar_chat_open_tools`, `pulsar_chat_append_turn`,
  `pulsar_chat_end_assistant`.
- **No chat entry ends the process.**  A think mode or role a family's template cannot express was an exit; it is a
  refusal now (false, nothing appended, one line ending "-- refusing"), on all five families.  pulsar-agent fails the
  turn; the CLI's REPL keeps the conversation.
- `pulsar_session_sync_mm` can return `PULSAR_SESSION_SYNC_BUSY`.
- `PULSAR_IMAGE_BLOCKS_MAX` is public.  `PULSAR_SESSION_KV_PAYLOAD_VERSION` 8, `PULSAR_SESSION_SEGMENT_VERSION` 9.
- The speculative round entries refuse by name on an engine with no drafter, at the context's end, and on a round
  ended with fewer rows than it began.

## Tests

- The state contract runs on every hosted family in the release tier: `session-contract`, `bank-isolation`,
  `block-spec`, `scratch-poison`, `bank-adopt` and `graph-eager` on Gemma 4's pack (it had its reference gates only);
  `session-contract` and `bank-isolation` on Nemotron's.  On 0.5.1's tree the Gemma contract fails at F1 / F5.
- `gemma4-jev-gate` joins the release tier.
- `gqa_attn_gate`: Gemma's window layer at its real 1024-key window, bounded against unbounded, a tile across a chunk
  boundary (fails on 0.5.1 at position 1280); an out-of-range operand on the widening arm.
- `cuda-sampler-device-gate`: rows with NaN / +Inf logits refuse; their neighbours draw the reference's bytes.
- `kernel-contract-gate` case 13: a scratch, activation-cache and expert-table grow inside a capture.
- New live leg `tests/encode_gpu_thread_live.py` (100 requests a model on separate connections: no plan rebuilt).
- Host tests for each server, agent, store and engine fix, in `unit-test-gate` and `agent-test-gate`.

## Known limits

- The tensor-parallel paths these fixes touch (the mirrored sync's busy result, the digest readers, the speculative
  block entry's refusal) are compiled and unit-tested only: no pair is in the release run.
- On sm_75 an attention operand too small for fp16 still rounds or flushes to zero, as before; only overflow and
  non-finite operands refuse.
- A readmission that fails while the pool is parked on the pool's only resident bank leaves that bank parked; no step
  runs on it (refused by name).
