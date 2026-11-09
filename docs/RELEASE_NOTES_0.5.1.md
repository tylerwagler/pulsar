# Pulsar 0.5.1 release notes

0.5.1 is 0.5.0 with pulsar-agent's tool loop working on every served family (L365).  The full list is
`git log v0.5.0..v0.5.1`.

## Fixed

- **pulsar-agent calls tools on Qwen, Nemotron 3.5 and Gemma 4** (L365).  The agent spoke DeepSeek's DSML only: on any
  other family the model's calls were refused as malformed DSML and no tool ever ran (0.5.0's known issue).  The agent
  now takes the server's path on every family but DeepSeek: its head declares the tools in the family's own template,
  and each round runs the family's output parser over the sampled text.  The parser's calls run as the agent's tool
  calls, one named result per call; a malformed call is answered with "Tool error: invalid tool call" and the model
  retries.  A call's structure is decoded greedily and its string arguments sampled, as the server does.
- **Gemma 4 answers calls inside the model turn that made them.**  The agent leaves that turn open, and the responses
  follow the calls in the same turn, as Gemma's template writes them.  A user turn, a compaction request or an earlier
  failure closes an open turn first.
- DeepSeek keeps DSML.  Its tool prompt is byte-identical to 0.5.0's.

## Upgrade notes

- Each tool result is now its own tool message, named after its call, on every family but DeepSeek.  In 0.5.0 one
  combined message answered all of a turn's calls.
- pulsar-agent refuses Inkling by name (the family is parked).

## API (`src/pulsar.h`)

- `pulsar_chat_open_tools(e, tokens, trusted, system, tools_json, think_mode)`: the head also declares an OpenAI
  tools array in the family's template.  DeepSeek refuses it: its tools are DSML text passed as `trusted`.
- `pulsar_chat_message.name`: a tool message's function.
- `pulsar_chat_calls_continue_turn(e)`: whether the family answers calls inside the calling turn (Gemma 4).
- `pulsar_chat_opens_reasoning(e, tokens, think_mode)`: whether the transcript leaves the turn inside its reasoning,
  which is where the output parser starts.
- The lib's `qwen_` / `nemotron_` / `gemma4_chat_render_head` take a `tools_json`; new `gemma4_chat_render_tool_turn`.

## Tests

- New release test `agent-tools-gate` (`tests/agent_tools_gate.py`): pulsar-agent, headless, on Qwen, Nemotron and
  Gemma 4.  It must read a file, then write the file's code word to `answer.txt`, through the family's own call
  syntax.
- The Qwen, Nemotron and Gemma chat gates compose a head with tools plus turns and compare it with the full render
  with tools.  Gemma's also composes the tool loop turn by turn (C2).
- `agent-test-gate` adds the family test: the tools array, both prompt variants, the call conversion, and whole Qwen
  and Gemma turns through their parsers.
