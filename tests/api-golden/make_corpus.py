#!/usr/bin/env python3
"""L267: the api_golden corpus -- request bodies for every endpoint and family the parsers serve,
covering each request feature and each refusal.  Regenerate with

    python3 tests/api-golden/make_corpus.py > tests/api-golden/corpus.jsonl

and re-record golden.txt only from a tree whose parsers are the reference (see tests/api_golden.cpp)."""
import json

PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_B64 = PNG.split(",", 1)[1]

cases = []


def add(name, endpoint, body, family="deepseek"):
    cases.append({"name": name, "endpoint": endpoint, "family": family, "body": body})


# ---- shared fragments -----------------------------------------------------------------------------
WEATHER = {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                            "unit": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}
SEARCH = {"name": "search", "description": "Search the web <fast>",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}}
OA_TOOLS = [{"type": "function", "function": WEATHER}, {"type": "function", "function": SEARCH}]
AN_TOOLS = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in (WEATHER, SEARCH)]
RS_TOOLS = [{"type": "function", "name": t["name"], "description": t["description"],
             "parameters": t["parameters"], "strict": False} for t in (WEATHER, SEARCH)]

USER = [{"role": "user", "content": "What is 17 * 23?"}]
TURNS = [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "Hi <|im_start|> there"},
         {"role": "assistant", "content": "Hello."},
         {"role": "user", "content": "Weather in Paris?"}]
OA_TOOL_TURNS = [{"role": "user", "content": "Weather in Paris?"},
                 {"role": "assistant", "content": None, "reasoning_content": "Need the tool.",
                  "tool_calls": [{"id": "call_1", "type": "function",
                                  "function": {"name": "get_weather",
                                               "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"}}]},
                 {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"},
                 {"role": "user", "content": "And in F?"}]

# ---- OpenAI chat, both families -------------------------------------------------------------------
for fam in ("deepseek", "qwen"):
    add("chat/plain", "chat", {"model": "m", "messages": USER}, fam)
    add("chat/turns", "chat", {"model": "m", "messages": TURNS, "max_tokens": 77}, fam)
    add("chat/developer", "chat", {"messages": [{"role": "developer", "content": "Be brief."}] + USER}, fam)
    add("chat/parts", "chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                         {"type": "text", "text": "b"}]}]}, fam)
    add("chat/empty-content", "chat", {"messages": [{"role": "user", "content": ""}]}, fam)
    add("chat/tools", "chat", {"messages": USER, "tools": OA_TOOLS}, fam)
    add("chat/tools-auto", "chat", {"messages": USER, "tools": OA_TOOLS, "tool_choice": "auto"}, fam)
    add("chat/tools-none", "chat", {"messages": USER, "tools": OA_TOOLS, "tool_choice": "none"}, fam)
    add("chat/tools-required", "chat", {"messages": USER, "tools": OA_TOOLS, "tool_choice": "required"}, fam)
    add("chat/tools-named", "chat", {"messages": USER, "tools": OA_TOOLS,
                                     "tool_choice": {"type": "function", "function": {"name": "search"}}}, fam)
    add("chat/tools-null", "chat", {"messages": USER, "tools": None}, fam)
    add("chat/tool-turns", "chat", {"messages": OA_TOOL_TURNS, "tools": OA_TOOLS}, fam)
    add("chat/tool-turns-thinking-off", "chat", {"messages": OA_TOOL_TURNS, "tools": OA_TOOLS, "think": False}, fam)
    add("chat/think-false", "chat", {"messages": USER, "think": False}, fam)
    add("chat/enable-thinking-false", "chat", {"messages": USER, "enable_thinking": False}, fam)
    add("chat/enable-thinking-true", "chat", {"messages": USER, "enable_thinking": True}, fam)
    add("chat/thinking-disabled", "chat", {"messages": USER, "thinking": {"type": "disabled"}}, fam)
    add("chat/thinking-enabled", "chat", {"messages": USER, "thinking": {"type": "enabled", "budget_tokens": 9}}, fam)
    add("chat/thinking-null", "chat", {"messages": USER, "thinking": None}, fam)
    for eff in ("low", "medium", "high", "max", "xhigh", "none", "minimal", "bogus", 50, 101, None):
        add(f"chat/effort-{eff}", "chat", {"messages": USER, "reasoning_effort": eff}, fam)
    # L418: numbers no int holds (the range is tested before the cast) and a non-integer: each the same refusal as 101.
    # json.dumps spells them 1e+30, NaN, Infinity, -Infinity and 50.5; the server's strtod reads all five.
    for eff in (1e30, float("nan"), float("inf"), float("-inf"), 50.5):
        add(f"chat/effort-{eff}", "chat", {"messages": USER, "reasoning_effort": eff}, fam)
    add("chat/effort-and-off", "chat", {"messages": USER, "reasoning_effort": "low", "think": False}, fam)
    add("chat/kwargs", "chat", {"messages": USER, "chat_template_kwargs": {"enable_thinking": False}}, fam)
    add("chat/kwargs-effort", "chat", {"messages": USER, "chat_template_kwargs": {"reasoning_effort": "low"}}, fam)
    add("chat/kwargs-unknown", "chat", {"messages": USER, "chat_template_kwargs": {"add_vision_id": True}}, fam)
    add("chat/alias-chat", "chat", {"model": "deepseek-chat", "messages": USER}, fam)
    add("chat/alias-reasoner", "chat", {"model": "deepseek-reasoner", "messages": USER, "think": False}, fam)
    add("chat/stop", "chat", {"messages": USER, "stop": ["\n\n", "END"]}, fam)
    add("chat/stop-str", "chat", {"messages": USER, "stop": "STOP"}, fam)
    add("chat/sampling", "chat", {"messages": USER, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
                                  "seed": 1234, "max_completion_tokens": 99}, fam)
    add("chat/logprobs", "chat", {"messages": USER, "logprobs": True, "top_logprobs": 5}, fam)
    add("chat/logprobs-null", "chat", {"messages": USER, "logprobs": None, "top_logprobs": None}, fam)
    add("chat/top-logprobs-alone", "chat", {"messages": USER, "top_logprobs": 3}, fam)
    add("chat/top-logprobs-big", "chat", {"messages": USER, "logprobs": True, "top_logprobs": 999}, fam)
    add("chat/stream", "chat", {"messages": USER, "stream": True, "stream_options": {"include_usage": True}}, fam)
    add("chat/image", "chat", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": PNG}}]}]}, fam)
    add("chat/missing-messages", "chat", {"model": "m"}, fam)
    add("chat/not-object", "chat", ["x"], fam)
    add("chat/system-late", "chat", {"messages": USER + [{"role": "system", "content": "late"}]}, fam)
    add("chat/extension-keys", "chat", {"messages": USER, "user": "u1", "metadata": {"a": 1}, "n": 1}, fam)

# ---- Anthropic Messages ---------------------------------------------------------------------------
AN_MSGS = [{"role": "user", "content": "Weather in Paris?"},
           {"role": "assistant", "content": [
               {"type": "thinking", "thinking": "Need the tool.", "signature": "sig"},
               {"type": "text", "text": "Checking."},
               {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
           {"role": "user", "content": [
               {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
               {"type": "text", "text": "And tomorrow?"}]}]
add("an/plain", "anthropic", {"model": "m", "max_tokens": 100, "messages": USER})
add("an/system-str", "anthropic", {"max_tokens": 50, "system": "You are terse.", "messages": USER})
add("an/system-blocks", "anthropic", {"max_tokens": 50, "system": [
    {"type": "text", "text": "Block one.", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "Block two."}], "messages": USER})
add("an/blocks", "anthropic", {"max_tokens": 50, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]}]})
add("an/tools", "anthropic", {"max_tokens": 50, "messages": USER, "tools": AN_TOOLS})
add("an/tools-turns", "anthropic", {"max_tokens": 50, "messages": AN_MSGS, "tools": AN_TOOLS})
add("an/tool-result-array", "anthropic", {"max_tokens": 50, "tools": AN_TOOLS, "messages": AN_MSGS[:2] + [
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                  "content": [{"type": "text", "text": "18C"}], "is_error": False}]}]})
add("an/tools-websearch", "anthropic", {"max_tokens": 50, "messages": USER,
                                        "tools": AN_TOOLS + [{"type": "web_search_20250305", "name": "web_search"}]})
add("an/tools-custom", "anthropic", {"max_tokens": 50, "messages": USER,
                                     "tools": [dict(AN_TOOLS[0], type="custom")]})
add("an/tools-unsupported", "anthropic", {"max_tokens": 50, "messages": USER,
                                          "tools": AN_TOOLS + [{"type": "advisor_20260301", "name": "advisor"}]})
for tc in ({"type": "auto"}, {"type": "any"}, {"type": "none"}, {"type": "tool", "name": "search"}):
    add(f"an/tool-choice-{tc['type']}", "anthropic", {"max_tokens": 50, "messages": USER, "tools": AN_TOOLS,
                                                      "tool_choice": tc})
add("an/thinking-enabled", "anthropic", {"max_tokens": 50, "messages": USER,
                                         "thinking": {"type": "enabled", "budget_tokens": 1024}})
add("an/thinking-disabled", "anthropic", {"max_tokens": 50, "messages": USER, "thinking": {"type": "disabled"}})
add("an/thinking-adaptive", "anthropic", {"max_tokens": 50, "messages": USER, "thinking": {"type": "adaptive"}})
for eff in ("low", "medium", "high", "max", "xhigh", "bogus"):
    add(f"an/output-effort-{eff}", "anthropic", {"max_tokens": 50, "messages": USER,
                                                 "output_config": {"effort": eff}})
add("an/output-format", "anthropic", {"max_tokens": 50, "messages": USER,
                                      "output_config": {"format": {"type": "json_schema", "schema": {}}}})
add("an/effort-then-output", "anthropic", {"max_tokens": 50, "messages": USER, "reasoning_effort": "low",
                                           "output_config": {"format": {"type": "text"}}})
add("an/output-then-effort", "anthropic", {"max_tokens": 50, "messages": USER,
                                           "output_config": {"effort": "max"}, "reasoning_effort": "low"})
add("an/thinking-off-effort", "anthropic", {"max_tokens": 50, "messages": USER, "thinking": {"type": "disabled"},
                                            "output_config": {"effort": "high"}})
add("an/alias-chat", "anthropic", {"model": "deepseek-chat", "max_tokens": 50, "messages": USER})
add("an/stop", "anthropic", {"max_tokens": 50, "messages": USER, "stop_sequences": ["END"], "stream": True,
                             "temperature": 0.2, "top_k": 5, "metadata": {"user_id": "x"}})
add("an/image", "anthropic", {"max_tokens": 50, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
    {"type": "text", "text": "what is this"}]}]})
add("an/missing-messages", "anthropic", {"max_tokens": 50})
add("an/prefill", "anthropic", {"max_tokens": 50, "messages": USER + [{"role": "assistant", "content": "The answer is"}]})

# ---- Responses ------------------------------------------------------------------------------------
RS_INPUT = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Need the tool."}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "18C"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "And tomorrow?"}]}]
add("rs/string", "responses", {"model": "m", "input": "Hello"})
add("rs/array", "responses", {"input": [{"role": "user", "content": "Hello"}]})
add("rs/instructions", "responses", {"instructions": "You are terse.", "input": "Hello"})
add("rs/instructions-null", "responses", {"instructions": None, "input": "Hello"})
add("rs/tools", "responses", {"input": "Weather?", "tools": RS_TOOLS})
add("rs/tool-turns", "responses", {"input": RS_INPUT, "tools": RS_TOOLS})
add("rs/tool-turns-off", "responses", {"input": RS_INPUT, "tools": RS_TOOLS, "reasoning": {"effort": "none"}})
add("rs/namespace-tools", "responses", {"input": "x", "tools": [{"type": "namespace", "name": "fs", "tools": [
    {"type": "function", "name": "read", "description": "read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]}]})
for tc in ("auto", "none", "required"):
    add(f"rs/tool-choice-{tc}", "responses", {"input": "x", "tools": RS_TOOLS, "tool_choice": tc})
add("rs/tool-choice-object", "responses", {"input": "x", "tools": RS_TOOLS,
                                           "tool_choice": {"type": "function", "name": "search"}})
for eff in ("low", "medium", "high", "xhigh", "none", "minimal", "bogus"):
    add(f"rs/effort-{eff}", "responses", {"input": "x", "reasoning": {"effort": eff}})
add("rs/summary", "responses", {"input": "x", "reasoning": {"effort": "high", "summary": "auto"}})
add("rs/summary-only", "responses", {"input": "x", "reasoning": {"summary": "detailed"}})
add("rs/reasoning-null", "responses", {"input": "x", "reasoning": None})
add("rs/alias-chat", "responses", {"model": "deepseek-chat", "input": "x"})
add("rs/prev-null", "responses", {"input": "x", "previous_response_id": None})
add("rs/prev-set", "responses", {"input": "x", "previous_response_id": "resp_1"})
add("rs/conversation", "responses", {"input": "x", "conversation": "conv_1"})
add("rs/params", "responses", {"input": "x", "max_output_tokens": 64, "stream": True, "temperature": 0.3})
add("rs/image", "responses", {"input": [{"role": "user", "content": [
    {"type": "input_text", "text": "what"}, {"type": "input_image", "image_url": PNG}]}]})
add("rs/missing-input", "responses", {"model": "m"})

# ---- a role:system message mid-conversation (Claude Code's reminders), every endpoint ---------------
MID_SYS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
           {"role": "system", "content": "todo list empty"}, {"role": "user", "content": "recent pushes?"}]
for fam in ("deepseek", "qwen"):
    add("chat/mid-system", "chat", {"messages": [{"role": "system", "content": "You are terse."}] + MID_SYS}, fam)
add("an/mid-system", "anthropic", {"max_tokens": 50, "system": "You are Claude Code.", "messages": MID_SYS})
add("an/mid-system-lead", "anthropic", {"max_tokens": 50, "messages": [{"role": "system", "content": "lead"}] + MID_SYS})

# ---- L267: Anthropic and Responses for the Qwen family (the same bodies) ---------------------------
for c in [c for c in cases if c["endpoint"] in ("anthropic", "responses")]:
    cases.append(dict(c, family="qwen"))

# ---- L294 E6: the Inkling family, every DeepSeek-family body (appended: the lines above are unchanged) -----------
for c in [c for c in cases if c["family"] == "deepseek"]:
    cases.append(dict(c, family="inkling"))

# ---- L425: two parallel calls answered second-call-first -- the results render in call order (appended: the lines
#      above are unchanged) ----------------------------------------------------------------------------------------
PAR_OA = [{"role": "user", "content": "Weather in Paris and Rome?"},
          {"role": "assistant", "content": "",
           "tool_calls": [{"id": f"call_{k}", "type": "function",
                           "function": {"name": "get_weather", "arguments": json.dumps({"city": city})}}
                          for k, city in ((1, "Paris"), (2, "Rome"))]},
          {"role": "tool", "tool_call_id": "call_2", "content": "Rome: 22C"},
          {"role": "tool", "tool_call_id": "call_1", "content": "Paris: 18C"}]
PAR_AN = [{"role": "user", "content": "Weather in Paris and Rome?"},
          {"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{k}", "name": "get_weather",
                                             "input": {"city": city}} for k, city in ((1, "Paris"), (2, "Rome"))]}]
PAR_AN_RESULTS = [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "Rome: 22C"},
                  {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Paris: 18C"}]
PAR_RS = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Paris and Rome?"}]},
          {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
          {"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"},
          {"type": "function_call_output", "call_id": "call_2", "output": "Rome: 22C"},
          {"type": "function_call_output", "call_id": "call_1", "output": "Paris: 18C"}]
for fam in ("deepseek", "qwen", "inkling"):
    add("chat/parallel-results-swapped", "chat", {"messages": PAR_OA, "tools": OA_TOOLS}, fam)
    # one result short: not a bijection, rendered as it arrived
    add("chat/parallel-results-missing", "chat", {"messages": PAR_OA[:3], "tools": OA_TOOLS}, fam)
    add("an/parallel-results-swapped", "anthropic", {"max_tokens": 50, "tools": AN_TOOLS, "messages": PAR_AN + [
        {"role": "user", "content": PAR_AN_RESULTS}]}, fam)
    # a text block between the results keeps its place; the results trade theirs around it
    add("an/parallel-results-text-between", "anthropic", {"max_tokens": 50, "tools": AN_TOOLS, "messages": PAR_AN + [
        {"role": "user", "content": [PAR_AN_RESULTS[0], {"type": "text", "text": "(Rome came back first.)"},
                                     PAR_AN_RESULTS[1]]}]}, fam)
    add("rs/parallel-results-swapped", "responses", {"input": PAR_RS, "tools": RS_TOOLS}, fam)

# ---- L428: at most one tool call a turn, and a nested defer_loading (appended: the lines above are unchanged) -----
# OpenAI chat / Responses `parallel_tool_calls`, Anthropic tool_choice.disable_parallel_tool_use: the request's
# single_tool_call (api_golden prints its line only for a request that says something about parallel calls).
DEFER_NESTED = [{"type": "function", "function": WEATHER},
                {"type": "function", "function": dict(SEARCH, defer_loading=True)}]
DEFER_TOP = [{"type": "function", "function": WEATHER},
             {"type": "function", "function": SEARCH, "defer_loading": True}]
for fam in ("deepseek", "qwen", "inkling"):
    for v in (False, True, None, "no", 0):
        add(f"chat/parallel-{json.dumps(v)}", "chat", {"messages": USER, "tools": OA_TOOLS, "parallel_tool_calls": v}, fam)
    add("chat/parallel-false-no-tools", "chat", {"messages": USER, "parallel_tool_calls": False}, fam)
    add("chat/parallel-false-none", "chat", {"messages": USER, "tools": OA_TOOLS, "tool_choice": "none",
                                             "parallel_tool_calls": False}, fam)
    add("chat/parallel-false-required", "chat", {"messages": USER, "tools": OA_TOOLS, "tool_choice": "required",
                                                 "parallel_tool_calls": False}, fam)
    add("chat/parallel-false-named", "chat", {"messages": USER, "tools": OA_TOOLS, "parallel_tool_calls": False,
                                              "tool_choice": {"type": "function", "function": {"name": "search"}}}, fam)
    add("chat/parallel-false-tool-turns", "chat", {"messages": OA_TOOL_TURNS, "tools": OA_TOOLS,
                                                   "parallel_tool_calls": False}, fam)
    for choice in ({"type": "auto", "disable_parallel_tool_use": True},
                   {"type": "any", "disable_parallel_tool_use": True},
                   {"type": "tool", "name": "search", "disable_parallel_tool_use": True},
                   {"type": "none", "disable_parallel_tool_use": True},
                   {"type": "auto", "disable_parallel_tool_use": False},
                   {"type": "auto", "disable_parallel_tool_use": None},
                   {"type": "auto", "disable_parallel_tool_use": "yes"}):
        add(f"an/disable-parallel-{choice['type']}-{json.dumps(choice['disable_parallel_tool_use'])}", "anthropic",
            {"max_tokens": 50, "messages": USER, "tools": AN_TOOLS, "tool_choice": choice}, fam)
    add("an/disable-parallel-turns", "anthropic", {"max_tokens": 50, "messages": AN_MSGS, "tools": AN_TOOLS,
                                                   "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}, fam)
    for v in (False, True, None, []):
        add(f"rs/parallel-{json.dumps(v)}", "responses", {"input": "x", "tools": RS_TOOLS, "parallel_tool_calls": v}, fam)
    add("rs/parallel-false-required", "responses", {"input": "x", "tools": RS_TOOLS, "tool_choice": "required",
                                                    "parallel_tool_calls": False}, fam)
    add("rs/parallel-false-turns", "responses", {"input": RS_INPUT, "tools": RS_TOOLS, "parallel_tool_calls": False}, fam)
    # defer_loading inside OpenAI's "function" object and beside it: the tool is declared, not in the tools block,
    # and the key is not in any schema
    add("chat/tools-defer-nested", "chat", {"messages": USER, "tools": DEFER_NESTED}, fam)
    add("chat/tools-defer-top", "chat", {"messages": USER, "tools": DEFER_TOP}, fam)
    add("rs/tools-defer-nested", "responses", {"input": "x", "tools": DEFER_NESTED}, fam)
    add("rs/tools-defer-flat", "responses", {"input": "x", "tools": [RS_TOOLS[0], dict(RS_TOOLS[1], defer_loading=True)]}, fam)
    add("an/tools-defer", "anthropic", {"max_tokens": 50, "messages": USER,
                                        "tools": [AN_TOOLS[0], dict(AN_TOOLS[1], defer_loading=True)]}, fam)

for c in cases:
    print(json.dumps(c, ensure_ascii=False))
