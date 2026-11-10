/** @file
 *  pulsar-agent: system prompt rendering and the worker's output queues. */
#include "pulsar_agent_internal.h"



/* ============================================================================
 * System Prompt Rendering And Worker Output Queues
 * ============================================================================
 */

/** The system prompt's opening: the agent's role. */
static const char agent_tools_prompt_role[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "## Tools\n\n";

/** DeepSeek's tool-call shape: DSML, spelled out in the prompt (the family has no tools template). */
static const char agent_tools_prompt_dsml_shape[] =
    "You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n"
    PULSAR_TOOL_CALLS_START "\n"
    PULSAR_INVOKE_START " name=\"$TOOL_NAME\">\n"
    PULSAR_PARAM_START " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" PULSAR_PARAM_END "\n"
    PULSAR_INVOKE_END "\n"
    PULSAR_TOOL_CALLS_END "\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n"
    "String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n";

/** Every other family (L365): the template declares the tools and their call syntax; the prompt only says when. */
static const char agent_tools_prompt_family_shape[] =
    "Call the tools declared in this prompt in the format it describes. "
    "Finish thinking before calling a tool; a call inside the thinking is not run.\n\n";

/** The read tools. */
static const char agent_tools_prompt_read[] =
    "Read defaults to a bounded chunk: path alone returns the first 500 lines, not the whole file. "
    "If read says more lines are available, call more with count=<lines> to read the next chunk; "
    "more defaults to the next 500 lines. "
    "The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. "
    "If the user explicitly asks you to read a complete file into context, call read with whole=true. "
    "A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";



/** The system prompt's file-editing guidance (write, edit, anchored [upto] old text). */
static const char agent_tools_prompt_edit_line[] =
    "## Editing files\n\n"
    "Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. "
    "For edit, always put the edited file path as the first parameter. "
    "The old text must match exactly once in the current file; otherwise edit fails for safety.\n"
    "For large replacements, prefer anchored old text: write the first lines, then [upto], then the final lines. "
    "The tool replaces everything from the head through the tail. If the head or tail is ambiguous, the edit fails.\n"
    "After [upto], always write unique final lines before closing old; never close old immediately after [upto].\n"
    "Do not use a generic tail anchor like:\n"
    "- BigNum bignum_add(BigNum *a, BigNum *b) {\n"
    "- [upto]\n"
    "- }\n"
    "because the closing brace may match many functions. Instead include final lines that are unique near that function, "
    "for example its last calculation and return line before the brace.\n";

/** The anchored-edit example in DSML. */
static const char agent_tools_prompt_edit_example_dsml[] =
    "Example anchored edit:\n"
    PULSAR_TOOL_CALLS_START "\n"
    PULSAR_INVOKE_START " name=\"edit\">\n"
    PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/example.c" PULSAR_PARAM_END "\n"
    PULSAR_PARAM_START " name=\"old\" string=\"true\">static int parse(void) {\n"
    "    int ok = 0;\n"
    "[upto]\n"
    "    return ok;\n"
    "}" PULSAR_PARAM_END "\n"
    PULSAR_PARAM_START " name=\"new\" string=\"true\">static int parse(void) {\n"
    "    return parse_impl();\n"
    "}" PULSAR_PARAM_END "\n"
    PULSAR_INVOKE_END "\n"
    PULSAR_TOOL_CALLS_END "\n";

/** The anchored-edit example as arguments (any family's call syntax). */
static const char agent_tools_prompt_edit_example_family[] =
    "Example anchored edit: call edit with path \"/tmp/example.c\", old\n"
    "static int parse(void) {\n"
    "    int ok = 0;\n"
    "[upto]\n"
    "    return ok;\n"
    "}\n"
    "and new\n"
    "static int parse(void) {\n"
    "    return parse_impl();\n"
    "}\n";

/** The edit guidance after the example. */
static const char agent_tools_prompt_edit_tail[] =
    "To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n"
    "Use read raw=true only when you need plain file text without line numbers or read annotations.\n\n";



/** The bash-job guidance. */
static const char agent_tools_prompt_bash[] =
    "For long-running bash commands, pass refresh_sec. If a bash job is still running, use "
    "bash_status to check it early or bash_stop to terminate it.\n\n";

/** The tool schemas, OpenAI function objects: the DSML prompt lists them, the other families' templates
 *  declare them (agent_build_tools_json). */
static const char *const agent_tool_schemas[] = {
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash\",\n"
    "    \"description\": \"Run a shell command.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"command\": {\"type\": \"string\"},\n"
    "        \"timeout_sec\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"command\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_status\",\n"
    "    \"description\": \"Report current status and new output for a bash job.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"number\"},\n"
    "        \"pid\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_stop\",\n"
    "    \"description\": \"Terminate a running bash job and report its final output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"number\"},\n"
    "        \"pid\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"read\",\n"
    "    \"description\": \"Read a text file or a range of lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"start_line\": {\"type\": \"number\"},\n"
    "        \"max_lines\": {\"type\": \"number\"},\n"
    "        \"whole\": {\"type\": \"boolean\"},\n"
    "        \"raw\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"more\",\n"
    "    \"description\": \"Continue the previous read-like output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"count\": {\"type\": \"number\"}\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"write\",\n"
    "    \"description\": \"Create or overwrite a text file.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"content\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"content\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"edit\",\n"
    "    \"description\": \"Replace exactly one old text match; old may contain [upto] between unique head and tail anchors.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"old\": {\"type\": \"string\"},\n"
    "        \"new\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"old\", \"new\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"search\",\n"
    "    \"description\": \"Search files and return compact edit-friendly matches.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"},\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"mode\": {\"type\": \"string\"},\n"
    "        \"glob\": {\"type\": \"string\"},\n"
    "        \"context\": {\"type\": \"number\"},\n"
    "        \"max_results\": {\"type\": \"number\"},\n"
    "        \"case_sensitive\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}",
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"list\",\n"
    "    \"description\": \"List one directory compactly.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    
    "}",
};

/** The DSML rule. */
static const char agent_tools_prompt_dsml_rule[] =
    "- Always use strict syntax for DSML tool stanzas.\n";

/** The rules every family shares. */
static const char agent_tools_prompt_rules[] =
    "- This system runs on local inference of a few hundred tokens/s of prefill, "
    "and a few tens of tokens/s decoding speed. Use read/search to get the "
    "anchors you need, then use anchored edit to avoid having to "
    "retype large text.\n"
    "- Write code that is reliable and works well; always have a mental model of "
    "what is going on in complex parts of the code.\n"
    "- Work in a way that preserves the current system configuration integrity, "
    "unless explicitly asked otherwise by the user.\n";



char *agent_build_tools_prompt(bool dsml) {
    agent_buf b = {0};
    agent_buf_puts(&b, agent_tools_prompt_role);
    agent_buf_puts(&b, dsml ? agent_tools_prompt_dsml_shape : agent_tools_prompt_family_shape);
    agent_buf_puts(&b, agent_tools_prompt_read);
    agent_buf_puts(&b, agent_tools_prompt_edit_line);
    agent_buf_puts(&b, dsml ? agent_tools_prompt_edit_example_dsml : agent_tools_prompt_edit_example_family);
    agent_buf_puts(&b, agent_tools_prompt_edit_tail);
    agent_buf_puts(&b, agent_tools_prompt_bash);
    if (dsml) {   /* DeepSeek reads the schemas here; every other family's template declares them */
        agent_buf_puts(&b, "### Available Tool Schemas\n\n");
        const size_t n = sizeof agent_tool_schemas / sizeof agent_tool_schemas[0];
        for (size_t i = 0; i < n; i++) {
            agent_buf_puts(&b, agent_tool_schemas[i]);
            agent_buf_puts(&b, i + 1 < n ? "\n\n" : "\n");
        }
        agent_buf_puts(&b, "\n");
    }
    agent_buf_puts(&b, "# Rules\n\n");
    if (dsml) agent_buf_puts(&b, agent_tools_prompt_dsml_rule);
    agent_buf_puts(&b, agent_tools_prompt_rules);
    return agent_buf_take(&b);
}

char *agent_build_tools_json(void) {
    agent_buf b = {0};
    agent_buf_puts(&b, "[");
    const size_t n = sizeof agent_tool_schemas / sizeof agent_tool_schemas[0];
    for (size_t i = 0; i < n; i++) {
        if (i) agent_buf_puts(&b, ",");
        agent_buf_puts(&b, agent_tool_schemas[i]);
    }
    agent_buf_puts(&b, "]");
    return agent_buf_take(&b);
}



const char agent_dsml_syntax_reminder[] =
    "DSML syntax reminder:\n"
    PULSAR_TOOL_CALLS_START "\n"
    PULSAR_INVOKE_START " name=\"$TOOL_NAME\">\n"
    PULSAR_PARAM_START " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" PULSAR_PARAM_END "\n"
    PULSAR_INVOKE_END "\n"
    PULSAR_TOOL_CALLS_END "\n";


/** @brief The tools prompt (agent_build_tools_prompt) wrapped in "[System prompt reminder ...]" markers.
 *  @param dsml the family calls tools in DSML (agent_build_tools_prompt)
 *  @return the text, malloc'd; caller frees */
static char *agent_build_system_prompt_reminder(bool dsml) {
    char *tools = agent_build_tools_prompt(dsml);
    const char *start = "\n\n[System prompt reminder follows.]\n";
    const char *end = "[End system prompt reminder.]\n\n";
    size_t len = strlen(start) + strlen(tools) + strlen(end) + 1;
    char *out = (char *)agent_xmalloc(len);
    out[0] = '\0';
    strcat(out, start);
    strcat(out, tools);
    strcat(out, end);
    free(tools);
    return out;
}



void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}



void agent_worker_maybe_append_datetime_context(agent_worker *w) {
    if (w->datetime_context_injected) return;

    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);

    char when[128];
    if (strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", &tm) == 0)
        snprintf(when, sizeof(when), "%lld", (long long)now);

    char msg[256];
    snprintf(msg, sizeof(msg),
             "Current local date and time at session start: %s. "
             "Use this only when date or time matters.", when);
    agent_turn_add(w, "system", msg, false);
    agent_trace_text(w, "datetime-context", msg, strlen(msg));
    w->datetime_context_injected = true;
}



/* The full tool/system reminder is separate from DSML syntax errors: it is a
 * pressure-controlled refresh of the same trusted prompt shape used at startup,
 * added to the turn as system notes: the built-in prompt trusted (DeepSeek keeps
 * its DSML markers native control tokens), arbitrary -sys text ordinary text. */
void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w) {
    if (w->last_system_prompt_reminder_at <= 0) {
        agent_worker_note_system_prompt_seen(w);
        return;
    }
    if (w->transcript.len - w->last_system_prompt_reminder_at <
        AGENT_SYSTEM_PROMPT_REMINDER_TOKENS)
    {
        return;
    }

    char *reminder = agent_build_system_prompt_reminder(agent_worker_dsml(w));
    agent_publish_system_status(w, "Re-injecting system prompt reminder...");
    agent_trace(w, "system prompt reminder injected at transcript=%d",
                w->transcript.len);
    agent_turn_add(w, "system", reminder, true);
    free(reminder);

    const char *extra = w->cfg->gen.system;
    if (extra && extra[0]) {
        agent_buf b = {0};
        agent_buf_puts(&b, "\nAdditional system instructions reminder:\n");
        agent_buf_puts(&b, extra);
        agent_buf_puts(&b, "\n[End additional system instructions reminder.]\n\n");
        char *note = agent_buf_take(&b);
        agent_turn_add(w, "system", note, false);
        free(note);
    }
    agent_worker_note_system_prompt_seen(w);
}



/* Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe. */
void agent_wake_locked(agent_worker *w) {
    if (w->wake_pending) return;
    w->wake_pending = true;
    char c = 'x';
    ssize_t wr = write(w->wake_fd[1], &c, 1);
    (void)wr;
}



size_t agent_terminal_sanitize(char *s, size_t n) {
    size_t out = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\r') {
            if (i + 1 < n && s[i + 1] == '\n') s[out++] = '\r';   /* a bare return would overwrite the row */
        } else if ((c < 0x20 && c != '\n' && c != '\t') || c == 0x7f) {
            s[out++] = '?';
        } else if (c == 0xc2 && i + 1 < n && (unsigned char)s[i + 1] >= 0x80 && (unsigned char)s[i + 1] < 0xa0) {
            s[out++] = '?';
            s[out++] = '?';
            i++;
        } else {
            s[out++] = s[i];
        }
    }
    return out;
}

void agent_publish_untrusted(agent_worker *w, const char *s, size_t n) {
    if (!n) return;
    char stack[1024];
    char *copy = n <= sizeof stack ? stack : (char *)malloc(n);
    if (!copy) return;   /* out of memory: nothing unsanitized goes out */
    memcpy(copy, s, n);
    agent_publish(w, copy, agent_terminal_sanitize(copy, n));
    if (copy != stack) free(copy);
}

/* Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place. */
void agent_publish(agent_worker *w, const char *s, size_t n) {
    if (!n) return;
    pthread_mutex_lock(&w->mu);
    if (w->out_len + n + 1 > w->out_cap) {
        size_t cap = w->out_cap ? w->out_cap * 2 : 4096;
        while (cap < w->out_len + n + 1) cap *= 2;
        char *p = (char *)realloc(w->out, cap);
        if (!p) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        w->out = p;
        w->out_cap = cap;
    }
    memcpy(w->out + w->out_len, s, n);
    w->out_len += n;
    w->out[w->out_len] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish(w, stack, (size_t)n);
        return;
    }

    char *heap = (char *)agent_xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish(w, heap, (size_t)n);
    free(heap);
}



void agent_set_status(agent_worker *w, agent_worker_state state) {
    pthread_mutex_lock(&w->mu);
    w->status.state = state;
    if (state != AGENT_WORKER_PREFILL)
        w->status.prefill_tps = 0.0;
    if (state != AGENT_WORKER_GENERATING)
        w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



void agent_set_error(agent_worker *w, const char *msg) {
    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_ERROR;
    w->status.prefill_tps = 0.0;
    w->status.greedy_sampling = false;
    snprintf(w->status.error, sizeof(w->status.error), "%s", msg ? msg : "unknown error");
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

