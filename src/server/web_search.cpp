#include "pulsar_server_internal.h"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

/* Server-executed web_search tool (Anthropic server-tool protocol), backed by
 * a SearXNG instance reached over plain LAN HTTP.
 *
 * Clients like Claude Code advertise {"type":"web_search_20250305"} and expect
 * the BACKEND to run the search mid-request: the model's tool call must never
 * reach the client as a stop_reason=tool_use turn, because no client-side
 * executor exists for it.  This TU owns everything that must stay byte-stable
 * across turns:
 *
 *   - the per-result text chunk the model sees (web_search_render_chunk) is
 *     ALSO stored verbatim in each result block's encrypted_content field, and
 *     the next-turn replay rebuilds the model-visible tool_result text by
 *     joining those echoed fields (web_search_rebuild_result_text).  Changing
 *     the chunk format breaks byte-exact KV replay for stored conversations,
 *     so treat it like a wire format.
 *   - error results are deterministic text keyed by error_code only, for the
 *     same reason; the interesting detail goes to the server log instead. */

#define WEB_SEARCH_TOP_N 8
#define WEB_SEARCH_IO_TIMEOUT_SEC 8
#define WEB_SEARCH_MAX_BODY (2u << 20)

static const char WEB_SEARCH_NO_RESULTS_TEXT[] = "No search results found.";



bool web_search_tool_entry(const char *raw_tool_json, int *max_uses) {
    const char *p = raw_tool_json ? raw_tool_json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    char *type = NULL;
    char *name = NULL;
    int uses = 0;
    bool ok = true;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) { ok = false; break; }
        json_ws(&p);
        if (*p != ':') { free(key); ok = false; break; }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) { free(key); ok = false; break; }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) { free(key); ok = false; break; }
        } else if (!strcmp(key, "max_uses")) {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) { free(key); ok = false; break; }
            uses = atoi(raw);
            free(raw);
        } else if (!json_skip_value(&p)) {
            free(key);
            ok = false;
            break;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    /* An Anthropic server tool is versioned by type ("web_search_20250305",
     * later dates to come) and always named web_search. */
    bool is_web_search = ok && type &&
        !strncmp(type, "web_search", strlen("web_search")) &&
        (!name || !strcmp(name, "web_search"));
    if (is_web_search && max_uses) *max_uses = uses > 0 ? uses : WEB_SEARCH_DEFAULT_MAX_USES;
    free(type);
    free(name);
    return is_web_search;
}



const char *web_search_schema_line(void) {
    /* The wire tool carries no input_schema (the real executor is server-side),
     * so the DSML prompt needs a synthesized one or the model invents its own
     * argument names. */
    return "{\"name\":\"web_search\","
           "\"description\":\"Search the web. Returns titles, URLs and snippets "
           "of the top results for a query. Use a focused query; follow up on "
           "promising URLs with your other tools if deeper content is needed.\","
           "\"parameters\":{\"type\":\"object\",\"properties\":{"
           "\"query\":{\"type\":\"string\",\"description\":\"The search query\"}},"
           "\"required\":[\"query\"]}}";
}



char *web_search_query_from_arguments(const char *arguments_json) {
    const char *p = arguments_json ? arguments_json : "";
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    char *query = NULL;
    char *first_string = NULL;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) break;
        json_ws(&p);
        if (*p != ':') { free(key); break; }
        p++;
        json_ws(&p);
        if (*p == '"') {
            char *val = NULL;
            if (!json_string(&p, &val)) { free(key); break; }
            if (!strcmp(key, "query")) {
                free(query);
                query = val;
                val = NULL;
            } else if (!first_string) {
                first_string = val;
                val = NULL;
            }
            free(val);
        } else if (!json_skip_value(&p)) {
            free(key);
            break;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    /* Lenient fallback: a model that ignored the schema and used its own key
     * ("search_query" was observed) still gets its intent executed. */
    if (!query && first_string) return first_string;
    free(first_string);
    return query;
}



/* ---- SearXNG HTTP fetch --------------------------------------------------- */

/** A SearXNG endpoint split out of its configured URL. Fixed-size fields: a
 * URL longer than these bounds is rejected rather than allocated for. */
typedef struct {
    char host[256];   ///< hostname, NUL-terminated
    char port[16];    ///< port as text, ready for getaddrinfo
    char path[512];   ///< request path, including any query prefix
} web_search_endpoint;

static bool web_search_parse_url(const char *url, web_search_endpoint *ep) {
    const char *scheme = "http://";
    if (!url || strncmp(url, scheme, strlen(scheme)) != 0) return false;
    const char *host = url + strlen(scheme);
    const char *slash = strchr(host, '/');
    const char *colon = strchr(host, ':');
    if (colon && slash && colon > slash) colon = NULL;
    const char *host_end = colon ? colon : (slash ? slash : host + strlen(host));
    if (host_end == host || (size_t)(host_end - host) >= sizeof(ep->host)) return false;
    memcpy(ep->host, host, (size_t)(host_end - host));
    ep->host[host_end - host] = '\0';
    if (colon) {
        const char *port = colon + 1;
        const char *port_end = slash ? slash : port + strlen(port);
        if (port_end == port || (size_t)(port_end - port) >= sizeof(ep->port)) return false;
        memcpy(ep->port, port, (size_t)(port_end - port));
        ep->port[port_end - port] = '\0';
    } else {
        snprintf(ep->port, sizeof(ep->port), "80");
    }
    size_t path_len = slash ? strlen(slash) : 0;
    if (path_len >= sizeof(ep->path)) return false;
    if (slash) memcpy(ep->path, slash, path_len);
    ep->path[path_len] = '\0';
    /* Normalize away a trailing slash so "/search" concatenation is stable. */
    if (path_len && ep->path[path_len - 1] == '/') ep->path[path_len - 1] = '\0';
    return true;
}

static void append_url_encoded(buf *b, const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    for (s = s ? s : ""; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~')
        {
            buf_putc(b, (char)c);
        } else if (c == ' ') {
            buf_putc(b, '+');
        } else {
            buf_putc(b, '%');
            buf_putc(b, hex[c >> 4]);
            buf_putc(b, hex[c & 15]);
        }
    }
}

/* Blocking GET on the worker thread; bounded hard by socket timeouts.  The
 * search box is LAN-local, so the common case is tens of milliseconds; the
 * worst case stalls other sessions for WEB_SEARCH_IO_TIMEOUT_SEC, which is the
 * accepted v1 trade-off (noted in the tool's design discussion). */
static char *web_search_http_get(const web_search_endpoint *ep, const char *query_path,
                                 char *err, size_t errlen) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = NULL;
    int rc = getaddrinfo(ep->host, ep->port, &hints, &res);
    if (rc != 0 || !res) {
        snprintf(err, errlen, "resolve %s: %s", ep->host, gai_strerror(rc));
        if (res) freeaddrinfo(res);
        return NULL;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = {WEB_SEARCH_IO_TIMEOUT_SEC, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        snprintf(err, errlen, "connect %s:%s failed", ep->host, ep->port);
        return NULL;
    }

    /* HTTP/1.0 + Connection: close keeps the response un-chunked and
     * EOF-delimited, so no transfer-coding parser is needed. */
    buf req = {0};
    buf_printf(&req, "GET %s HTTP/1.0\r\nHost: %s:%s\r\n"
                     "User-Agent: pulsar-server\r\nAccept: application/json\r\n"
                     "Connection: close\r\n\r\n",
               query_path, ep->host, ep->port);
    size_t off = 0;
    while (off < req.len) {
        ssize_t n = send(fd, req.ptr + off, req.len - off, MSG_NOSIGNAL);
        if (n <= 0) {
            snprintf(err, errlen, "send failed");
            buf_free(&req);
            close(fd);
            return NULL;
        }
        off += (size_t)n;
    }
    buf_free(&req);

    buf resp = {0};
    char chunk[8192];
    for (;;) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0) {
            snprintf(err, errlen, "recv failed (timeout?)");
            buf_free(&resp);
            close(fd);
            return NULL;
        }
        if (n == 0) break;
        if (resp.len + (size_t)n > WEB_SEARCH_MAX_BODY) {
            snprintf(err, errlen, "response too large");
            buf_free(&resp);
            close(fd);
            return NULL;
        }
        buf_append(&resp, chunk, (size_t)n);
    }
    close(fd);

    if (!resp.ptr || strncmp(resp.ptr, "HTTP/1.", 7) != 0) {
        snprintf(err, errlen, "malformed response");
        buf_free(&resp);
        return NULL;
    }
    int status = atoi(resp.ptr + 9);
    char *body = strstr(resp.ptr, "\r\n\r\n");
    if (status != 200 || !body) {
        snprintf(err, errlen, "http status %d", status);
        buf_free(&resp);
        return NULL;
    }
    char *out = xstrdup(body + 4);
    buf_free(&resp);
    return out;
}



/* ---- SearXNG JSON -> canonical result chunks ------------------------------ */

/** One search result, as rendered back into the model's context. */
typedef struct {
    char *title;    ///< result title, owned
    char *url;      ///< result URL, owned
    char *snippet;  ///< result summary text, owned
} web_search_hit;

static void web_search_render_chunk(buf *b, const web_search_hit *hit) {
    buf_puts(b, "Result:\nTitle: ");
    buf_puts(b, hit->title ? hit->title : "");
    buf_puts(b, "\nURL: ");
    buf_puts(b, hit->url ? hit->url : "");
    buf_puts(b, "\nSnippet: ");
    buf_puts(b, hit->snippet ? hit->snippet : "");
}

static bool web_search_parse_hit(const char **p, web_search_hit *hit) {
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        json_ws(p);
        char **dst = NULL;
        if (!strcmp(key, "title")) dst = &hit->title;
        else if (!strcmp(key, "url")) dst = &hit->url;
        else if (!strcmp(key, "content")) dst = &hit->snippet;
        if (dst && **p == '"') {
            free(*dst);
            *dst = NULL;
            if (!json_string(p, dst)) { free(key); return false; }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static void web_search_hit_free(web_search_hit *hit) {
    free(hit->title);
    free(hit->url);
    free(hit->snippet);
    memset(hit, 0, sizeof(*hit));
}

/* Fill model_text (the exact bytes the model sees inside <tool_result>) and
 * client_content_json (the web_search_tool_result "content" value echoed back
 * to us next turn).  Every model-text chunk is carried verbatim in its
 * result's encrypted_content, which is what makes replay byte-exact. */
static bool web_search_format_results(const char *body, buf *model_text,
                                      buf *client_content_json) {
    const char *p = body ? body : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    int emitted = 0;
    bool found_results = false;
    buf_putc(client_content_json, '[');
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) break;
        json_ws(&p);
        if (*p != ':') { free(key); break; }
        p++;
        json_ws(&p);
        if (!strcmp(key, "results") && *p == '[') {
            found_results = true;
            p++;
            json_ws(&p);
            while (*p && *p != ']') {
                web_search_hit hit = {0};
                if (!web_search_parse_hit(&p, &hit)) {
                    web_search_hit_free(&hit);
                    free(key);
                    goto done;
                }
                if (emitted < WEB_SEARCH_TOP_N && hit.url && hit.url[0]) {
                    buf chunk = {0};
                    web_search_render_chunk(&chunk, &hit);
                    if (emitted) buf_puts(model_text, "\n\n");
                    buf_puts(model_text, chunk.ptr ? chunk.ptr : "");
                    if (emitted) buf_putc(client_content_json, ',');
                    buf_puts(client_content_json, "{\"type\":\"web_search_result\",\"url\":");
                    json_escape(client_content_json, hit.url);
                    buf_puts(client_content_json, ",\"title\":");
                    json_escape(client_content_json, hit.title ? hit.title : "");
                    buf_puts(client_content_json, ",\"encrypted_content\":");
                    json_escape(client_content_json, chunk.ptr ? chunk.ptr : "");
                    buf_puts(client_content_json, ",\"page_age\":null}");
                    buf_free(&chunk);
                    emitted++;
                }
                web_search_hit_free(&hit);
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            free(key);
            break; /* results consumed; the rest of the body is irrelevant */
        } else if (!json_skip_value(&p)) {
            free(key);
            break;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
done:
    buf_putc(client_content_json, ']');
    if (emitted == 0) buf_puts(model_text, WEB_SEARCH_NO_RESULTS_TEXT);
    return found_results;
}

static void web_search_format_error(const char *error_code, buf *model_text,
                                    buf *client_content_json) {
    buf_puts(model_text, "Web search failed: ");
    buf_puts(model_text, error_code);
    buf_puts(model_text, ". Answer from available information.");
    buf_puts(client_content_json,
             "{\"type\":\"web_search_tool_result_error\",\"error_code\":");
    json_escape(client_content_json, error_code);
    buf_putc(client_content_json, '}');
}



bool web_search_run(const char *base_url, const char *query,
                    buf *model_text, buf *client_content_json,
                    char *logerr, size_t errlen) {
    web_search_endpoint ep;
    memset(&ep, 0, sizeof(ep));
    if (!query || !query[0]) {
        snprintf(logerr, errlen, "empty query");
        web_search_format_error("invalid_input", model_text, client_content_json);
        return false;
    }
    if (!web_search_parse_url(base_url, &ep)) {
        snprintf(logerr, errlen, "bad --web-search-url %s", base_url ? base_url : "(null)");
        web_search_format_error("unavailable", model_text, client_content_json);
        return false;
    }
    buf path = {0};
    buf_printf(&path, "%s/search?format=json&q=", ep.path);
    append_url_encoded(&path, query);
    char *body = web_search_http_get(&ep, path.ptr, logerr, errlen);
    buf_free(&path);
    if (!body) {
        web_search_format_error("unavailable", model_text, client_content_json);
        return false;
    }
    bool ok = web_search_format_results(body, model_text, client_content_json);
    free(body);
    if (!ok) {
        /* Body was 200 but not the expected shape (limiter HTML, etc.). */
        buf_free(model_text);
        buf_free(client_content_json);
        snprintf(logerr, errlen, "unexpected response shape");
        web_search_format_error("unavailable", model_text, client_content_json);
        return false;
    }
    return true;
}

void web_search_run_exhausted(buf *model_text, buf *client_content_json) {
    web_search_format_error("max_uses_exceeded", model_text, client_content_json);
}



/* ---- next-turn replay: web_search_tool_result content -> model text ------- */

char *web_search_rebuild_result_text(const char *content_json) {
    const char *p = content_json ? content_json : "";
    json_ws(&p);
    buf out = {0};
    if (*p == '{') {
        /* Error block: deterministic text keyed by error_code. */
        p++;
        char *code = NULL;
        json_ws(&p);
        while (*p && *p != '}') {
            char *key = NULL;
            if (!json_string(&p, &key)) break;
            json_ws(&p);
            if (*p != ':') { free(key); break; }
            p++;
            if (!strcmp(key, "error_code")) {
                free(code);
                if (!json_string(&p, &code)) { free(key); break; }
            } else if (!json_skip_value(&p)) {
                free(key);
                break;
            }
            free(key);
            json_ws(&p);
            if (*p == ',') p++;
            json_ws(&p);
        }
        buf ignored = {0};
        web_search_format_error(code && code[0] ? code : "unavailable", &out, &ignored);
        buf_free(&ignored);
        free(code);
        return buf_take(&out);
    }
    if (*p != '[') return xstrdup(WEB_SEARCH_NO_RESULTS_TEXT);
    p++;
    int emitted = 0;
    json_ws(&p);
    while (*p && *p != ']') {
        if (*p == '{') {
            p++;
            char *enc = NULL;
            json_ws(&p);
            while (*p && *p != '}') {
                char *key = NULL;
                if (!json_string(&p, &key)) goto bad;
                json_ws(&p);
                if (*p != ':') { free(key); goto bad; }
                p++;
                if (!strcmp(key, "encrypted_content")) {
                    free(enc);
                    if (!json_string(&p, &enc)) { free(key); goto bad; }
                } else if (!json_skip_value(&p)) {
                    free(key);
                    goto bad;
                }
                free(key);
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != '}') goto bad;
            p++;
            if (enc && enc[0]) {
                if (emitted) buf_puts(&out, "\n\n");
                buf_puts(&out, enc);
                emitted++;
            }
            free(enc);
            enc = NULL;
            json_ws(&p);
            if (*p == ',') p++;
            json_ws(&p);
            continue;
bad:
            free(enc);
            buf_free(&out);
            return xstrdup(WEB_SEARCH_NO_RESULTS_TEXT);
        }
        if (!json_skip_value(&p)) break;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (emitted == 0) {
        buf_free(&out);
        return xstrdup(WEB_SEARCH_NO_RESULTS_TEXT);
    }
    return buf_take(&out);
}
