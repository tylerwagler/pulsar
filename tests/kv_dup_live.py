#!/usr/bin/env python3
"""kv_dup_live.py -- L324: the duplicate-prefix census (/metrics pulsar:kv_dup_*) counts the prefix pages that fan-out
siblings hold twice in GPU memory, on a real pulsar-server.

    python3 tests/kv_dup_live.py MODEL --binary ./pulsar-server --work DIR [--port 8324] [--siblings 3]
                                 [--preamble 56000] [--ctx 65536] [--tag NAME]

Claude Code's fan-out shape (L320, 2026-10-23: 13 subagent siblings off one node shared ~24.5k tokens): siblings
share a long system prompt and part inside it (each sibling's own task closes the system prompt), so each one lands
on a FRESH bank (it diverges before the bank's last user marker; L322's in-place rule does not take it) and adopts the
shared prefix from the bank before it (L313/L322).  Every bank then holds its own copy of the shared grid pages.

Here: SIBLINGS Anthropic /v1/messages requests, one after another (greedy, 48 tokens each), each with the system
prompt [tag] + the first PREAMBLE characters of the story (~12.5k tokens at 56000 on Qwen and Nemotron: three whole
4096-token pages) + its own task.  Checks, from the server's log and /metrics after the last one:
  * DISTINCT BANKS: every sibling ran on a bank of its own (the resume line's bank), so the census has copies to count;
  * CENSUS RAN: pulsar:kv_dup_censuses_total grew by at least SIBLINGS (one census at each request's end);
  * PAGES: pulsar:kv_dup_page_tokens > 0 and pulsar:kv_dup_page_bytes > 0; at least SHARED = floor((min prompt - TAIL)
    / page) distinct pages held by >= 2 banks, and at least (SIBLINGS - 1) * SHARED extra copies.  min prompt - TAIL
    is a lower bound of the shared prefix (the prompts are the server's own counts; TAIL = 400 tokens over-covers each
    sibling's own task, the user turn and the template's close, ~30 tokens), and the census's pages sit on the grid
    from token 0, so its floor is a lower bound of the whole pages shared -- no further slack;
  * BYTES: pulsar:kv_dup_bytes == kv_dup_pages_duplicate * kv_dup_page_bytes;
  * PAIRS: at least SIBLINGS * (SIBLINGS - 1) / 2 pairs in the preamble class, the longest at least SHARED pages.
Other legs on a shared server may hold banks of their own; every check is a lower bound, so they only add.

A leg of tests/live_legs.py (L317 Leg shape: warm only).  Alone (this command line) it runs on an instance of its own,
PULSAR_MSEQ_BANKS=4.  Exit 0 only when every check held."""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg  # noqa: E402
from prefix_turns_live import STORY, post  # noqa: E402

TAIL = 400   # tokens: over-covers what follows the shared preamble (a sibling's task, the user turn, the template)
RESUME = re.compile(r"resume: bank (\d+) own ")
TASKS = [
    "Your sub-task: list the characters who appear in the first chapter.",
    "Your sub-task: list the places the first chapter describes.",
    "Your sub-task: name the first conflict of the story.",
    "Your sub-task: name the first object the story describes.",
    "Your sub-task: say in what season the story opens.",
    "Your sub-task: say who speaks first in the story.",
]


class KvDupLeg(Leg):
    NAME = "kv-dup"

    def __init__(self, g, work, siblings=3, preamble=56000, tag="kvdup"):
        Leg.__init__(self, g, work)
        self.siblings, self.tag = min(siblings, len(TASKS)), tag
        self.preamble = ("[%s] You are one of several sub-agents reading the story below; each of you has its own "
                         "sub-task, given after the story.\n\n" % tag) + STORY[:preamble]

    def warm(self, srv):
        m0 = srv.metrics()
        banks, prompts = [], []
        for k in range(self.siblings):
            body = {"model": "m", "max_tokens": 48, "temperature": 0,
                    "system": self.preamble + "\n\n" + TASKS[k],
                    "messages": [{"role": "user", "content": "Do your sub-task in one sentence."}]}
            mark = srv.mark()
            d, _ = post(srv, body)
            r = RESUME.search(srv.since(mark))
            banks.append(int(r.group(1)) if r else None)
            prompts.append(d["prompt_tokens"])
            print("# [%s] sibling %d: prompt %d tokens on bank %s" % (self.tag, k + 1, d["prompt_tokens"], banks[-1]),
                  flush=True)
        m = srv.metrics()
        get = lambda name: m.get("pulsar:" + name, -1.0)  # noqa: E731
        print("# [%s] census: %s" % (self.tag, {k[len("pulsar:"):]: v for k, v in m.items()
                                                if k.startswith("pulsar:kv_dup_")}), flush=True)
        n = self.siblings
        self.g.check(None not in banks and len(set(banks)) == n,
                     "every sibling ran on a bank of its own: banks %s" % banks)
        grew = get("kv_dup_censuses_total") - m0.get("pulsar:kv_dup_censuses_total", 0.0)
        self.g.check(grew >= n, "a census at each request's end: kv_dup_censuses_total grew by %d (>= %d)" % (grew, n))
        page, pbytes = get("kv_dup_page_tokens"), get("kv_dup_page_bytes")
        self.g.check(page > 0 and pbytes > 0, "the census priced its pages: %d tokens, %d bytes a page" % (page, pbytes))
        if page <= 0:
            return
        # the siblings share the system prompt up to their task: min prompt - TAIL is a lower bound of the shared prefix,
        # and the grid's pages start at token 0, so its floor is a lower bound of the whole pages shared
        shared = max(0, int((min(prompts) - TAIL) // page))
        self.g.check(shared >= 2, "the preamble spans at least 2 whole pages to share (%d)" % shared)
        self.g.check(get("kv_dup_pages_shared") >= shared,
                     "kv_dup_pages_shared %d >= %d" % (get("kv_dup_pages_shared"), shared))
        self.g.check(get("kv_dup_pages_duplicate") >= (n - 1) * shared,
                     "kv_dup_pages_duplicate %d >= %d" % (get("kv_dup_pages_duplicate"), (n - 1) * shared))
        self.g.check(get("kv_dup_bytes") == get("kv_dup_pages_duplicate") * pbytes and get("kv_dup_bytes") > 0,
                     "kv_dup_bytes %d == %d extra pages x %d bytes"
                     % (get("kv_dup_bytes"), get("kv_dup_pages_duplicate"), pbytes))
        pre = 'kv_dup_pairs{class="preamble"}'
        longest = 'kv_dup_longest_shared_tokens{class="preamble"}'
        self.g.check(get(pre) >= n * (n - 1) // 2, "%s %d >= %d" % (pre, get(pre), n * (n - 1) // 2))
        self.g.check(get(longest) >= shared * page, "%s %d >= %d" % (longest, get(longest), shared * page))


def _args(ap):
    ap.add_argument("--siblings", type=int, default=3)
    ap.add_argument("--preamble", type=int, default=56000)
    ap.add_argument("--tag", default="kvdup")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(KvDupLeg, sys.argv[1:], ctx=65536, banks=4, port=8324, add_args=_args))
