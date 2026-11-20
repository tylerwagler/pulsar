#!/usr/bin/env python3
"""L445: the strong-types series' enforcement gates (check tier).

  raw    every `.raw()` in src/ (outside src/lib/pulsar_units.h, src/vendor and the test sources; src/cuda's kernel
         bodies take plain integers, so its .raw() are host launch code and are read) is an argument of a log /
         format call or carries a `/* raw: <reason> */` tag on its line
  wrap   no unit constructed from another value's raw (`KvBytes{x.raw()}`) outside pulsar_units.h, except a tagged
         named crossing (`/* raw: <reason> */` on the line)
  decl   no bare int / uint32_t / size_t parameter or field whose name says token, position, bank, slot, layer,
         row, job id or byte count in the headers of src/server, src/engine, src/lib and src/families, beyond the
         allowlist (tests/units_decl_allow.txt: each entry with its reason; it can only shrink)

Python 3 stdlib only.  `--self-test` plants a violation of each kind and checks that the gate fails on it.
`--list-raw` / `--list-wrap` / `--list-decl` print what each scan finds.  UNITS_GATE_ROOT overrides the tree read.
"""
import os
import re
import sys

ROOT = os.environ.get('UNITS_GATE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS_H = 'src/lib/pulsar_units.h'
ALLOW = 'tests/units_decl_allow.txt'

# calls whose arguments are printed: a .raw() inside one is a log / format argument
# calls whose arguments are printed: the printf family, the log macros, and every function in src declared with a
# `const char *fmt` parameter (found below, so a new log helper is covered without editing this list)
FORMAT_BASE = {'printf', 'fprintf', 'snprintf', 'sprintf', 'vsnprintf', 'vfprintf', 'dprintf', 'fputs', 'puts',
               'perror', 'kv_line_int'}
FORMAT_MACRO = re.compile(r'^PULSAR_\w*(?:_ERR|REFUSE\w*)$|^DSPARK_REFUSE_IF$')
_format_calls = None


def format_calls():
    """The format functions: FORMAT_BASE and every src function with a `const char *fmt` parameter."""
    global _format_calls
    if _format_calls is None:
        names = set(FORMAT_BASE)
        decl = re.compile(r'\b(\w+)\s*\([^()]*const char \*\s*fmt\b[^()]*\)')
        for rel in source_files(['src'], ('.h', '.cpp', '.cu', '.hpp', '.cuh')):
            for m in decl.finditer(open(os.path.join(ROOT, rel), errors='replace').read()):
                names.add(m.group(1))
        _format_calls = names
    return _format_calls


def is_format_call(name):
    """Whether a call of `name` prints its arguments."""
    return name in format_calls() or FORMAT_MACRO.match(name) is not None
TAG = re.compile(r'/\*\s*raw:\s*\S.*?\*/')


def blank_comments_and_strings(text):
    """The text with comments and string / char literals replaced by spaces (newlines kept): positions unchanged."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            j = text.find('\n', i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
        elif c == '/' and i + 1 < n and text[i + 1] == '*':
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        elif c == 'R' and text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
            m = re.match(r'R"([^(\s]*)\(', text[i:])
            if not m:
                i += 1
                continue
            end = text.find(')' + m.group(1) + '"', i)
            end = n if end < 0 else end + len(m.group(1)) + 2
            for k in range(i, end):
                if out[k] != '\n':
                    out[k] = ' '
            i = end
        elif c in '"\'':
            if c == "'" and i > 0 and (text[i - 1].isalnum()) and i + 1 < n and text[i + 1].isalnum():
                i += 1   # a digit separator (1'000)
                continue
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == '\\':
                    j += 1
                elif text[j] == '\n':
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                out[k] = ' '
            i = j + 1
        else:
            i += 1
    return ''.join(out)


def source_files(subdirs, exts):
    """Repository-relative paths of the sources the gates read."""
    for sub in subdirs:
        for dp, dns, fns in os.walk(os.path.join(ROOT, sub)):
            dns[:] = sorted(d for d in dns if d != 'vendor')
            for fn in sorted(fns):
                if fn.endswith(exts):
                    yield os.path.relpath(os.path.join(dp, fn), ROOT)


def is_test_source(rel):
    """A test program's source inside src/."""
    b = os.path.basename(rel)
    return re.search(r'_tests?\.(cpp|h)$', b) is not None or '/testing.' in rel


def raw_scope(rel):
    """Whether the .raw() gate reads this file."""
    # src/cuda's .cu files are read too: their .raw() are all host code (the launchers' one raw name per value at the
    # top of a definition); a kernel body takes plain integers and has none to read
    return rel != UNITS_H and not is_test_source(rel)


def expression_brace(blank, i):
    """A `}` that closes a braced initialiser inside an expression (followed by an operator, a comma or a close)."""
    j = i + 1
    while j < len(blank) and blank[j] in ' \t\n':
        j += 1
    return j < len(blank) and blank[j] in ',)];.+-*/%<>=!&|?:^'


def enclosing_calls(blank, pos):
    """The names of the calls whose argument lists enclose `pos`, innermost first (stops at a statement end)."""
    names = []
    depth = 0
    i = pos - 1
    while i >= 0:
        c = blank[i]
        if c in ')]':
            depth += 1
        elif c == '}' and (depth > 0 or expression_brace(blank, i)):
            depth += 1
        elif c == '{' and depth > 0:
            depth -= 1
        elif c == '[':
            if depth == 0:
                names.append('[')
            else:
                depth -= 1
        elif c == '(':
            if depth == 0:
                j = i - 1
                while j >= 0 and blank[j] in ' \t\n':
                    j -= 1
                k = j
                while k >= 0 and (blank[k].isalnum() or blank[k] in '_:'):
                    k -= 1
                names.append(blank[k + 1:j + 1])
            else:
                depth -= 1
        elif c in ';{}' and depth == 0:
            if c == '{':
                # a braced initialiser T{...}: record its type name; a block opener ends the statement
                j = i - 1
                while j >= 0 and blank[j] in ' \t':
                    j -= 1
                k = j
                while k >= 0 and (blank[k].isalnum() or blank[k] == '_' or blank[k] == ':'):
                    k -= 1
                name = blank[k + 1:j + 1]
                if name and name[0].isalpha() and name not in ('else', 'do', 'try'):
                    names.append('{' + name)
                    i -= 1
                    continue
            break
        i -= 1
    return names


def line_of(text, pos):
    """1-based line number of `pos`."""
    return text.count('\n', 0, pos) + 1


def unit_type_names():
    """Every unit / index / offset / id type pulsar_units.h defines (typedef'd from a pulsar_* template)."""
    t = open(os.path.join(ROOT, UNITS_H)).read()
    names = set(re.findall(r'typedef\s+pulsar_\w+<[^;]*>\s+(\w+)\s*;', t))
    return names | {'BankCount'}


def scan_raw(files=None):
    """(untagged log-free .raw() sites, wrap sites) as (rel, line, text) lists."""
    units = unit_type_names()
    untagged, wraps = [], []
    for rel in files if files is not None else source_files(['src'], ('.h', '.cpp', '.cu', '.hpp', '.cuh')):
        if not raw_scope(rel) and files is None:
            continue
        text = open(os.path.join(ROOT, rel), errors='replace').read()
        blank = blank_comments_and_strings(text)
        lines = text.split('\n')
        seen_lines_u, seen_lines_w = set(), set()
        for m in re.finditer(r'\.raw\(\)', blank):
            ln = line_of(blank, m.start())
            src_line = lines[ln - 1]
            tagged = TAG.search(src_line) is not None
            calls = enclosing_calls(blank, m.start())
            wrap = next((c[1:] for c in calls if c.startswith('{') and c[1:].split('::')[-1] in units), None)
            if wrap is None:
                # Unit(x.raw()), Unit::from_raw_unchecked(x.raw()) and Unit::from_wire(x.raw()) are the same construction
                wrap = next((c for c in calls if c.split('::')[-1] in units or
                             (c.split('::')[-1] in ('from_raw_unchecked', 'from_wire') and
                              len(c.split('::')) >= 2 and c.split('::')[-2] in units)), None)
            if wrap is not None and not tagged and ln not in seen_lines_w:
                seen_lines_w.add(ln)
                wraps.append((rel, ln, src_line.strip()))
            fmt = any(is_format_call(c.split('::')[-1]) for c in calls if c and c[0] not in '[{')
            if not fmt and not tagged and ln not in seen_lines_u:
                seen_lines_u.add(ln)
                untagged.append((rel, ln, src_line.strip(), calls))
    return untagged, wraps


DECL_DIRS = ['src/server', 'src/engine', 'src/lib', 'src/families']
BARE = r'(?:int|unsigned|unsigned\s+int|int32_t|uint32_t|int64_t|uint64_t|size_t|long|unsigned\s+long|uint16_t|int16_t)'
NAME_PAT = re.compile(r'^(?:\w*_)?(?:tok|token|tokens|pos|position|bank|banks|slot|slots|layer|layers|il|row|rows|'
                      r'n_rows|job_id|jobid|job_seq|seq_id|bytes|nbytes|n_bytes)(?:_\w*)?$|'
                      r'^\w*(?:_tok|_token|_tokens|_pos|_bank|_slot|_layer|_row|_rows|_bytes|_job)$')
DECL_RE = re.compile(r'(?<![\w:.>])(?:const\s+)?' + BARE + r'\s+(?:const\s+)?(\w+)\s*(?=[,;)=\[]|$)')


def scan_decls():
    """Every bare integer parameter / field with a banned name in the listed headers: set of 'rel:name:context'."""
    found = []
    for rel in source_files(DECL_DIRS, ('.h', '.hpp', '.cuh')):
        if rel == UNITS_H or is_test_source(rel):
            continue
        text = open(os.path.join(ROOT, rel), errors='replace').read()
        blank = blank_comments_and_strings(text)
        for ln, line in enumerate(blank.split('\n'), 1):
            s = line.strip()
            if s.startswith('#'):
                continue
            for m in DECL_RE.finditer(line):
                name = m.group(1)
                if NAME_PAT.match(name) and not name.isupper():
                    found.append((rel, name, ln))
    return found


def load_allow():
    """The allowlist: {(rel, name): (count, reason)} from tests/units_decl_allow.txt."""
    allow = {}
    path = os.path.join(ROOT, ALLOW)
    if not os.path.exists(path):
        return allow
    for raw in open(path):
        line = raw.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split('|', 3)
        if len(parts) != 4 or not parts[3].strip():
            print('units-decl-gate: malformed allowlist line (want `file | name | count | reason`): ' + line)
            sys.exit(2)
        allow[(parts[0].strip(), parts[1].strip())] = (int(parts[2]), parts[3].strip())
    return allow


def gate_raw():
    untagged, _ = scan_raw()
    for rel, ln, s, calls in untagged[:60]:
        print(f'  {rel}:{ln}: {s}')
    if untagged:
        print(f'RAW GATE: FAIL ({len(untagged)} `.raw()` lines neither a log / format argument nor tagged '
              f'`/* raw: <reason> */`)')
        return 1
    print('RAW GATE: PASS (every .raw() in src outside the units header, the kernels and the tests is a log / '
          'format argument or a tagged boundary)')
    return 0


def gate_wrap():
    _, wraps = scan_raw()
    for rel, ln, s in wraps[:60]:
        print(f'  {rel}:{ln}: {s}')
    if wraps:
        print(f'WRAP GATE: FAIL ({len(wraps)} units constructed from another value\'s .raw() without a '
              f'`/* raw: <reason> */` named crossing)')
        return 1
    print('WRAP GATE: PASS (no Unit{x.raw()} outside pulsar_units.h but tagged named crossings)')
    return 0


def gate_decl():
    allow = load_allow()
    counts = {}
    where = {}
    for rel, name, ln in scan_decls():
        counts[(rel, name)] = counts.get((rel, name), 0) + 1
        where.setdefault((rel, name), ln)
    bad = []
    for key, n in sorted(counts.items()):
        cap = allow.get(key, (0, ''))[0]
        if n > cap:
            bad.append(f'  {key[0]}:{where[key]}: `{key[1]}` x{n} (allowlist: {cap})')
    stale = [k for k, (cap, _) in allow.items() if counts.get(k, 0) < cap]
    for line in bad[:60]:
        print(line)
    for k in stale[:60]:
        print(f'  allowlist entry {k[0]} | {k[1]} holds {allow[k][0]}, the tree has {counts.get(k, 0)}: '
              f'lower it (the list only shrinks)')
    if bad or stale:
        print(f'DECL GATE: FAIL ({len(bad)} new bare declarations named for a typed quantity, {len(stale)} '
              f'allowlist entries above the tree)')
        return 1
    print(f'DECL GATE: PASS ({sum(counts.values())} bare declarations, all on the allowlist ({len(allow)} entries))')
    return 0


def self_test():
    """Plant one violation of each kind in a scratch copy of a source and check that the scan reports it."""
    import tempfile
    rc = 0
    global ROOT
    real = ROOT
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'src/server'))
        os.makedirs(os.path.join(d, 'src/lib'))
        os.makedirs(os.path.join(d, 'tests'))
        open(os.path.join(d, UNITS_H), 'w').write(open(os.path.join(real, UNITS_H)).read())
        open(os.path.join(d, 'src/server/planted.cpp'), 'w').write(
            'static int f(KvRows r) { return (int)r.raw(); }\n'
            'static int g(KvRows r) { return (int)r.raw(); /* raw: planted, tagged */ }\n'
            'static void h(KvRows r) { fprintf(stderr, "%llu\\n",\n    (unsigned long long)r.raw()); }\n'
            'static KvBytes w(KvRows r) { return KvBytes{r.raw()}; }\n')
        open(os.path.join(d, 'src/server/planted.h'), 'w').write('struct p { int bank_index; };\n')
        ROOT = d
        global _format_calls
        _format_calls = None
        try:
            untagged, wraps = scan_raw()
            ok_raw = [x[1] for x in untagged] == [1, 5]
            ok_wrap = [x[1] for x in wraps] == [5]
            decls = scan_decls()
            ok_decl = [(x[0], x[1]) for x in decls] == [('src/server/planted.h', 'bank_index')]
            ok_gate = gate_raw() == 1 and gate_wrap() == 1 and gate_decl() == 1
        finally:
            ROOT = real
    for name, ok in (('raw', ok_raw), ('wrap', ok_wrap), ('decl', ok_decl), ('gates fail', ok_gate)):
        print(f'self-test {name}: {"ok" if ok else "FAILED"}')
        rc |= 0 if ok else 1
    return rc


def main(argv):
    what = argv[1:] or ['raw', 'wrap', 'decl']
    if what == ['--self-test']:
        return self_test()
    if what == ['--list-raw']:
        for rel, ln, s, calls in scan_raw()[0]:
            print(f'{rel}:{ln}:{"|".join(calls)}:{s}')
        return 0
    if what == ['--list-wrap']:
        for rel, ln, s in scan_raw()[1]:
            print(f'{rel}:{ln}:{s}')
        return 0
    if what == ['--list-decl']:
        for rel, name, ln in scan_decls():
            print(f'{rel}:{ln}:{name}')
        return 0
    rc = 0
    for w in what:
        rc |= {'raw': gate_raw, 'wrap': gate_wrap, 'decl': gate_decl}[w]()
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
