"""
link_refs.py — resolve <smallcaps> cross-references in the CLDF descriptions to entry IDs and wrap
them as `<a data-entry="ID">…</a>` markers (the webapp turns data-entry into a real href).

CDIAL marks referenced headwords with <smallcaps>; a trailing superscript (¹²³) disambiguates
homographs. Roots (√<smallcaps>…</smallcaps>) are NOT entries and are left untouched. A reference
is only linked when it resolves to exactly one entry (via the superscript when needed), so we never
create a wrong link. Idempotent: already-linked spans are skipped.

Run AFTER make_cldf.py:  python link_refs.py
"""

import csv
import html
import re
import sys
from collections import defaultdict

SUP = {"¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5"}
_TAGS = re.compile(r"<[^>]+>")
_SUPCH = re.compile(r"[¹²³⁴⁵]")
_REF = re.compile(r"(√\s*)?<smallcaps>(.*?)</smallcaps>([¹²³⁴⁵]?)")


def _base(s):
    """Normalised headword key: strip markup, superscripts/numbers, edge punctuation; lowercase."""
    s = html.unescape(_TAGS.sub("", s))
    s = re.sub(r"[¹²³⁴⁵\d]", "", s)
    return s.strip().strip("-–—*°,;. ").lower()


def _sup(s):
    m = _SUPCH.search(html.unescape(s))
    return SUP[m.group(0)] if m else ""


def build_resolver(param_rows):
    """headword-base → {homograph: id}, plus base → {all ids}. Returns a resolve(base, sup) fn."""
    byhom = defaultdict(dict)
    bybase = defaultdict(set)
    for p in param_rows:
        m = re.search(r"<b>(.*?)</b>", p.get("Description") or "")
        if not m:
            continue
        b, h = _base(m.group(1)), _sup(m.group(1))
        if not b:
            continue
        byhom[b].setdefault(h, p["ID"])
        bybase[b].add(p["ID"])

    def resolve(base, sup):
        if not base or base not in bybase:
            return None
        ids = bybase[base]
        if sup and sup in byhom[base]:
            return byhom[base][sup]  # homograph-disambiguated
        if len(ids) == 1:
            return next(iter(ids))  # unambiguous
        return None  # ambiguous and undisambiguated → leave unlinked

    return resolve


def linkify(desc, resolve):
    """Wrap resolvable <smallcaps> references in <a data-entry>. Returns (new_desc, n_linked)."""
    if not desc or "<smallcaps>" not in desc:
        return desc, 0
    n = [0]

    def repl(m):
        root, content, trail = m.group(1), m.group(2), m.group(3)
        if root or "data-entry" in content:
            return m.group(0)  # a root, or already linked
        parts = re.split(r"(,\s*)", content)  # keep the separators
        real = [i for i, t in enumerate(parts) if t.strip() and not re.fullmatch(r",\s*", t)]
        last = real[-1] if real else -1
        out = []
        for i, tok in enumerate(parts):
            if i not in real:
                out.append(tok)
                continue
            b = _base(tok)
            sup = _sup(tok) or (trail if i == last else "")
            eid = resolve(b, sup)
            if eid:
                n[0] += 1
                out.append(f'<a data-entry="{eid}">{tok}</a>')
            else:
                out.append(tok)
        return (root or "") + "<smallcaps>" + "".join(out) + "</smallcaps>" + trail

    return _REF.sub(repl, desc), n[0]


def process(path, resolve, col="Description"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0
    fields = list(rows[0].keys())
    total = 0
    for r in rows:
        r[col], k = linkify(r.get(col, ""), resolve)
        total += k
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return total, len(rows)


def main():
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    resolve = build_resolver(params)
    for path in ("cldf/parameters.csv", "cldf/forms.csv"):
        n, rows = process(path, resolve)
        print(f"{path}: linked {n} references across {rows} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
