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


ADDENDA_LO, ADDENDA_HI = 14190, 14845


def _is_addendum(pid):
    return pid.isdigit() and ADDENDA_LO <= int(pid) <= ADDENDA_HI


def compute_merges(param_rows):
    """Match each addendum to the main entry it supplements by head-word (same base + homograph
    superscript, exactly one non-addendum candidate). Returns {addendum id -> main id}. These get
    folded together downstream; for reference resolution the addendum is not a separate homograph."""
    mains = defaultdict(list)  # (base, sup) -> [non-addendum ids]
    adds = []  # (addendum id, key)
    for p in param_rows:
        m = re.search(r"<b>(.*?)</b>", p.get("Description") or "")
        if not m:
            continue
        key = (_base(m.group(1)), _sup(m.group(1)))
        if not key[0]:
            continue
        if _is_addendum(p["ID"]):
            adds.append((p["ID"], key))
        else:
            mains[key].append(p["ID"])
    merges = {}
    for aid, key in adds:
        cands = mains.get(key, [])
        if len(cands) == 1:
            merges[aid] = cands[0]
    return merges


def build_resolver(param_rows, skip=frozenset()):
    """headword-base → {homograph: id}, plus base → {all ids}. Returns a resolve(base, sup) fn."""
    byhom = defaultdict(dict)
    bybase = defaultdict(set)
    for p in param_rows:
        if p["ID"] in skip:
            continue
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


def linkify(desc, resolve, root_map=None):
    """Wrap resolvable <smallcaps> references in <a data-entry>. Returns (new_desc, n_linked).
    `root_map` (base, sup)→root-id links √root references to their synthesised root entries."""
    if not desc or "<smallcaps>" not in desc:
        return desc, 0
    n = [0]

    def repl(m):
        root, content, trail = m.group(1), m.group(2), m.group(3)
        if "data-entry" in content:
            return m.group(0)  # already linked
        if root:
            # a √root reference — link it to its root entry if we have one
            rid = root_map.get((_base(content), SUP.get(trail, ""))) if root_map else None
            if rid:
                n[0] += 1
                return f'{root}<smallcaps><a data-entry="{rid}">{content}</a></smallcaps>{trail}'
            return m.group(0)
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


def extract_derivations(param_rows):
    """Edges (child derived-term → parent/ancestor etymon) from the etymology brackets in headers.
    Only bare ancestry brackets count: `[<X->]` / `[<X->, <Y->]` — NOT `[√root]` and NOT
    `[Cf. …]` (see-also). Relies on the descriptions already being linkified (data-entry markers)."""
    edges = []
    seen = set()
    for p in param_rows:
        for b in re.findall(r"\[([^\[\]]*)\]", p.get("Description") or ""):
            plain = html.unescape(_TAGS.sub("", b)).strip()
            if plain.startswith("√"):
                continue
            frag = b
            # "[Cf. …commentary… — X-, Y-]": the em-dash splits a see-also aside from the actual
            # ancestry that follows it — parse only the part after the dash.
            if "—" in b:
                frag = b.rsplit("—", 1)[-1]
            elif re.match(r"Cf\.", plain):
                continue  # pure see-also, no ancestry to record
            for eid in re.findall(r'data-entry="([^"]+)"', frag):
                if eid != p["ID"] and (p["ID"], eid) not in seen:
                    seen.add((p["ID"], eid))
                    edges.append((p["ID"], eid))
    return edges


ROOT_REF = re.compile(r"√\s*<smallcaps>(.*?)</smallcaps>([¹²³⁴⁵]?)")


def extract_roots(param_rows):
    """CDIAL cites verbal roots as `[√<smallcaps>X</smallcaps>]` inside the etymology bracket.
    Roots are not attested entries, so we synthesise one node per distinct root (base + optional
    homograph superscript) and record a derivation edge from every citing entry to it — the root
    then surfaces its citing entries as 'derived terms'. Returns (root_param_rows, edges)."""
    occ = defaultdict(list)  # (base, sup) -> [citing entry ids]
    disp = {}  # (base, sup) -> display string ("aś¹")
    for p in param_rows:
        for b in re.findall(r"\[([^\[\]]*)\]", p.get("Description") or ""):
            for m in ROOT_REF.finditer(b):
                base = _base(m.group(1))
                if not base:
                    continue
                supch = m.group(2)
                key = (base, SUP.get(supch, ""))
                disp[key] = base + supch
                occ[key].append(p["ID"])
    keys = sorted(occ)
    rid = {k: f"r{i}" for i, k in enumerate(keys, 1)}
    edges, seen = [], set()
    for k in keys:
        for eid in occ[k]:
            if (eid, rid[k]) not in seen:
                seen.add((eid, rid[k]))
                edges.append((eid, rid[k]))
    root_params = [
        {
            "ID": rid[k],
            "Name": "√" + disp[k],
            "Language_ID": "Indo-Aryan",
            "Description": f"<html><body><b>√{disp[k]}</b></body></html>",
            "Etyma": "",
        }
        for k in keys
    ]
    return root_params, edges, rid  # rid: (base, sup) -> root id, for linking √ refs


def process(path, resolve, root_map=None, col="Description"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0
    fields = list(rows[0].keys())
    total = 0
    for r in rows:
        r[col], k = linkify(r.get(col, ""), resolve, root_map)
        total += k
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return total, len(rows)


def main():
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    fields = list(params[0].keys())
    # addenda folded into a main entry aren't separate homographs for reference resolution
    merges = compute_merges(params)  # addendum id -> main id
    resolve = build_resolver(params, skip=set(merges))
    with open("cldf/merges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Addendum_ID", "Main_ID"])
        w.writerows(sorted(merges.items()))
    print(f"cldf/merges.csv: {len(merges)} addenda → main entries", file=sys.stderr)
    # root entries + the (base, sup)→root-id map so √root references get linked in the text too
    root_params, root_edges, root_map = extract_roots(params)

    for path in ("cldf/parameters.csv", "cldf/forms.csv"):
        n, rows = process(path, resolve, root_map)
        print(f"{path}: linked {n} references across {rows} rows", file=sys.stderr)

    # append the synthesised √root entries to parameters.csv
    if root_params:
        with open("cldf/parameters.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(root_params)

    # derivation graph: child derived-term → parent etymon (from ancestry brackets) + root edges
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))  # re-read the now-linkified descriptions
    edges = extract_derivations(params) + root_edges
    # merged addenda fold into their main entry — re-point any edge touching one, drop self-loops
    deduped, seen = [], set()
    for c, pa in edges:
        c, pa = merges.get(c, c), merges.get(pa, pa)
        if c != pa and (c, pa) not in seen:
            seen.add((c, pa))
            deduped.append((c, pa))
    edges = deduped

    with open("cldf/derivation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Child_ID", "Parent_ID"])
        w.writerows(edges)
    kids = len({c for c, _ in edges})
    print(
        f"cldf/derivation.csv: {len(edges)} edges ({len(root_edges)} to {len(root_params)} roots), "
        f"{kids} derived-term entries",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
