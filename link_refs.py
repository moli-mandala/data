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
import unicodedata
from collections import defaultdict

SUP = {"¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5"}
_TAGS = re.compile(r"<[^>]+>")
_SUPCH = re.compile(r"[¹²³⁴⁵]")
# the homograph superscript may sit after a stem hyphen, e.g. `<smallcaps>varta</smallcaps>-²`
_REF = re.compile(r"(√\s*\*?\s*)?<smallcaps>(.*?)</smallcaps>(-?)([¹²³⁴⁵]?)")


def _base(s):
    """Normalised headword key: strip markup, superscripts/numbers, edge punctuation; lowercase.
    Strip the Vedic pitch accents (acute udātta / grave anudātta) FROM VOWELS only — a reference
    usually omits them (`akṣa-²`) while the headword carries them (`akṣá²`). The SAME combining
    acute on a consonant is phonemic (ś, ć, ń — distinct letters, not pitch), so it must be kept:
    only strip acute/grave when it sits on a vowel. Other marks (macron, dot-below) are preserved."""
    s = html.unescape(_TAGS.sub("", s))
    s = re.sub(r"[¹²³⁴⁵\d]", "", s)
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"([aeiouAEIOU][̧̣̱̄̆]*)[́̀]", r"\1", s)
    s = unicodedata.normalize("NFC", s)
    return s.strip().strip("-–—*°,;. ").lower()


def _base_acc(s):
    """Like _base but KEEPS the pitch accent, so it can disambiguate accent-only homograph pairs
    (uṣṇá¹ vs uṣṇa²) when a reference actually carries the accent."""
    s = html.unescape(_TAGS.sub("", s))
    s = re.sub(r"[¹²³⁴⁵\d]", "", s)
    return s.strip().strip("-–—*°,;. ").lower()


def _sup(s):
    m = _SUPCH.search(html.unescape(s))
    return SUP[m.group(0)] if m else ""


# the headword's homograph superscript may sit inside OR just after the bold: `<b>varta</b>²`
_HEAD = re.compile(r"<b>(.*?)</b>\s*([¹²³⁴⁵]?)")

# an italic form inside an [etymology bracket]; a homograph superscript may trail it (`<i>akṣa</i>-²`)
_IREF = re.compile(r"<i>([^<]*?)</i>(-?)([¹²³⁴⁵]?)")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")


def _headword(desc):
    """(base, homograph-sup) of an entry's bold head-word, or None if it has none."""
    m = _HEAD.search(desc or "")
    if not m:
        return None
    return _base(m.group(1)), (_sup(m.group(1)) or SUP.get(m.group(2), ""))


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
        key = _headword(p.get("Description") or "")
        if not key:
            continue
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
    """headword-base → {homograph: id}, base → {all ids}, and accent-preserving base → {ids}.
    Returns a resolve(base, sup, acc) fn (acc = the accent-preserving reference base, optional)."""
    byhom = defaultdict(dict)
    bybase = defaultdict(set)
    byacc = defaultdict(set)
    for p in param_rows:
        if p["ID"] in skip:
            continue
        m = _HEAD.search(p.get("Description") or "")
        if not m:
            continue
        b = _base(m.group(1))
        if not b:
            continue
        h = _sup(m.group(1)) or SUP.get(m.group(2), "")
        byhom[b].setdefault(h, p["ID"])
        bybase[b].add(p["ID"])
        byacc[_base_acc(m.group(1))].add(p["ID"])

    def resolve(base, sup, acc=None):
        if not base or base not in bybase:
            return None
        ids = bybase[base]
        if sup and sup in byhom[base]:
            return byhom[base][sup]  # homograph-disambiguated by superscript
        # if the reference carries an accent (or lacks one) and that pins exactly one candidate,
        # use it — the accent disambiguates uṣṇá¹ from uṣṇa² without a superscript
        if acc and len(byacc.get(acc, ())) == 1:
            only = next(iter(byacc[acc]))
            if only in ids:
                return only
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
        root, content, hyph, trail = m.group(1), m.group(2), m.group(3), m.group(4)
        tail = hyph + trail  # re-emitted verbatim so display (e.g. "-²") is preserved
        if "data-entry" in content:
            return m.group(0)  # already linked
        if root:
            # a √root reference — link it to its root entry if we have one
            rid = root_map.get((_base(content), SUP.get(trail, ""))) if root_map else None
            if rid:
                n[0] += 1
                return f'{root}<smallcaps><a data-entry="{rid}">{content}</a></smallcaps>{tail}'
            return m.group(0)
        # references inside one <smallcaps> are separated by commas, colons (`X-: √root`), or an
        # em-dash aside (`samakṣá-. — sa-², ákṣi-`) — split on all three so each ref resolves.
        parts = re.split(r"(,\s*|:\s*|\s*—\s*)", content)
        real = [
            i for i, t in enumerate(parts) if t.strip() and not re.fullmatch(r"[,:]\s*|\s*—\s*", t)
        ]
        last = real[-1] if real else -1
        out = []
        for i, tok in enumerate(parts):
            if i not in real:
                out.append(tok)
                continue
            rm = re.match(r"(\s*√\s*\*?\s*)(.+)", tok)  # an inline √root reference
            if rm and root_map:
                rid = root_map.get((_base(rm.group(2)), _sup(tok)))
                if rid:
                    n[0] += 1
                    out.append(f'{rm.group(1)}<a data-entry="{rid}">{rm.group(2)}</a>')
                    continue
            b = _base(tok)
            sup = _sup(tok) or (SUP.get(trail, "") if i == last else "")
            eid = resolve(b, sup, _base_acc(tok))
            if eid:
                n[0] += 1
                out.append(f'<a data-entry="{eid}">{tok}</a>')
            else:
                out.append(tok)
        return (root or "") + "<smallcaps>" + "".join(out) + "</smallcaps>" + tail

    linked, count = _REF.sub(repl, desc), n[0]

    # Also link italic head-word references inside [etymology brackets] to their entry, best-effort:
    # only unambiguous resolutions are linked (inflected examples that don't resolve stay plain).
    def bracket(bm):
        def irepl(im):
            form = im.group(1)
            if not form.strip() or "data-entry" in form:
                return im.group(0)
            eid = resolve(_base(form), _sup(im.group(0)), _base_acc(form))
            if eid:
                n[0] += 1
                return f'<i><a data-entry="{eid}">{form}</a></i>{im.group(2)}{im.group(3)}'
            return im.group(0)
        return "[" + _IREF.sub(irepl, bm.group(1)) + "]"

    if "[" in linked:
        n[0] = count
        linked = _BRACKET.sub(bracket, linked)
        count = n[0]
    return linked, count


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
                # "[Cf. …aside… : X-, Y-]": a colon likewise splits the see-also from the real
                # ancestry that follows it; otherwise it's a pure see-also with nothing to record.
                if ":" in b:
                    frag = b.rsplit(":", 1)[-1]
                else:
                    continue
            # italic references (`<i>…</i>`) inside a bracket are cross-links only, never ancestry —
            # only the <smallcaps> ancestry refs become derivation edges. Drop italics first.
            frag = re.sub(r"<i>.*?</i>", "", frag)
            for eid in re.findall(r'data-entry="([^"]+)"', frag):
                if eid != p["ID"] and (p["ID"], eid) not in seen:
                    seen.add((p["ID"], eid))
                    edges.append((p["ID"], eid))
    return edges


ROOT_REF = re.compile(r"√\s*\*?\s*<smallcaps>(.*?)</smallcaps>([¹²³⁴⁵]?)")


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
            # roots also appear *inside* a <smallcaps> span, separated by a comma, colon, or
            # em-dash aside — e.g. `<smallcaps>X-: √root</smallcaps>` or `<smallcaps>X¹. — √root`
            for sc in re.findall(r"<smallcaps>(.*?)</smallcaps>", b):
                for tok in re.split(r"[,:—]", sc):
                    rm = re.match(r"\s*√\s*\*?\s*(.+)", tok)
                    if not rm:
                        continue
                    base = _base(rm.group(1))
                    if not base:
                        continue
                    scm = _SUPCH.search(tok)
                    supch = scm.group(0) if scm else ""
                    key = (base, SUP.get(supch, ""))
                    disp.setdefault(key, base + supch)
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
