"""Infer CDIAL reflexes that continue a pleonastically extended stem (-kk-, -ḍ-, -l-, -r-) but
sit on the bare etymon because Turner did not split them into an "ext. -X-" section.

The signal is purely surface shape: a NIA reflex whose consonant skeleton equals that of an
unextended sibling under the same etymon (a *witness*) plus one trailing consonant from the
extension set. Citation endings (-nā, -ṇu, -iki, …) are stripped first, and a hit is rejected
when the etymon itself already carries that consonant class after its first consonant (then the
extra consonant is cluster residue, e.g. káḍāra → G. karāṛⁱ).

Metrics come from Turner's own labels. Reflexes in an "ext. -X-" section are positives (the
detector is run on them as if they sat on the base); reflexes that Turner left on the base of an
entry that HAS an extension section of the same class are the negative proxy (he was attending
to extensions in that entry and did not move them). Tiers:

    same-language   the witness is in the reflex's own language
    other-language  the witness is another NIA language
    etymon          the only witness is the OIA head-word itself

Only tiers listed in APPLY_TIERS are written to data/cdial/inferred-extensions.csv, which
unify_cldf.py applies (rows are re-homed to the entry's extension node, created if needed, and
tagged ``ext:<morph> inferred``). Regenerate with ``make inferred-extensions``; run with
``--metrics`` to print the evaluation only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from unify_cldf import ext_morpheme, is_derivation_section, section_kind, strip_accent  # noqa: E402

ROOT = Path(__file__).parent
CDIAL_CSV = ROOT / "data/cdial/cdial.csv"
PARAMS_CSV = ROOT / "data/cdial/params.csv"
OUT_CSV = ROOT / "data/cdial/inferred-extensions.csv"
OUT_COLS = ["Row", "Language_ID", "Parameter_ID", "Original", "Cognateset", "Morpheme", "Class",
            "Witness_Language", "Witness", "Tier"]  # Row = 0-based line in cdial.csv (its legacy id)

APPLY_TIERS = ("same-language",)

# old / literary stages: never candidates, never witnesses (their shapes predate the extensions)
OLD = {
    "Indo-Aryan", "MIA", "Pa", "Pk", "OIA", "Sk", "OSi", "NiDoc", "Ap", "Aś", "KharI", "Dhp",
    "OAw", "OMarw", "OG", "OM", "OB", "OH", "OMth", "OP", "OK", "OOr", "OA", "OBi", "OL", "OS",
}
# section headers whose forms are morphologically something else (derivatives, causatives,
# compounds, inflected sub-paradigms): not eligible as candidates
_INELIGIBLE = re.compile(
    r"(?i)deriv|possess|compar|caus|cmpd|comp\.|adj\.|fem|\bpl\b|obl|pres|\bpp\b|redup|\bX\b|ppp|inf\."
)

VOWELS = set("aāiīuūeēoōɛɔəʌɪʊæεᵘⁱᵃᵒᵉ")  # incl. Kashmiri superscript finals
# consonant classes; the extension set is K T L R
CLS = {
    "k": "K", "g": "K", "q": "K", "x": "K", "ɣ": "K", "ɠ": "K",
    "ṭ": "T", "ḍ": "T", "ṛ": "T", "ᶑ": "T", "ɖ": "T", "ɽ": "T", "ʈ": "T",
    "l": "L", "ḷ": "L", "ɭ": "L",
    "r": "R", "ṟ": "R",
    "t": "t", "d": "t", "n": "N", "ṇ": "N", "ñ": "N", "ṅ": "N", "ŋ": "N", "ɳ": "N", "m": "M",
    "c": "C", "j": "C", "ʦ": "C", "ʣ": "C", "č": "C", "ǰ": "C", "ʧ": "C", "ʤ": "C",
    "ś": "S", "ṣ": "S", "s": "S", "z": "S", "š": "S", "ž": "S", "ʂ": "S", "ʃ": "S",
    "h": "H", "v": "W", "w": "W", "y": "Y", "p": "P", "b": "P", "f": "P", "ɓ": "P", "β": "P",
}
EXT_CLASSES = {"K", "T", "L", "R"}
# the morpheme an inferred node is named after, from the reflex's actual final consonant
MORPHEME = {
    "k": "kk", "g": "kk", "q": "kk", "x": "kk", "ɣ": "kk", "ɠ": "kk",
    "ṭ": "ṭṭ", "ʈ": "ṭṭ", "ḍ": "ḍ", "ṛ": "ḍ", "ᶑ": "ḍ", "ɖ": "ḍ", "ɽ": "ḍ",
    "l": "l", "ḷ": "l", "ɭ": "l", "r": "r", "ṟ": "r",
}
# citation endings that add a consonant (verb infinitives, -nu/-ṇu, Shina/Khowar -iki/-ik …)
ENDINGS = sorted(
    ["ṇā", "nā", "ṇu", "ṇo", "nu", "no", "vũ", "ivũ", "ibā", "ibō", "ibo", "bā", "ba", "anu",
     "ṇē", "iba", "iva", "uṇ", "aṇ", "ṇ", "n", "va", "vā", "oiki", "iki", "ik", "aiki", "ōnu",
     "unu", "un", "ab", "eb"],
    key=len, reverse=True,
)


def segments(word: str) -> list[str]:
    """Letters of a CDIAL form with diacritics attached; ``kh`` etc. collapse to the stop."""
    w = strip_accent(unicodedata.normalize("NFC", word)).lstrip("*")
    w = re.sub(r"<[^>]+>|[-°()\[\]?~,.;:!ʼ'‘’]", "", w)
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", w):
        if unicodedata.combining(ch) or ch in "ʰʱ":
            if out:
                out[-1] += ch
            continue
        out.append(ch)
    out = [unicodedata.normalize("NFC", s) for s in out]
    merged: list[str] = []
    for s in out:
        if s == "h" and merged and merged[-1][0] in "kgcjṭḍtdpb":
            continue  # aspiration digraph
        merged.append(s)
    return merged


def _consonant_class(seg: str) -> str | None:
    base = unicodedata.normalize("NFD", seg)[0]
    if base in VOWELS or seg in ("ṁ", "ṃ", "ⁿ", "̃"):
        return None
    return CLS.get(seg) or CLS.get(base) or "?"


def skeleton(word: str) -> list[str]:
    """Consonant classes of a form, geminates (same class, no vowel between) collapsed,
    aspiration and voicing ignored."""
    out: list[str] = []
    after_vowel = True
    for s in segments(word):
        c = _consonant_class(s)
        if c is None:
            after_vowel = True
            continue
        if c != "?" and (after_vowel or not out or out[-1] != c):
            out.append(c)
        after_vowel = False
    return out


def final_consonant(word: str) -> str | None:
    """The last consonant letter of a form (after ending stripping), for naming the morpheme."""
    last = None
    for s in segments(word):
        if _consonant_class(s) not in (None, "?"):
            last = unicodedata.normalize("NFC", "".join(
                ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch) or ch == "\u0323"
            ))
    return last


def strip_ending(word: str) -> str:
    for e in ENDINGS:
        if word.endswith(e) and len(word) > len(e) + 1:
            return word[: -len(e)]
    return word


def etymon_stem(head: str) -> str:
    w = strip_accent(head).lstrip("*")
    return re.sub(r"(ati|āti|ayati|ayatē|atē|ti|ana|aka|ika)$", "", w) if len(w) > 4 else w


def detect(form: str, witnesses: list[tuple[str, str]], head: str, lang: str):
    """→ (class, witness_lang, witness_form, tier) or None.

    ``witnesses`` are (language, form) pairs of unextended siblings homed on the same head."""
    if " " in form or "-" in form.strip("-"):
        return None
    sk = skeleton(strip_ending(form))
    if len(sk) < 2 or sk[-1] not in EXT_CLASSES:
        return None
    cls, want = sk[-1], sk[:-1]
    head_sk = skeleton(etymon_stem(head))
    if cls in head_sk[1:]:
        return None  # the etymon already has this class: cluster residue, not an extension
    best = None
    for wl, wf in witnesses:
        if " " in wf or wl in OLD:
            continue
        if skeleton(strip_ending(wf)) == want:
            if wl == lang:
                return (cls, wl, wf, "same-language")
            if best is None:
                best = (cls, wl, wf, "other-language")
    if best:
        return best
    if head_sk == want:
        return (cls, "Indo-Aryan", head, "etymon")
    return None


def load_entries():
    """CDIAL rows grouped by entry, each row annotated with its home: ('base',), ('form', n),
    ('ext', morpheme) or ('other',), mirroring unify_cldf's section logic."""
    heads = {}
    with open(PARAMS_CSV, newline="", encoding="utf-8") as fh:
        for pid, name, *_ in csv.reader(fh):
            heads[pid] = name
    entries = collections.defaultdict(list)
    with open(CDIAL_CSV, newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.reader(fh)):
            entries[r[1]].append(r + [i])  # the row index is the row's legacy id in make_cldf
    annotated = {}
    for pid, rows in entries.items():
        last_num = 1
        out = []
        # pronoun / particle / numeral heads inflect and suffix too freely for a shape test
        head_note = next((r[6] for r in rows if r[0] == "Indo-Aryan"), "")
        if re.match(r"(?i)\s*(pron|adv|prep|postp|conj|interj|num|indecl)\b", head_note):
            annotated[pid] = (heads.get(pid, ""), [(r, ("other",), "") for r in rows])
            continue
        for r in rows:
            cog = r[8]
            info = cog.split(":", 1)[1] if ":" in cog else ""
            kind, suffix, _tag = section_kind(info)
            if kind == "ext":
                home = ("ext", suffix)
            elif kind:
                home = ("other",)
            elif info.isdigit():
                last_num = int(info)
                home = ("base",) if last_num == 1 else ("form", last_num)
            elif is_derivation_section(info):
                # Generic derivatives reset the numbered-form context, but are not
                # unextended base witnesses or candidates for surface-shape inference.
                last_num = 1
                home = ("other",)
            elif info == "":
                last_num = 1
                home = ("base",)
            elif _INELIGIBLE.search(re.sub(r"<[^>]+>", "", info)) or re.search(r"(?i)in place of|^with\s+<i>|^replaced", info):
                home = ("other",)
            else:
                home = ("base",) if last_num == 1 else ("form", last_num)
            out.append((r, home, info))
        annotated[pid] = (heads.get(pid, ""), out)
    return annotated


def run(entries, evaluate: bool):
    """Yield candidate dicts; when ``evaluate`` also return the metrics."""
    results = []
    stats = collections.defaultdict(collections.Counter)
    for pid, (head, rows) in entries.items():
        base = [(r[0], r[2]) for r, home, _ in rows if home == ("base",) and r[0] not in OLD]
        ext_classes = {skeleton(m)[-1] if skeleton(m) else None for _, home, _ in rows if home[0] == "ext" for m in [home[1]]}
        for r, home, info in rows:
            lang, form = r[0], r[2]
            if lang in OLD:
                continue
            if home == ("base",):
                witnesses = [w for w in base if w != (lang, form)]
                hit = detect(form, witnesses, head, lang)
                if hit:
                    cls, wl, wf, tier = hit
                    results.append({
                        "Row": r[-1], "Language_ID": lang, "Parameter_ID": pid, "Original": form,
                        "Cognateset": r[8], "Morpheme": MORPHEME.get(final_consonant(strip_ending(form)) or "", ""),
                        "Class": cls, "Witness_Language": wl, "Witness": wf, "Tier": tier,
                    })
                    if evaluate and cls in ext_classes:
                        stats[tier]["neg_fp"] += 1  # Turner kept it on the base despite an ext section
                if evaluate and ext_classes:
                    for c in ext_classes:
                        stats["_"]["neg_" + str(c)] += 1
            elif evaluate and home[0] == "ext":
                msk = skeleton(home[1])
                if not msk or msk[-1] not in EXT_CLASSES:
                    continue
                truth = msk[-1]
                stats["_"]["pos"] += 1
                hit = detect(form, base, head, lang)
                if hit:
                    tier = hit[3]
                    stats[tier]["pos_hit"] += 1
                    stats[tier]["pos_tp" if hit[0] == truth else "pos_wrong"] += 1
    return results, stats


def print_metrics(results, stats):
    pos = stats["_"]["pos"]
    print(f"positives (reflexes in Turner's ext. sections): {pos}")
    print(f"negative proxy (base reflexes in entries with an ext. section): "
          f"{sum(v for k, v in stats['_'].items() if k.startswith('neg_'))} class-slots")
    print(f"{'tier':15} {'cands':>6} {'recall':>7} {'class-acc':>9} {'proxy-FP':>9}")
    for tier in ("same-language", "other-language", "etymon"):
        n = sum(1 for r in results if r["Tier"] == tier)
        hit, tp = stats[tier]["pos_hit"], stats[tier]["pos_tp"]
        fp = stats[tier]["neg_fp"]
        acc = f"{tp / hit:.2f}" if hit else "-"
        print(f"{tier:15} {n:6} {hit / pos:7.2f} {acc:>9} {fp:9}")
    # precision estimate: hits in entries where Turner labelled the class, split by whether he
    # put the reflex in the extension section (tp) or left it on the base (proxy fp)
    for tier in APPLY_TIERS:
        tp, fp = stats[tier]["pos_tp"], stats[tier]["neg_fp"]
        if tp + fp:
            print(f"estimated precision of {tier}: {tp / (tp + fp):.2f} ({tp} labelled hits vs {fp} left on base)")
    by_class = collections.Counter(r["Class"] for r in results if r["Tier"] in APPLY_TIERS)
    print("applied tiers:", ", ".join(APPLY_TIERS), "| by class:", dict(by_class))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", action="store_true", help="evaluate only; do not write the table")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    entries = load_entries()
    results, stats = run(entries, evaluate=True)
    print_metrics(results, stats)
    if args.metrics:
        return
    applied = [r for r in results if r["Tier"] in APPLY_TIERS]
    applied.sort(key=lambda r: (int(re.sub(r"\D", "", r["Parameter_ID"]) or 0), r["Parameter_ID"], r["Language_ID"], r["Original"]))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(applied)
    print(f"wrote {len(applied)} rows to {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
