"""
tags.py — lift structured tokens out of the free-text `Description` (notes) field.

CDIAL notes pack a leading, semicolon-delimited run of structured tokens (gender, grammatical
category, and attestation loci) ahead of any prose. This module separates those into a `Tags` field
so `Description` keeps only free text (etymological cross-references, source citations, etc.).

Conservative by design: a ";"-delimited field is only lifted when it consists ENTIRELY of known
tokens, so prose is never mangled. Extracts:
  • gender:        m, f, n  (and combos mn, fn, mf)
  • grammatical:   part of speech, valency/voice, number, case, verb forms
  • source:        attestation loci — every Sanskrit work abbreviation in sanskrit.txt, plus a few
                   dictionaries/lexicographers; a cited work also contributes an ERA tag
                   (Early-Vedic / Late-Vedic / Epic / Classical / Medieval) from sanskrit_works.tsv.
"""

import html
import os
import re

GENDER_TAGS = {"m", "f", "n", "mn", "fn", "mf"}
GRAMMATICAL_TAGS = {
    # valency / voice
    "tr", "intr", "caus", "pass", "refl", "denom",
    # number
    "sg", "pl", "du",
    # part of speech
    "adj", "adv", "pron", "num", "postp", "prep", "conj", "interj", "part", "indecl", "ord",
    # case
    "nom", "acc", "dat", "gen", "loc", "abl", "instr", "voc", "obl",
    # verb forms
    "pp", "ppp", "pres", "fut", "inf", "impv", "ind", "ger", "verb",
    # Tamil verb morphology
    "weak", "middle", "strong", "Tamil-class-1", "Tamil-class-2", "Tamil-class-3",
    "Tamil-class-4", "Tamil-class-5", "Tamil-class-6", "Tamil-class-7",
}

# Non-work attestation sources kept explicitly: dictionaries / lexicographers not listed as
# individual works in sanskrit.txt.
_EXTRA_SOURCES = {"MW", "Apte", "W", "Gal", "Cat", "lex", "DNM", "Uṇ", "BHSk", "Bhpr", "Naigh"}


def _load_works():
    """(work abbreviations, {abbrev: era-tag}) from sanskrit.txt + sanskrit_works.tsv (data dir)."""
    here = os.path.dirname(os.path.abspath(__file__))
    abbrevs = set()
    try:
        with open(os.path.join(here, "sanskrit.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() and " " in line:
                    abbrevs.add(line.split(" ", 1)[0].rstrip("."))
    except FileNotFoundError:
        pass
    era = {}
    try:
        with open(os.path.join(here, "sanskrit_works.tsv"), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[0]:
                    era[parts[0]] = parts[1]
    except FileNotFoundError:
        pass
    return abbrevs, era


_WORK_ABBREVS, WORK_ERA = _load_works()
# Attestation sources keep their case (RV, MBh, ŚBr). The dotted per-language dialect codes
# (S., F., Mu.) are excluded because _classify strips only a single trailing dot then matches.
SOURCE_TAGS = _EXTRA_SOURCES | _WORK_ABBREVS
ERA_TAGS = set(WORK_ERA.values())

_ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#\d+|#x[0-9a-fA-F]+);")
_HOLE = re.compile(r"\x00(\d+)\x00")
_TAGS = re.compile(r"<[^>]+>")


def _split_fields(note):
    """Split on ';' WITHOUT breaking HTML entities (e.g. `&lt;`, which contain a ';')."""
    ents = []

    def stash(m):
        ents.append(m.group(0))
        return f"\x00{len(ents) - 1}\x00"

    protected = _ENTITY.sub(stash, note)
    restore = lambda s: _HOLE.sub(lambda m: ents[int(m.group(1))], s)
    return [restore(p) for p in protected.split(";")]


def _classify(field):
    """Tag list for a field if it is ENTIRELY gender/grammatical/source tokens, else None."""
    plain = html.unescape(_TAGS.sub("", field)).strip()
    toks = plain.split()
    if not toks:
        return None
    out = []
    for tok in toks:
        base = tok.rstrip(".")
        if base in GENDER_TAGS:
            out.append(base)
        elif base.lower() in GRAMMATICAL_TAGS:
            out.append(base.lower())
        elif base in SOURCE_TAGS:
            out.append(base)  # sources keep their case (RV, MBh, ŚBr)
        else:
            return None
    return out


def _category(tag):
    if tag in GENDER_TAGS:
        return 0
    if tag in GRAMMATICAL_TAGS:
        return 1
    if tag in ERA_TAGS:
        return 3
    return 2  # attestation source


def extract_tags(note):
    """(tags, cleaned_notes): `tags` is a space-separated list (gender, grammatical, source, era);
    `cleaned_notes` keeps every field that was not purely structured tokens."""
    if not note:
        return "", note or ""
    kept, tags = [], []
    for field in _split_fields(note):
        if not field.strip():
            continue
        cls = _classify(field)
        if cls is None:
            kept.append(field.strip())
        else:
            tags += cls
    # a cited Sanskrit work also contributes the era of that work
    for t in list(tags):
        e = WORK_ERA.get(t)
        if e:
            tags.append(e)
    seen = set()
    ordered = [t for t in tags if not (t in seen or seen.add(t))]
    ordered.sort(key=_category)  # gender, grammatical, source, then era
    return " ".join(ordered), "; ".join(kept)
