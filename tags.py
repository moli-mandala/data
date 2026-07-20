"""
tags.py — lift structured tokens out of the free-text `Description` (notes) field.

CDIAL notes pack a leading, semicolon-delimited run of structured tokens (gender, grammatical
category) ahead of any prose. This module separates those into a `Tags` field so `Description`
keeps only free text (etymological cross-references, source citations, etc.).

Conservative by design: a ";"-delimited field is only lifted when it consists ENTIRELY of known
tokens, so prose is never mangled. Currently extracts:
  • gender:        m, f, n  (and combos mn, fn, mf)
  • grammatical:   part of speech, valency/voice, number, case, verb forms
"""

import html
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
    "pp", "ppp", "pres", "fut", "inf", "impv", "ind", "ger",
}

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
    """Return the tag list for a field if it is ENTIRELY gender/grammatical tokens, else None."""
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
        else:
            return None
    return out


def extract_tags(note):
    """(tags, cleaned_notes): `tags` is a space-separated list (gender first, then grammatical);
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
    seen = set()
    ordered = [t for t in tags if not (t in seen or seen.add(t))]
    ordered.sort(key=lambda t: 0 if t in GENDER_TAGS else 1)
    return " ".join(ordered), "; ".join(kept)
