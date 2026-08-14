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

from dialects import dialect_tag

GENDER_TAGS = {"m", "f", "n", "mn", "fn", "mf"}
GRAMMATICAL_TAGS = {
    # valency / voice
    "tr", "intr", "caus", "pass", "refl", "denom",
    # number
    "sg", "pl", "du", "double-plural",
    # part of speech
    "noun", "adj", "adv", "pron", "num", "postp", "prep", "conj", "interj", "part", "indecl", "ord",
    # case
    "nom", "acc", "dat", "gen", "loc", "abl", "instr", "voc", "obl",
    # verb forms
    "pp", "ppp", "pres", "fut", "inf", "impv", "ind", "ger", "verb",
    "ipfv", "pfv", "neg", "participle", "conjunctive-participle",
    "subj", "obj", "direct-object", "indirect-object",
    "abs", "erg", "ade", "ine", "ess",
    "prox", "dist", "indef", "finalis",
    "poss", "conditional", "prefix", "suffix", "emph", "interr", "dir", "uncertain",
    "1sg", "2sg", "3sg", "1pl", "2pl", "3pl",
    "pret", "aor", "opt", "perfect", "stem",
    "derived", "inherited", "loanword", "diminutive", "intensive", "compound",
    "alternate", "replaced", "reduplicated", "sound-variant",
    "poetic", "dialectal", "archaic", "modern", "colloquial", "vulgar",
    # Language-specific inflection / noun classes
    "weak", "middle", "strong", "Tamil-class-1", "Tamil-class-2", "Tamil-class-3",
    "Tamil-class-4", "Tamil-class-5", "Tamil-class-6", "Tamil-class-7",
    "Kalasha-class-1", "Kalasha-class-2", "Kalasha-class-3", "Kalasha-class-4",
    "Burushaski-class-H", "Burushaski-class-HM", "Burushaski-class-HF",
    "Burushaski-class-X", "Burushaski-class-Y", "Burushaski-class-Z",
    "Palula-noun-class-a", "Palula-noun-class-i", "Palula-noun-class-m",
    "Palula-noun-class-aan", "Palula-noun-class-ee", "Palula-noun-class-irregular",
    "Palula-verb-class-L-a", "Palula-verb-class-L-e",
    "Palula-verb-class-L-consonant", "Palula-verb-class-L-minor",
    "Palula-verb-class-T", "Palula-verb-class-suppletive",
    # Additional lexical subclasses used by rich dictionary importers.
    "determiner", "discourse-marker", "auxiliary", "negator", "mood-marker",
    "honorific", "proper-noun", "multiword-expression", "demonstrative",
    "personal", "reciprocal", "copula", "modal", "conjunct-verb",
    "incorporating", "non-incorporating", "temporal", "spatial", "manner",
    "degree", "sentential", "onomatopoeia",
}

# CDIAL abbreviations normalized into the shared schema. They are applied only when the complete
# semicolon-delimited field is structured, so prose such as ``pret. of ...`` remains untouched.
GRAMMATICAL_ALIASES = {
    "absol": "abs",
    "inst": "instr",
    "imper": "impv",
    "vb": "verb",
    "subst": "noun",
    "sb": "noun",
    "st": "stem",
    "perf": "perfect",
    "part.": "participle",  # dotted CDIAL ``part.``; bare ``part`` means particle
    "poet": "poetic",
    "dial": "dialectal",
    "mod": "modern",
    "old": "archaic",
    "colloq": "colloquial",
    "vulg": "vulgar",
    "hon": "honorific",
}

_PERSON_NUMBER = re.compile(r"([123])(?:st|nd|rd)?\s+(sg|pl)\.?", re.IGNORECASE)
_DOTTED_GENDERS = {"m.n": "mn", "m.f": "mf", "f.m": "mf", "f.n": "fn", "n.f": "fn"}

# Printed regional lect labels which occur as complete parenthesized note fields. Directional
# distinctions are retained because CDIAL uses them contrastively. Compiled tags are qualified by
# normalized language ID and registered, with geography, in cldf/dialects.csv.
REGIONAL_LABELS = {
    "bastar": "Bastar",
    "camparan": "Camparan",
    "etirhut": "East Tirhut",
    "gaya": "Gaya",
    "kamdesh": "Kamdesh",
    "manbhum": "Manbhum",
    "netirhut": "Northeast Tirhut",
    "ntirhut": "North Tirhut",
    "patna": "Patna",
    "saltrange": "Salt Range",
    "sambhalpur": "Sambhalpur",
    "saran": "Saran",
    "sbhagalpur": "South Bhagalpur",
    "setirhut": "Southeast Tirhut",
    "shahabad": "Shahabad",
    "shahpur": "Shahpur",
    "smunger": "South Munger",
    "swshahabad": "Southwest Shahabad",
    "tarai": "Tarai",
    "wama": "Wama",
}

# Reuse an existing registry identity where CDIAL's regional label is already a named source lect.
REGIONAL_SOURCE_IDS = {("Kt", "Kamdesh"): "Kamd"}


def _regional_label(plain):
    key = re.sub(r"\s+", "", plain.strip().strip("()").strip().lower())
    return REGIONAL_LABELS.get(key)


def _regional_tag(plain, language_id):
    label = _regional_label(plain)
    if not label or not language_id:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    source_id = REGIONAL_SOURCE_IDS.get(
        (language_id, label), f"cdial-{language_id}-{slug}"
    )
    return dialect_tag(language_id, source_id, label)

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
_WORK_IN_PROSE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(x) for x in sorted(_WORK_ABBREVS, key=len, reverse=True))
    + r")\."
)


def _split_fields(note):
    """Split on ';' WITHOUT breaking HTML entities (e.g. `&lt;`, which contain a ';')."""
    ents = []

    def stash(m):
        ents.append(m.group(0))
        return f"\x00{len(ents) - 1}\x00"

    protected = _ENTITY.sub(stash, note)
    restore = lambda s: _HOLE.sub(lambda m: ents[int(m.group(1))], s)
    return [restore(p) for p in protected.split(";")]


def _classify(field, language_id=None):
    """Tag list for a field if it is ENTIRELY gender/grammatical/source tokens, else None."""
    plain = html.unescape(_TAGS.sub("", field)).strip()
    regional_tag = _regional_tag(plain, language_id)
    if regional_tag:
        return [regional_tag]
    regional_dialect = re.fullmatch(r"(.+?)\s+dial\.?", plain, re.IGNORECASE)
    if regional_dialect:
        regional_tag = _regional_tag(regional_dialect.group(1), language_id)
        if regional_tag:
            return ["dialectal", regional_tag]
    dotted_gender = _DOTTED_GENDERS.get(plain.rstrip(".").lower())
    if dotted_gender:
        return [dotted_gender]
    person_number = _PERSON_NUMBER.fullmatch(plain)
    if person_number:
        return ["".join(person_number.groups()).lower()]
    toks = plain.split()
    if not toks:
        return None
    out = []
    for tok in toks:
        base = tok.rstrip(".")
        # Alias matching stays lowercase-only for the same reason as canonical grammar matching:
        # uppercase source/dialect abbreviations must retain their existing interpretation.
        alias = None
        if tok == tok.lower():
            alias = GRAMMATICAL_ALIASES.get(tok, GRAMMATICAL_ALIASES.get(base))
        if base in GENDER_TAGS:
            out.append(base)
        elif alias:
            out.append(alias)
        # Grammar abbreviations are lowercase in source data.  Preserve case
        # here so source/dialect codes such as DEDR ``Tr.`` do not become the
        # grammatical tag ``tr``.
        elif base in GRAMMATICAL_TAGS:
            out.append(base)
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


def extract_tags(note, language_id=None):
    """(tags, cleaned_notes): `tags` is a space-separated list (gender, grammatical, source, era);
    `cleaned_notes` keeps every field that was not purely structured tokens."""
    if not note:
        return "", note or ""
    # A work locus can be embedded in a parenthetical or other scholarly prose, e.g.
    # ``('devotion' Prab.com.)`` or ``(sudhyatē ṢaḍvBr.)``.  Discover dotted work markers there,
    # but retain the prose verbatim; only wholly structured fields are removed below.
    plain_note = html.unescape(_TAGS.sub("", note))
    tags = [m.group(1) for m in _WORK_IN_PROSE.finditer(plain_note)]
    kept = []
    for field in _split_fields(note):
        if not field.strip():
            continue
        cls = _classify(field, language_id=language_id)
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
