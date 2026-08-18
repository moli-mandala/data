"""
unify_cldf.py — fold the etyma (parameters.csv) and the attested reflexes (forms.csv) into ONE
table: cldf/forms.csv, one row per node in the etymon graph. Reflexes point at their etymon via a
self-referential `Origin_ID`; etyma have an empty `Origin_ID`. parameters.csv is then removed.

The etymon is NOT duplicated as its own reflex. For each etymon we find its self-reflex — the form
in the etymon's own language whose Form equals the head-word — and fold that reflex's parsed data
(gloss, tags, native/phonemic/original, source) up onto the etymon node, then drop it. The etymon's
free-text etymological header (the CDIAL entry HTML) is stored in a dedicated `Etymology` column,
leaving `Gloss` for the parsed short meaning.

Same-language, non-head-word forms (e.g. OIA variant spellings / reconstructions) are kept but
marked `Relation = variant`; genuine daughter-language reflexes are `Relation = reflex`; etyma have
an empty `Relation`.

Column mapping:
    etymon : Form=headword, Gloss=parsed meaning (from self-reflex), Etymology=CDIAL entry HTML,
             Tags/Native/Phonemic/Original/Source folded from the self-reflex, Description=Etyma,
             Origin_ID="", Relation=""
    reflex : Form=form, Gloss=meaning, Description=notes, Origin_ID=<etymon id>,
             Relation="reflex" | "variant"

Run after make_cldf.py and link_refs.py. ``assign_form_ids.py`` follows this pass, then alignment
runs against the final persistent-ID graph.
"""

import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

from edges_build import build_edges, write_edges
from burushaski_cognates import apply_catalog as apply_burushaski_catalog
from burushaski_cognates import load_catalog as load_burushaski_catalog

_ADD_PTR = re.compile(r"\s*Add\.\s*\d+\.?")  # the now-defunct "Add. N" pointer after a merge
# separates a main entry's etymology snippet from a merged addendum's; the webapp splits on it and
# renders one accented block per snippet (so no snippet is dropped when addenda fold into a main).
ADD_DELIM = "<!--addendum-->"

DRAVIDIAN_CLADES = {
    "Old Dravidian", "S. Dravidian I", "S. Dravidian II", "C. Dravidian",
    "N. Dravidian", "Brahui",
}
INDO_ARYAN_CLADES = {
    "OIA", "MIA", "Early NIA", "Nuristani", "Pashai", "Chitrali", "Shinaic",
    "Kohistani", "Kunar", "Kashmiric", "Sindhic", "Lahndic", "Punjabic",
    "W. Pahari", "C. Pahari", "E. Pahari", "Eastern", "Bihari", "E. Hindi",
    "W. Hindi", "Rajasthanic", "Gujaratic", "Bhil", "Khandeshi",
    "Marathi-Konkani", "Halbic", "Insular", "Migratory",
}

# Three multi-head routes have identical automatic evidence and are resolved from the lexical
# semantics/topology rather than whichever PNur row happens to occur first in the catalog.
NURISTANI_REFLEX_ROUTE_OVERRIDES = {
    ("125", "Wg", "æ̃r̆"): "n1090",   # 'fire': general *āŋā branch, not Prasun-only n1091
    ("726", "Kt", "ū"): "n3371",       # 'down': *voi with Wg/Kata/Kam, not Ashkun-only *vo
    ("13969", "Kt", "jut"): "n3779",  # 'panther': leopard branch, not 'brown speckled goat'
}

# Internal row layout (positional): the graph columns (Origin_ID/Relation/Variant_Of/
# Borrowed_From) exist only in memory — the serialized forms.csv drops them in favour of
# cldf/edges.csv (see edges_build.py), keeping Redirect plus a node Status column.
UNIFIED = [
    "ID", "Language_ID", "Form", "Gloss", "Native", "Phonemic", "Original", "Cognateset",
    "Description", "Tags", "Source", "Origin_ID", "Etymology", "Relation", "Redirect", "Variant_Of",
    "Borrowed_From",
]
UNIFIED_SERIALIZED = [
    "ID", "Language_ID", "Form", "Gloss", "Native", "Phonemic", "Original", "Cognateset",
    "Description", "Tags", "Source", "Etymology", "Redirect", "Status",
]


def load_borrowings(path="data/borrowings.csv"):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["Borrower_ID"]: r["Source_ID"] for r in csv.DictReader(f)}


def load_nuristani_cognates(path="data/nuristani_cognates.csv"):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_nuristani_borrowings(path="data/nuristani_borrowings.csv"):
    with open(path, encoding="utf-8") as f:
        return {
            row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
            for row in csv.DictReader(f)
        }


def load_strand_oia_redirects(path="data/strand_oia_redirects.csv"):
    with open(path, encoding="utf-8") as f:
        return {
            row["Strand_ID"]: row["CDIAL_ID"]
            for row in csv.DictReader(f)
        }


def load_dbia_redirects(path="data/dbia/cdial_redirects.csv"):
    with open(path, encoding="utf-8") as f:
        return {
            row["DBIA_ID"]: row["CDIAL_ID"]
            for row in csv.DictReader(f)
        }


def apply_dbia_redirects(etyma_rows, reflex_rows, redirects):
    """Re-home DBIA loans on CDIAL heads while retaining cited DBIA redirect stubs."""
    by_id = {row[0]: row for row in etyma_rows + reflex_rows}
    missing = sorted(
        (dbia, cdial) for dbia, cdial in redirects.items()
        if dbia not in by_id or cdial not in by_id
    )
    if missing:
        raise ValueError(f"Unknown DBIA/CDIAL redirects: {missing}")

    references = 0
    for row in etyma_rows + reflex_rows:
        if row[0] in redirects:
            continue
        for column in (11, 14, 15, 16):
            target = redirects.get(row[column])
            if target:
                row[column] = target
                references += 1

    for dbia, cdial in redirects.items():
        source_row = by_id[dbia]
        target_row = by_id[cdial]
        if source_row[1] != "Indo-Aryan" or target_row[1] != "Indo-Aryan":
            raise ValueError(f"DBIA redirect must join Indo-Aryan entries: {dbia}, {cdial}")
        source_etymology = (source_row[12] or "").strip()
        if source_etymology and source_etymology not in (target_row[12] or ""):
            target_row[12] = (
                target_row[12] + ADD_DELIM + source_etymology
                if target_row[12] else source_etymology
            )
        source_row[14] = cdial
    return len(redirects), references


def apply_borrowings(rows, borrowings):
    ids = {r[0] for r in rows}
    missing = sorted((borrower, source) for borrower, source in borrowings.items()
                     if borrower not in ids or source not in ids)
    if missing:
        raise ValueError(f"Unknown borrowing IDs: {missing}")
    applied = 0
    for row in rows:
        source = borrowings.get(row[0])
        if source:
            row[11] = source
            row[13] = "borrowed"
            row[16] = source
            applied += 1
    return applied


def apply_nuristani_cognates(rows, cognates):
    origins = {}
    for cognate in cognates:
        ancestor = cognate["Ancestor_ID"]
        for child in (cognate["Proto_Nuristani_ID"], cognate["Indo_Aryan_ID"]):
            existing = origins.setdefault(child, ancestor)
            if existing != ancestor:
                raise ValueError(f"Conflicting Proto-Indo-Iranian ancestors for {child}: {existing}, {ancestor}")

    by_id = {row[0]: row for row in rows}
    expected = set(origins) | {r["Ancestor_ID"] for r in cognates}
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(f"Unknown Nuristani cognate IDs: {missing}")
    for child, ancestor in origins.items():
        row = by_id[child]
        if row[11] and row[11] != ancestor:
            raise ValueError(f"Cannot attach {child} to {ancestor}; it already has origin {row[11]}")
        row[11] = ancestor
        row[13] = "reflex"
    return len(origins)


def comparable_nuristani_form(value):
    """A deliberately light comparison key for routing duplicate Nuristani reflexes.

    This is not used to infer cognacy: ``nuristani_cognates.csv`` already supplies that reviewed
    relationship.  It only distinguishes between two or more Strand PNur branches already linked
    to the same Indo-Aryan entry.  Diacritics and the house/source affricate spellings are folded so
    that, for example, CDIAL ``dost`` can be compared with Strand ``dost``/``dast`` evidence.
    """
    value = re.sub(r"<[^>]+>", "", value)
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.translate(str.maketrans({
        "ʦ": "ts", "ʣ": "dz", "č": "c", "ǰ": "j", "š": "s", "ž": "z",
        "ṣ": "s", "ṭ": "t", "ḍ": "d", "ṇ": "n", "ṅ": "n", "ñ": "n",
        "ṛ": "r", "ṝ": "r", "ḷ": "l", "ʹ": "", "′": "", "˜": "",
    }))
    return re.sub(r"[^a-z]", "", value)


def nuristani_form_similarity(left, right):
    left, right = comparable_nuristani_form(left), comparable_nuristani_form(right)
    return SequenceMatcher(None, left, right).ratio() if left and right else 0


def _pnur_order(entry_id):
    match = re.fullmatch(r"n(\d+)", entry_id)
    return (0, int(match.group(1))) if match else (1, entry_id)


def reparent_cdial_nuristani_reflexes(rows, cognates, language_clades):
    """Move CDIAL's inherited Nuristani reflexes from IA heads to Strand PNur heads.

    Turner groups Nuristani forms inside Indo-Aryan entries.  For the reviewed cases where Strand
    instead reconstructs inheritance from Proto-Indo-Iranian through Proto-Nuristani, keeping those
    forms on the Indo-Aryan sibling duplicates the Nuristani branch and asserts the wrong immediate
    ancestor.  A single PNur sibling is unambiguous.  When several PNur reconstructions share one IA
    comparison, route each CDIAL form using, in order: a Strand descendant in the same language,
    similarity to that same-language evidence, similarity to any evidence in the branch, and
    similarity to the PNur head.  Source order is only the final deterministic tie-break.

    Only direct ``reflex`` rows cited to CDIAL are changed.  Variants remain transitively attached
    to their lemma, and PNur heads that Strand places beneath OIA are handled separately as loans.
    """
    by_id = {row[0]: row for row in rows}
    pnur_by_ia = defaultdict(list)
    for cognate in cognates:
        ia = cognate["Indo_Aryan_ID"]
        pnur = cognate["Proto_Nuristani_ID"]
        if pnur not in pnur_by_ia[ia]:
            pnur_by_ia[ia].append(pnur)

    # Snapshot the Strand branches before any CDIAL form is moved into them, so routing one form
    # cannot influence the score of the next.
    pnur_ids = {pnur for candidates in pnur_by_ia.values() for pnur in candidates}
    branch_rows = defaultdict(list)
    for row in rows:
        if row[11] in pnur_ids:
            branch_rows[row[11]].append(row)

    moved = single = multi = overridden = tied = 0
    for row in rows:
        ia = row[11]
        candidates = pnur_by_ia.get(ia, ())
        if (
            not candidates
            or row[13] != "reflex"
            or row[1] == "PNur"
            or language_clades.get(row[1]) != "Nuristani"
            or "CDIAL" not in row[10].split(";")
        ):
            continue

        if len(candidates) == 1:
            target = candidates[0]
            single += 1
        else:
            override_key = (ia, row[1], unicodedata.normalize("NFC", row[2]))
            target = NURISTANI_REFLEX_ROUTE_OVERRIDES.get(override_key)
            if target is not None:
                if target not in candidates:
                    raise ValueError(
                        f"Nuristani reflex override {override_key} targets unrelated PNur {target}"
                    )
                overridden += 1
                multi += 1
                row[11] = target
                moved += 1
                continue

            ranked = []
            for pnur in candidates:
                branch = branch_rows[pnur]
                same_language = [child for child in branch if child[1] == row[1]]
                score = (
                    bool(same_language),
                    max(
                        (nuristani_form_similarity(row[2], child[2]) for child in same_language),
                        default=0,
                    ),
                    max(
                        (nuristani_form_similarity(row[2], child[2]) for child in branch),
                        default=0,
                    ),
                    nuristani_form_similarity(row[2], by_id[pnur][2]),
                )
                ranked.append((score, pnur))
            best_score = max(score for score, _ in ranked)
            winners = [pnur for score, pnur in ranked if score == best_score]
            if len(winners) > 1:
                tied += 1
            target = min(winners, key=_pnur_order)
            multi += 1

        row[11] = target
        moved += 1

    return {
        "moved": moved,
        "single": single,
        "multi": multi,
        "overridden": overridden,
        "tied": tied,
    }


def apply_nuristani_borrowings(rows, borrowings):
    by_id = {row[0]: row for row in rows}
    missing = sorted(
        (nuristani, indo_aryan)
        for nuristani, indo_aryan in borrowings.items()
        if nuristani not in by_id or indo_aryan not in by_id
    )
    if missing:
        raise ValueError(f"Unknown Nuristani borrowing IDs: {missing}")

    descendants = 0
    for nuristani, indo_aryan in borrowings.items():
        branch = [
            row
            for row in rows
            if row[0] == nuristani or row[11] == nuristani
        ]
        if not branch:
            raise ValueError(f"No Nuristani borrowing branch found for {nuristani}")
        for row in branch:
            row[11] = indo_aryan
            row[13] = "borrowed"
            row[16] = indo_aryan
            if row[0] != nuristani:
                descendants += 1
    return len(borrowings), descendants


def apply_strand_oia_redirects(etyma_rows, reflex_rows, redirects):
    rows = etyma_rows + reflex_rows
    by_id = {row[0]: row for row in rows}
    missing = sorted(
        (strand, cdial)
        for strand, cdial in redirects.items()
        if strand not in by_id or cdial not in by_id
    )
    if missing:
        raise ValueError(f"Unknown Strand OIA redirect IDs: {missing}")

    for strand, cdial in redirects.items():
        if by_id[strand][1] != "Indo-Aryan" or by_id[cdial][1] != "Indo-Aryan":
            raise ValueError(f"Strand OIA redirect must join Indo-Aryan entries: {strand}, {cdial}")

    redirected = 0
    for row in rows:
        if row[0] in redirects:
            continue
        for column in (11, 14, 15, 16):
            target = redirects.get(row[column])
            if target:
                row[column] = target
                redirected += 1

    etyma_rows[:] = [row for row in etyma_rows if row[0] not in redirects]
    return len(redirects), redirected


def load_language_clades(path="cldf/languages.csv"):
    with open(path, encoding="utf-8") as f:
        return {row["ID"]: row["Clade"] for row in csv.DictReader(f)}


def mark_cross_family_borrowings(rows, language_clades):
    """Classify structurally impossible inheritance edges as loans.

    Dravidian forms only inherit from Proto-Dravidian; forms in all other families cannot inherit
    from an Indo-Aryan node. Existing curated loans remain unchanged, while newly detected edges
    receive the same Relation/Borrowed_From representation as every other borrowing.
    """
    by_id = {row[0]: row for row in rows}
    newly_marked = 0
    for row in rows:
        origin_id = row[11]
        origin = by_id.get(origin_id)
        if not origin:
            continue
        child_clade = language_clades.get(row[1])
        origin_clade = language_clades.get(origin[1])
        dravidian_from_non_proto = (
            child_clade in DRAVIDIAN_CLADES and origin[1] != "PDr"
        )
        non_ia_from_ia = (
            origin_clade in INDO_ARYAN_CLADES and child_clade not in INDO_ARYAN_CLADES
        )
        if not (dravidian_from_non_proto or non_ia_from_ia):
            continue
        if row[13] != "borrowed":
            newly_marked += 1
        row[13] = "borrowed"
        row[16] = origin_id
    return newly_marked


def apply_borrowings_to_unified():
    """Standalone re-apply of data/borrowings.csv onto the built cldf/ (edge-model form).

    Post-cutover the unified file carries no graph columns, so the curated borrowings are
    patched directly into cldf/edges.csv: the borrower's rank-1 attestation edge is replaced
    by (borrower, source, borrowed, 1). Node Status is untouched (an etymon that gains a loan
    source stays an entry — matching the in-pipeline behaviour of apply_borrowings, which only
    rewrites the graph columns of etyma rows)."""
    import edges_build

    with open("cldf/edges.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != edges_build.EDGES_HEADER:
            raise ValueError("cldf/edges.csv is not in edge-table format")
        edges = list(reader)
    borrowings = load_borrowings()
    node_ids = set()
    with open("cldf/forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node_ids.add(row["ID"])
    applied = 0
    by_child_rank1 = {}
    for e in edges:
        if e[3] == "1" and e[2] in ("reflex", "borrowed", "variant"):
            by_child_rank1[e[0]] = e
    for borrower, source in borrowings.items():
        if borrower not in node_ids or source not in node_ids:
            continue
        existing = by_child_rank1.get(borrower)
        if existing is not None:
            existing[1], existing[2] = source, "borrowed"
        else:
            edge = [borrower, source, "borrowed", "1", "", "", ""]
            edges.append(edge)
            by_child_rank1[borrower] = edge
        applied += 1
    edges.sort(key=lambda e: (e[0], e[2], int(e[3]), int(e[4]) if e[4] else 0, e[1]))
    with open("cldf/edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(edges_build.EDGES_HEADER)
        w.writerows(edges)
    print(f"re-applied {applied} curated borrowings to cldf/edges.csv", file=sys.stderr)


def strip_marker(pid: str) -> str:
    """Reflex Parameter_IDs may carry a borrowing / semi-tatsama marker (>, ~)."""
    return pid[1:] if pid and pid[0] in ">~" else pid


_EXT_MORPH = re.compile(r"<i>([^<]+)</i>")


def ext_morpheme(info: str) -> str | None:
    """The extension suffix in a CDIAL section header — ``ext. -<i>kk</i>-`` or a bare
    ``-<i>kk</i>-`` — returned as ``-kk-`` (tags stripped), else None. These sections mark reflexes
    descending from a morphologically extended stem of the headword."""
    if not (re.search(r"\bext\b", info, re.I) or re.fullmatch(r"-<i>[^<]+</i>-", info)):
        return None
    m = _EXT_MORPH.search(info)
    return f"-{m.group(1)}-" if m else None


def is_derivation_section(info: str) -> bool:
    """Whether a CDIAL descendant-group heading labels the forms as derivatives."""
    plain = re.sub(r"<[^>]+>", "", info).strip()
    return re.match(r"(?i)^deriv", plain) is not None


def derivation_morpheme(info: str) -> str | None:
    """An explicit suffix from ``Deriv. ... with -<i>X</i>-`` headings.

    Generic ``Deriv.`` groups do not posit an intermediate form and are represented by a
    ``derived`` tag on their reflexes.  Only a printed ``with -X-`` supplies enough structure to
    retain a separate derived branch.
    """
    if not is_derivation_section(info):
        return None
    match = re.search(r"(?i)\bwith\s+-\s*<i>([^<]+)</i>\s*-", info)
    return f"-{match.group(1)}-" if match else None


def section_kind(info: str):
    """Classify a section header as a derived-form section that we promote to its own entry →
    (kind, suffix, tag). Generic derivative sections are deliberately excluded: without a printed
    morpheme they do not identify an intermediate lexeme. Returns (None, None, None) otherwise."""
    m = ext_morpheme(info)
    if m:
        return ("ext", m, "ext:" + m.strip("-"))
    plain = re.sub(r"<[^>]+>", "", info).strip()
    if re.match(r"(?i)^caus", plain):
        return ("caus", "-áya-", "caus")  # OIA causative morpheme
    m = derivation_morpheme(info)
    if m:
        return ("deriv-morph", m, "ext:" + m.strip("-"))
    return (None, None, None)


_CROSS = re.compile(r"\bX\b[^<]*<smallcaps>([^<]+)</smallcaps>")


def contamination(info: str) -> str | None:
    """A CDIAL "X ⟨Y⟩" header marks reflexes crossed/blended with another etymon Y (contamination);
    return Y's head-word (NFC), else None."""
    m = _CROSS.search(info)
    return nfc(m.group(1).strip().rstrip("-").strip()) if m else None


_OIA_VOWELS = set("aāiīuūeēoōṛṝḷ")

_UNCERTAIN = re.compile(r"(?i)^(poss|prob|perh|dub|doubtful|maybe|uncertain)\b")  # \b excludes "Possessive"
_REDUP = re.compile(r"(?i)^redup")
# a leading grammatical label on a section header ("Adj. forms", "f. *X", "tr. with e") → tag token.
# Causatives and explicit-morpheme derivative sections are promoted to their own entries. Generic
# derivative sections receive the canonical ``derived`` tag below.
_GRAM_TAG = {
    "adj": "adj", "adv": "adv", "tr": "tr", "intr": "intr", "pron": "pron", "postp": "postp",
    "num": "num", "fem": "f", "masc": "m", "neut": "n", "f": "f", "m": "m", "n": "n",
    "pres": "pres", "pret": "pret", "pp": "pp", "ppp": "ppp", "pass": "pass", "fut": "fut",
    "inf": "inf", "ger": "ger", "impv": "impv", "opt": "opt",
    "sg": "sg", "pl": "pl", "du": "du",
    "nom": "nom", "acc": "acc", "gen": "gen", "dat": "dat", "loc": "loc", "abl": "abl",
    "instr": "instr", "voc": "voc", "obl": "obl",
}


def section_flags(info: str) -> list[str]:
    """Structured flags a section header carries: etymological hedge (uncertain), reduplication,
    alternate form, and a leading grammatical label (Adj., pres., pp., obl., …). Additive tags."""
    p = re.sub(r"<[^>]+>", "", info).strip()
    out = []
    if is_derivation_section(info):
        out.append("derived")
    if _UNCERTAIN.match(p):
        out.append("uncertain")
    if _REDUP.match(p):
        out.append("reduplicated")
    if re.match(r"(?i)^altern", p):
        out.append("alternate")
    m = re.match(r"(?i)^([a-z]+)\.", p)  # a grammatical abbreviation followed by a period
    if m and _GRAM_TAG.get(m.group(1).lower()):
        out.append(_GRAM_TAG[m.group(1).lower()])
    return out


def strip_accent(s: str) -> str:
    """Drop the Vedic pitch accent (combining acute / grave) from an OIA head-word."""
    d = unicodedata.normalize("NFD", s)
    d = "".join(c for c in d if c not in ("́", "̀"))
    return unicodedata.normalize("NFC", d)


_WITH_SUB = re.compile(r"(?i)-?<i>([^<]+)</i>-?\s+in place of\s+-?<i>([^<]+)</i>-?")
_WITH_INIT = re.compile(r"(?i)^with\s+<i>([^<]+)</i>-")
# a "< <form>" reference (italic or smallcaps) — the source of an alternate/replacement etymology
_ETYM_REF = re.compile(r"(?:&lt;|<)\s*-?(?:<i>(\*?[^<]+?)</i>|<smallcaps>(\*?[^<]+?)</smallcaps>)")
_REPLACED_BY = re.compile(
    r"(?i)^replaced\s+by\b[^<]*?(?:<i>(\*?[^<]+?)</i>|<smallcaps>(\*?[^<]+?)</smallcaps>)"
)


def _ref_form(m) -> str | None:
    """The head-word captured by an _ETYM_REF / _REPLACED_BY match (either group), tidied."""
    if not m:
        return None
    w = (m.group(1) or m.group(2) or "").strip().rstrip("-").strip()
    return nfc(w) or None


def sound_variant(info: str, base_bare: str):
    """Generate a sound-variant proto-form from a section header applied to the (accent-stripped)
    base head-word: 'with A in place of B' → substitute B→A; initial 'With X-' → prepend a consonant,
    or replace the leading vowel with a vowel. Returns (word, short-label) or None."""
    m = _WITH_SUB.search(info)
    if m:
        new = strip_accent(m.group(1)).strip("-").strip()
        old = strip_accent(m.group(2)).strip("-").strip()
        if new and old and old in base_bare:
            return ("*" + base_bare.replace(old, new), f"with {new} for {old}")
        return None
    m = _WITH_INIT.match(info)
    if m:
        x = strip_accent(m.group(1)).strip("-").strip()
        if not x or not base_bare:
            return None
        has_vowel = any(c in _OIA_VOWELS for c in x)
        has_cons = any(c.isalpha() and c not in _OIA_VOWELS for c in x)
        base_vowel_initial = base_bare[0] in _OIA_VOWELS
        if has_vowel and not has_cons and base_vowel_initial:
            return ("*" + x + base_bare[1:], f"with {x}-")  # vowel: replace the leading vowel
        if has_cons and not has_vowel and base_vowel_initial:
            return ("*" + x + base_bare, f"with {x}-")  # single consonant onto a vowel-initial base
        return None  # CV syllable (a compound member) or would form a consonant cluster — skip
    return None


def nfc(s: str) -> str:
    """Head-words and forms can disagree on Unicode normalisation (precomposed vs combining); compare
    them NFC-folded so a self-reflex like OIA aṅgúli is recognised as the head-word."""
    return unicodedata.normalize("NFC", s)


def main():
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = list(csv.DictReader(f))

    # Preserve rich importers' immutable source-local record keys outside the unified CLDF table.
    # assign_form_ids.py consumes this sidecar after graph construction; keeping it separate avoids
    # exposing ingestion bookkeeping as linguistic columns in the published wordlist.
    source_key_counts = defaultdict(int)
    for row in forms:
        if row.get("Entry_Key"):
            source_key_counts[row["Entry_Key"]] += 1
    source_keys = [
        (row["ID"], row["Entry_Key"])
        for row in forms
        # A key identifies a form only when it is unique. Some compound analyses repeat one lexical
        # entry once per proposed etymon; those need a future edge-model migration, so retain their
        # existing registry identity instead of pretending the shared entry key is node-unique.
        if row.get("Entry_Key") and source_key_counts[row["Entry_Key"]] == 1
    ]

    # Rich manual sources can describe graph relations with stable source-local
    # keys before make_cldf assigns numeric IDs. Resolve those keys here.
    source_id_by_key = {}
    for r in forms:
        if r.get("Entry_Key"):
            source_id_by_key.setdefault(r["Entry_Key"], r["ID"])

    params_by_id = {p["ID"]: p for p in params}
    graph_node_ids = set(params_by_id) | {r["ID"] for r in forms}

    def resolve_graph_ref(value):
        if not value:
            return ""
        if value.startswith("id:"):
            target = value[3:]
            return target if target in graph_node_ids else ""
        return source_id_by_key.get(value, "")

    forms_by_param = defaultdict(list)
    for r in forms:
        forms_by_param[r["Parameter_ID"]].append(r)

    # For each etymon, locate its self-reflex: the form in the etymon's own language whose Form is
    # the head-word. Its parsed data folds up onto the etymon and the reflex row is then dropped.
    self_reflex_ids = set()
    folded = {}  # etymon ID -> the self-reflex form row
    for p in params:
        for r in forms_by_param.get(p["ID"], ()):
            if r["Language_ID"] == p["Language_ID"] and nfc(r["Form"]) == nfc(p["Name"]):
                self_reflex_ids.add(r["ID"])
                folded.setdefault(p["ID"], r)

    # addenda→main merges (computed by head-word in link_refs.py)
    merges = {}
    if os.path.exists("cldf/merges.csv"):
        with open("cldf/merges.csv", encoding="utf-8") as f:
            merges = {r["Addendum_ID"]: r["Main_ID"] for r in csv.DictReader(f)}

    # ---- build etymon rows -------------------------------------------------
    G, NA, PH, OR, DE, TG, SR_, ET, RD = 3, 4, 5, 6, 8, 9, 10, 12, 14  # column indices
    etyma_rows, etyma_by_id = [], {}
    for p in params:
        header = p["Description"]
        # CDIAL entries carry the full dictionary entry as HTML (starting with a tag — <html><body>
        # or a bare <number>/<b> depending on the bs4 parser); other sources (Dravidian/Munda/
        # Nuristani) put the plain meaning there instead.
        is_html = header.lstrip().startswith("<")
        gloss = "" if is_html else header
        etymology = p.get("Etymology", "") or (header if is_html else "")
        native = phonemic = original = tags = source = ""
        sr = folded.get(p["ID"])
        if sr:  # fold the self-reflex's parsed data into empty etymon fields
            gloss = gloss or sr["Gloss"]
            tags = tags or sr.get("Tags", "")
            native = native or sr["Native"]
            phonemic = phonemic or sr["Phonemic"]
            original = original or sr["Original"]
            source = source or sr["Source"]
        row = [p["ID"], p["Language_ID"], p["Name"], gloss, native, phonemic, original,
               "", p.get("Etyma", ""), tags, source, "", etymology, "", "", "", ""]
        etyma_rows.append(row)
        etyma_by_id[p["ID"]] = row

    # ---- build reflex rows (self-reflexes dropped; addenda reflexes re-parented) ------
    # A CDIAL entry's header lists numbered derived forms (`2. *kṣata-². 3. *kṣaṇana-. …`); each is a
    # lexeme in its own right, so we promote it to an entry derived from the head (form 1 = the
    # etymon). Reflexes are grouped into those forms by the `info` half of their Cognateset
    # ("subnum:info" → info is the form number). Non-numeric info carries forward the most recent
    # form number; form 1 (or no numbered form) stays on the head.
    n_reflex = n_variant = n_section = n_borrowed = n_lone = n_ext = n_crossed = n_svar = 0
    n_deriv_flat = 0
    n_replaced = n_altern = 0
    reflex_rows = []
    ext_entry_rows = []  # synthetic extension/caus/morphemic-derivative + shared morpheme entries
    morpheme_id = {}  # suffix -> shared morpheme entry id (one `-kk-`/`-áya-`/… entry, reused)
    section_edges = []  # (derived-form id -> parent id); an extension has TWO (base + morpheme)
    for r in forms:
        for parent_key in (r.get("Derivation_Parent_Keys") or "").split("|"):
            parent_id = resolve_graph_ref(parent_key)
            if parent_id and parent_id != r["ID"]:
                section_edges.append((r["ID"], parent_id))
    # contaminating-etymon head-word -> its id, for "X ⟨Y⟩" cross links
    oia_id_by_form = {
        nfc(p.get("Name", "")): p["ID"] for p in params if p.get("Language_ID") == "Indo-Aryan"
    }
    all_ids = {r["ID"] for r in forms}  # to keep promoted `<etymon>-<n>` ids collision-free

    # Borrowed sub-reflexes: a CDIAL note "(→ H. …, B. …)" lists forms borrowed FROM that reflex.
    # parse.py already split them into rows tagged with Cognateset "<subnum>:<parent-lang> →"; here we
    # link each back to its parent reflex — the one on the same etymon + section number, in the named
    # language, whose note carries the "(→" marker.
    borrow_parent = {}  # (pid, subnum, lang) -> parent reflex id
    for r in forms:
        if "(→" in (r["Description"] or ""):
            key = (strip_marker(r["Parameter_ID"]), (r["Cognateset"] or "").split(":", 1)[0],
                   r["Language_ID"])
            borrow_parent.setdefault(key, r["ID"])

    for pid_key, group in forms_by_param.items():
        pid = strip_marker(pid_key)
        parent = params_by_id.get(pid)
        cdial = parent is not None and parent["Language_ID"] == "Indo-Aryan" and pid not in merges

        # enumerate the numbered head-forms (same language as the etymon, not the self-reflex, not a
        # comma-alternate) in header order → form 2, 3, …. A promoted form is re-id'd `<etymon>-<n>`.
        section_by_num, promoted_id, ext_by_morph, variant_by_info = {}, {}, {}, {}
        variant_tag = {}  # info -> the tag for its generated entry ('sound-variant' | 'replaced')
        section_word = {}  # CDIAL form number -> its head-word (for the "altern. < N" link text)
        if cdial:
            num = 2
            prev_or = False  # the previous head-form ended in "or" → the next is its alternate
            for r in group:
                if r["Language_ID"] != parent["Language_ID"] or r.get("Variant_Of"):
                    continue
                # italic head-forms (parse.py marks them "@variant") are alternate spellings of the
                # head, never numbered section forms — skip promotion; they fall through to the
                # same-language branch below and become variants of the etymon.
                if (r.get("Cognateset") or "").endswith("@variant"):
                    continue
                # a head-form joined to the previous by "or" (e.g. "*dr̥kṣati or *drakṣati") is an
                # alternate of the SAME form-slot, not a new numbered form; skip it so CDIAL's own
                # form numbering — which the reflex sections index into via Cognateset info — is kept.
                alternate = prev_or
                prev_or = re.search(r"\bor$", (r.get("Description") or "").strip()) is not None
                if r["ID"] in self_reflex_ids or alternate:
                    continue
                new_id = f"{pid}-{num}"
                while new_id in all_ids:  # rare clash with a make_cldf `<file>-<row>` id
                    new_id += "x"
                all_ids.add(new_id)
                section_by_num[num] = new_id
                section_word[num] = r.get("Form", "")
                promoted_id[r["ID"]] = new_id
                num += 1

            # Promote derived-form sections with an explicit morpheme (extension "ext. -kk-",
            # causative "caus", or "Deriv. ... with -X-") to entries — one node per distinct suffix,
            # headword = base head-word + suffix. Generic "Deriv." groups stay on the base etymon.
            # The suffix morpheme is itself a shared entry (one "-kk-" reused everywhere), so each
            # derived entry has a COMPOUND etymology: base + morpheme (two derivation edges).
            _MLABEL = {
                "ext": "extension suffix", "caus": "causative suffix",
                "deriv-morph": "derivational suffix",
            }
            _MTAG = {
                "ext": "morpheme:extension", "caus": "morpheme:causative",
                "deriv-morph": "morpheme:derivative",
            }
            base_row = etyma_by_id.get(pid)
            base_form = base_row[2] if base_row else ""
            for r in group:
                cg = r.get("Cognateset") or ""
                info = cg.split(":", 1)[1] if ":" in cg else ""
                kind, suffix, tag = section_kind(info)
                if not kind or suffix in ext_by_morph:
                    continue
                new_id = f"{pid}-{num}"
                while new_id in all_ids:
                    new_id += "x"
                all_ids.add(new_id)
                ext_by_morph[suffix] = new_id
                num += 1
                # One shared morpheme node is reused for every occurrence of the suffix.
                mo_id = morpheme_id.get(suffix)
                if mo_id is None:
                    clean = re.sub(r"[\s().\-]+", "", suffix) or f"m{len(morpheme_id)}"
                    mo_id = f"mo-{clean}"
                    while mo_id in all_ids:
                        mo_id += "x"
                    all_ids.add(mo_id)
                    morpheme_id[suffix] = mo_id
                    ext_entry_rows.append([
                        mo_id, "Indo-Aryan", suffix.strip(), _MLABEL[kind],
                        "", "", "", "", "", _MTAG[kind], "CDIAL", "", "", "", "", "", "",
                    ])
                # construct the headword: accent-stripped base + morpheme, always reconstructed (*).
                # An extension of an Indo-Aryan -ati verb infixes the morpheme into the present stem
                # (dravati + -ḍ- -> *dravaḍati, a complete verb); otherwise it is a bound stem and
                # keeps a trailing hyphen (*katukk-).
                base_bare = strip_accent(base_form).lstrip("*")
                morph_bare = strip_accent(suffix).strip().strip("-")  # 'kk', 'ḍ', 'aya'
                is_verb = parent["Language_ID"] == "Indo-Aryan" and base_bare.endswith("ati")
                root = base_bare[:-3] if is_verb else base_bare  # strip the -ati verb ending
                if not is_verb and root.endswith("n") and not strip_accent(base_form).startswith("*"):
                    root = root[:-1]  # attested an-/in-stem noun: drop -n (reconstructed roots keep it)
                if root.endswith("ā"):
                    root = root[:-1] + "a"  # shorten a stem-final long ā to short a
                # break a root(C)+morpheme(C) cluster with an epenthetic -a-
                sep = (
                    "a"
                    if root and root[-1] not in _OIA_VOWELS
                    and morph_bare and morph_bare[0] not in _OIA_VOWELS
                    else ""
                )
                if is_verb:
                    word = ("*" + root + sep + morph_bare + "ati").replace("aati", "ati")
                else:
                    word = "*" + root + sep + morph_bare + "-"
                # readable, linked etymology (data-entry links render like CDIAL cross-refs), e.g.
                # "kk-extension of ⟨kaṭu⟩" with both the morpheme and the base head clickable.
                base_link = f'<smallcaps><a data-entry="{pid}">{strip_accent(base_form)}</a></smallcaps>'
                mo_link = f'<a data-entry="{mo_id}">{morph_bare}</a>' if mo_id else ""
                etym = {
                    "ext": f"{mo_link}-extension of {base_link}",
                    "caus": f"{mo_link}-causative of {base_link}",
                    "deriv-morph": f"{mo_link}-derivative of {base_link}; CDIAL section: {info}",
                }[kind]
                entry_tags = f"derived {tag}" if kind == "deriv-morph" else tag
                ext_entry_rows.append([
                    new_id, parent["Language_ID"], word, base_row[3] if base_row else "",
                    "", "", "", "", "", entry_tags, "CDIAL", "", etym, "", "", "", "",
                ])
                section_edges.append((new_id, pid))     # derived from the base head
                if mo_id:
                    section_edges.append((new_id, mo_id))  # …and from the morpheme (ext/caus compound)
                n_ext += 1

            # promote sound-variant sections ("with A in place of B", initial "With X-") — head-word
            # generated from the base — and "Replaced by ⟨Y⟩" sections (head-word = the named
            # replacement Y) to their own entries; the section's reflexes home to them below.
            blink = f'<smallcaps><a data-entry="{pid}">{strip_accent(base_form)}</a></smallcaps>'
            for r in group:
                cg = r.get("Cognateset") or ""
                info = cg.split(":", 1)[1] if ":" in cg else ""
                if info in variant_by_info or not info:
                    continue
                v = sound_variant(info, strip_accent(base_form).lstrip("*"))
                rep = None if v else _ref_form(_REPLACED_BY.match(info))
                if not v and not rep:
                    continue
                new_id = f"{pid}-{num}"
                while new_id in all_ids:
                    new_id += "x"
                all_ids.add(new_id)
                variant_by_info[info] = new_id
                variant_tag[info] = "sound-variant" if v else "replaced"
                num += 1
                if v:
                    word, label = v
                    tag, etym = "sound-variant", f"{label} of {blink}"
                else:
                    word = rep if rep.startswith("*") else "*" + rep
                    tag, etym = "replaced", f"replaced {blink}"
                ext_entry_rows.append([
                    new_id, "Indo-Aryan", word, base_row[3] if base_row else "",
                    "", "", "", "", "", tag, "CDIAL", "", etym, "", "", "", "",
                ])
                section_edges.append((new_id, pid))
                if tag == "sound-variant":
                    n_svar += 1
                else:
                    n_replaced += 1

        last_num = 1  # carry-forward form number within this entry (1 = the head itself)
        for r in group:
            if r["ID"] in self_reflex_ids:
                continue
            # Unetymologised manual-import form (blank Param_ID) → standalone "lone" node: kept in the
            # DB and searchable, empty Origin_ID, Relation="local" so it never enters the entries list.
            if not pid:
                source_variant = resolve_graph_ref(r.get("Variant_Of_Key", ""))
                source_donor = resolve_graph_ref(r.get("Borrowed_From_Key", ""))
                if source_donor:
                    local_origin, local_relation = source_donor, "borrowed"
                    local_variant, local_borrowed = "", source_donor
                    n_borrowed += 1
                elif source_variant:
                    local_origin, local_relation = source_variant, "variant"
                    local_variant, local_borrowed = source_variant, ""
                    n_variant += 1
                else:
                    local_origin, local_relation = "", "local"
                    legacy_variant = r.get("Variant_Of", "")
                    local_variant = legacy_variant if legacy_variant in graph_node_ids else ""
                    local_borrowed = ""
                    n_lone += 1
                reflex_rows.append([
                    r["ID"], r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                    r["Original"], r["Cognateset"], r["Description"], r.get("Tags", ""), r["Source"],
                    local_origin, r.get("Etymology", ""), local_relation, "",
                    local_variant, local_borrowed,
                ])
                continue
            vof = (
                resolve_graph_ref(r.get("Variant_Of_Key", ""))
                or r.get("Variant_Of", "")
            )
            if vof not in graph_node_ids:
                vof = ""
            source_donor = resolve_graph_ref(r.get("Borrowed_From_Key", ""))
            origin = r["Parameter_ID"]
            marker = origin[:1] if origin[:1] in (">", "~") else ""
            borrowed_from = ""

            # a CDIAL numbered head-form → promote to an entry (id `<etymon>-<n>`) derived from the head
            if r["ID"] in promoted_id:
                new_id = promoted_id[r["ID"]]
                section_edges.append((new_id, pid))
                n_section += 1
                reflex_rows.append([
                    new_id, r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                    r["Original"], "", r["Description"], r.get("Tags", ""), r["Source"],
                    "", r.get("Etymology", ""), "", "", "", "",
                ])
                continue

            cog = r["Cognateset"] or ""
            # a borrowed sub-reflex ("<subnum>:<lang> →") → child of the reflex it was borrowed from
            arrow_lang = ""
            if "→" in cog:
                sub, _, rest = cog.partition(":")
                arrow_lang = rest.split("→")[0].strip()
                borrowed_from = borrow_parent.get((pid, sub, arrow_lang), "")

            # two kinds of variant: a comma-listed alternate of a main reflex (Variant_Of set by
            # make_cldf), or a same-language non-head-word form on a non-CDIAL etymon.
            if source_donor:
                relation = "borrowed"
                origin = source_donor
                borrowed_from = source_donor
                vof = ""
                n_borrowed += 1
            elif borrowed_from:
                # the reflex it was borrowed from becomes its parent (origin) — a proper node with
                # this form as a child — so ancestry recurses through it and it owns its borrowings.
                relation = "borrowed"
                vof = ""
                origin = borrowed_from
                n_borrowed += 1
            elif marker:
                origin = strip_marker(origin)
                relation = "borrowed"
                borrowed_from = origin
                vof = ""
                marker_tag = "semi-tatsama" if marker == "~" else "marked borrowing"
                tags = [tag for tag in (r.get("Tags", "") or "").split() if tag]
                if marker_tag not in tags:
                    tags.append(marker_tag)
                r["Tags"] = " ".join(tags)
                n_borrowed += 1
            elif arrow_lang:
                # a "X →" borrowing section whose exact source reflex wasn't matched — still a loan;
                # parent it on the etymon as a fallback and tag its target (borrowed-into) language.
                relation = "borrowed"
                origin = pid
                borrowed_from = pid
                vof = ""
                tags = [tag for tag in (r.get("Tags", "") or "").split() if tag]
                if f"loan:{arrow_lang}" not in tags:
                    tags.append(f"loan:{arrow_lang}")
                r["Tags"] = " ".join(tags)
                n_borrowed += 1
            elif vof and vof not in self_reflex_ids:
                relation = "variant"
                n_variant += 1
            elif parent and r["Language_ID"] == parent["Language_ID"] and not cog.endswith("@variant"):
                relation = "variant"
                vof = ""
                n_variant += 1
            else:
                # italic head-forms (cog "@variant") fall through here: they attach as reflexes of
                # the head (their non-numeric info carries last_num=1, so they home to the etymon).
                relation = "reflex"
                vof = ""
                n_reflex += 1
                if cdial:  # re-home to its numbered head-form or derived node via the Cognateset info
                    info = cog.split(":", 1)[1].strip() if ":" in cog else ""
                    _k, sfx, sfx_tag = section_kind(info)
                    raw_info = cog.split(":", 1)[1] if ":" in cog else ""
                    if sfx and sfx in ext_by_morph:
                        mk = origin[0] if origin and origin[0] in ">~" else ""
                        origin = mk + ext_by_morph[sfx]  # home to the promoted morpheme-bearing entry
                        rtags = [t for t in (r.get("Tags", "") or "").split() if t]
                        if sfx_tag not in rtags:  # the extension morpheme as a searchable tag on the reflex
                            rtags.append(sfx_tag)
                        r["Tags"] = " ".join(rtags)
                    elif raw_info in variant_by_info:
                        mk = origin[0] if origin and origin[0] in ">~" else ""
                        origin = mk + variant_by_info[raw_info]  # generated sound-variant / replacement
                        vt = variant_tag.get(raw_info, "sound-variant")
                        rtags = [t for t in (r.get("Tags", "") or "").split() if t]
                        if vt not in rtags:
                            rtags.append(vt)
                        r["Tags"] = " ".join(rtags)
                    else:
                        m_add = re.match(r"Addenda.*?(\d+)\s*$", info)  # "Addenda: *X. N" → form N
                        if info.isdigit():
                            last_num = int(info)
                        elif m_add:
                            last_num = int(m_add.group(1))
                        elif is_derivation_section(info):
                            # A generic CDIAL "Deriv." heading supplies no intermediate protoform.
                            # Keep its forms as reflexes of the base.
                            last_num = 1
                        elif info == "":
                            last_num = 1  # section-less paragraph → the main entry (the head)
                        # else: a non-numeric label (e.g. "prob") carries forward the last form number
                        if last_num in section_by_num:
                            mk = origin[0] if origin and origin[0] in ">~" else ""
                            origin = mk + section_by_num[last_num]

            # Preserve a generic derivative heading on every affected row, including rows whose
            # legacy Variant_Of encoding bypassed the ordinary reflex branch above.
            section_info = cog.split(":", 1)[1] if ":" in cog else ""
            if is_derivation_section(section_info) and not derivation_morpheme(section_info):
                detail = f"CDIAL section: {section_info}"
                etymology = r.get("Etymology", "") or ""
                if detail not in etymology:
                    r["Etymology"] = f"{etymology}; {detail}" if etymology else detail
                n_deriv_flat += 1

            if pid in merges and not borrowed_from:  # merged addendum → hang on the main entry
                mk = origin[0] if origin and origin[0] in ">~" else ""
                origin = mk + merges[pid]

            # "X ⟨Y⟩" contamination: these reflexes are crossed/blended with another etymon Y (a
            # secondary origin). Tag it (queryable) and, when Y resolves to an OIA etymon, add a
            # navigable "× ⟨Y⟩" note (the webapp turns data-entry into a link, as for Turner refs).
            cross = contamination(cog.split(":", 1)[1] if ":" in cog else "")
            if cross:
                tags = [t for t in (r.get("Tags", "") or "").split() if t]
                if "contaminated" not in tags:  # flag only; the crossing etymon is in the note below
                    tags.append("contaminated")
                r["Tags"] = " ".join(tags)
                cid = oia_id_by_form.get(cross)
                if cid:
                    note = f'× <smallcaps><a data-entry="{cid}">{cross}</a></smallcaps>'
                    desc = r.get("Description") or ""
                    if cid not in desc:
                        r["Description"] = (desc + " " if desc else "") + note
                n_crossed += 1

            # alternate etymology: "altern. < N" (another CDIAL sub-form of this entry) or
            # "altern./or/poss. < ⟨form⟩" (another etymon) — a second, competing derivation. The
            # primary origin stays in Origin_ID; the alternate is a real second parent in the
            # derivation graph, so getAncestryChain shows BOTH etyma (as it does for compounds).
            araw = cog.split(":", 1)[1] if ":" in cog else ""
            aplain = re.sub(r"<[^>]+>", "", araw)
            alt_id = None
            mN = re.match(r"(?i)^altern\.?\s*(?:&lt;|<)\s*(\d+)", aplain)
            if cdial and mN and int(mN.group(1)) in section_by_num:
                alt_id = section_by_num[int(mN.group(1))]  # CDIAL sub-form N of this entry
            elif re.match(r"(?i)^(altern|or|poss|perh)\b.*(?:&lt;|<)", aplain):
                alt_id = oia_id_by_form.get(_ref_form(_ETYM_REF.search(araw)))
            if alt_id and alt_id != strip_marker(origin):
                atags = [t for t in (r.get("Tags", "") or "").split() if t]
                if "alternate" not in atags:
                    atags.append("alternate")
                r["Tags"] = " ".join(atags)
                section_edges.append((r["ID"], alt_id))  # second parent in the derivation graph
                n_altern += 1

            # structured flags from the section header: etymological hedge (uncertain), reduplication,
            # and a grammatical POS label — added as searchable tags on the section's reflexes.
            flags = section_flags(cog.split(":", 1)[1] if ":" in cog else "")
            if flags:
                rtags = [t for t in (r.get("Tags", "") or "").split() if t]
                for fl in flags:
                    if fl not in rtags:
                        rtags.append(fl)
                r["Tags"] = " ".join(rtags)

            reflex_rows.append([
                r["ID"], r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                r["Original"], r["Cognateset"], r["Description"], r.get("Tags", ""), r["Source"],
                origin, r.get("Etymology", ""), relation, "", vof, borrowed_from,
            ])

    # ---- fold each addendum's content up onto its main entry, then redirect it -------
    n_merged = 0
    for n, m in merges.items():
        nrow, mrow = etyma_by_id.get(n), etyma_by_id.get(m)
        if not nrow or not mrow:
            continue
        for i in (G, NA, PH, OR, TG):
            mrow[i] = mrow[i] or nrow[i]
        mrow[SR_] = ";".join(x for x in (mrow[SR_], nrow[SR_]) if x)
        # keep BOTH etymology snippets as separate blocks — the main entry's own header, then each
        # merged addendum's — joined by ADD_DELIM (the webapp renders one accented block each).
        # Previously the main's snippet was overwritten by the addendum's, silently dropping it for
        # the ~91 mains whose header lacked a "[Cf. …]" note.
        add_et = _ADD_PTR.sub("", nrow[ET] or "").strip()
        mrow[ET] = _ADD_PTR.sub("", mrow[ET] or "").strip()
        if add_et:
            mrow[ET] = mrow[ET] + ADD_DELIM + add_et if mrow[ET] else add_et
        nrow[RD] = m  # the addendum redirects to its main entry
        n_merged += 1

    n_curated_borrowings = apply_borrowings(etyma_rows, load_borrowings())
    language_clades = load_language_clades()
    nuristani_cognates = load_nuristani_cognates()
    n_nuristani_reflexes = apply_nuristani_cognates(
        etyma_rows + reflex_rows, nuristani_cognates
    )
    nuristani_reparented = reparent_cdial_nuristani_reflexes(
        etyma_rows + reflex_rows, nuristani_cognates, language_clades
    )
    n_nuristani_borrowings, n_nuristani_borrowed_descendants = apply_nuristani_borrowings(
        etyma_rows + reflex_rows, load_nuristani_borrowings()
    )
    burushaski_catalog = load_burushaski_catalog()
    burushaski_rows, burushaski_source_keys = apply_burushaski_catalog(
        etyma_rows + reflex_rows + ext_entry_rows,
        source_id_by_key,
        burushaski_catalog,
    )
    ext_entry_rows.extend(burushaski_rows)
    source_keys.extend(burushaski_source_keys)
    n_strand_oia_redirects, n_strand_oia_references = apply_strand_oia_redirects(
        etyma_rows, reflex_rows, load_strand_oia_redirects()
    )
    n_dbia_redirects, n_dbia_references = apply_dbia_redirects(
        etyma_rows, reflex_rows, load_dbia_redirects()
    )
    n_cross_family_borrowings = mark_cross_family_borrowings(
        etyma_rows + reflex_rows + ext_entry_rows, language_clades
    )

    # ---- derive the single typed edge table + serialize ------------------------------------
    # link_refs.py wrote cldf/derivation.csv; append the promoted numbered-form → head edges,
    # then classify everything into cldf/edges.csv (edges_build.py is the serialization
    # boundary of the edge model; classification + invariants live there). derivation.csv is a
    # build intermediate from this point on — consumed here and removed like parameters.csv.
    deriv_path = "cldf/derivation.csv"
    existing = []
    if os.path.exists(deriv_path):
        with open(deriv_path, encoding="utf-8") as f:
            existing = list(csv.reader(f))[1:]  # drop header
    seen = set(map(tuple, existing))
    added = [e for e in section_edges if e not in seen]
    combined_deriv = [tuple(e) for e in existing] + list(added)

    all_rows = etyma_rows + reflex_rows + ext_entry_rows
    edge_rows, edge_status, edge_stats = build_edges(all_rows, combined_deriv)
    write_edges(edge_rows)

    def serialize(row):
        # attested rows shed a stale Redirect (the v1/v2 DB builders already dropped it there)
        redirect = row[RD] if not row[13] else ""
        return row[:11] + [row[12], redirect, edge_status[row[0]]]

    with open("cldf/forms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(UNIFIED_SERIALIZED)
        w.writerows(serialize(r) for r in all_rows)

    with open("cldf/form-source-keys.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Legacy_ID", "Source_Key"])
        w.writerows(source_keys)

    if "--legacy-cols" in sys.argv:
        # development cross-check: the pre-cutover 17-column format + the raw derivation list,
        # so tests/test_edges.py can diff every edge against the legacy encoding
        with open("cldf/forms-legacy.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(UNIFIED)
            w.writerows(all_rows)
        with open(deriv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Child_ID", "Parent_ID"])
            w.writerows(combined_deriv)
    else:
        try:
            os.remove(deriv_path)
        except FileNotFoundError:
            pass

    print(
        f"cldf/edges.csv: {len(edge_rows)} typed edges "
        f"({edge_stats.get('component_groups', 0)} compounds, "
        f"{edge_stats.get('alt_edges', 0)} alternate hypotheses, "
        f"{edge_stats.get('alt_reviewable', 0)} flagged for review)",
        file=sys.stderr,
    )

    # Another watcher may already have removed the split-stage table after the
    # unified file is atomically written; final cleanup is intentionally idempotent.
    try:
        os.remove("cldf/parameters.csv")
    except FileNotFoundError:
        pass
    print(
        f"unified cldf/forms.csv: {len(etyma_rows)} etyma "
        f"({len(folded)} folded self-reflexes, {n_merged} merged addenda) + {n_reflex} reflexes "
        f"+ {n_variant} variants + {n_section} promoted section-forms + {n_ext} morpheme-bearing "
        f"derived entries + {n_deriv_flat} generic-derived reflexes + {n_svar} generated sound-variants "
        f"+ {n_replaced} replacement entries "
        f"+ {n_altern} alternate-etymology links + {n_borrowed} borrowed "
        f"+ {n_crossed} contamination-tagged + {n_lone} lone nodes; "
        f"applied {n_curated_borrowings} curated cross-dictionary borrowings; "
        f"attached {n_nuristani_reflexes} PNur/IA nodes as Proto-II reflexes; "
        f"moved {nuristani_reparented['moved']} CDIAL Nuristani reflexes from IA to PNur "
        f"({nuristani_reparented['single']} unambiguous, "
        f"{nuristani_reparented['multi']} routed among multiple PNur heads, "
        f"{nuristani_reparented['overridden']} manually disambiguated, "
        f"{nuristani_reparented['tied']} unresolved score ties); "
        f"built {len(burushaski_rows)} Proto-Burushaski entries from "
        f"{sum(len(row['Evidence_Keys'].split('|')) for row in burushaski_catalog)} dialect attestations; "
        f"applied {n_nuristani_borrowings} Strand OIA loan branches "
        f"with {n_nuristani_borrowed_descendants} direct borrowed descendants; "
        f"merged {n_strand_oia_redirects} duplicate Strand OIA heads and redirected "
        f"{n_strand_oia_references} references; "
        f"redirected {n_dbia_redirects} DBIA heads and {n_dbia_references} references; "
        f"marked {n_cross_family_borrowings} inferred cross-family borrowings; "
        f"removed parameters.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--borrowings-only"]:
        apply_borrowings_to_unified()
    else:
        main()
