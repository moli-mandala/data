"""Build Jambu rows from Liljegren's complete Palula CLDF dictionary.

The old ``20220913-palula.csv`` was an etymology concordance: it retained only
entries which could be attached to a CDIAL number.  This importer instead reads
all 2,700 dictionary entries.  Entries without a Turner/CDIAL reference receive
a blank Parameter_ID and are therefore retained by ``make_cldf.py`` as lone,
unetymologised nodes.

Download the source data from https://github.com/dictionaria/palula (tag v1.2),
then run, for example::

    uv run python data/other/forms/raw_data/liljegren.py /path/to/palula/cldf

The generated CSV uses Jambu's rich 15-column manual-ingestion schema.  It
preserves Liljegren's inflectional description in Notes while also exposing
parts of speech, nominal declensions, verbal classes, and loans as searchable
tags.  The source dataset is CC BY 4.0 and is archived as Liljegren (2019),
Dictionaria 3, version 1.2: https://doi.org/10.5281/zenodo.5526477.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Sequence
from pathlib import Path


SOURCE_ID = "liljegren"
LANGUAGE_ID = "Phal"
BIORI_LANGUAGE_ID = "biori"
DATA_ROOT = Path(__file__).resolve().parents[4]

TURNER_GROUP_RE = re.compile(r"\(\s*T\s*[:.]?\s*([^)]+)\)", re.I)

POS_TAGS: dict[str, tuple[str, ...]] = {
    "n": ("noun",),
    "adj": ("adj",),
    "adv": ("adv",),
    "v": ("verb",),
    "pron": ("pron",),
    "num": ("num",),
    "post": ("postp",),
    "conj": ("conj",),
    "interj": ("interj",),
    "det": ("determiner",),
    "disc": ("discourse-marker",),
    "aux": ("auxiliary",),
    "neg": ("negator", "neg"),
    "mood": ("mood-marker",),
    "hon": ("honorific",),
    "sfx": ("suffix",),
    "MWE": ("multiword-expression",),
}

POS_FEATURE_TAGS: dict[str, tuple[str, ...]] = {
    "masc": ("m",),
    "fem": ("f",),
    "tr": ("tr",),
    "intr": ("intr",),
    "pass": ("pass",),
    "caus": ("caus",),
    "pn": ("proper-noun",),
    "inv": ("indecl",),
    "dem": ("demonstrative",),
    # Liljegren describes this class as indefinite-interrogative pro-forms.
    "ind": ("indef", "interr"),
    "pers": ("personal",),
    "refl": ("refl",),
    "recp": ("reciprocal",),
    "cop": ("copula",),
    "mod": ("modal",),
    "cjt": ("compound", "conjunct-verb"),
    "inc": ("incorporating",),
    "ninc": ("non-incorporating",),
}

ADVERB_FEATURE_TAGS = {
    "tm": "temporal",
    "sp": "spatial",
    "mann": "manner",
    "deg": "degree",
    "sent": "sentential",
}

NOUN_CLASS_TAGS = {
    "a": "Palula-noun-class-a",
    "i": "Palula-noun-class-i",
    "m": "Palula-noun-class-m",
    "aan": "Palula-noun-class-aan",
    "ee": "Palula-noun-class-ee",
}

VERB_CLASS_TAGS = {
    "a": "Palula-verb-class-L-a",
    "e": "Palula-verb-class-L-e",
    "cons": "Palula-verb-class-L-consonant",
    "con": "Palula-verb-class-L-consonant",  # one source typo
    "minor": "Palula-verb-class-L-minor",
}

LOAN_LANGUAGES = {
    "Arabic", "English", "Khowar", "Panjabi", "Pashto", "Persian",
    "Sinitic", "Turkish", "Turkic", "Urdu",
}


def canonical(tags: Iterable[str]) -> list[str]:
    """Return nonempty tags in source order without duplicates."""
    return list(dict.fromkeys(tag for tag in tags if tag))


def grammatical_tags(part_of_speech: str, inflection: str = "") -> list[str]:
    """Translate Liljegren's POS and inflection labels into Jambu tags."""
    parts = part_of_speech.split(":") if part_of_speech else []
    base = parts[0] if parts else ""
    tags = list(POS_TAGS.get(base, ()))

    # Dotted labels encode both the major category and a first subclass,
    # e.g. n.masc, v.tr, and adv.tm.
    dotted = base.split(".")
    if dotted[0] in POS_TAGS:
        tags = list(POS_TAGS[dotted[0]])
    features = dotted[1:] + [
        feature for section in parts[1:] for feature in section.split(".")
    ]
    for feature in features:
        tags.extend(POS_FEATURE_TAGS.get(feature, ()))
        if dotted[0] == "adv" and feature in ADVERB_FEATURE_TAGS:
            tags.append(ADVERB_FEATURE_TAGS[feature])

    if "noun" in tags:
        for noun_class in re.findall(r"(?:^|[/ ])(aan|ee|a|i|m)-decl", inflection):
            tags.append(NOUN_CLASS_TAGS[noun_class])
        if re.search(r"\bIrr\b|/irr\b", inflection, re.I):
            tags.append("Palula-noun-class-irregular")

    if "verb" in tags or "auxiliary" in tags:
        for verb_class in re.findall(r"\bL:(cons|con|minor|e|a)\b", inflection):
            tags.append(VERB_CLASS_TAGS[verb_class])
        if re.search(r"(?:^|[/ ])T(?:\s|$|/)", inflection):
            tags.append("Palula-verb-class-T")
        if re.search(r"\bSuppl\b", inflection):
            tags.append("Palula-verb-class-suppletive")

    return canonical(tags)


def loan_tags(origin: str) -> list[str]:
    """Expose every explicitly named donor/source language in Origin."""
    languages = [language for language in sorted(LOAN_LANGUAGES) if re.search(
        rf"\b{re.escape(language)}\b", origin, re.I
    )]
    if not languages:
        return []
    return ["loanword", *(f"loan:{language}" for language in languages)]


def turner_parameters(proto_form: str) -> list[str]:
    """Return every Turner number cited by the source, in source order."""
    parameters = []
    for group in TURNER_GROUP_RE.findall(proto_form):
        parameters.extend(re.findall(r"\d+[a-z]?", group, re.I))
    if not parameters:
        # One entry has a bare number in an explicitly Indo-Aryan analysis.
        match = re.search(r"\((\d{3,5}[a-z]?)\)\s*$", proto_form)
        if match:
            parameters.append(match.group(1))
    return canonical(parameters)


def turner_parameter(proto_form: str) -> str:
    """Backward-compatible convenience accessor for the first Turner number."""
    parameters = turner_parameters(proto_form)
    return parameters[0] if parameters else ""


def split_variants(value: str) -> list[tuple[str, str]]:
    """Return ``(form, qualifier)`` pairs from Variant_Form.

    Semicolons delimit qualifier groups.  A final parenthesis qualifies every
    comma-separated form in its group, so ``ak, a (Biori)`` yields two Biori
    variants rather than attaching the dialect only to the second one.
    """
    variants: list[tuple[str, str]] = []
    for group in re.split(r"\s*;\s*", value.strip()):
        if not group:
            continue
        qualifier = ""
        # Qualifiers can themselves contain parentheses (e.g. ``(With a
        # closed class of (motion) verbs)``), so find the opening mate of the
        # final close parenthesis rather than using a flat regular expression.
        if group.endswith(")"):
            depth = 0
            for position in range(len(group) - 1, -1, -1):
                if group[position] == ")":
                    depth += 1
                elif group[position] == "(":
                    depth -= 1
                    if depth == 0:
                        qualifier = group[position + 1 : -1].strip()
                        group = group[:position].strip()
                        break
        for form in re.split(r"\s*,\s*", group):
            if form:
                variants.append((form, qualifier))
    return variants


def notes_for(entry: dict[str, str], qualifier: str = "") -> str:
    fields = []
    for label, column in (
        ("Inflection", "Inflection_Class"),
        ("Morphemic form", "Morphemic_Form"),
        ("Restriction", "Restrictions"),
        ("Usage", "Usage"),
    ):
        if entry[column]:
            fields.append(f"{label}: {entry[column]}")
    if qualifier:
        fields.append(f"Variant restriction: {qualifier}")
    if entry["Part_Of_Speech"] == "?":
        fields.append("Dictionary POS: ?")
    return "; ".join(fields)


def etymology_for(entry: dict[str, str]) -> str:
    parts = []
    if entry["IndoAryan_Proto_Form"]:
        parts.append(entry["IndoAryan_Proto_Form"])
    if entry["Origin"]:
        parts.append("Origin: " + entry["Origin"])
    return "; ".join(parts)


def import_rows(cldf_dir: Path) -> Iterable[list[str]]:
    """Yield complete rich-schema Jambu rows from a Palula CLDF directory."""
    with (cldf_dir / "senses.csv").open(encoding="utf-8", newline="") as stream:
        senses = {row["Entry_ID"]: row["Description"] for row in csv.DictReader(stream)}

    with (cldf_dir / "entries.csv").open(encoding="utf-8", newline="") as stream:
        for entry in csv.DictReader(stream):
            source_key = f"liljegren-{entry['ID']}"
            parameters = turner_parameters(entry["IndoAryan_Proto_Form"]) or [""]
            tags = grammatical_tags(entry["Part_Of_Speech"], entry["Inflection_Class"])
            tags.extend(loan_tags(entry["Origin"]))
            tags = canonical(tags)
            gloss = senses[entry["ID"]]
            etymology = etymology_for(entry)

            variants = split_variants(entry["Variant_Form"])
            for parameter_index, parameter in enumerate(parameters):
                parameter_key = (
                    source_key if parameter_index == 0 else f"{source_key}-turner-{parameter}"
                )
                yield [
                    LANGUAGE_ID, parameter, entry["Headword"], gloss,
                    entry["Vernacular"], entry["Phonetic"], notes_for(entry), SOURCE_ID,
                    "", etymology, parameter_key, "", "", "", " ".join(tags),
                ]

                seen = {entry["Headword"]}
                for index, (form, qualifier) in enumerate(variants, 1):
                    if form in seen:
                        continue
                    seen.add(form)
                    is_biori = bool(re.search(r"\bB(?:iori)?\b", qualifier, re.I))
                    variant_tags = canonical(tags + (["dialect:Biori"] if is_biori else []))
                    yield [
                        BIORI_LANGUAGE_ID if is_biori else LANGUAGE_ID,
                        parameter, form, gloss, "", "", notes_for(entry, qualifier), SOURCE_ID,
                        "", etymology, f"{parameter_key}-variant-{index}", parameter_key,
                        "", "", " ".join(variant_tags),
                    ]


def write_csv(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cldf_dir", type=Path, help="palula v1.2 cldf directory")
    parser.add_argument(
        "--output", type=Path,
        default=DATA_ROOT / "data/other/forms/20220913-palula.csv",
    )
    args = parser.parse_args()
    rows = list(import_rows(args.cldf_dir))
    write_csv(args.output, rows)
    print(
        f"Wrote {len(rows):,} rows from 2,700 entries to {args.output} "
        f"({sum(not row[1] for row in rows):,} unetymologised rows)"
    )


if __name__ == "__main__":
    main()
