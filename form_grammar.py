"""Conservative grammatical-tag extraction for legacy form-source glosses.

Several survey and glossary importers predate the rich 15-column schema.  They preserve
source-supplied grammatical labels in ``Gloss`` rather than ``Tags``.  This module provides the
single, source-scoped normalization layer used by ``make_cldf.py`` and by source importers as they
are modernized.  It deliberately does not infer categories from ordinary English meanings.
"""

from __future__ import annotations

import re


# Only inputs whose YAML settings declare ``gloss.grammar_tags: true`` have been checked against
# their parser/source representation.  The scope matters: parenthetical prose in an unrelated
# dictionary must not turn into grammar merely because it contains a short token such as ``v``.
import source_meta


_BRACKETED = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]")
_SPACE = re.compile(r"\s+")

_WORD_TAGS = {
    "v": ("verb",),
    "vb": ("verb",),
    "verb": ("verb",),
    "n": ("noun",),
    "noun": ("noun",),
    "a": ("adj",),
    "adj": ("adj",),
    "adjective": ("adj",),
    "adv": ("adv",),
    "adverb": ("adv",),
    "pro": ("pron",),
    "pron": ("pron",),
    "pronoun": ("pron",),
    "num": ("num",),
    "numeral": ("num",),
    "tr": ("verb", "tr"),
    "transitive": ("verb", "tr"),
    "intr": ("verb", "intr"),
    "intransitive": ("verb", "intr"),
    "sg": ("sg",),
    "sing": ("sg",),
    "singular": ("sg",),
    "pl": ("pl",),
    "plural": ("pl",),
    "dual": ("du",),
    "m": ("m",),
    "masc": ("m",),
    "masculine": ("m",),
    "f": ("f",),
    "fem": ("f",),
    "feminine": ("f",),
    "neut": ("n",),
    "neuter": ("n",),
    "past": ("pret",),
    "pt": ("pret",),
    "present": ("pres",),
    "prs": ("pres",),
    "future": ("fut",),
    "fut": ("fut",),
    "neg": ("neg",),
    "negative": ("neg",),
    "cmd": ("impv",),
    "command": ("impv",),
    "imperative": ("impv",),
    "formal": ("formal",),
    "form": ("formal",),
    "informal": ("informal",),
    "inform": ("informal",),
    "honorific": ("honorific",),
    "h": ("honorific",),
    "inclusive": ("inclusive",),
    "incl": ("inclusive",),
    "exclusive": ("exclusive",),
    "excl": ("exclusive",),
}

_FILLER = {"and", "or", "many", "more", "tense"}


def _normalized_annotation(value: str) -> str:
    return (
        value.casefold()
        .replace("sɡ", "sg")
        .replace("pʟ", "pl")
        .replace("–", "-")
        .replace("—", "-")
        .strip(" .;:")
    )


def annotation_tags(value: str) -> tuple[str, ...]:
    """Return tags only when *all* annotation material is grammatical."""
    text = _normalized_annotation(value)
    if not text:
        return ()

    tags: list[str] = []

    # Compact person-number and survey tense labels: 3S-PT, 2S-neg, 1p, 2 sg.
    person_number = r"(?<![a-z0-9])([123])(?:st|nd|rd)?\s*(s|p|sg|pl|singular|plural)(?![a-z])"
    for person, number in re.findall(person_number, text):
        tags.append(person + ("sg" if number in {"s", "sg", "singular"} else "pl"))
    text = re.sub(person_number, " ", text)

    # A bare ordinal person occurs in a few prompts whose number is genuinely unspecified.
    # Keep it in the gloss rather than inventing singular/plural.
    if re.search(r"(?<![a-z0-9])[123](?:st|nd|rd)(?![a-z0-9])", text):
        return ()

    if "near future" in text:
        tags.extend(("fut", "near-future"))
        text = text.replace("near future", " ")
    if re.fullmatch(r"3\s+or\s+more", text):
        return ("pl",)

    # In person-marked survey prompts ``inf`` means informal, not infinitive.
    person_context = any(re.fullmatch(r"[123](?:sg|pl)", tag) for tag in tags)
    tokens = [token for token in re.split(r"[\s,/+-]+", text) if token]
    for raw in tokens:
        token = raw.rstrip(".")
        if token in _FILLER:
            continue
        if token == "inf" and person_context:
            tags.append("informal")
            continue
        if token == "inf" and not person_context:
            tags.append("inf")
            continue
        mapped = _WORD_TAGS.get(token)
        if not mapped:
            return ()
        tags.extend(mapped)

    return tuple(dict.fromkeys(tags))


def _source_defined_gloss_tags(gloss: str, source_key: str) -> tuple[str, ...]:
    """Labels printed in source tables but omitted by the earliest snapshot scripts."""
    if not source_meta.load().flag(source_key, "gloss", "source_defined_tags"):
        return ()
    lexical = re.sub(r"\s+", " ", gloss.casefold()).strip()
    lexical = re.sub(r"^to\s+", "", lexical)
    verbs = {
        "drink", "eat", "bite", "see", "hear", "know", "sleep", "die", "kill",
        "swim", "fly", "walk", "come", "lie (down)", "sit", "stand", "give", "say",
        "burn", "sleep, lie down",
    }
    return ("verb",) if lexical in verbs else ()


def extract_gloss_tags(
    gloss: str,
    *,
    input_file: str,
    source_key: str,
    full_input_path: str = "",
) -> tuple[str, tuple[str, ...]]:
    """Separate checked source grammatical annotations from a lexical gloss."""
    if not source_meta.load().file_flag(full_input_path or input_file, "gloss", "grammar_tags"):
        return gloss, ()

    tags = list(_source_defined_gloss_tags(gloss, source_key))

    def remove_if_grammar(match: re.Match[str]) -> str:
        annotation = match.group(1) or match.group(2) or ""
        parsed = annotation_tags(annotation)
        if not parsed:
            return match.group(0)
        tags.extend(parsed)
        return ""

    cleaned = _BRACKETED.sub(remove_if_grammar, gloss)

    # Some comparative tables use the grammatical prompt itself as the entire gloss (``1st pl.``).
    if cleaned.strip() == gloss.strip():
        parsed = annotation_tags(gloss)
        if parsed:
            tags.extend(parsed)
            cleaned = ""

    # A few table headings print bare terminal number labels (not parenthesized), notably
    # Eastern Magar ``you plural``.  Restrict this to the unambiguous full words.
    terminal = re.search(r"\s+(singular|plural)$", cleaned, re.IGNORECASE)
    if terminal:
        tags.extend(_WORD_TAGS[terminal.group(1).casefold()])
        cleaned = cleaned[: terminal.start()]

    cleaned = _SPACE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+([,;:.])", r"\1", cleaned).strip(" ,;")
    return cleaned, tuple(dict.fromkeys(tags))
