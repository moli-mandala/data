"""Parse OAD raw units (see ``segment.py``) into structured records.

A unit is flattened into a character string with a parallel font-class list so that bold
headwords, bold-italic cross-reference forms and roman prose stay distinguishable.
"""

from __future__ import annotations

import re
import unicodedata as ud

SUPERSCRIPT_DIGITS = "¹²³⁴⁵⁶⁷⁸⁹⁰"

# Printed grammatical labels (Symbols and Abbreviations, pp. XIII-XV) -> Jambu tags.
# Only labels with an existing canonical tag are mapped; ``preverb`` is proposed (proposals.md)
# and emitted only once registered (``EMIT_PROPOSED_TAGS``).
LABELS = {
    "adj": ["adj"], "m": ["m"], "n": ["n"], "f": ["f"], "mn": ["mn"],
    "adv": ["adv"], "indecl": ["indecl"], "ptcl": ["part"], "conj": ["conj"],
    "prep": ["prep"], "postp": ["postp"], "interj": ["interj"], "num": ["num"],
    "neg": ["neg"], "prefix": ["prefix"], "pron": ["pron"], "encl": ["clitic"],
    "pers.pron": ["pron", "personal"], "dem.pron": ["pron", "demonstrative"],
    "rel.pron": ["pron", "relative"], "poss.pron": ["pron", "poss"],
    "indef.pron": ["pron", "indef"], "interrog.pron": ["pron", "interr"],
    "interrog": ["interr"], "interrog.adv": ["adv", "interr"], "rel.adv": ["adv", "relative"],
    "rel": ["relative"],
    "pron.adj": ["pron", "adj"], "verb.adj": ["adj"], "verb. adj": ["adj"],
    "des.adj": ["adj", "desiderative"], "ptcpl": ["participle"], "denom": ["denom"],
    "pres": ["pres"], "perf": ["perfect"], "PN": ["proper-noun"],
    "personal name": ["proper-noun"], "pvb": ["preverb"], "sg": ["sg"], "pl": ["pl"],
    "du": ["du"], "superl": [], "comp": [], "compar": [], "passim": [],
    "intensive": ["intensive"], "emph": ["emph"], "enclit": ["clitic"], "negation": ["neg"],
}
PROPOSED_TAGS = {"preverb"}
# words that may accompany labels without carrying a tag (government, scope)
LABEL_FILLERS = {"with", "instr", "acc", "loc", "gen", "dat", "abl", "+", "or", "and",
                 "short", "passim", "governing", "of"}

CASES = {"N": "nom", "A": "acc", "G": "gen", "D": "dat", "Abl": "abl", "I": "instr",
         "Inst": "instr", "L": "loc", "V": "voc"}
PERSON = re.compile(r"^([123])([sp])\.?(?:pers\.pron\.?)?$")
LOCATOR = re.compile(r"(?:―\s*)?(?:\([^()]{1,30}\)\s*)?\[?(?:YH|Y)?\d+[,.]\d+")


def flatten(unit: dict) -> tuple[str, list[str]]:
    """Join a unit's lines into one string plus a per-character font class list.

    A line-final hyphen followed by a lowercase letter is joined without a space (the
    print's line-wrap of hyphenated words keeps its hyphen); other breaks become one space."""
    chars: list[str] = []
    classes: list[str] = []
    for line in unit["lines"]:
        spans = line["spans"]
        first = "".join(t for _, t in spans).lstrip()
        if chars:
            while chars and chars[-1] == " ":
                chars.pop(); classes.pop()
            if not (chars and chars[-1] == "-" and first[:1].islower()):
                chars.append(" "); classes.append("R")
        for cls, text in spans:
            for ch in text:
                if ch.isspace():
                    if chars and chars[-1] == " ":
                        continue
                    ch = " "
                chars.append(ch); classes.append(cls)
    while chars and chars[-1] == " ":
        chars.pop(); classes.pop()
    # extraction already NFC-normalizes each span; per-character classes must stay aligned
    return "".join(chars), classes


def raw_text(unit: dict) -> str:
    return " ⏎ ".join("".join(t for _, t in l["spans"]).strip() for l in unit["lines"])


def top_level_groups(text: str, start: int, end: int):
    """Yield ``(kind, begin, stop)`` items at bracket depth 0: ``paren``/``quote``/``text``.

    Square brackets (non-Old-Avestan material) are transparent. A ``’`` followed by a letter is
    an apostrophe, not a closing quote."""
    i = start
    while i < end:
        ch = text[i]
        if ch == "(":
            depth, j = 1, i + 1
            while j < end and depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            yield "paren", i, j
            i = j
        elif ch == "‘":
            depth, j = 1, i + 1
            while j < end and depth:
                if text[j] == "‘":
                    depth += 1
                elif text[j] == "’" and not (j + 1 < end and text[j + 1].isalpha()):
                    depth -= 1
                j += 1
            yield "quote", i, j
            i = j
        else:
            j = i
            while j < end and text[j] not in "(‘":
                j += 1
            yield "text", i, j
            i = j


def _lookup(token: str) -> list[str] | None:
    token = token.strip("()")
    if token in LABELS:
        return LABELS[token]
    person = re.fullmatch(r"([123])([sdp])", token)
    if person:  # no person-dual tags exist, so a dual person code adds nothing
        return [] if person.group(2) == "d" else [
            person.group(1) + ("sg" if person.group(2) == "s" else "pl")]
    return None


def _segment(pieces: list[str]) -> list[str] | None:
    """Cover dot-separated abbreviations with known labels (``1s.pers.pron`` -> 1s + pers.pron)."""
    if not pieces:
        return []
    for cut in range(len(pieces), 0, -1):
        head = _lookup(".".join(pieces[:cut]))
        if head is not None:
            rest = _segment(pieces[cut:])
            if rest is not None:
                return head + rest
    return None


def label_tags(content: str) -> tuple[list[str], bool]:
    """Map a parenthesised label group to tags; ``ok`` is False when any token is unknown."""
    content = content.strip()
    if not content:
        return [], False
    tags: list[str] = []
    norm = re.sub(r"\s+or\s+", "/", content)
    if norm in ("m./n.", "m/n", "mn"):
        return ["mn"], True
    if norm in ("m./f.", "m/f", "f./m."):
        return ["mf"], True
    if norm.rstrip(".") in LABELS:
        return list(LABELS[norm.rstrip(".")]), True
    for token in re.split(r"[/,]\s*|\s+", norm):
        token = token.strip().rstrip(".")
        if not token:
            continue
        found = _lookup(token)
        if found is None and "." in token:
            found = _segment([p for p in token.split(".") if p])
        if found is None:
            if token.lower() in LABEL_FILLERS:
                continue
            return tags, False
        for tag in found:
            if tag not in tags:
                tags.append(tag)
    return tags, bool(tags) or content.rstrip(".") in ("superl", "comp")


def grammar_code_tags(content: str) -> tuple[list[str], bool]:
    """``(2s.N)``, ``(1sD)``, ``(2s.Abl)`` on pronoun lemmata -> person-number + case tags."""
    m = re.fullmatch(r"([123])([sp])\.?\s*(Abl|Inst|[NAGDILV])", content.strip())
    if not m:
        return [], False
    return [f"{m.group(1)}{'sg' if m.group(2) == 's' else 'pl'}", CASES[m.group(3)]], True


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
