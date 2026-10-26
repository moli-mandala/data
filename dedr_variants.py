import itertools
import re
import unicodedata


_ATTACHED_PARENTHETICAL = re.compile(r"(?<=\S)\(([^()]*)\)")

_MACRON = "̄"
_BREVE = "̆"
_TILDE = "̃"
_MACRON_BREVE = _MACRON + _BREVE
# Burrow and Emeneau mark Toda, Kota, Kodagu and Kolami vowel length with a raised dot
# (``Ko. a·k``); the website prints it as the Greek ano teleia or a middle dot, sometimes
# after a space. House transcription is the macron.
_LENGTH_DOT = re.compile(r"([aeiouy])([\u0300-\u036f]*)\s*[\u0387\u00b7]")


def _dot_to_macron(match):
    letter, marks = match.group(1), match.group(2)
    if _MACRON in marks:
        return letter + marks  # already long
    return letter + _MACRON + marks


# Burrow and Emeneau's editorial parentheticals inside a cited form: source citations, usage,
# grammatical and transcription notes.  They belong in the row's notes, not the display form
# (the profile would otherwise transliterate "before vowel" as "before vovel").  A note either
# opens with one of these words or carries markup or a numeral (``<i>Voc.</i> 84``, ``3 pers.``,
# ``Su. 1971``).  Stem classes and alternate stems -- ``(-pp-, -tt-)``, ``(a·c-)`` -- are the
# dictionary's morphological notation and stay in the form.
_EDITORIAL_NOTE_START = re.compile("|".join((
    r"LSB ", r"MBE ", r"Gai, ", r"Burrow ", r"cf\. ", r"see ", r"for ", r"before consonant",
    r"before vowel", r"same meaning", r"in a song", r"in song", r"in all usages", r"coll\.",
    r"commonly ", r"construed ", r"in (?:some )?cpds", r"also ", r"[a-z] = [a-z]+$", r"\?$",
    r"irregular", r"i\.e\.", r"s\.v\.", r"past stem", r"pair word", r"used ", r"or with ",
)))
_PARENTHETICAL = re.compile(r"\s*\(([^()]*)\)")


def _is_editorial(content):
    return bool(
        _EDITORIAL_NOTE_START.match(content)
        or re.search(r"<|&lt;|&gt;|\d", content)
    )


def split_editorial_notes(form):
    """``(display form, note)``: the form without its editorial parentheticals, and those
    notes joined with ``; `` (markup stripped)."""
    notes = []

    def drop(match):
        if not _is_editorial(match.group(1)):
            return match.group(0)
        note = re.sub(r"<[^>]+>", "", match.group(1)).replace("&lt;", "<").replace("&gt;", ">")
        notes.append(note.strip())
        return ""

    display = _PARENTHETICAL.sub(drop, form).strip()
    return (display, "; ".join(notes)) if notes else (form, "")


def normalize_dedr_marks(form):
    """Canonicalise length and nasalisation marks so the profile's vowels match: the raised
    length dot becomes a macron (``a·k`` → ``āk``, ``ï·`` → ``ï̄``), a spacing
    tilde (``˜``, as in ``ī˜``) becomes a combining tilde (matching the CDIAL parser), and a
    tilde written before a macron (``ã̄`` = tilde+macron) is reordered to macron-then-tilde
    (``ā̃``), the order the profile lists."""
    s = unicodedata.normalize("NFD", canonical_dedr_marks(form))
    s = _LENGTH_DOT.sub(_dot_to_macron, s)  # raised length dot -> macron next to the letter
    s = s.replace(_TILDE + _MACRON, _MACRON + _TILDE)  # tilde+macron -> macron+tilde
    return unicodedata.normalize("NFC", s)


def canonical_dedr_marks(form):
    """The notation-only canonicalisation that Original keeps as well (so identities do not
    depend on how the website encoded a nasal mark): spacing tilde to combining tilde, and a
    tilde written before a macron reordered after it."""
    s = unicodedata.normalize("NFD", form)
    s = s.replace("˜", _TILDE)
    s = s.replace(_TILDE + _MACRON, _MACRON + _TILDE)
    return unicodedata.normalize("NFC", s)


def expand_length_variants(form):
    """A vowel written long-or-short (macron+breve, e.g. ``ā̆``) is attested with either
    length; emit it as two forms -- long (keep the macron) and short (drop both marks) --
    mirroring the CDIAL parser. Several such vowels expand combinatorially."""
    nfd = unicodedata.normalize("NFD", form)
    # canonicalise spacing breve and reversed mark order to a single macron+breve sequence
    nfd = nfd.replace(_MACRON + "˘", _MACRON_BREVE).replace(_BREVE + _MACRON, _MACRON_BREVE)
    if _MACRON_BREVE not in nfd:
        return [form]
    parts = nfd.split(_MACRON_BREVE)
    variants = [parts[0]]
    for part in parts[1:]:
        variants = [v + mark + part for v in variants for mark in (_MACRON, "")]
    return list(dict.fromkeys(unicodedata.normalize("NFC", v) for v in variants))


def _is_optional_sound(content):
    if not content or len(content) > 4 or content != content.lower():
        return False
    return all(
        character.isalpha() or unicodedata.category(character).startswith("M")
        for character in content
    )


def expand_attached_sound_variants(form):
    matches = [
        match
        for match in _ATTACHED_PARENTHETICAL.finditer(form)
        if _is_optional_sound(match.group(1))
    ]
    if not matches:
        return [form]

    pieces = []
    cursor = 0
    for match in matches:
        pieces.append(form[cursor : match.start()])
        cursor = match.end()
    pieces.append(form[cursor:])

    expanded = []
    for included in itertools.product((False, True), repeat=len(matches)):
        candidate = pieces[0]
        for index, include in enumerate(included):
            if include:
                candidate += matches[index].group(1)
            candidate += pieces[index + 1]
        if candidate not in expanded:
            expanded.append(candidate)
    return expanded
