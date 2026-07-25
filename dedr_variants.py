import itertools
import re
import unicodedata


_ATTACHED_PARENTHETICAL = re.compile(r"(?<=\S)\(([^()]*)\)")

_MACRON = "̄"
_BREVE = "̆"
_TILDE = "̃"
_MACRON_BREVE = _MACRON + _BREVE


def normalize_dedr_marks(form):
    """Canonicalise nasalisation marks so the profile's nasalised vowels match: a spacing
    tilde (``˜``, as in ``ī˜``) becomes a combining tilde (matching the CDIAL parser), and a
    tilde written before a macron (``ã̄`` = tilde+macron) is reordered to macron-then-tilde
    (``ā̃``), the order the profile lists."""
    s = unicodedata.normalize("NFD", form)
    s = s.replace("˜", _TILDE)  # spacing tilde -> combining tilde
    s = s.replace(_TILDE + _MACRON, _MACRON + _TILDE)  # tilde+macron -> macron+tilde
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
