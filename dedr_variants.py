import itertools
import re
import unicodedata


_ATTACHED_PARENTHETICAL = re.compile(r"(?<=\S)\(([^()]*)\)")


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
