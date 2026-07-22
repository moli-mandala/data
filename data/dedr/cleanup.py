import re


_FOOTER_REFERENCE = re.compile(r"\b(?:DED(?:S|\s*\([^)]*\))?|DEN)\b")


def is_footer_misparse(form):
    return bool(_FOOTER_REFERENCE.search(form))


def footer_note(form, gloss):
    note = " ".join(part.strip() for part in (form, gloss) if part.strip())
    note = re.sub(r"</?div[^>]*>", "", note)
    note = re.sub(r"\s+", " ", note)
    return note.lstrip(" .,;").strip()
