import re
from dataclasses import dataclass


_PARADIGM = re.compile(r"^(?P<citation>.+?)\s+\((?P<morphology>[^()]*)\)$")
_TOKEN = re.compile(r"-?([^\s,;()-]+)-?")

_CLASS_BY_STEMS = {
    ("v", "t"): ("Tamil-class-1", "weak"),
    ("v", "nt"): ("Tamil-class-2", "weak"),
    ("v", "in"): ("Tamil-class-3", "weak"),
    ("p", "ṭ"): ("Tamil-class-5", "middle"),
    ("pp", "tt"): ("Tamil-class-6", "strong"),
    ("pp", "nt"): ("Tamil-class-7", "strong"),
}


@dataclass(frozen=True)
class TamilVerbMorphology:
    citation_form: str
    note: str
    tags: tuple[str, ...]
    review_reason: str = ""


def _classify_segment(segment):
    tokens = tuple(token for token in _TOKEN.findall(segment) if token)
    if len(tokens) != 2:
        return None
    return _CLASS_BY_STEMS.get(tokens)


def _classify_full_paradigm(citation, morphology):
    if citation.endswith("u") and morphology == f"{citation[:-1]}i-":
        return ("Tamil-class-3", "weak")

    if citation.endswith("ṭu") and morphology == f"{citation}v-, {citation[:-2]}ṭṭ-":
        return ("Tamil-class-4", "weak")
    if citation.endswith("ṟu") and morphology == f"{citation}v-, {citation[:-2]}ṟṟ-":
        return ("Tamil-class-4", "weak")

    stems = [stem.strip().removesuffix("-") for stem in morphology.split(",")]
    if (
        len(stems) == 2
        and stems[0].endswith("p")
        and not stems[0].endswith("pp")
        and stems[1]
    ):
        return ("Tamil-class-5", "middle")

    return None


def extract_tamil_verb_morphology(form):
    match = _PARADIGM.match(form.strip())
    if not match:
        return None

    morphology = match.group("morphology").strip()
    if not morphology.endswith("-") or not re.search(r"\w", morphology):
        return None

    citation = match.group("citation").strip()
    classifications = []
    unclassified = []
    full_classification = _classify_full_paradigm(citation, morphology)
    if full_classification:
        classifications.append(full_classification)
    else:
        for segment in morphology.split(";"):
            classification = _classify_segment(segment.strip())
            if classification is None:
                unclassified.append(segment.strip())
            else:
                classifications.append(classification)

    tags = ["verb"]
    for class_tag, strength_tag in classifications:
        if class_tag not in tags:
            tags.append(class_tag)
        if strength_tag not in tags:
            tags.append(strength_tag)

    if not classifications:
        review_reason = "unclassified full-stem or irregular paradigm"
    elif unclassified:
        review_reason = "partially classified paradigm"
    else:
        review_reason = ""

    return TamilVerbMorphology(
        citation_form=citation,
        note=f"({morphology})",
        tags=tuple(tags),
        review_reason=review_reason,
    )


def append_note(notes, note):
    return "; ".join(part for part in (notes.strip(), note) if part)
