"""Keep ingestion provenance out of public form notes.

The source CSVs and per-record audits deliberately retain extraction method, review state,
source-list classification, similarity-group labels, contributor names, and other information
needed to reproduce an ingest.  Those fields are useful locally but are not lexical notes and
must not be copied into CLDF ``Description`` or the browser database's ``notes`` column.

Sources are listed explicitly rather than filtered with prose heuristics.  That makes the public
boundary reviewable and avoids deleting genuine usage, grammatical, or etymological commentary
from unrelated dictionaries.  Add a source here only when its installed ``Notes`` payload is
fully represented by its checked-in audit/source artifacts.
"""

from __future__ import annotations

import re


# Sources whose Notes are exclusively reproducible extraction/audit metadata declare
# ``notes.audit_only: true`` in their YAML settings (source_meta.py).  Page, item, site, and
# upstream record identifiers already live in their CLDF citation locators, except for the three
# legacy sources normalized specially below.
import source_meta

AUDIT_ONLY_NOTE_SOURCES = source_meta.load().keys_where("notes", "audit_only")


def citation_keys(citation: str) -> tuple[str, ...]:
    """Return the bibliography keys from a semicolon-separated CLDF citation field."""
    return tuple(
        part.strip().split("[", 1)[0]
        for part in citation.split(";")
        if part.strip()
    )


def _append_citations(source: str, additions: list[str]) -> str:
    citations = [part.strip() for part in source.split(";") if part.strip()]
    for citation in additions:
        if citation not in citations:
            citations.append(citation)
    return ";".join(citations)


def _without_key(source: str, key: str) -> str:
    return ";".join(
        part.strip()
        for part in source.split(";")
        if part.strip() and part.strip().split("[", 1)[0] != key
    )


def apply_form_note_policy(
    notes: str, source: str, etymology: str
) -> tuple[str, str, str, tuple[str, ...]]:
    """Return public notes, citations, etymology, and any promoted tags.

    Call this after structured leading tokens have been extracted from ``notes``.  Unlisted
    sources pass through byte-for-byte.  Listed sources keep the full original payload in their
    local source/audit files, while the compiled public note is blank.
    """
    keys = citation_keys(source)
    policy_keys = [key for key in keys if key in AUDIT_ONLY_NOTE_SOURCES]
    if not policy_keys:
        return notes, source, etymology, ()

    key = policy_keys[0]

    promoted_tags: tuple[str, ...] = ()

    if key == "zoller2005":
        locations = []
        for chapter, page in re.findall(r"Zoller 2005 ch\. (\d+), p\. (\d+)", notes):
            location = f"ch. {chapter} p. {page}"
            if location not in locations:
                locations.append(location)
        if locations:
            # The legacy row cites a bare source key; replace it with exact printed locations.
            source = _append_citations(
                _without_key(source, key),
                [f"zoller2005[{', '.join(locations)}]"],
            )

    elif key == "shackle-auto":
        printed_pages = list(dict.fromkeys(re.findall(r"printed p\. (\d+)", notes)))
        if printed_pages:
            label = "p." if len(printed_pages) == 1 else "pp."
            source = _append_citations(
                _without_key(source, key),
                [f"shackle-auto[{label} {', '.join(printed_pages)}]"],
            )
        # A few auto rows merge with an older hand-entered/CDIAL attestation. Remove only the
        # generated OCR provenance there and retain the other source's genuine residual note.
        if any(other not in AUDIT_ONLY_NOTE_SOURCES for other in keys):
            notes = re.sub(
                r"(?:^|; )Shackle PDF p\. \d+ \(printed p\. \d+\)", "", notes
            )
            notes = re.sub(r"(?:^|; )etym\. \[.*?\](?=; |$)", "", notes)
            notes = re.sub(r"(?:^|; )auto-review: [^;]+", "", notes)
            notes = re.sub(r"^(?:;\s*)+|(?:;\s*)+$", "", notes).strip()
            notes = re.sub(r";\s*;", ";", notes).strip()

    elif key == "zubair":
        wordlists = list(dict.fromkeys(re.findall(r"Wordlist no\. (\d+)", notes)))
        if wordlists:
            label = "wordlist" if len(wordlists) == 1 else "wordlists"
            source = _append_citations(
                _without_key(source, key),
                [f"zubair[{label} {', '.join(wordlists)}]"],
            )
        if re.search(r"(?:^|; )\?(?:;|$)", notes):
            promoted_tags = ("uncertain",)

    if key == "shackle-auto" and any(
        other not in AUDIT_ONLY_NOTE_SOURCES for other in keys
    ):
        return notes, source, etymology, promoted_tags
    return "", source, etymology, promoted_tags
