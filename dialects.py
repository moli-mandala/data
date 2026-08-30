"""Shared helpers for the explicit dialect registry.

Raw importers may retain source-specific lect IDs.  The compiled CLDF uses the
parent language as ``Language_ID`` and records the lect as a structured tag.
``cldf/dialects.csv`` is the translation and metadata boundary between them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import quote, unquote


def dialect_tag(language_id: str, source_language_id: str, name: str) -> str:
    """Return a stable, globally unique tag for a source-defined dialect."""
    return (
        f"dialect:{quote(language_id, safe='')}:{quote(source_language_id, safe='')}:"
        f"{quote(name, safe='')}"
    )


def tag_label(tag: str) -> str:
    """Return the display label carried by either simple or qualified dialect tags."""
    if not tag.startswith("dialect:"):
        return tag
    return unquote(tag.rsplit(":", 1)[-1])


def load_dialect_aliases(path: str | Path = "cldf/dialects.csv") -> dict[str, dict[str, str]]:
    """Index source lect IDs which must be normalized during compilation."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            row["Source_Language_ID"]: row
            for row in csv.DictReader(stream)
            if row.get("Source_Language_ID")
        }


def normalize_dialect(
    language_id: str,
    tags: str,
    aliases: dict[str, dict[str, str]],
    registered_language_ids: set[str] | None = None,
) -> tuple[str, str]:
    """Map a source lect to its parent language and preserve it as a dialect tag.

    A source importer may already have emitted a short tag such as
    ``dialect:Biori``.  When the registry supplies the qualified canonical tag,
    replace only that same-label tag and retain any genuinely additional dialect
    information.
    """
    # New importers emit a globally qualified dialect tag. It is source-local and therefore
    # authoritative even when the raw lect code is a tiny value such as M, L or K that occurs
    # in several unrelated surveys. The second tag component is the registered parent language.
    qualified_parents = {
        unquote(parts[1])
        for token in (tags or "").split()
        if token.startswith("dialect:")
        and len((parts := token.split(":", 3))) == 4
    }
    if len(qualified_parents) == 1:
        return qualified_parents.pop(), tags

    # A real registered language ID must not be reinterpreted as another source's short site
    # code. Legacy unregistered source lect IDs continue through the alias table below.
    if registered_language_ids is not None and language_id in registered_language_ids:
        return language_id, tags

    dialect = aliases.get(language_id)
    if not dialect:
        return language_id, tags

    canonical_tag = dialect["Tag"]
    label = dialect["Name"]
    tokens = [
        tag
        for tag in (tags or "").split()
        if not (tag.startswith("dialect:") and tag_label(tag) == label)
    ]
    tokens.append(canonical_tag)
    return dialect["Language_ID"], " ".join(dict.fromkeys(tokens))
