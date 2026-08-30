"""Preserve hand-curated Parameter_ID values across legacy survey re-extraction."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def lexical_key(language: str, form: str, gloss: str, source: str) -> tuple[str, ...]:
    return (
        language.strip(),
        unicodedata.normalize("NFC", form).strip(),
        unicodedata.normalize("NFC", gloss).strip(),
        source.split("[", 1)[0].strip(),
    )


@dataclass
class CuratedParameters:
    exact: dict[tuple[str, ...], str]
    without_language: dict[tuple[str, ...], set[str]]


def load(path: Path) -> CuratedParameters:
    parameters: dict[tuple[str, ...], str] = {}
    without_language: dict[tuple[str, ...], set[str]] = {}
    if not path.exists():
        return CuratedParameters(parameters, without_language)
    with path.open(encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            if len(row) < 8 or not row[1]:
                continue
            key = lexical_key(row[0], row[2], row[3], row[7])
            previous = parameters.setdefault(key, row[1])
            if previous != row[1]:
                raise ValueError(
                    f"conflicting curated Parameter_ID values at {path}:{row_number}: "
                    f"{previous!r} versus {row[1]!r} for {key!r}"
                )
            without_language.setdefault(key[1:], set()).add(row[1])
    return CuratedParameters(parameters, without_language)


def apply(rows: list[list[str]], parameters: CuratedParameters) -> None:
    emitted_exact = set()
    emitted_without_language = set()
    for row in rows:
        key = lexical_key(row[0], row[2], row[3], row[7])
        emitted_exact.add(key)
        emitted_without_language.add(key[1:])
        candidates = parameters.without_language.get(key[1:], set())
        row[1] = parameters.exact.get(key, "")
        if not row[1] and len(candidates) == 1:
            row[1] = next(iter(candidates))
    missing = {
        key for key in parameters.exact
        if key not in emitted_exact and key[1:] not in emitted_without_language
    }
    if missing:
        raise ValueError(
            f"re-extraction dropped {len(missing)} curated survey responses; "
            f"examples: {sorted(missing)[:5]!r}"
        )
