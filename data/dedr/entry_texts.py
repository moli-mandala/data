#!/usr/bin/env python3
"""Preserve DEDR article commentary without duplicating its parsed reflex inventory.

The ordinary DEDR importer expands language-labelled lexical material into reflex rows, while
``data.cross_family`` owns every comparison with a resolved CDIAL target.  This companion
extractor keeps only the residual source prose: unresolved/source-only comparison notes, inline
same-family ``Cf.`` notes, botanical/editorial brackets, and old DED/DEDS/DEN locators.  Each
block retains the current DEDR article locator and promotes any printed older-edition locator
into its structured CLDF citation.

Run without ``--install`` to inspect ``tmp/dedr-entry-texts``.  ``--install`` writes the entry-text
sidecar plus its complete per-article audit and fixed review sample.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from data.cross_family import (
    SourceEntry,
    _citation,
    dedr_citation_locators,
    dedr_source_segments,
    source_entries,
    source_text,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMPARISON_AUDIT = ROOT / "data/cross-family-comparisons-audit.csv"
INSTALLED_TEXTS = ROOT / "data/other/entry_texts/20260819-dedr.csv"
INSTALLED_REFERENCES = HERE / "entry-references.csv"
INSTALLED_AUDIT = HERE / "entry-texts-audit.csv.gz"
INSTALLED_SAMPLE = HERE / "entry-texts-sample.csv"
INSTALLED_MANIFEST = HERE / "entry-texts-manifest.json"
TMP_OUTPUT = ROOT / "tmp/dedr-entry-texts"
SAMPLE_SEED = 20260819
SAMPLE_SIZE = 20

ENTRY_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]
REFERENCE_FIELDS = ["Form_ID", "Source"]
AUDIT_FIELDS = [
    "Entry_ID",
    "Page",
    "Citation",
    "Status",
    "Block_Count",
    "Structured_Comparison_Count",
    "Kinds",
    "Extracted_Content",
    "Raw_Article",
    "Reason",
]

INLINE_CF_RE = re.compile(r"(?<!\w)(?:\?\s*)?Cf\.\s+")
SUBSECTION_RE = re.compile(r"(?<!\w)\(\s*[b-z]\s*\)\s+(?=[A-Z][A-Za-z]{0,6}\.)")


@dataclass(frozen=True)
class EntryText:
    Form_ID: str
    Position: int
    Kind: str
    Format: str
    Content: str
    Source: str


@dataclass(frozen=True)
class EntryReference:
    Form_ID: str
    Source: str


@dataclass(frozen=True)
class Audit:
    Entry_ID: str
    Page: int
    Citation: str
    Status: str
    Block_Count: int
    Structured_Comparison_Count: int
    Kinds: str
    Extracted_Content: str
    Raw_Article: str
    Reason: str


@dataclass(frozen=True)
class Fragment:
    start: int
    kind: str
    content: str


def _structured_evidence() -> dict[str, set[str]]:
    by_entry: dict[str, set[str]] = defaultdict(set)
    with COMPARISON_AUDIT.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Source_Dictionary"] == "dedr" and row["Status"] == "installed":
                by_entry[row["Source_Entry_ID"]].add(row["Evidence"].strip())
    return by_entry


def _locate(text: str, fragment: str, cursor: int = 0) -> tuple[int, int]:
    start = text.find(fragment, cursor)
    if start < 0:
        start = text.find(fragment)
    return (start, start + len(fragment)) if start >= 0 else (-1, -1)


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in ranges)


def _terminal_source_note(
    text: str, occupied: list[tuple[int, int]]
) -> Fragment | None:
    """Return a terminal old-edition locator, with an immediately preceding bracketed note."""
    matches = list(re.finditer(
        r"\b(?:DEDS|DED\s*(?:\(\s*[SN](?:\s*[,\.]\s*N)?\s*\))?|DEN)\s+"
        r"(?:\d+(?:\s*\(\s*[a-z]\s*\))?|DBIA\s+[A-Z]+)",
        text,
        re.IGNORECASE,
    ))
    if not matches:
        return None
    final = matches[-1]
    if text[final.end():].strip(" .;,"):
        return None
    first = final
    # A footer may list several adjacent older-edition locators.
    for candidate in reversed(matches[:-1]):
        between = text[candidate.end():first.start()]
        if re.fullmatch(r"[\s,;]*(?:and\s+)?", between, re.IGNORECASE):
            first = candidate
        else:
            break
    start = first.start()
    prefix = text[:start].rstrip()
    if prefix.endswith("]"):
        bracket_start = prefix.rfind("[")
        if bracket_start >= 0:
            start = bracket_start
    # A bare old-edition number belongs on the entry's structured citation, not in a display
    # block.  Only preserve a source note here when the printed article also contains prose.
    if start == first.start() or _overlaps(start, len(text), occupied):
        return None
    return Fragment(start, "source-note", text[start:].strip(" /"))


def extract_entry(
    entry: SourceEntry, structured: set[str]
) -> tuple[list[EntryText], Audit]:
    full_text = source_text(entry.html, keep_lines=True)
    fragments: list[Fragment] = []
    occupied: list[tuple[int, int]] = []

    # Mark every installed structured claim, including inline CDIAL notes, so it never reappears
    # as generic prose.  Several targets may share the same printed passage.
    for evidence in sorted(structured, key=len, reverse=True):
        start, end = _locate(full_text, evidence)
        if start >= 0:
            occupied.append((start, end))

    # Source-level slash notes are commentary, not reflexes.  Keep only those which could not be
    # represented as a resolved cross-family pair; installed claims already have their own block.
    cursor = 0
    for segment in dedr_source_segments(entry):
        start, end = _locate(full_text, segment, cursor)
        if start >= 0:
            cursor = end
            occupied.append((start, end))
        if segment not in structured:
            fragments.append(Fragment(max(start, 0), "comparison", segment))

    # DEDR also prints same-family comparisons inline at the end of its ordinary reflex run.
    # Capture from the cue to the next source note/subsection, never the preceding reflexes.
    claimed = occupied[:]
    for match in INLINE_CF_RE.finditer(full_text):
        if _overlaps(match.start(), match.end(), claimed):
            continue
        ends = [start for start, _ in claimed if start > match.start()]
        subsection = SUBSECTION_RE.search(full_text, match.end())
        if subsection:
            ends.append(subsection.start())
        end = min(ends) if ends else len(full_text)
        content = full_text[match.start():end].strip(" /\n")
        if not content:
            continue
        fragments.append(Fragment(match.start(), "comparison", content))
        claimed.append((match.start(), end))

    occupied.extend((fragment.start, fragment.start + len(fragment.content)) for fragment in fragments)
    terminal = _terminal_source_note(full_text, occupied)
    if terminal:
        fragments.append(terminal)

    # A malformed source article can expose the same tail through two structural paths.  Retain
    # its first source position only and keep output order deterministic.
    unique: dict[str, Fragment] = {}
    for fragment in sorted(fragments, key=lambda value: (value.start, value.kind, value.content)):
        content = re.sub(r"\s+", " ", fragment.content).strip()
        if content and content not in unique:
            unique[content] = Fragment(fragment.start, fragment.kind, content)

    blocks = [
        EntryText(
            entry.entry_id,
            fragment.start,
            fragment.kind,
            "text",
            fragment.content,
            _citation(entry, fragment.content),
        )
        for fragment in unique.values()
    ]
    if blocks:
        status = "installed"
        reason = "residual DEDR commentary preserved after reflex and structured-comparison removal"
    elif structured:
        status = "structured-only"
        reason = "all non-reflex prose is represented by structured cross-family comparisons"
    else:
        status = "reflex-only"
        reason = "article contains no detected residual commentary"
    audit = Audit(
        entry.entry_id,
        entry.page,
        _citation(entry, full_text),
        status,
        len(blocks),
        len(structured),
        "|".join(block.Kind for block in blocks),
        " || ".join(block.Content for block in blocks),
        full_text,
        reason,
    )
    return blocks, audit


def build() -> tuple[list[EntryText], list[EntryReference], list[Audit]]:
    evidence = _structured_evidence()
    blocks: list[EntryText] = []
    references: list[EntryReference] = []
    audits: list[Audit] = []
    seen_entries: dict[str, int] = {}
    for entry in source_entries("dedr"):
        if entry.entry_id in seen_entries:
            raw_article = source_text(entry.html, keep_lines=True)
            audits.append(Audit(
                entry.entry_id,
                entry.page,
                _citation(entry, raw_article),
                "duplicate-excluded",
                0,
                0,
                "",
                "",
                raw_article,
                f"duplicate website article number; canonical occurrence is on page {seen_entries[entry.entry_id]}",
            ))
            continue
        seen_entries[entry.entry_id] = entry.page
        entry_blocks, audit = extract_entry(entry, evidence.get(entry.entry_id, set()))
        blocks.extend(entry_blocks)
        references.append(EntryReference(entry.entry_id, _citation(entry, audit.Raw_Article)))
        audits.append(audit)
    blocks.sort(key=lambda row: (row.Form_ID, row.Position, row.Content))
    audits.sort(key=lambda row: row.Entry_ID)
    references.sort(key=lambda row: row.Form_ID)
    return blocks, references, audits


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_audit(path: Path, audits: list[Audit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in audits)


def write_outputs(
    output: Path | None,
    blocks: list[EntryText],
    references: list[EntryReference],
    audits: list[Audit],
) -> None:
    if output is None:
        texts_path = INSTALLED_TEXTS
        references_path = INSTALLED_REFERENCES
        audit_path = INSTALLED_AUDIT
        sample_path = INSTALLED_SAMPLE
        manifest_path = INSTALLED_MANIFEST
    else:
        texts_path = output / INSTALLED_TEXTS.name
        references_path = output / INSTALLED_REFERENCES.name
        audit_path = output / INSTALLED_AUDIT.name
        sample_path = output / INSTALLED_SAMPLE.name
        manifest_path = output / INSTALLED_MANIFEST.name

    _write_csv(texts_path, ENTRY_FIELDS, [asdict(row) for row in blocks])
    _write_csv(references_path, REFERENCE_FIELDS, [asdict(row) for row in references])
    _write_audit(audit_path, audits)
    installed = [row for row in audits if row.Status == "installed"]
    sample = random.Random(SAMPLE_SEED).sample(installed, min(SAMPLE_SIZE, len(installed)))
    sample_rows = [{**asdict(row), "Review": "ok"} for row in sample]
    _write_csv(sample_path, AUDIT_FIELDS + ["Review"], sample_rows)

    status_counts = Counter(row.Status for row in audits)
    kind_counts = Counter(block.Kind for block in blocks)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_articles": len(audits),
                "installed_blocks": len(blocks),
                "entry_references": len(references),
                "entries_with_blocks": len({block.Form_ID for block in blocks}),
                "status_counts": dict(sorted(status_counts.items())),
                "kind_counts": dict(sorted(kind_counts.items())),
                "sample_seed": SAMPLE_SEED,
                "sample_size": len(sample),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    blocks, references, audits = build()
    output = None if args.install else TMP_OUTPUT
    write_outputs(output, blocks, references, audits)
    destination = INSTALLED_TEXTS.parent if args.install else output
    counts = Counter(row.Status for row in audits)
    print(f"wrote {len(blocks):,} DEDR text blocks to {destination}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count:,}")


if __name__ == "__main__":
    main()
