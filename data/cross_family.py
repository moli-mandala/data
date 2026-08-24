#!/usr/bin/env python3
"""Extract source-attributed Dravidian<->Indo-Aryan comparison claims.

This is deliberately separate from the ancestry graph.  DEDR's Sanskrit/Indo-Aryan examples and
CDIAL's Dravidian examples are evidence for article-level comparisons; they are not ordinary
reflex rows, and a hedged comparison is not an accepted borrowing edge.  Southworth's paired
DEDR and Turner references are likewise retained as claims by that paper, independently of the
Marathi borrowing edges installed from its first table.

Run without ``--install`` to inspect regenerated files under ``tmp/cross-family-comparisons``.
Use ``--install`` to replace the checked-in source table and audits under ``data/``.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import pickle
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEDR = HERE / "dedr"
CDIAL = HERE / "cdial"
OVERRIDES = HERE / "cross-family-comparison-overrides.csv"
SOUTHWORTH_IMPORTER = HERE / "other/forms/raw_data/southworth_marathi.py"
TMP_OUTPUT = ROOT / "tmp" / "cross-family-comparisons"
APPENDIX_PAGE = 509
SAMPLE_SEED = 20260818
SAMPLE_SIZE = 20

COMPARISON_FIELDS = [
    "ID", "Entry_ID", "Compared_Entry_ID", "Relation", "Direction", "Confidence",
    "Source", "Evidence",
]
AUDIT_FIELDS = [
    "Source_Dictionary", "Source_Entry_ID", "Source_Page", "Printed_Target_ID",
    "Resolved_Target_ID", "Status", "Reason", "Resolution", "Relation", "Direction",
    "Confidence", "Citation", "Evidence",
]
OVERRIDE_FIELDS = [
    "Source_Dictionary", "Source_Entry_ID", "Printed_Target_ID", "Resolved_Target_ID",
    "Reason",
]

DEDR_CITATION_RE = re.compile(r"\b(DED(?:S)?)(?:\s*\([^)]*\))?\s+(\d+)")
DEDR_LOCATOR_RE = re.compile(
    r"\b(?P<label>DEDS|DED\s*(?:\(\s*[SN](?:\s*[,\.]\s*N)?\s*\))?|DEN)"
    r"\s+(?P<item>\d+(?:\s*\(\s*[a-z]\s*\))?|DBIA\s+[A-Z]+)",
    re.IGNORECASE,
)
CDIAL_MARKER_RE = re.compile(r"\bCDIAL\b", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"\bnos?\s*[.;,:]?\s*(\d+[a-z]?)(?:\s*[\-–]\s*(\d+[a-z]?))?",
    re.IGNORECASE,
)
DEDR_FOOTER_RE = re.compile(r"\bDED(?:S)?(?:\s*\([^)]*\))?\s+\d+", re.IGNORECASE)
IA_MARKER_RE = re.compile(
    r"\b(?:CDIAL|Turner|Skt|Sanskrit|Pkt|Prakrit|Pali|IA|NIA|Indo-Aryan)\b",
    re.IGNORECASE,
)
DRAV_MARKER_RE = re.compile(
    r"\b(?:Drav|Dravidian|Tam|Tamil|Kan|Kannada|Tel|Telugu|Mal|Malayalam|Tu|Tulu|"
    r"Brah|Brahui|Kur|Kurukh|Kol|Kolami|Gond[iī]?|Parji|Naiki|Gadba)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceEntry:
    dictionary: str
    entry_id: str
    printed_id: str
    page: int
    html: str


@dataclass
class Comparison:
    ID: str
    Entry_ID: str
    Compared_Entry_ID: str
    Relation: str
    Direction: str
    Confidence: str
    Source: str
    Evidence: str


@dataclass
class Audit:
    Source_Dictionary: str
    Source_Entry_ID: str
    Source_Page: str
    Printed_Target_ID: str
    Resolved_Target_ID: str
    Status: str
    Reason: str
    Resolution: str
    Relation: str
    Direction: str
    Confidence: str
    Citation: str
    Evidence: str


def _clean_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def source_text(fragment: str, *, keep_lines: bool = False) -> str:
    if keep_lines:
        # Source HTML contains layout newlines inside paragraphs.  Only <br> is a semantic article
        # boundary, so remove those layout breaks before giving <br> its own sentinel newline.
        fragment = fragment.replace("\n", " ")
    soup = BeautifulSoup(fragment, "html.parser")
    if keep_lines:
        for br in soup.find_all("br"):
            br.replace_with("\n")
    return _clean_text(soup.get_text(" ", strip=True))


def source_entries(dictionary: str) -> list[SourceEntry]:
    source_dir = DEDR if dictionary == "dedr" else CDIAL
    with (source_dir / f"{dictionary}.pickle").open("rb") as handle:
        pages = pickle.load(handle)

    result: list[SourceEntry] = []
    for page_number, page in enumerate(pages, 1):
        chunks = str(page).split("<number>")[1:]
        numbers = [
            chunk.split("</number>", 1)[0].strip()
            for chunk in chunks
            if "</number>" in chunk
        ]
        for index, chunk in enumerate(chunks):
            if "</number>" not in chunk:
                continue
            printed = chunk.split("</number>", 1)[0].strip()
            # DEDR page snapshots occasionally carry a truncated record immediately before its
            # complete continuation.  Match parse.py and retain only the final occurrence.
            if dictionary == "dedr" and printed in numbers[index + 1:]:
                continue
            fragment = "<number>" + chunk
            if "</hw>" in fragment:
                fragment = fragment.split("</hw>", 1)[0]
            local_id = ("a" if dictionary == "dedr" and page_number >= APPENDIX_PAGE else "") + printed
            entry_id = ("d" + local_id) if dictionary == "dedr" else printed
            result.append(SourceEntry(dictionary, entry_id, printed, page_number, fragment))
    return result


def read_entry_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0] for row in csv.reader(handle) if row}


def read_overrides(path: Path = OVERRIDES) -> dict[tuple[str, str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0].keys()) != OVERRIDE_FIELDS:
        raise ValueError(f"unexpected override schema in {path}")
    return {
        (row["Source_Dictionary"], row["Source_Entry_ID"], row["Printed_Target_ID"]):
        (row["Resolved_Target_ID"], row["Reason"])
        for row in rows
    }


def _expand_number(first: str, last: str | None) -> list[str]:
    if not last:
        return [first]
    if not first.isdigit() or not last.isdigit():
        return [first, last]
    start = int(first)
    if len(last) < len(first):
        prefix = first[: len(first) - len(last)]
        end = int(prefix + last)
    else:
        end = int(last)
    if end < start or end - start > 20:
        return [first, last]
    return [str(number) for number in range(start, end + 1)]


def cdial_numbers(text: str) -> list[str]:
    """Return numbers scoped to printed CDIAL mentions, not unrelated numbered notes."""
    found: list[str] = []
    markers = list(CDIAL_MARKER_RE.finditer(text))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        footer = DEDR_FOOTER_RE.search(text, marker.end(), end)
        if footer:
            end = footer.start()
        other_source = re.search(r"\bDBIA\b", text[marker.end():end], re.IGNORECASE)
        if other_source:
            end = marker.end() + other_source.start()
        for number in NUMBER_RE.finditer(text, marker.end(), end):
            found.extend(_expand_number(number.group(1), number.group(2)))
    return list(dict.fromkeys(found))


def _uncertain_direction(text: str) -> bool:
    return bool(re.search(r"direction.{0,45}uncertain|immediate relationship.{0,80}problem", text, re.I))


def classify_claim(text: str, dictionary: str) -> tuple[str, str, str]:
    """Classify the source's wording, not the parser's confidence in resolving its target."""
    compact = " ".join(text.split())
    lower = compact.lower()

    if "influenc" in lower or re.search(r"\binfl\.?\s+by\b", lower):
        relation = "influence"
    elif re.search(r"\b(?:borrow|loan|deriv)", lower) or "←" in compact or "→" in compact \
            or re.search(r"(?:^|[;/()])\s*(?:prob\.?|poss\.?|possibly|probably|\?)?\s*[<>]", compact, re.I) \
            or re.search(r"\b(?:may\s+be|prob(?:ably)?\.?|poss(?:ibly)?\.?)\s*[<>]", compact, re.I) \
            or re.search(
                r"(?:<|from)\s*(?:IA|NIA|Skt|Sanskrit|Pali|Pkt|Prakrit|Panj|Lahnda|Mar)\b|"
                r"\bor\s+(?:prob(?:ably)?\.?\s+|poss(?:ibly)?\.?\s+)?(?:IA|NIA|Indo-Aryan)\b",
                compact,
                re.I,
            ):
        relation = "loan"
    else:
        relation = "related"

    direction = "undetermined"
    if not _uncertain_direction(compact):
        if dictionary == "dedr":
            # A statement that the IA comparanda themselves come from Dravidian has priority:
            # an entry can also contain ``< IA`` for one of its individual reflexes.
            if re.search(
                r"\b(?:IA|NIA|Indo-Aryan)\s+(?:words?|forms?|items?)\s+(?:are\s+)?"
                r"(?:prob(?:ably)?\.?\s+|poss(?:ibly)?\.?\s+)?(?:<|←|from)\s*(?:Dr|Drav|Dravidian)\b",
                compact,
                re.I,
            ):
                direction = "compared-from-entry"
            elif re.search(r"(?:^|[;/()])\s*(?:prob\.?|poss\.?|possibly|probably|\?)?\s*(?:<|←)", compact, re.I) \
                    or re.search(r"\bmay\s+be\s*(?:<|←)\s*(?:IA|NIA|Skt|Sanskrit)", compact, re.I) \
                    or re.search(
                        r"\b(?:borrow\w*|deriv\w*)\s+from\s+(?:IA|NIA|Skt|Sanskrit|Pali|Pkt|Prakrit)",
                        compact,
                        re.I,
                    ) \
                    or re.search(
                        r"(?:<|from)\s*(?:IA|NIA|Skt|Sanskrit|Pali|Pkt|Prakrit|Panj|Lahnda|Mar)\b",
                        compact,
                        re.I,
                    ) \
                    or re.search(
                        r"\bor\s+(?:prob(?:ably)?\.?\s+|poss(?:ibly)?\.?\s+)?(?:IA|NIA|Indo-Aryan)\b",
                        compact,
                        re.I,
                    ) \
                    or re.search(r"\bfrom which (?:these words are|are borrowed)", compact, re.I):
                direction = "entry-from-compared"
            elif re.search(r"(?:^|[;/])\s*(?:prob\.?|poss\.?|possibly|probably|\?)?\s*(?:>|→)", compact, re.I):
                direction = "compared-from-entry"
        else:
            # Prefer an explicit statement that the cited Dravidian material is from IA over an
            # earlier alternative in the same passage (CDIAL 6087 is the canonical example).
            if re.search(r"\bDrav\.?\s+(?:rather\s+)?←\s*(?:IA|NIA|Sk|Skt|Sanskrit)\b", compact, re.I):
                direction = "compared-from-entry"
            elif re.search(
                r"(?:Drav|Tamil|Tam|Kannada|Kan|Telugu|Tel|Mal|Tulu|Tu)[^.;\]]{0,180}"
                r"(?:←|borrowed from|derived from)\s*(?:IA|NIA|Sk|Skt|Sanskrit)",
                compact,
                re.I,
            ):
                direction = "compared-from-entry"
            elif re.search(
                r"(?:←|borrowed from|derived from|influenced by)\s*(?:Drav|Dravidian|Tamil|Tam|"
                r"Kannada|Kan|Telugu|Tel|Mal|Tulu|Tu)", compact, re.I,
            ):
                direction = "entry-from-compared"
            elif re.search(
                r"(?:borrow\w*\s+from|infl\.?\s+by)\s*(?:Drav|Dravidian|Tamil|Tam|Kannada|Kan|"
                r"Telugu|Tel|Mal|Tulu|Tu)", compact, re.I,
            ):
                direction = "entry-from-compared"
            elif re.search(r"(?:are\s+)?←\s*(?:IA|NIA|Sk|Skt|Sanskrit)\b", compact, re.I):
                direction = "compared-from-entry"
            elif re.search(r"(?:→)\s*(?:Drav|Dravidian|Tamil|Tam|Kannada|Kan|Telugu|Tel|Mal|Tulu|Tu)", compact, re.I):
                direction = "compared-from-entry"

    if _uncertain_direction(compact) or re.search(
        r"\bdoubt(?:ful)?\b|\bproblems?\b|\buncertain\b|\bwith \(\?\)|\?\s*(?:<|←)|\bCf\.",
        compact,
        re.I,
    ):
        confidence = "low"
    elif re.search(r"\b(?:prob|poss|perh|perhaps|possibly|probably|may)\b|influenc|\binfl\.?\b|\brather\b", lower):
        confidence = "medium"
    elif relation == "loan" and direction != "undetermined":
        confidence = "high"
    else:
        confidence = "medium"
    return relation, direction, confidence


def _comparison_id(dictionary: str, entry_id: str, target_id: str) -> str:
    other = "cdial" if dictionary == "dedr" else "dedr"
    return f"{dictionary}:{entry_id}:{other}:{target_id}"


def dedr_citation_locators(text: str) -> list[str]:
    """Return source-faithful old DED/DEDS/DEN article locators in print order."""
    locators: list[str] = []
    for match in DEDR_LOCATOR_RE.finditer(text):
        label = re.sub(r"\s+", "", match.group("label").upper())
        label = label.replace(".", ",")
        if label.startswith("DED("):
            inside = label[4:-1].replace(",", ", ")
            label = f"DED({inside})"
        item = re.sub(r"\s+", "", match.group("item"))
        if item.startswith("DBIA"):
            item = re.sub(r"\s+", " ", match.group("item").upper())
        locator = f"{label} {item}"
        if locator not in locators:
            locators.append(locator)
    return locators


def _citation(entry: SourceEntry, evidence: str = "") -> str:
    key = "dedr" if entry.dictionary == "dedr" else "CDIAL"
    locator = (
        f"appendix entry {entry.printed_id}"
        if entry.dictionary == "dedr" and entry.entry_id.startswith("da")
        else f"entry {entry.printed_id}"
    )
    locators = [locator]
    if entry.dictionary == "dedr":
        locators.extend(dedr_citation_locators(evidence))
    return f"{key}[{', '.join(locators)}]"


def _load_southworth_importer():
    """Load the checked Southworth table transcription as the single source of row data."""
    module_name = "jambu_southworth_comparison_source"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SOUTHWORTH_IMPORTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Southworth importer at {SOUTHWORTH_IMPORTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_southworth(
    dedr_ids: set[str], cdial_ids: set[str]
) -> tuple[list[Comparison], list[Audit]]:
    """Pair the DEDR and Turner entries printed for the same Southworth table item.

    Table 1 identifies the proposed Dravidian source and Table 2 identifies an Indo-Aryan
    comparison only for seven cross-referenced items. Other Table 2 rows lack a printed DEDR
    counterpart in the ingested scope and therefore remain prose blocks rather than inferred
    graph or comparison links.
    """
    source = _load_southworth_importer()
    table1 = {row.ordinal: row for row in source.TABLE1}
    comparisons: list[Comparison] = []
    audits: list[Audit] = []

    for row in source.TABLE2:
        if row.table1_ordinal is None or not row.targets:
            continue
        dravidian = table1[row.table1_ordinal]
        dedr_id = f"d{dravidian.dedr}"
        confidence = "low" if dravidian.uncertain else "high"
        evidence = (
            f"Table 1 row {dravidian.ordinal}: Marathi {dravidian.form} "
            f"'{dravidian.gloss}', {dravidian.source_class}, {dravidian.proto_label} "
            f"{dravidian.proto_form}, DEDR {dravidian.dedr}. "
            f"Table 2{row.section} item {row.item}: "
            f"{row.source_class or 'source class not repeated'}, {row.form} "
            f"'{row.gloss}', Turner {row.printed_citation}."
        )
        citation = (
            f"{source.SOURCE_ID}[p. 9, table 1, row {dravidian.ordinal}; "
            f"p. 10, table 2{row.section}, item {row.item}]"
        )
        for cdial_id in row.targets:
            missing = [
                entry_id
                for entry_id, known in ((dedr_id, dedr_ids), (cdial_id, cdial_ids))
                if entry_id not in known
            ]
            status = "unresolved" if missing else "installed"
            reason = (
                f"resolved checked cross-table IDs {dedr_id} and {cdial_id}"
                if not missing
                else "missing compiled endpoint(s): " + "|".join(missing)
            )
            audits.append(Audit(
                "southworth", row.record_key, "9--10",
                f"DEDR {dravidian.dedr}; Turner {row.printed_citation}",
                f"{dedr_id}|{cdial_id}" if not missing else "",
                status, reason, "source-image-verified-cross-table-pair",
                "loan", "compared-from-entry", confidence, citation, evidence,
            ))
            if missing:
                continue
            comparisons.append(Comparison(
                f"southworth2005m:t2{row.section.lower()}:{row.item:02d}:"
                f"dedr:{dedr_id}:cdial:{cdial_id}",
                dedr_id, cdial_id, "loan", "compared-from-entry", confidence,
                citation, evidence,
            ))
    return comparisons, audits


def _dedr_source_slash(markup: str, match: re.Match) -> bool:
    """Whether a spaced slash is a comparison-note boundary rather than a form/gloss slash."""
    inside_bold = (
        markup.rfind("<b>", 0, match.start())
        > markup.rfind("</b>", 0, match.start())
    )
    if inside_bold:
        return False
    following = source_text(markup[match.end():])
    if not following:
        return False
    comparison_cue = bool(re.match(
        r"(?:cf|perhaps|prob|poss|borrow|loan|influenc|relationship|areal)\b",
        following,
        re.IGNORECASE,
    ))
    return not following[0].islower() or comparison_cue


def dedr_source_segments(entry: SourceEntry) -> list[str]:
    """Return only DEDR's source-level slash notes, excluding lexical/gloss alternatives."""
    markup = entry.html.replace("\n", " ")
    separators = [
        match for match in re.finditer(r"(?:\s+|(?<=[.;:]))/\s*", markup)
        if _dedr_source_slash(markup, match)
    ]
    segments = []
    for index, separator in enumerate(separators):
        end = separators[index + 1].start() if index + 1 < len(separators) else len(markup)
        fragment = markup[separator.end():end]
        # A new labelled DEDR subsection can follow the comparison note without another slash.
        subsection = re.search(r"<b><i>\([a-z]\)(?:\s|</i>)", fragment)
        if subsection:
            fragment = fragment[:subsection.start()]
        text = source_text(fragment, keep_lines=True)
        # The source marks later lexical subsections in several malformed-HTML shapes
        # (``<b><i>(b) Ka.</i>``, ``<i>(b) <b>Ta.</b></i>``, and plain variants).  The rendered
        # text is more stable than those tags: stop before the next labelled reflex inventory.
        text_subsection = re.search(
            r"(?<!\w)\(\s*[b-z]\s*\)\s+(?=[A-Z][^\s]{0,12}\.?\s)", text
        )
        if text_subsection:
            text = text[:text_subsection.start()].rstrip()
        if text:
            segments.append(text)
    return segments


def _negative_cdial(text: str) -> bool:
    return bool(re.search(r"(?:not in|no entry in)\s+Turner,?\s*CDIAL", text, re.I))


def extract_dedr(
    entries: list[SourceEntry], cdial_ids: set[str]
) -> tuple[list[Comparison], list[Audit]]:
    comparisons: list[Comparison] = []
    audits: list[Audit] = []
    seen_unresolved: set[tuple[str, str]] = set()
    for entry in entries:
        seen_numbers: set[str] = set()
        seen_contexts: set[str] = set()
        for segment in dedr_source_segments(entry):
            seen_contexts.add(segment)
            relation, direction, confidence = classify_claim(segment, "dedr")
            numbers = cdial_numbers(segment)
            if numbers:
                for number in numbers:
                    seen_numbers.add(number)
                    resolved = number if number in cdial_ids else ""
                    status = "installed" if resolved else "unresolved"
                    reason = "printed CDIAL ID" if resolved else "printed CDIAL ID is absent from CDIAL"
                    audits.append(Audit(
                        "dedr", entry.entry_id, str(entry.page), number, resolved, status, reason,
                        "printed-id" if resolved else "missing-target", relation, direction,
                        confidence, _citation(entry, segment), segment,
                    ))
                    if resolved:
                        comparisons.append(Comparison(
                            _comparison_id("dedr", entry.entry_id, resolved), entry.entry_id,
                            resolved, relation, direction, confidence, _citation(entry, segment), segment,
                        ))
            else:
                key = (entry.entry_id, segment)
                if key in seen_unresolved:
                    continue
                seen_unresolved.add(key)
                negative = _negative_cdial(segment)
                audits.append(Audit(
                    "dedr", entry.entry_id, str(entry.page), "", "",
                    "excluded" if negative else "unresolved",
                    "source explicitly says the comparison is absent from CDIAL" if negative
                    else "Indo-Aryan comparison has no resolvable CDIAL entry ID",
                    "negative-citation" if negative else "no-target-id", relation, direction,
                    confidence, _citation(entry, segment), segment,
                ))

        # Some DEDR comparisons are parenthetical or introduced by ``Cf.`` rather than a slash.
        # Scan every explicit CDIAL marker so those claims are not dependent on layout punctuation.
        full_text = source_text(entry.html, keep_lines=True)
        for marker in CDIAL_MARKER_RE.finditer(full_text):
            context = _claim_context(full_text, marker.start(), marker.end())
            numbers = cdial_numbers(context)
            relation, direction, confidence = classify_claim(context, "dedr")
            for number in numbers:
                if number in seen_numbers:
                    continue
                seen_numbers.add(number)
                resolved = number if number in cdial_ids else ""
                audits.append(Audit(
                    "dedr", entry.entry_id, str(entry.page), number, resolved,
                    "installed" if resolved else "unresolved",
                    "printed CDIAL ID" if resolved else "printed CDIAL ID is absent from CDIAL",
                    "printed-id" if resolved else "missing-target", relation, direction,
                    confidence, _citation(entry, context), context,
                ))
                if resolved:
                    comparisons.append(Comparison(
                        _comparison_id("dedr", entry.entry_id, resolved), entry.entry_id,
                        resolved, relation, direction, confidence, _citation(entry, context), context,
                    ))
            if not numbers and context not in seen_contexts:
                key = (entry.entry_id, context)
                if key in seen_unresolved:
                    continue
                seen_unresolved.add(key)
                negative = _negative_cdial(context)
                audits.append(Audit(
                    "dedr", entry.entry_id, str(entry.page), "", "",
                    "excluded" if negative else "unresolved",
                    "source explicitly says the comparison is absent from CDIAL" if negative
                    else "CDIAL comparison has no resolvable entry ID",
                    "negative-citation" if negative else "no-target-id", relation, direction,
                    confidence, _citation(entry, context), context,
                ))
    return comparisons, audits


def build_ded_resolver(entries: list[SourceEntry]) -> dict[tuple[str, str], set[str]]:
    """Map CDIAL's old DED/DEDS numbers onto current DEDR entry IDs.

    ``DED(S)`` and ``DED(S,N)`` are annotations on the main DED numbering, whereas the printed
    abbreviation ``DEDS`` denotes the separately numbered supplement.
    """
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in entries:
        text = source_text(entry.html)
        for match in DEDR_CITATION_RE.finditer(text):
            edition = "supplement" if match.group(1) == "DEDS" else "main"
            result[(edition, match.group(2))].add(entry.entry_id)
    return result


def _claim_context(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    bracket_start = text.rfind("[", line_start, start)
    if bracket_start >= 0 and text.rfind("]", line_start, start) < bracket_start:
        line_start = bracket_start
        bracket_end = text.find("]", end)
        if bracket_end >= 0:
            line_end = min(line_end, bracket_end + 1)
    # DEDR often puts an entire borrowing assertion in parentheses inside a long article.  Keep
    # that assertion instead of hundreds of unrelated lexical words, but do not reduce ordinary
    # parenthetical CDIAL citations whose comparison wording occurs outside the parentheses.
    stack: list[int] = []
    for position, character in enumerate(text[line_start:start], line_start):
        if character == "(":
            stack.append(position)
        elif character == ")" and stack:
            stack.pop()
    if stack:
        paren_start = stack[-1]
        depth = 1
        paren_end = -1
        for position in range(start, line_end):
            if text[position] == "(":
                depth += 1
            elif text[position] == ")":
                depth -= 1
                if depth == 0:
                    paren_end = position + 1
                    break
        if paren_end > 0:
            candidate = text[paren_start:paren_end]
            without_marker = CDIAL_MARKER_RE.sub("", candidate)
            if re.search(
                r"(?:<|>|←|→|\bIA\b|\bNIA\b|Indo-Aryan|Dravidian|\bDrav\.?\b|"
                r"borrow|loan|origin|deriv|influenc)",
                without_marker,
                re.I,
            ):
                line_start, line_end = paren_start, paren_end
    if line_end - line_start > 1400:
        line_start = max(line_start, start - 650)
        line_end = min(line_end, end + 650)
    return " ".join(text[line_start:line_end].split()).strip(" ;")


def extract_cdial(
    entries: list[SourceEntry], resolver: dict[tuple[str, str], set[str]],
    overrides: dict[tuple[str, str, str], tuple[str, str]],
) -> tuple[list[Comparison], list[Audit]]:
    comparisons: list[Comparison] = []
    audits: list[Audit] = []
    articles_with_claim = set()
    articles_with_drav = set()
    for entry in entries:
        text = source_text(entry.html, keep_lines=True)
        if DRAV_MARKER_RE.search(text):
            articles_with_drav.add(entry.entry_id)
        matches = list(DEDR_CITATION_RE.finditer(text))
        if matches:
            articles_with_claim.add(entry.entry_id)
        for match in matches:
            evidence = _claim_context(text, match.start(), match.end())
            relation, direction, confidence = classify_claim(evidence, "cdial")
            printed = match.group(2)
            edition = "supplement" if match.group(1) == "DEDS" else "main"
            candidates = resolver.get((edition, printed), set())
            override = overrides.get(("cdial", entry.entry_id, printed))
            resolved = ""
            resolution = ""
            reason = ""
            if override:
                target, override_reason = override
                if candidates and target not in candidates:
                    raise ValueError(
                        f"override {entry.entry_id}/{printed} -> {target} is not among {sorted(candidates)}"
                    )
                resolved = target
                resolution = "manual-override"
                reason = override_reason
            elif len(candidates) == 1:
                resolved = next(iter(candidates))
                resolution = "unique-legacy-id"
                reason = "printed DED/DEDS ID uniquely resolves through the DEDR footer"
            elif not candidates:
                resolution = "missing-target"
                reason = "printed DED/DEDS ID has no DEDR footer match"
            else:
                resolution = "ambiguous-target"
                reason = "printed DED/DEDS ID resolves to multiple DEDR entries: " + "|".join(sorted(candidates))

            audits.append(Audit(
                "cdial", entry.entry_id, str(entry.page),
                ("DEDS " if edition == "supplement" else "DED ") + printed,
                resolved, "installed" if resolved else "unresolved", reason, resolution,
                relation, direction, confidence, _citation(entry, evidence), evidence,
            ))
            if resolved:
                comparisons.append(Comparison(
                    _comparison_id("cdial", entry.entry_id, resolved), entry.entry_id,
                    resolved, relation, direction, confidence, _citation(entry, evidence), evidence,
                ))

    # A Dravidian passage without an old DED number cannot be linked conservatively to a DEDR
    # entry. Keep it visible in the audit rather than turning its cited word into a reflex row.
    unresolved_articles = articles_with_drav - articles_with_claim
    for entry in entries:
        if entry.entry_id not in unresolved_articles:
            continue
        text = source_text(entry.html, keep_lines=True)
        marker = DRAV_MARKER_RE.search(text)
        if not marker:
            continue
        evidence = _claim_context(text, marker.start(), marker.end())
        relation, direction, confidence = classify_claim(evidence, "cdial")
        audits.append(Audit(
            "cdial", entry.entry_id, str(entry.page), "", "", "unresolved",
            "Dravidian comparison has no resolvable DED/DEDS entry ID", "no-target-id",
            relation, direction, confidence, _citation(entry, evidence), evidence,
        ))
    return comparisons, audits


def dedupe_comparisons(rows: list[Comparison]) -> list[Comparison]:
    merged: dict[tuple[str, str, str], Comparison] = {}
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    for row in rows:
        key = (row.Entry_ID, row.Compared_Entry_ID, row.Source)
        old = merged.get(key)
        if old is None:
            merged[key] = row
            continue
        if row.Evidence not in old.Evidence:
            old.Evidence += " | " + row.Evidence
        if old.Direction != row.Direction:
            old.Direction = "undetermined"
            old.Confidence = "low"
        elif confidence_rank[row.Confidence] < confidence_rank[old.Confidence]:
            old.Confidence = row.Confidence
        if old.Relation != row.Relation:
            old.Relation = "related"
    return sorted(merged.values(), key=lambda row: (row.Entry_ID, row.Compared_Entry_ID, row.Source))


def build() -> tuple[list[Comparison], list[Audit]]:
    dedr_entries = source_entries("dedr")
    cdial_entries = source_entries("cdial")
    cdial_ids = read_entry_ids(CDIAL / "params.csv")
    dedr_ids = read_entry_ids(DEDR / "params.csv")
    overrides = read_overrides()

    dedr_rows, dedr_audit = extract_dedr(dedr_entries, cdial_ids)
    resolver = build_ded_resolver(dedr_entries)
    cdial_rows, cdial_audit = extract_cdial(cdial_entries, resolver, overrides)
    southworth_rows, southworth_audit = extract_southworth(dedr_ids, cdial_ids)
    comparisons = dedupe_comparisons(dedr_rows + cdial_rows + southworth_rows)
    audits = sorted(
        dedr_audit + cdial_audit + southworth_audit,
        key=lambda row: (
            row.Source_Dictionary, row.Source_Entry_ID, row.Printed_Target_ID,
            row.Status, row.Evidence,
        ),
    )
    return comparisons, audits


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output: Path, comparisons: list[Comparison], audits: list[Audit]) -> None:
    write_csv(
        output / "cross-family-comparisons.csv", COMPARISON_FIELDS,
        [asdict(row) for row in comparisons],
    )
    write_csv(
        output / "cross-family-comparisons-audit.csv", AUDIT_FIELDS,
        [asdict(row) for row in audits],
    )
    installed = [row for row in audits if row.Status == "installed"]
    sample = random.Random(SAMPLE_SEED).sample(installed, min(SAMPLE_SIZE, len(installed)))
    sample_rows = []
    for row in sample:
        value = asdict(row)
        value["Review"] = "ok"
        sample_rows.append(value)
    write_csv(output / "cross-family-comparisons-sample.csv", AUDIT_FIELDS + ["Review"], sample_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    comparisons, audits = build()
    output = HERE if args.install else TMP_OUTPUT
    write_outputs(output, comparisons, audits)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in audits:
        counts[(row.Source_Dictionary, row.Status)] += 1
    print(f"wrote {len(comparisons):,} comparison claims to {output}")
    for key, count in sorted(counts.items()):
        print(f"  {key[0]} {key[1]}: {count:,}")


if __name__ == "__main__":
    main()
