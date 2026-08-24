#!/usr/bin/env python3
"""Install Burrow & Emeneau's 1972 *Dravidian Etymological Notes*, part II.

The copyrighted PDF is not redistributed. One page-local JSON file per printed article page is
the reproducible raw layer. This importer reuses the deliberately conservative Part I resolver:
only active/corrected Dravidian forms independently corroborated by the later DEDR are installed;
index pages, source prose, deletions, comparisons, loans, and unresolved readings remain audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[4]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from data.other.forms.raw_data import burrow_emeneau_1972_den1 as shared


SOURCE_ID = "burrow-emeneau1972den2"
SNAPSHOT_DATE = "2026-08-19"
PDF_SHA256 = "da65837343c4811224cd5fa9da4580d0adcdd13190a0b9be29c4ac9d1ec8d64e"
PDF_PAGES = 18
PRINTED_PAGES = tuple(range(475, 492))
LEXICAL_PAGES = frozenset(range(475, 480))
EXPECTED_RECORD_COUNTS = {475: 29, 476: 38, 477: 31, 478: 17, 479: 4}

ROOT = DATA_ROOT
RAW_DIR = ROOT / "data/other/forms/raw_data"
AGENT_DIR = RAW_DIR / "burrow_emeneau_1972_den2_agent"
FORM_OUTPUT = ROOT / "data/other/forms/20260819-burrow-emeneau-den2.csv"
TEXT_OUTPUT = ROOT / "data/other/entry_texts/20260819-burrow-emeneau-den2.csv"
AUDIT_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den2-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den2-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den2-manifest.json"
RECONCILIATION_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den2-reconciliation.json"

FORM_FIELDS = shared.FORM_FIELDS
TEXT_FIELDS = shared.TEXT_FIELDS
AUDIT_FIELDS = shared.AUDIT_FIELDS

# Raw agent evidence is never rewritten. Any source-image reconciliation discovered during review
# is recorded here and copied into the audit. Populate only with checked page-local decisions.
STATUS_CORRECTIONS: dict[tuple[str, int], tuple[str, str]] = {}
LINK_RELATION_CORRECTIONS: dict[tuple[str, int], tuple[str, str]] = {}
FORM_CORRECTIONS: dict[tuple[str, int], tuple[str, str]] = {
    ("p476:u011", 0): (
        "sūri",
        "source-image/current-DEDR review repairs the agent's spurious medial f",
    ),
    ("p476:u015", 1): (
        "jammō",
        "source-image/current-DEDR review restores final vowel length in the Gondi form",
    ),
    ("p476:u020", 2): (
        "tōṛa (tōṛi-)",
        "source-image/current-DEDR review resolves the agent's S/6 glyph placeholders",
    ),
    ("p477:u005", 1): (
        "ñiṉiru",
        "source-image review restores the queried Tamil alternant's initial and medial ñ",
    ),
}
TARGET_OVERRIDES: dict[tuple[str, int], tuple[str, str]] = {}
FORCE_EXCLUDE: dict[tuple[str, int], str] = {}
RECORD_TARGET_OVERRIDES: dict[str, list[str]] = {}
AUTO_TARGET_OVERRIDES: dict[tuple[str, int], tuple[str, str]] = {}
AUTO_RECORD_TARGET_OVERRIDES: dict[str, list[str]] = {}


@contextmanager
def shared_configuration():
    """Temporarily point the reviewed Part I reconciliation engine at Part II evidence."""
    values = {
        "SOURCE_ID": SOURCE_ID,
        "SNAPSHOT_DATE": SNAPSHOT_DATE,
        "PDF_SHA256": PDF_SHA256,
        "PDF_PAGES": PDF_PAGES,
        "PRINTED_PAGES": PRINTED_PAGES,
        "AGENT_DIR": AGENT_DIR,
        "FORM_OUTPUT": FORM_OUTPUT,
        "TEXT_OUTPUT": TEXT_OUTPUT,
        "AUDIT_OUTPUT": AUDIT_OUTPUT,
        "SAMPLE_OUTPUT": SAMPLE_OUTPUT,
        "MANIFEST_OUTPUT": MANIFEST_OUTPUT,
        "RECONCILIATION_OUTPUT": RECONCILIATION_OUTPUT,
        "STATUS_CORRECTIONS": STATUS_CORRECTIONS,
        "LINK_RELATION_CORRECTIONS": LINK_RELATION_CORRECTIONS,
        "FORM_CORRECTIONS": FORM_CORRECTIONS,
        "TARGET_OVERRIDES": {**AUTO_TARGET_OVERRIDES, **TARGET_OVERRIDES},
        "FORCE_EXCLUDE": FORCE_EXCLUDE,
        "RECORD_TARGET_OVERRIDES": {
            **AUTO_RECORD_TARGET_OVERRIDES, **RECORD_TARGET_OVERRIDES,
        },
    }
    previous = {name: getattr(shared, name) for name in values}
    for name, value in values.items():
        setattr(shared, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def load_pages() -> list[dict]:
    pages: list[dict] = []
    seen_units: set[str] = set()
    for printed_page in PRINTED_PAGES:
        path = AGENT_DIR / f"p{printed_page}.json"
        with path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        assert page["printed_page"] == printed_page, path
        assert page["pdf_page"] == printed_page - 473, path
        assert page["page_kind"] in shared.PAGE_KINDS, path
        if printed_page in LEXICAL_PAGES:
            assert page["page_kind"] == "lexical_entries", path
            assert page["records"], f"lexical page has no records: {path}"
            assert len(page["records"]) == EXPECTED_RECORD_COUNTS[printed_page], (
                path, len(page["records"]), EXPECTED_RECORD_COUNTS[printed_page]
            )
        else:
            assert page["page_kind"] == "bibliography", path
            assert not page["records"], f"index page unexpectedly has records: {path}"
            assert page.get("page_notes", "").strip(), f"index scope is undocumented: {path}"

        for ordinal, record in enumerate(page["records"], 1):
            expected = f"p{printed_page}:u{ordinal:03d}"
            assert record["unit_id"] == expected, (path, record["unit_id"], expected)
            assert expected not in seen_units, expected
            seen_units.add(expected)
            assert record.get("raw_entry_text", "").strip(), (
                expected, "missing page-local raw entry transcription"
            )
            assert record["series"] in shared.SERIES, expected
            assert set(record["operations"]) <= shared.OPERATIONS, expected
            for index, form in enumerate(record.get("forms", [])):
                assert form.get("language_abbrev", "").strip(), (
                    expected, index, "missing printed language abbreviation"
                )
                assert form.get("language_name", "").strip(), (
                    expected, index, "missing resolved language name"
                )
                assert form.get("form_original", "").strip(), (
                    expected, index, "blank lexical form"
                )
                status = form.get("form_status", "")
                if (expected, index) not in STATUS_CORRECTIONS:
                    assert status in shared.FORM_STATUSES, (expected, index, status)
                assert form.get("relation_to_entry", "") in shared.FORM_RELATIONS, (
                    expected, index, form.get("relation_to_entry", "")
                )
            for index, link in enumerate(record.get("links", [])):
                assert link.get("target_system", "") in shared.TARGET_SYSTEMS, (expected, link)
                relation = link.get("relation", "")
                if (expected, index) not in LINK_RELATION_CORRECTIONS:
                    assert relation in shared.LINK_RELATIONS, (expected, index, relation)
                assert link.get("claim_status", "") in shared.CLAIM_STATUSES, (expected, link)
                assert link.get("editorial_action", "") in shared.EDITORIAL_ACTIONS, (expected, link)
        pages.append(page)
    return pages


def _candidate_for_new_entry(
    value: str,
    gloss: str,
    language: str,
    inventory: dict[tuple[str, str], list[tuple[str, str]]],
) -> tuple[str, str]:
    """Resolve a DEN-II new-entry form across the current DEDR inventory.

    The printed S21--S277 labels are the DEN new-entry sequence, not historical DEDS numbers.
    Consequently they must be aligned by independently corroborated language/form evidence rather
    than passed through the old-number resolver used for Part I.
    """
    candidates = [
        (target, candidate_form, candidate_gloss)
        for (target, candidate_language), values in inventory.items()
        if candidate_language == language
        for candidate_form, candidate_gloss in values
    ]
    normalized = shared.normalized_form(value)
    exact = [row for row in candidates if shared.normalized_form(row[1]) == normalized]
    skeleton = shared.form_skeleton(value)
    skeleton_matches = [
        row for row in candidates if skeleton and shared.form_skeleton(row[1]) == skeleton
    ]
    matches = exact or skeleton_matches
    match_kind = (
        "exact language-and-form" if exact
        else "unique diacritic-insensitive language-and-form"
    )
    if not matches:
        return "", "no current-DEDR language-and-form corroboration"

    # A cheap page agent can erase a contrastive DEDR mark and thereby make the resulting string
    # exactly equal to an unrelated homonym (e.g. Kodagu puḷi 'mist' versus puḷi 'sour').  When
    # the broader skeleton set contains one uniquely supported gloss, prefer that candidate over
    # the accidental unmarked exact match.
    wanted = shared.gloss_words(gloss)
    if exact and wanted:
        skeleton_targets = sorted({row[0] for row in skeleton_matches})
        skeleton_scores = {
            target: max(
                len(wanted & shared.gloss_words(candidate_gloss))
                for candidate_target, _, candidate_gloss in skeleton_matches
                if candidate_target == target
            )
            for target in skeleton_targets
        }
        best_skeleton = max(skeleton_scores.values(), default=0)
        skeleton_winners = [
            target for target, score in skeleton_scores.items()
            if score == best_skeleton and score > 0
        ]
        exact_targets = {row[0] for row in exact}
        if len(skeleton_winners) == 1 and skeleton_winners[0] not in exact_targets:
            return (
                skeleton_winners[0],
                "DEN new entry resolved by diacritic-insensitive form plus unique gloss overlap; "
                "an unmarked exact homonym has an incompatible gloss",
            )

    targets = sorted({row[0] for row in matches})
    if len(targets) == 1:
        return targets[0], f"DEN new entry resolved by {match_kind} corroboration"

    scored = {
        target: max(
            len(wanted & shared.gloss_words(candidate_gloss))
            for candidate_target, _, candidate_gloss in matches
            if candidate_target == target
        )
        for target in targets
    }
    best = max(scored.values(), default=0)
    winners = [target for target, score in scored.items() if score == best and score > 0]
    if len(winners) == 1:
        return winners[0], f"DEN new entry resolved by {match_kind} plus unique gloss overlap"
    return "", "current-DEDR form occurs in multiple entries without unique gloss support"


def prepare_new_entry_targets(pages: list[dict]) -> None:
    """Build reproducible current-DEDR target sets for the article's new S entries."""
    language_ids, language_by_name = shared.load_language_ids()
    inventory = shared.load_dedr_inventory(language_ids, language_by_name)
    AUTO_TARGET_OVERRIDES.clear()
    AUTO_RECORD_TARGET_OVERRIDES.clear()
    for page in pages:
        for record in page["records"]:
            if record.get("series") != "DEDS":
                continue
            targets: set[str] = set()
            unit_id = record["unit_id"]
            for index, form in enumerate(record.get("forms", [])):
                status = STATUS_CORRECTIONS.get(
                    (unit_id, index), (form.get("form_status", ""), "")
                )[0]
                relation = form.get("relation_to_entry", "")
                if status not in {"active", "corrected"} or relation not in {
                    "reflex", "variant", "derived",
                }:
                    continue
                language = shared.resolve_language(
                    form.get("language_name", ""), form.get("language_abbrev", ""),
                    language_ids, language_by_name,
                )
                if not language:
                    continue
                target, reason = _candidate_for_new_entry(
                    FORM_CORRECTIONS.get(
                        (unit_id, index), (form.get("form_original", ""), "")
                    )[0],
                    form.get("gloss", ""),
                    language,
                    inventory,
                )
                if target:
                    AUTO_TARGET_OVERRIDES[(unit_id, index)] = (target, reason)
                    targets.add(target)
            # Presence, including an empty set, suppresses the inappropriate historical-DEDS
            # resolver for every printed DEN new-entry segment.
            AUTO_RECORD_TARGET_OVERRIDES[unit_id] = sorted(targets)


def build(pages: list[dict]):
    prepare_new_entry_targets(pages)
    with shared_configuration():
        forms, texts, audits, summary = shared.build(pages)
    unit_meta = {
        record["unit_id"]: (page["printed_page"], record["entry_label"], record["series"])
        for page in pages for record in page["records"]
    }

    def display_label(raw: str) -> str:
        return f"S²{raw[2:]}" if raw.startswith("S2") else raw

    for row in forms:
        unit_id = row["Entry_Key"].removeprefix(f"{SOURCE_ID}:").rsplit(":f", 1)[0]
        printed_page, raw_label, _ = unit_meta[unit_id]
        label = display_label(raw_label)
        row["Source"] = f"{SOURCE_ID}[p. {printed_page}, entry {label}]"
        row["Etymology"] = re.sub(
            r"DEN I \(1972\) records this (.+?) under old supplement \d+;",
            rf"DEN II (1972) records this \1 under new-entry label {label};",
            row["Etymology"],
        )
    for row in audits:
        parent_unit = row["Parent_Unit_ID"] or row["Unit_ID"]
        if parent_unit in unit_meta and unit_meta[parent_unit][2] == "DEDS":
            printed_page, raw_label, _ = unit_meta[parent_unit]
            label = display_label(raw_label)
            row["Old_Targets"] = f"den2-new:{label}"
            row["Source"] = f"{SOURCE_ID}[p. {printed_page}, entry {label}]"
        if (
            row["Series"] == "DBIA"
            and row["Final_Status"] == "unresolved_target"
            and row["Resolution"] == "old DED/DEDS number is absent from current DEDR"
        ):
            row["Resolution"] = (
                "DBIA addition/correction retained for a separate loan-entry reconciliation; "
                "the current structural pilot installs only independently DEDR-corroborated forms"
            )
        elif row["Series"] == "DEDS" and row["Item_Type"] == "form":
            if row["Resolution"] == "old DED/DEDS number is absent from current DEDR":
                row["Resolution"] = (
                    "DEN new-entry form has no current-DEDR language/form corroboration"
                )
            elif row["Resolution"].startswith("unique old-number resolution;"):
                row["Resolution"] = row["Resolution"].replace(
                    "unique old-number resolution",
                    "current-DEDR target inherited from the same DEN new-entry segment",
                    1,
                )
    resolved_statuses = {
        "installed_form", "unresolved_target", "unreconciled_transcription",
        "duplicate_excluded",
    }
    summary["target_resolutions"] = dict(sorted(Counter(
        row["Resolution"].split("; ", 1)[0]
        for row in audits
        if row["Item_Type"] == "form" and row["Final_Status"] in resolved_statuses
    ).items()))
    return forms, texts, audits, summary


def write_metadata(summary: dict, sample_count: int) -> None:
    RECONCILIATION_OUTPUT.write_text(
        json.dumps({
            "source_id": SOURCE_ID,
            "snapshot_date": SNAPSHOT_DATE,
            "policy": (
                "Page-agent JSON is raw evidence. Only active/corrected direct Dravidian reflex, "
                "variant, or derivational forms with conservative current-DEDR resolution are installed. "
                "The printed S-squared labels are DEN-II new-entry numbers, not historical DEDS IDs; "
                "DBIA additions remain audited pending their separate loan-entry reconciliation."
            ),
            "known_agent_corrections": [
                {"unit_id": unit, "form_index": index + 1, "normalized_status": value[0], "decision": value[1]}
                for (unit, index), value in sorted(STATUS_CORRECTIONS.items())
            ] + [
                {"unit_id": unit, "link_index": index + 1, "normalized_relation": value[0], "decision": value[1]}
                for (unit, index), value in sorted(LINK_RELATION_CORRECTIONS.items())
            ] + [
                {"unit_id": unit, "form_index": index + 1, "normalized_form": value[0], "decision": value[1]}
                for (unit, index), value in sorted(FORM_CORRECTIONS.items())
            ],
            **summary,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MANIFEST_OUTPUT.write_text(
        json.dumps({
            "source_id": SOURCE_ID,
            "snapshot_date": SNAPSHOT_DATE,
            "title": "Dravidian Etymological Notes: Supplement to DED, DEDS, and DBIA, Pt. II",
            "authors": ["T. Burrow", "M. B. Emeneau"],
            "year": 1972,
            "stable_url": "https://www.jstor.org/stable/599958",
            "doi": "10.2307/599958",
            "pdf_sha256": PDF_SHA256,
            "pdf_pages": PDF_PAGES,
            "article_printed_pages": [475, 491],
            "lexical_printed_pages": [475, 479],
            "index_printed_pages": [480, 491],
            "pdf_redistributed": False,
            "rights": "Copyright JSTOR/JAOS scan; only extracted linguistic facts and audit metadata are checked in.",
            "extraction": {
                "method": "page-isolated gpt-5.6-luna extraction followed by editorial reconciliation",
                "contract": "data/other/forms/raw_data/burrow_emeneau_1972_den2_prompt.md",
                "raw_page_directory": "data/other/forms/raw_data/burrow_emeneau_1972_den2_agent",
            },
            "outputs": {
                "forms": str(FORM_OUTPUT.relative_to(ROOT)),
                "entry_texts": str(TEXT_OUTPUT.relative_to(ROOT)),
                "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
                "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
                "reconciliation": str(RECONCILIATION_OUTPUT.relative_to(ROOT)),
            },
            **summary,
            "sample_count": sample_count,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="optional original PDF for identity verification")
    args = parser.parse_args()
    if args.pdf:
        assert args.pdf.is_file(), args.pdf
        actual = shared.sha256(args.pdf)
        if actual != PDF_SHA256:
            raise ValueError(f"PDF SHA-256 {actual} does not match expected {PDF_SHA256}")

    pages = load_pages()
    forms, texts, audits, summary = build(pages)
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert len(audits) == summary["record_count"] + summary["raw_form_count"]

    shared.write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    shared.write_csv(TEXT_OUTPUT, TEXT_FIELDS, texts)
    shared.write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audits)
    sample = sorted(
        audits, key=lambda row: hashlib.sha256(row["Unit_ID"].encode()).hexdigest()
    )[:20]
    shared.write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample)
    write_metadata(summary, len(sample))
    print(
        f"installed {len(forms)} forms and {len(texts)} entry-text blocks from "
        f"{summary['record_count']} numbered page segments; audited {len(audits)} units"
    )


if __name__ == "__main__":
    main()
