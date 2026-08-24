import csv
import gzip
from collections import Counter
from dataclasses import asdict

from data.cross_family import dedr_citation_locators
from data.dedr.entry_texts import (
    INSTALLED_AUDIT,
    INSTALLED_REFERENCES,
    INSTALLED_SAMPLE,
    INSTALLED_TEXTS,
    build,
)


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_rows():
    with gzip.open(INSTALLED_AUDIT, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialized(records):
    return [
        {key: str(value) for key, value in asdict(record).items()}
        for record in records
    ]


def test_old_ded_locators_are_parsed_in_print_order():
    assert dedr_citation_locators(
        "DED 57; DED (S, N) 930( b ); DEDS 687; DEN DBIA SI; DED 57"
    ) == ["DED 57", "DED(S, N) 930(b)", "DEDS 687", "DEN DBIA SI"]


def test_cached_dedr_commentary_extraction_is_reproducible_and_complete():
    blocks, references, audits = build()
    assert serialized(blocks) == rows(INSTALLED_TEXTS)
    assert serialized(references) == rows(INSTALLED_REFERENCES)
    assert serialized(audits) == audit_rows()

    assert len(audits) == 5627
    assert len(references) == 5623 == len({row.Form_ID for row in references})
    assert len(blocks) == 1572
    assert len({row.Form_ID for row in blocks}) == 1470
    assert Counter(row.Kind for row in blocks) == {
        "comparison": 1528,
        "source-note": 44,
    }
    assert Counter(row.Status for row in audits) == {
        "installed": 1470,
        "reflex-only": 3827,
        "structured-only": 326,
        "duplicate-excluded": 4,
    }
    assert all(row.Raw_Article for row in audits)


def test_dedr_blocks_exclude_reflex_inventory_and_structured_cross_family_prose():
    blocks, references, _ = build()
    by_entry = {}
    for block in blocks:
        by_entry.setdefault(block.Form_ID, []).append(block)
    refs = {row.Form_ID: row.Source for row in references}

    assert [row.Content for row in by_entry["d64"]] == [
        "? Cf. Ta. aṭai-ppai, s.v. 88 Ta. aṭaikkāy."
    ]
    assert "aṭappam" not in by_entry["d64"][0].Content
    assert "CDIAL" not in by_entry["d64"][0].Content
    assert refs["d64"] == "dedr[entry 64, DED 57]"

    # This article has no commentary beyond the parsed reflexes and its old-edition locator.
    assert "d4229" not in by_entry
    assert refs["d4229"] == "dedr[entry 4229, DEDS 687]"

    # A comparison note immediately before subsection (b) must not absorb subsection reflexes.
    assert [row.Content for row in by_entry["d2949"]] == [
        "Cf. Skt. ḍam- to sound (as a drum)."
    ]
    assert [row.Content for row in by_entry["d4711"]] == [
        "Cf. 4714 Ta. maravai. DED(S) 3856."
    ]


def test_dedr_review_sample_is_fixed_and_signed_off():
    sample = rows(INSTALLED_SAMPLE)
    assert len(sample) == 20
    assert {row["Review"] for row in sample} == {"ok"}
