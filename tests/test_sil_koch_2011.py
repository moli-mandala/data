"""Regression tests for the SIL 2011-033 (Koch survey wordlists) ingest."""

import csv
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
INSTALLED = ROOT / "data/other/forms/20260826-sil-koch.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260826-sil-koch-audit.csv"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "kondakov2011koch"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
# the report models Margan, Harigaya, Wanang, Tintekiya and Koch-Rabha as one Koch language
SITES_PER_LANGUAGE = {"Koch": 9, "Garo": 1, "Rabha": 1}


def rows():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, r)) for r in csv.reader(stream)]


def audit():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


INSTALLED_ROWS = rows()


def test_every_row_is_a_survey_lect_with_one_site_tag():
    for row in INSTALLED_ROWS:
        assert row["Language_ID"] in SITES_PER_LANGUAGE
        tags = row["Tags"].split()
        assert len(tags) == 1 and tags[0].startswith(f"dialect:{row['Language_ID']}:")
        # a survey wordlist asserts no etymology
        assert row["Parameter_ID"] == row["Cognateset"] == row["Etymology"] == ""
        assert row["Form"] and row["Form"] == row["Phonemic"] and row["Native"] == ""


def test_site_counts_match_the_key():
    sites = {language: set() for language in SITES_PER_LANGUAGE}
    for row in INSTALLED_ROWS:
        sites[row["Language_ID"]].add(row["Tags"].split(":")[2])
    assert {k: len(v) for k, v in sites.items()} == SITES_PER_LANGUAGE
    # eleven survey sites in total, matching the key on the first wordlist page
    assert sum(len(v) for v in sites.values()) == 11


def test_every_line_is_accounted_for_and_keys_are_unique():
    status = Counter(row["Status"] for row in audit())
    assert status["unparsed"] == 0
    assert status["unmapped"] == 0
    keys = Counter(row["Entry_Key"] for row in INSTALLED_ROWS)
    assert not [key for key, n in keys.items() if n > 1]


def test_each_gloss_is_attested_across_the_eleven_sites():
    by_gloss = Counter(row["Gloss"] for row in INSTALLED_ROWS)
    # the appendix prints 194 of the standard 210 items; the report notes that items too
    # problematic for consistent elicitation were dropped
    assert len(by_gloss) == 194
    # every printed item carries at least one form from every site
    assert min(by_gloss.values()) >= 11


def test_vertically_centred_labels_resolve_to_the_right_site():
    # 'body' prints B. RR's two forms above and below its label; both must land on Rabha
    body = [r for r in INSTALLED_ROWS if r["Gloss"] == "body"]
    rabha = [r["Form"] for r in body if r["Language_ID"] == "Rabha"]
    assert "kɑń" in rabha and "kɑnɡɑnd͡ʒi" in rabha


def test_languages_and_sites_are_registered():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["Koch"]["Glottocode"] == "koch1250"
    assert languages["Garo"]["Glottocode"] == "garo1247"
    for site in {row["Tags"].split(":")[2] for row in INSTALLED_ROWS}:
        assert site in dialects, site
        # the report prints no coordinates, so every site point is an approximation
        assert dialects[site]["Quality"] == "C"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {p.split("[", 1)[0].strip() for p in row["Source"].split(";")}
        ]
    assert len(compiled) == len(INSTALLED_ROWS)
    assert {row["Language_ID"] for row in compiled} == set(SITES_PER_LANGUAGE)
    assert all(row["Status"] == "unlinked" for row in compiled)
