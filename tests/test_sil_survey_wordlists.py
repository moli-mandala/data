"""Regression tests for the SIL India Appendix B3 survey-wordlist ingests."""

import csv
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
COMPILED = ROOT / "cldf/forms.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
CONTROLS = {"Tamil", "Malayalam"}

# report -> (installed csv, audit csv, source key, sites per language, glosses with no form)
REPORTS = {
    "idukki": (
        "20260826-sil-idukki.csv", "20260826-sil-idukki-audit.csv", "varghese-mathew2015idukki",
        {"Muthuvan": 8, "Mannan": 7, "Urali": 2, "MalaPulaya": 1, "Paliya": 1},
        # items 11 "breast" and 79 "cauliflower" were elicited at no site
        ["g011", "g079"],
    ),
    "palakkad": (
        "20260826-sil-palakkad.csv", "20260826-sil-palakkad-audit.csv", "varghese2015palakkad",
        # the appendix labels the Ellakkadu Malasar site "Malasar pasha", so Malasar has two
        {"Irula": 9, "Muduga": 2, "Kurumba": 2, "Kadar": 2, "Eravallan": 2,
         "AluKurumba": 1, "Malasar": 2, "MalaMalasar": 1},
        None,   # asserted from the audit rather than hard-coded
    ),
}


def installed(name):
    path = ROOT / "data/other/forms" / REPORTS[name][0]
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited(name):
    path = ROOT / "data/other/forms/raw_data" / REPORTS[name][1]
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_rows_are_target_lects_with_one_site_dialect_tag(name):
    for row in installed(name):
        assert row["Language_ID"] in REPORTS[name][3]
        tags = row["Tags"].split()
        assert len(tags) == 1 and tags[0].startswith(f"dialect:{row['Language_ID']}:")
        # a survey wordlist asserts no etymology, so every row stays unlinked
        assert row["Parameter_ID"] == row["Cognateset"] == row["Etymology"] == ""
        assert row["Form"] and row["Form"] == row["Phonemic"] and row["Native"] == ""


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_site_counts_match_the_report(name):
    sites = {language: set() for language in REPORTS[name][3]}
    for row in installed(name):
        sites[row["Language_ID"]].add(row["Tags"].split(":")[2])
    assert {k: len(v) for k, v in sites.items()} == REPORTS[name][3]


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_the_whole_210_item_list_is_parsed_and_keys_are_unique(name):
    rows = installed(name)
    keys = Counter(row["Entry_Key"] for row in rows)
    assert not [key for key, n in keys.items() if n > 1]
    seen = {f"g{int(row['Gloss_Number']):03d}" for row in audited(name) if row["Gloss_Number"]}
    assert len(seen) == 210, "the parser must see the whole 210-item list"
    have = {row["Entry_Key"].split(":")[1] for row in rows}
    expected_gaps = REPORTS[name][4]
    if expected_gaps is not None:
        assert sorted(seen - have) == expected_gaps
    # any gloss without an installed form must be one where every target record is a printed gap
    for gloss in seen - have:
        records = [r for r in audited(name)
                   if r["Gloss_Number"] and f"g{int(r['Gloss_Number']):03d}" == gloss
                   and r["Lect"] not in CONTROLS]
        assert records and all(r["Group"] == "0" for r in records), gloss


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_audit_accounts_for_every_line(name):
    audit = audited(name)
    status = Counter(row["Status"] for row in audit)
    # a two-part answer ("a, b") is one audited record but two installed rows, so the audit
    # carries every key it produced
    keys = [k for row in audit if row["Status"] == "installed" for k in row["Entry_Key"].split()]
    assert len(keys) == len(installed(name))
    assert status["installed"] <= len(installed(name))
    assert status["unparsed"] == 0      # no line of Appendix B3 may go unexplained
    assert status["unmapped"] == 0


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_controls_and_printed_gaps_never_become_forms(name):
    audit = audited(name)
    controls = [row for row in audit if row["Lect"] in CONTROLS]
    assert controls and all(row["Status"] == "excluded" for row in controls)
    assert not [row for row in installed(name) if row["Language_ID"] in CONTROLS]
    gaps = [row for row in audit if row["Group"] == "0"]
    assert gaps and all(row["Status"] == "excluded" for row in gaps)


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_languages_and_survey_sites_are_registered(name):
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    for language in REPORTS[name][3]:
        assert languages[language]["Clade"] == "S. Dravidian I"
        assert languages[language]["Glottocode"]
    for site in {row["Tags"].split(":")[2] for row in installed(name)}:
        assert site in dialects, site
        assert dialects[site]["Language_ID"] in REPORTS[name][3]
        # the reports print no coordinates, so every site point is an approximation
        assert dialects[site]["Quality"] == "C"


@pytest.mark.parametrize("name", sorted(REPORTS))
@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build(name):
    key = REPORTS[name][2]
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if key in {p.split("[", 1)[0].strip() for p in row["Source"].split(";")}
        ]
    assert len(compiled) == len(installed(name))
    assert {row["Language_ID"] for row in compiled} == set(REPORTS[name][3])
    assert all(row["Status"] == "unlinked" for row in compiled)
