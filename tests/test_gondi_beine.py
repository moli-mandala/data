"""Regression tests for the Beine (1994) Gondi survey word lists (Rama et al. 2017 digitization)."""

import csv
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
INSTALLED = ROOT / "data/other/forms/20260825-gondi-beine.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260825-gondi-beine-audit.csv"
SAMPLE = ROOT / "data/other/forms/raw_data/20260825-gondi-beine-sample.csv"
PROFILE = ROOT / "conversion/gondi-beine.txt"

SOURCE = "rama-coltekin-sofroniev2017gondi"
FIELDWORK = "beine1994gondi"

SITES = 46
CONCEPTS = 210
SOURCE_CELLS = SITES * CONCEPTS          # the upstream matrix is complete
MISSING_CELLS = 158                      # cells printed "-----": no word elicited
INSTALLED_ROWS = 10264                   # 9502 attested cells, 762 of which list alternates
AUDIT_ROWS = INSTALLED_ROWS + MISSING_CELLS


@pytest.fixture(scope="module")
def installed():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


@pytest.fixture(scope="module")
def audit():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def dialects():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        return {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}


def convert(value):
    tokenizer = Tokenizer(str(PROFILE))
    return unicodedata.normalize(
        "NFC",
        tokenizer(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_installed_counts_and_row_shape(installed):
    assert len(installed) == INSTALLED_ROWS
    assert {len(row) for row in installed} == {15}
    assert all(row[2] for row in installed)                       # no blank forms
    assert all(row[1] == "" for row in installed)                 # unetymologised: blank Param_ID
    assert all(row[4] == "" for row in installed)                 # no native script in the source
    assert all(row[8] == "" and row[9] == "" for row in installed)
    assert len({row[0] for row in installed}) == SITES


def test_entry_keys_are_unique_stable_and_source_shaped(installed):
    keys = [row[10] for row in installed]
    assert len(set(keys)) == INSTALLED_ROWS
    assert all(key.startswith("beine:") and len(key.split(":")) == 4 for key in keys)
    by_key = {row[10]: row for row in installed}
    # 'eat' at the Raj Gondi site of Hetitola: one response, dental unmarked in the source
    assert by_key["beine:ght:eat:1"][2] == "wortitur"
    # the same prompt at Rui, where the source does mark the dental
    assert by_key["beine:rui:eat:1"][2] == "dʒɛkit̪on"
    # a cell listing two responses becomes two rows, not one variant chain
    assert by_key["beine:gja:bad:1"][2] == "kʰarab"
    assert by_key["beine:gja:bad:2"][2] == "beshile"
    assert by_key["beine:gja:bad:1"][11] == "" and by_key["beine:gja:bad:2"][11] == ""


def test_source_transcription_is_kept_in_original_and_phonemic(installed):
    # Form (converted downstream by the profile) and Phonemic both start as the source IPA
    assert all(row[2] == row[5] for row in installed)
    assert all(row[2] == unicodedata.normalize("NFC", row[2]) for row in installed)
    assert not any("," in row[2] for row in installed)            # alternates were split out
    parenthesised = [row for row in installed if "(" in row[2]]
    assert len(parenthesised) == 308                              # printed optional material kept


def test_every_row_cites_both_the_digitization_and_the_fieldwork(installed):
    for row in installed:
        assert row[7].startswith(f"{SOURCE}[data/gondi_combined_cognates.csv, row ")
        assert f"; {FIELDWORK}[site " in row[7]
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@inproceedings{{{SOURCE}," in bib
    assert f"@mastersthesis{{{FIELDWORK}," in bib


def test_glosses_are_readable_lexical_prompts(installed):
    glosses = {row[3] for row in installed}
    assert len(glosses) == CONCEPTS
    assert not any("_" in gloss for gloss in glosses)
    assert {"we (exclusive)", "we (inclusive)", "you (singular, informal)",
            "you (singular, formal)", "you (plural)", "he was hungry", "he was thirsty",
            "evening, afternoon", "what kind", "how many"} <= glosses


def test_every_site_is_registered_as_a_georeferenced_gondi_dialect(installed, dialects):
    codes = {row[0] for row in installed}
    assert len(codes) == SITES
    for code in codes:
        dialect = dialects[code]
        assert dialect["Language_ID"] == "Gondi"
        assert dialect["Clade"] == "S. Dravidian II"
        assert dialect["Quality"] == "B"       # the digitizers' geolocation of Beine's sites
        assert dialect["Glottocode"] == ""     # Beine's sites are not Glottolog languoids
        assert float(dialect["Latitude"]) and float(dialect["Longitude"])
        assert "Beine (1994) survey site" in dialect["Location"]
        assert "Glottolog subgroup per Rama et al. (2017)" in dialect["Location"]
    assert dialects["beine_ght"]["Name"] == "Hetitola (Raj Gondi, ght)"
    assert dialects["beine_bhb"]["Name"] != dialects["beine_bhm"]["Name"]  # same tehsil, two sites


def test_audit_accounts_for_every_source_cell(audit, installed):
    assert len(audit) == AUDIT_ROWS
    statuses = Counter(row["Status"] for row in audit)
    assert statuses == {"ingested": INSTALLED_ROWS, "missing": MISSING_CELLS}
    assert len({row["Source_Row"] for row in audit}) == SOURCE_CELLS
    assert all(row["Raw_IPA"] == "-----" for row in audit if row["Status"] == "missing")
    assert {row["Entry_Key"] for row in audit if row["Status"] == "ingested"} == {
        row[10] for row in installed
    }
    # Rama's Gondi-internal cognate classes are not installed, but are preserved per record
    assert {row["Cognate_Class"] for row in audit} >= {"1", "2", "?", "A", "8?"}
    assert all(row["ASJP"] and row["SCA"] for row in audit if row["Status"] == "ingested")


def test_seeded_sample_is_drawn_from_the_audit(audit):
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    assert len(sample) == 20
    keys = {row["Entry_Key"] for row in audit}
    assert all(row["Entry_Key"] in keys for row in sample)


def test_profile_covers_every_installed_form(installed):
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed:
        assert "�" not in tokenizer(unicodedata.normalize("NFC", row[2]), column="IPA")


def test_profile_maps_the_difficult_source_notation():
    # the source marks dentality only sporadically; both spellings are the house dental
    assert convert("wort̪itor") == "vortitor"
    assert convert("wortitur") == "vortitur"
    assert convert("k̪an") == "kan"
    # retroflexes, affricates and sibilants take Dravidianist house graphemes
    assert convert("dʒɛkit̪on") == "jɛkiton"
    assert convert("tʃʰat̪i") == "cʰati"
    assert convert("(hor)und̪ʒetol") == "(hor)unjetol"
    assert convert("boʈa") == "boṭa"
    assert convert("dʰoɖa") == "dʰoḍa"
    assert convert("ʃɪɖ") == "śɪḍ"
    assert convert("ɾoʒu") == "rožu"
    assert convert("dʒʰunaɭ") == "jʰunaḷ"
    assert convert("ɡʌɖːi") == "gʌḍḍi"          # a consonant plus length is a geminate
    assert convert("nʌlːoʈʌ") == "nʌlloṭʌ"
    assert convert("kaɾabaːt̪ʌ") == "karabātʌ"  # a vowel plus length takes a macron
    # w/ʋ/v and j/y are one phoneme each in this transcription; vowel qualities are kept
    assert convert("wat̪ʌ") == "vatʌ"
    assert convert("i̯awal") == "i̯aval"
    assert convert("səri") == "səri"
    assert convert("mɛkra") == "mɛkra"
    # printed optional material, half-length and nasality survive conversion
    assert convert("(akˑi)ɡʰobi") == "(akˑi)gʰobi"
    assert convert("kẽst̪or") == "kẽstor"


def test_compiled_forms_are_unlinked_gondi_dialect_records():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if SOURCE in row["Source"]]
    assert len(rows) == INSTALLED_ROWS
    assert all(row["Language_ID"] == "Gondi" for row in rows)
    assert all(row["Status"] == "unlinked" for row in rows)
    assert all(row["Cognateset"] == "" for row in rows)
    tags = {tag for row in rows for tag in row["Tags"].split()}
    assert len(tags) == SITES
    assert all(tag.startswith("dialect:Gondi:beine_") for tag in tags)

    # unify_cldf moves the importer's immutable keys into the form-source-keys sidecar
    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as stream:
        keys = {row["Source_Key"]: row["Legacy_ID"] for row in csv.DictReader(stream)}
    assert len({key for key in keys if key.startswith("beine:")}) == INSTALLED_ROWS
    eat = {row["ID"]: row for row in rows}[keys["beine:rui:eat:1"]]
    assert (eat["Form"], eat["Original"], eat["Phonemic"]) == (
        "jɛkiton", "dʒɛkit̪on", "dʒɛkit̪on",
    )
    assert eat["Gloss"] == "eat"
    assert eat["Tags"] == "dialect:Gondi:beine_rui:Rui%20%28Koitur%20Gondi%2C%20rui%29"
