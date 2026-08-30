"""Regression tests for SSNP volume 2's complete Northern Areas wordlists."""

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).parents[1]
RAW_FORMS = ROOT / "data/other/forms/raw_data/northern"
RAW_PARAMETERS = ROOT / "data/other/forms/raw_data/northern_param"
INSTALLED = ROOT / "data/other/forms/20230416-northern.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20230416-northern-manifest.json"


def read_dicts(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_frozen_cldf_is_the_complete_cc_by_v1_1_release():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_release"] == "v1.1"
    assert manifest["upstream_release_doi"] == "10.5281/zenodo.13149113"
    assert manifest["upstream_commit"] == "377d157614c706b2fcb61eccd5c839f394b9aa6c"
    assert manifest["license"] == "CC-BY-4.0"
    assert hashlib.sha256(RAW_FORMS.read_bytes()).hexdigest() == manifest["raw_forms_sha256"]
    assert hashlib.sha256(RAW_PARAMETERS.read_bytes()).hexdigest() == manifest["raw_parameters_sha256"]


def test_all_source_rows_and_controls_are_accounted_for():
    rows = read_dicts(RAW_FORMS)
    assert len(rows) == 11_343
    assert len({row["ID"] for row in rows}) == len(rows)
    assert len({row["Language_ID"] for row in rows}) == 51
    assert all(row["Form"] and row["Source"] == "Backstrom1992" for row in rows)
    controls = Counter(row["Language_ID"] for row in rows if row["Language_ID"] in {"Urdu", "Pashto"})
    assert controls == Counter(Urdu=261, Pashto=246)
    assert len(rows) - sum(controls.values()) == 10_836


def test_current_concepticon_mappings_and_all_six_lists_are_preserved():
    params = {row["ID"]: row for row in read_dicts(RAW_PARAMETERS)}
    assert len(params) == 1_233
    assert len({row["Concepticon_ID"] for row in params.values() if row["Concepticon_ID"]}) == 224
    for suffix in "abcdef":
        assert params[f"Backstrom-1992-210{suffix}-174"]["Concepticon_ID"] == "3985"
        assert params[f"Backstrom-1992-210{suffix}-200"]["Concepticon_ID"] == "3862"
    for suffix in "abcdf":
        assert params[f"Backstrom-1992-210{suffix}-173"]["Concepticon_ID"] == "3984"


def test_installed_file_preserves_every_released_lexeme():
    raw = read_dicts(RAW_FORMS)
    params = {row["ID"]: row["Name"] for row in read_dicts(RAW_PARAMETERS)}
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        installed = list(csv.reader(stream))
    assert len(installed) == len(raw) == 11_343
    installed_lexemes = Counter((row[0], row[2], row[3]) for row in installed)
    released_lexemes = Counter(
        (row["Language_ID"], row["Form"], params[row["Parameter_ID"]]) for row in raw
    )
    # Three Domaaki predicate responses have their overt subject/imperative
    # support word separated from the lexical form in the legacy Jambu import.
    assert released_lexemes - installed_lexemes == Counter({
        ("Domaaki", "tu kʰɑ", "you eat!"): 1,
        ("Domaaki", "šunɑin ǰʌṇɛrẓ̌ɪn", "the dog bites"): 1,
        ("Domaaki", "mum bɛṣ̌", "sit down!"): 1,
    })
    assert installed_lexemes - released_lexemes == Counter({
        ("Domaaki", "kʰɑ", "(you) eat!"): 1,
        ("Domaaki", "ǰʌṇɛrẓ̌ɪn", "(the dog) bites"): 1,
        ("Domaaki", "bɛṣ̌", "sit down!"): 1,
    })
    assert all(row[7] == "backstrom1992" for row in installed)


def test_every_target_variety_has_canonical_language_or_dialect_metadata():
    source_ids = {
        row["Language_ID"] for row in read_dicts(RAW_FORMS)
        if row["Language_ID"] not in {"Urdu", "Pashto"}
    }
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        language_ids = {row["ID"] for row in csv.DictReader(stream)}
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialect_ids = {row["Source_Language_ID"] for row in csv.DictReader(stream)}
    legacy_aliases = {
        "Domaaki": "D",
        "Dras": "dr",
        "Gilgit": "gil",
        "Punial": "punl",
        "Palas": "pales",
    }
    canonical_source_ids = {legacy_aliases.get(source_id, source_id) for source_id in source_ids}
    assert canonical_source_ids <= language_ids | dialect_ids


def test_northern_profile_covers_every_installed_transcription():
    profile = Tokenizer(str(ROOT / "conversion/northern.txt"))
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        forms = [row[2] for row in csv.reader(stream) if row[0] not in {"Urdu", "Pashto"}]
    converted = [
        unicodedata.normalize(
            "NFC",
            profile(unicodedata.normalize("NFD", form), column="IPA")
            .replace(" ", "")
            .replace("#", " "),
        )
        for form in forms
    ]
    bad = sorted({(source, display) for source, display in zip(forms, converted) if "�" in display})
    assert not bad, bad[:100]
    display_by_source = dict(zip(forms, converted))
    assert display_by_source["ṣ̌ʌq"] == "ṣaq"
    assert display_by_source["ǰ̣oŋ"] == "ʣ̣oŋ"
    assert display_by_source["èí"] == "eí"
