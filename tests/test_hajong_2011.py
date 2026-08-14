import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-hajong.csv"


def _convert(value: str) -> str:
    profile = Tokenizer("conversion/hajong-survey.txt")
    return unicodedata.normalize(
        "NFC",
        profile(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_hajong_clean_pdf_targets_are_complete():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 3311
    assert all(len(row) == 11 and row[2] == row[5] for row in rows)
    assert Counter(row[0] for row in rows) == {
        "hajong_nugapara": 337,
        "hajong_chilapara": 329,
        "hajong_nirghini": 330,
        "hajong_dalugau": 341,
        "hajong_balachanda": 336,
        "hajong_dhamor": 341,
        "hajong_gopalbari": 315,
        "hajong_gopalpur": 324,
        "hajong_bhalukapara": 331,
        "hajong_nokshi": 327,
    }
    concepts = {int(row[10].split(":")[1]) for row in rows}
    assert concepts == set(range(1, 308)) - {240, 303}
    assert len({row[10] for row in rows}) == 3311
    assert all(row[0] != "0" and "Bangla" not in row[0] for row in rows)
    assert all("/ch" not in row[2] and not any(f"/{n}" in row[2] for n in range(1000, 1800)) for row in rows)


def test_hajong_ipa_profile_and_full_source_coverage():
    assert _convert("brɪʃʈi") == "briśṭi"
    assert _convert("rɔŋdʰɔnu") == "roŋdʰonu"
    assert _convert("d̪al") == "dal"
    assert _convert("dui̯") == "dui̯"
    assert _convert("ɖim") == "ḍim"
    assert _convert("bɛŋun") == "beŋun"
    assert _convert("bɯla") == "bula"

    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in _convert(row[2])


def test_hajong_lects_are_registered_under_hajong():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}

    for lect in {
        "hajong_nugapara", "hajong_chilapara", "hajong_nirghini",
        "hajong_dalugau", "hajong_balachanda", "hajong_dhamor",
        "hajong_gopalbari", "hajong_gopalpur", "hajong_bhalukapara",
        "hajong_nokshi",
    }:
        assert dialects[lect]["Language_ID"] == "Hajong"
        assert dialects[lect]["Glottocode"] == "hajo1238"


def test_hajong_compiled_rows_keep_raw_source_transcription():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        source_keys = [
            row for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("hajong:")
        ]
    assert len(source_keys) == 3311
    assert len({row["Source_Key"] for row in source_keys}) == 3311

    legacy_ids = {row["Legacy_ID"] for row in source_keys}
    with open("cldf/form-id-aliases.csv", encoding="utf-8", newline="") as stream:
        aliases = {
            row["Legacy_ID"]: row["Form_ID"]
            for row in csv.DictReader(stream)
            if row["Legacy_ID"] in legacy_ids
        }
    assert aliases.keys() == legacy_ids
    form_ids = set(aliases.values())
    with open("cldf/forms.csv", encoding="utf-8", newline="") as stream:
        forms = [row for row in csv.DictReader(stream) if row["ID"] in form_ids]
    assert len(forms) == 3311
    assert all(row["Language_ID"] == "Hajong" and row["Phonemic"] for row in forms)
    assert sum(row["Form"] != row["Phonemic"] for row in forms) > 1000
