import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-western-tamang.csv"


def test_western_tamang_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 901
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("lipp2014western-tamang[p. ") for row in rows)
    assert all(row[10].startswith("western-tamang:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(76, 83))
    assert len({row[10].split(":")[1] for row in rows}) == 280


def test_western_tamang_import_keeps_all_three_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "eastern_gorkha_tamang_kashigaun": 296,
        "western_tamang_jharlang": 307,
        "western_tamang_sahugaun": 298,
    }


def test_western_tamang_lects_are_registered_under_current_languages():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert dialects["eastern_gorkha_tamang_kashigaun"]["Language_ID"] == "EasternGorkhaTamang"
    assert dialects["western_tamang_jharlang"]["Language_ID"] == "WesternTamang"
    assert dialects["western_tamang_sahugaun"]["Language_ID"] == "WesternTamang"


def test_western_tamang_ipa_profile_and_source_annotations():
    profile = Tokenizer("conversion/western-tamang.txt")
    cases = {
        "dʑiu": "ʣ̣iu",
        "tɕⁱam": "ʦ̣ⁱam",
        "ʔa:tɕabel": "ʔāʦ̣abel",
        "kʰalːa": "kʰalla",
        "mṳla": "mṳla",
        "ɕego": "śego",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected

    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert Counter(row[9] for row in rows)["Source marks this form as short."] == 2
    assert all(row[2] not in {"0", "-"} for row in rows)
    assert all(not row[2].lower().startswith("not elicited") for row in rows)
