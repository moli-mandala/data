import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-sampang.csv"


def test_sampang_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 907
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("rai-rai-thokar2015sampang[p. ") for row in rows)
    assert all(row[10].startswith("sampang:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(100, 111))
    assert {int(row[10].split(":")[1]) for row in rows} == set(range(1, 212))


def test_sampang_import_keeps_only_the_four_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "sampang_phedi": 235,
        "sampang_khartamchha": 225,
        "sampang_patheka": 227,
        "sampang_baspani": 220,
    }


def test_sampang_lects_are_registered_under_sampang():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Sampang"}
    assert all(dialects[lect]["Glottocode"] == "samp1249" for lect in lects)


def test_sampang_legacy_glyphs_and_profile():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert not any("(cid:" in row[2] for row in rows)
    assert any(row[2] == "tʌ̃" and row[3] == "horns" for row in rows)
    assert any(row[2] == "ṭano" and row[3] == "horns" for row in rows)
    assert any(row[3] == "above" and row[2] == "mutuni" for row in rows)

    profile = Tokenizer("conversion/sampang.txt")
    cases = {
        "tˢʰʌ̃wara": "ʦʰãvara",
        "pʌmtᶳʱu": "pamcʰu",
        "dᶽʰara": "jʰara",
        "ri:ma": "rīma",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
