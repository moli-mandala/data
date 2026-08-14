import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-eastern-magar.csv"


def test_eastern_magar_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 865
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("hilty2013eastern-magar[p. ") for row in rows)
    assert all(row[10].startswith("eastern-magar:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(40, 56))
    assert len({row[10].split(":")[1] for row in rows}) == 218


def test_eastern_magar_import_keeps_all_four_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "eastern_magar_dhankuta": 211,
        "eastern_magar_nawalparasi": 218,
        "eastern_magar_panchthar": 218,
        "eastern_magar_sarlahi": 218,
    }


def test_eastern_magar_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"EasternMagar"}


def test_eastern_magar_ipa_profile_and_pdf_font_repairs():
    profile = Tokenizer("conversion/eastern-magar.txt")
    cases = {
        "midʒaŋ": "mijaŋ",
        "tuk̚tʃʲo": "tuk̚cʸo",
        "miʃʲæk̚": "miśʸæk̚",
        "kʰeɾɛp̚": "kʰerep̚",
        "ŋɛ̃t": "ŋẽt",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected

    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        forms = [row[2] for row in csv.reader(stream)]
    assert all("+" not in form for form in forms)
    assert "miʃʲæk̚" in forms
