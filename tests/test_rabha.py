import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-rabha.csv"


def test_rabha_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 400
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("kondakov2013rabha[p. ") for row in rows)
    assert all(row[10].startswith("rabha:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)


def test_rabha_import_keeps_both_lects_and_alternates():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {"rabha_rongdani": 205, "rabha_maituri": 195}


def test_rabha_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Rabha"}


def test_rabha_ipa_profile():
    profile = Tokenizer("conversion/rabha.txt")
    cases = {
        "kɑ́nɡɑnd͡ʒi": "kā́ngānjī",
        "tʃɑ̑skɑm": "cā̑skām",
        "nuk̚d͡ʒo": "nūk̚jo",
        "t͡ʃɨŋ kɑmkɑʲ": "cɨŋ kāmkāʸ",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
