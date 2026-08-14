import csv
from collections import Counter


SOURCE_FILE = "data/other/forms/20260813-gurung.csv"


def test_gurung_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1574
    assert all(len(row) == 11 for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert len({row[10].split(":")[1] for row in rows}) == 281
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(34, 64))
    assert all(row[7].startswith("swenson2019gurung[p. ") for row in rows)


def test_gurung_import_keeps_all_six_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))
    assert counts == {
        "gurung_ajirkot": 254, "gurung_pyarjung": 231,
        "gurung_maling": 275, "gurung_yangjakot": 265,
        "gurung_birethanti": 276, "gurung_bhurdumpola": 273,
    }


def test_gurung_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Gurung"}
