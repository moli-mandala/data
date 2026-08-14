import csv
from collections import Counter


SOURCE_FILE = "data/other/forms/20260813-humla.csv"


def test_humla_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1512
    assert all(len(row) == 11 for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert len({row[10].split(":")[1] for row in rows}) == 205
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(49, 67))
    assert all(row[7].startswith("devries2020humla[p. ") for row in rows)


def test_humla_import_keeps_all_seven_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))
    assert set(counts) == {
        "humla_til", "humla_muchu", "humla_yalbang", "humla_kermi",
        "humla_yakpa", "humla_bargaun", "humla_dojam",
    }
    assert sum(counts.values()) == 1512


def test_humla_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Humla"}
