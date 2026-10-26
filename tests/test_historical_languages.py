"""cldf/historical-languages.csv: the hand-maintained list of languages that are not living ones.

jambu-static draws these with the historical rhombus (and the concept atlas never paints an area
for them); nothing is inferred from a language's name or clade.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def rows(name):
    with (ROOT / "cldf" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_historical_table_names_registered_languages():
    languages = {row["ID"]: row for row in rows("languages.csv")}
    table = rows("historical-languages.csv")
    ids = [row["ID"] for row in table]
    assert len(ids) == len(set(ids))
    assert {row["Basis"] for row in table} == {"attested", "reconstructed", "label"}
    for row in table:
        assert row["ID"] in languages, row["ID"]
        assert row["Name"] == languages[row["ID"]]["Name"], row["ID"]


def test_stage_and_reconstruction_names_are_in_the_table():
    flagged = {row["ID"] for row in rows("historical-languages.csv")}
    unflagged = [
        row["ID"] for row in rows("languages.csv")
        if (re.match(r"(Old|Middle|Proto|Pre)[ -]", row["Name"])
            or row["Clade"] in {"OIA", "MIA", "Early NIA"})
        and row["ID"] not in flagged
    ]
    assert unflagged == []


def test_ancient_comparanda_are_historical():
    basis = {row["ID"]: row["Basis"] for row in rows("historical-languages.csv")}
    for language_id in (
        "Lat", "Av", "MPrs", "Pahl", "Parth", "Goth", "Hit", "Toch", "Sogd", "Khot", "Chvar",
        "Akkadian", "Aram", "Dacian", "Phrygian", "Palaic", "MB", "MiddleChinese", "Gk",
    ):
        assert basis.get(language_id) == "attested", language_id
