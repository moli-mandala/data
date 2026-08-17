import csv
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_base_languages_use_available_dialect_coordinate_evidence():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = list(csv.DictReader(stream))

    parents_with_points = {
        row["Language_ID"]
        for row in dialects
        if row["Latitude"] and row["Longitude"]
    }
    assert all(
        languages[parent]["Latitude"] and languages[parent]["Longitude"]
        for parent in parents_with_points
    )


def test_only_geographically_undefined_languages_lack_coordinates():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        missing = {
            row["ID"] for row in csv.DictReader(stream)
            if not row["Latitude"] or not row["Longitude"]
        }
    assert missing == {"PBr", "TurkicUnspec"}
