import csv
from pathlib import Path

from dialects import load_dialect_aliases, normalize_dialect


ROOT = Path(__file__).parents[1]


def rows(name):
    with (ROOT / "cldf" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_languages_are_base_languages_and_dialects_are_explicit():
    languages = rows("languages.csv")
    dialects = rows("dialects.csv")
    language_ids = {row["ID"] for row in languages}

    assert all(": " not in row["Name"] for row in languages)
    assert len({row["Tag"] for row in dialects}) == len(dialects)
    assert all(row["Language_ID"] in language_ids for row in dialects)


def test_every_form_language_and_dialect_tag_is_registered():
    language_ids = {row["ID"] for row in rows("languages.csv")}
    dialect_tags = {row["Tag"] for row in rows("dialects.csv")}
    forms = rows("forms.csv")

    assert all(row["Language_ID"] in language_ids for row in forms)
    used_tags = {
        tag
        for row in forms
        for tag in row["Tags"].split()
        if tag.startswith("dialect:")
    }
    assert used_tags <= dialect_tags


def test_source_lect_normalization_replaces_redundant_short_tag():
    aliases = load_dialect_aliases(ROOT / "cldf/dialects.csv")
    language_id, tags = normalize_dialect("biori", "noun dialect:Biori", aliases)

    assert language_id == "Phal"
    assert tags.split() == ["noun", "dialect:Phal:biori:Biori"]


def test_cdial_regional_dialects_are_georeferenced():
    regional = [row for row in rows("dialects.csv") if row["ID"].startswith("cdial-")]

    assert regional
    assert all(row["Tag"].startswith("dialect:") for row in regional)
    assert all(row["Latitude"] and row["Longitude"] for row in regional)
    assert all(row["Location"] and row["Quality"] in {"A", "B", "C"} for row in regional)


def test_every_dialect_has_human_readable_location_metadata():
    dialects = rows("dialects.csv")

    assert all(row["Location"].strip() for row in dialects)
    assert all(row["Quality"] in {"A", "B", "C"} for row in dialects)
    assert all(bool(row["Latitude"].strip()) == bool(row["Longitude"].strip()) for row in dialects)


def test_every_dialect_has_coordinates_in_range():
    dialects = rows("dialects.csv")

    assert all(row["Latitude"].strip() and row["Longitude"].strip() for row in dialects)
    assert all(-90 <= float(row["Latitude"]) <= 90 for row in dialects)
    assert all(-180 <= float(row["Longitude"]) <= 180 for row in dialects)
