"""Shared assertions for the reviewed dialect-coordinate policy (September 2026).

Survey sites are no longer left blank or pinned to the base language's point: every located
site carries a point recorded, with its provenance, in ``data/dialect-coordinate-decisions.csv``
(quality ``B`` for a gazetteer village point, ``C`` for the nearest source-named
administrative unit or a map reading). Sites that no gazetteer knows may still be blank.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parents[1]


def decisions():
    with (ROOT / "data/dialect-coordinate-decisions.csv").open(encoding="utf-8", newline="") as stream:
        return {row["ID"]: row for row in csv.DictReader(stream)}


def assert_reviewed_point(row, table=None):
    """A located dialect row: coordinates present, quality graded, and provenance recorded."""
    table = decisions() if table is None else table
    assert row["Latitude"] and row["Longitude"], row["ID"]
    assert row["Quality"] in {"A", "B", "C"}, row["ID"]
    # points set by the coordinate review are recorded with provenance; a few approximate points
    # predate the table (e.g. a report's "Bangla" comparison list) and carry no decision row
    decision = table.get(row["ID"])
    if decision is not None:
        assert (decision["Latitude"], decision["Longitude"]) == (row["Latitude"], row["Longitude"]), row["ID"]


def assert_reviewed_or_blank(row, table=None):
    """A survey site: either a reviewed point, or honestly blank when no gazetteer knows it."""
    if row["Latitude"] or row["Longitude"]:
        assert_reviewed_point(row, table)
    else:
        assert row["Quality"] in {"", "A", "B", "C"}, row["ID"]
