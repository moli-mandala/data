import csv
from pathlib import Path


SURVEY = Path("data/other/forms/20230521-rajasthani.csv")
COMPILED_FORMS = Path("cldf/forms.csv")
COMPILED_EDGES = Path("cldf/edges.csv")
SOURCE_KEYS = {"bagri", "dhundari", "hadothi", "marwari", "mewari", "mewati"}


def survey_rows():
    with SURVEY.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_curated_etymologies_survive_reingestion():
    rows = survey_rows()
    linked = [row for row in rows if row[1]]

    # These are curated CDIAL assignments, not claims present in the raw survey files.  The
    # importer must carry them forward when it refreshes forms, rather than emitting blank
    # Parameter_ID values for the entire source.
    assert len(linked) == 11_749
    assert {row[7] for row in linked} == SOURCE_KEYS
    assert any(
        row[0] == "marwari_gomat" and row[2] == "hũ" and row[1] == "992"
        for row in rows
    )
    assert any(
        row[0] == "hadothi_patera" and row[2] == "mu" and row[1] == "9691"
        for row in rows
    )


def test_survey_lect_ids_are_registered_stable_ids():
    assert all(" " not in row[0] for row in survey_rows())
    assert any(row[0] == "mewari_kishanji" for row in survey_rows())


def test_curated_etymologies_survive_the_compiled_graph():
    with COMPILED_FORMS.open(encoding="utf-8", newline="") as stream:
        forms = [
            row for row in csv.DictReader(stream)
            if any(
                citation.split("[", 1)[0] in SOURCE_KEYS
                for citation in row["Source"].split(";")
            )
        ]
    linked_ids = {row["ID"] for row in forms if row["Status"] != "unlinked"}
    with COMPILED_EDGES.open(encoding="utf-8", newline="") as stream:
        linked_edges = {
            row["Child_ID"] for row in csv.DictReader(stream)
            if row["Child_ID"] in linked_ids
            and row["Kind"] == "reflex"
            and row["Rank"] == "1"
        }

    # Cross-source deduplication folds some of the 11,749 input attestations into shared nodes.
    assert len(linked_ids) == 11_245
    assert linked_edges == linked_ids
