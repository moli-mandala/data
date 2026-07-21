import csv

from pycldf import Dataset


def test_validate():
    d = Dataset.from_metadata("cldf/Wordlist-metadata.json")
    assert d.validate()


def test_curated_borrowings_are_applied():
    with open("data/borrowings.csv", encoding="utf-8") as f:
        borrowings = {r["Borrower_ID"]: r["Source_ID"] for r in csv.DictReader(f)}
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {r["ID"]: r for r in csv.DictReader(f)}

    assert borrowings
    assert all(source in forms for source in borrowings.values())
    for borrower, source in borrowings.items():
        assert forms[borrower]["Origin_ID"] == source
        assert forms[borrower]["Relation"] == "borrowed"
        assert forms[borrower]["Borrowed_From"] == source
