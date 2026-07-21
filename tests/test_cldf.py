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


def test_nuristani_cognates_are_proto_indo_iranian_reflexes():
    with open("data/nuristani_cognates.csv", encoding="utf-8") as f:
        cognates = list(csv.DictReader(f))
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {r["ID"]: r for r in csv.DictReader(f)}
    with open("cldf/derivation.csv", encoding="utf-8") as f:
        edges = {(r["Child_ID"], r["Parent_ID"]) for r in csv.DictReader(f)}

    assert cognates
    assert len({r["Proto_Nuristani_ID"] for r in cognates}) == len(cognates)
    for row in cognates:
        ancestor = row["Ancestor_ID"]
        nuristani = row["Proto_Nuristani_ID"]
        indo_aryan = row["Indo_Aryan_ID"]
        assert forms[ancestor]["Language_ID"] == "Indo-ir"
        assert forms[ancestor]["Form"] == ""
        assert forms[nuristani]["Language_ID"] == "PNur"
        assert forms[indo_aryan]["Language_ID"] == "Indo-Aryan"
        assert forms[indo_aryan]["Relation"] != "borrowed"
        assert forms[nuristani]["Origin_ID"] == ancestor
        assert forms[nuristani]["Relation"] == "reflex"
        assert forms[indo_aryan]["Origin_ID"] == ancestor
        assert forms[indo_aryan]["Relation"] == "reflex"
        assert (nuristani, ancestor) not in edges
        assert (indo_aryan, ancestor) not in edges


def test_strand_indo_aryan_loans_are_nuristani_borrowings():
    with open("data/nuristani_borrowings.csv", encoding="utf-8") as f:
        borrowings = list(csv.DictReader(f))
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {r["ID"]: r for r in csv.DictReader(f)}

    assert borrowings
    assert len({r["Proto_Nuristani_ID"] for r in borrowings}) == len(borrowings)
    borrowed_nuristani = {r["Proto_Nuristani_ID"] for r in borrowings}
    for row in borrowings:
        nuristani = row["Proto_Nuristani_ID"]
        indo_aryan = row["Indo_Aryan_ID"]
        descendants = [
            form for entry_id, form in forms.items()
            if entry_id.startswith(f"{nuristani}-")
        ]
        assert forms[nuristani]["Language_ID"] == "PNur"
        assert forms[indo_aryan]["Language_ID"] == "Indo-Aryan"
        assert forms[nuristani]["Origin_ID"] == indo_aryan
        assert forms[nuristani]["Relation"] == "borrowed"
        assert forms[nuristani]["Borrowed_From"] == indo_aryan
        assert descendants
        assert all(form["Origin_ID"] == indo_aryan for form in descendants)
        assert all(form["Relation"] == "borrowed" for form in descendants)
        assert all(form["Borrowed_From"] == indo_aryan for form in descendants)
    assert all(form["Origin_ID"] not in borrowed_nuristani for form in forms.values())

    yamaraja = next(r for r in borrowings if r["Proto_Nuristani_ID"] == "n2571")
    assert yamaraja["Indo_Aryan_ID"] == "10425"
