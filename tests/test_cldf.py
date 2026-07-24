import csv
import re
from pathlib import Path

import pytest
from pycldf import Dataset

from data.dedr.cleanup import is_footer_misparse


def test_validate():
    if not Path("cldf/parameters.csv").exists():
        pytest.skip("Wordlist metadata validates the pre-unification CLDF stage")
    d = Dataset.from_metadata("cldf/Wordlist-metadata.json")
    assert d.validate()


def test_every_used_reference_has_provenance():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        used = {
            token.split("[", 1)[0]
            for row in csv.DictReader(f)
            for token in row["Source"].split(";")
            if token
        }
    with open("cldf/references.csv", encoding="utf-8") as f:
        references = {row["ID"]: row for row in csv.DictReader(f)}

    assert used <= references.keys()
    assert all(references[key]["Provenance"].strip() for key in used)
    assert all(references[key]["Editor"].strip() for key in used)


def test_dedr_attached_parenthetical_is_a_variant():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}

    assert forms["d4993-33"]["Form"] == "mur̤ku"
    assert forms["d4993-33"]["Relation"] == "reflex"
    assert forms["d4993-34"]["Form"] == "mur̤uku"
    assert forms["d4993-34"]["Relation"] == "variant"
    assert forms["d4993-34"]["Variant_Of"] == "d4993-33"
    assert forms["d4993-33"]["Original"] == "mur̤(u)ku"
    assert forms["d4993-34"]["Original"] == "mur̤(u)ku"


def test_dedr_footer_references_are_not_forms():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = csv.DictReader(f)
        leaked = [row["Form"] for row in forms if is_footer_misparse(row["Form"])]

    assert leaked == []


def test_dedr_footer_references_are_preserved_on_entry():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        rows = {row["ID"]: row for row in csv.DictReader(f)}

    assert "DEDS 687" in rows["d4229"]["Etymology"]


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


def test_marked_origins_are_borrowings_with_valid_targets():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}

    marked = [
        row for row in forms.values()
        if row["Tags"] in {"marked borrowing", "semi-tatsama"}
    ]
    assert len(marked) == 221
    assert sum(row["Tags"] == "marked borrowing" for row in marked) == 158
    assert sum(row["Tags"] == "semi-tatsama" for row in marked) == 63
    for row in marked:
        assert row["Origin_ID"] in forms
        assert row["Relation"] == "borrowed"
        assert row["Borrowed_From"] == row["Origin_ID"]
        assert row["Origin_ID"][:1] not in {">", "~"}


def test_cross_family_descendants_are_borrowings():
    with open("cldf/languages.csv", encoding="utf-8") as f:
        clades = {row["ID"]: row["Clade"] for row in csv.DictReader(f)}
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}

    dravidian = {
        "Old Dravidian", "S. Dravidian I", "S. Dravidian II", "C. Dravidian",
        "N. Dravidian", "Brahui",
    }
    indo_aryan = {
        "OIA", "MIA", "Early NIA", "Nuristani", "Pashai", "Chitrali", "Shinaic",
        "Kohistani", "Kunar", "Kashmiric", "Sindhic", "Lahndic", "Punjabic",
        "W. Pahari", "C. Pahari", "E. Pahari", "Eastern", "Bihari", "E. Hindi",
        "W. Hindi", "Rajasthanic", "Gujaratic", "Bhil", "Khandeshi",
        "Marathi-Konkani", "Halbic", "Insular", "Migratory",
    }
    matched = []
    for row in forms.values():
        origin = forms.get(row["Origin_ID"])
        if not origin:
            continue
        child_clade = clades.get(row["Language_ID"])
        origin_clade = clades.get(origin["Language_ID"])
        dravidian_loan = child_clade in dravidian and origin["Language_ID"] != "PDr"
        ia_loan = (
            origin_clade in indo_aryan
            and child_clade not in indo_aryan
        )
        if dravidian_loan or ia_loan:
            matched.append(row)
            assert row["Relation"] == "borrowed"
            assert row["Borrowed_From"] == row["Origin_ID"]

    assert matched


def test_backstrom_control_wordlists_are_excluded():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        controls = [
            row for row in csv.DictReader(f)
            if "backstrom1992" in row["Source"].split(";")
            and row["Language_ID"] in {"H", "Psht"}
        ]
    assert not controls


def test_backstrom_poc_cloth_forms_link_to_pota_not_avajjharati():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = list(csv.DictReader(f))

    cloth = [
        row for row in forms
        if "backstrom1992" in row["Source"].split(";")
        and "cloth" in row["Gloss"].split(";")
        and row["Origin_ID"] in {"770", "8400"}
    ]
    assert len(cloth) == 19
    assert {row["Origin_ID"] for row in cloth} == {"8400"}


def test_audited_schmidt_assignments_are_applied_to_source_rows():
    with open(
        "data/other/forms/raw_data/schmidt_shina_database_etymologies.csv",
        encoding="utf-8",
    ) as f:
        review = list(csv.DictReader(f))
    with open("data/other/forms/20230621-shina.csv", encoding="utf-8") as f:
        source = list(csv.reader(f))

    accepted = [row for row in review if row["Decision"] == "yes"]
    held = [row for row in review if row["Decision"] == "no"]
    assert len(accepted) == 314
    assert len(held) == 5
    for row in accepted:
        assert source[int(row["Row"]) - 1][1] == row["Parameter_ID"]
    for row in held:
        assert not source[int(row["Row"]) - 1][1]


def test_unified_form_ids_are_unique():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        ids = [row["ID"] for row in csv.DictReader(f)]
    assert len(ids) == len(set(ids))


def test_reference_editor_credits_for_assisted_sources():
    with open("cldf/references.csv", encoding="utf-8") as f:
        references = {row["ID"]: row for row in csv.DictReader(f)}

    assert references["backstrom1992"]["Editor"] == "Aryaman Arora; OpenAI Codex"
    assert references["schmidt"]["Editor"] == "Aryaman Arora; OpenAI Codex"
    assert references["lehr"]["Editor"] == "Aryaman Arora; OpenAI Codex"
    assert references["shackle-auto"]["Editor"] == "OpenAI Codex"
    assert references["canvin2025"]["Editor"] == (
        "Aryaman Arora; OpenAI Codex; Claude Opus 4.8"
    )


def test_both_shackle_sources_use_cdial_phonetic_conversion_and_tags():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["Source"] in {"shackle", "shackle-auto"}
        ]

    assert {row["Source"] for row in rows} == {"shackle", "shackle-auto"}
    pos_prefix = re.compile(
        r"^\((?:vt|vi|vs|m|f|n|adj|adv|pr|pron|ppn|pp|num|cj|conj|intj|ger|inf)\)\s"
    )
    assert all(not pos_prefix.match(row["Gloss"]) for row in rows)
    assert any("ʰ" in row["Form"] for row in rows if row["Source"] == "shackle")
    assert any("ʰ" in row["Form"] for row in rows if row["Source"] == "shackle-auto")
    assert not any("ħ" in row["Form"] for row in rows)


def test_shina_etymologisers_target_indo_aryan_not_indo_iranian():
    reviews = [
        "data/other/forms/raw_data/northern_shina_database_etymologies.csv",
        "data/other/forms/raw_data/schmidt_shina_database_etymologies.csv",
    ]
    for filename in reviews:
        with open(filename, encoding="utf-8") as f:
            accepted = [
                row for row in csv.DictReader(f) if row["Decision"] == "yes"
            ]
        assert accepted
        assert not any(row["Parameter_ID"].startswith("pii-") for row in accepted)

    with open("cldf/forms.csv", encoding="utf-8") as f:
        languages = {row["ID"]: row["Language_ID"] for row in csv.DictReader(f)}
    with open(reviews[1], encoding="utf-8") as f:
        schmidt = [row for row in csv.DictReader(f) if row["Decision"] == "yes"]
    assert {languages[row["Parameter_ID"]] for row in schmidt} == {"Indo-Aryan"}


def test_duplicate_strand_oia_heads_are_merged_into_cdial():
    with open("data/strand_oia_redirects.csv", encoding="utf-8") as f:
        redirects = {
            row["Strand_ID"]: row["CDIAL_ID"]
            for row in csv.DictReader(f)
        }
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}

    assert redirects
    assert not set(redirects) & set(forms)
    assert set(redirects.values()) <= set(forms)
    for row in forms.values():
        assert row["Origin_ID"] not in redirects
        assert row["Borrowed_From"] not in redirects
        assert row["Variant_Of"] not in redirects
        assert row["Redirect"] not in redirects


def test_strand_borrowings_are_aligned_to_final_indo_aryan_donors():
    with open("data/nuristani_borrowings.csv", encoding="utf-8") as f:
        borrowings = {
            row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
            for row in csv.DictReader(f)
        }
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}

    wanted = set(borrowings)
    wanted.update(
        entry_id for entry_id in forms
        if entry_id.startswith("n2571-")
    )
    aligned_origins = {entry_id: set() for entry_id in wanted}
    with open("cldf/alignments.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Form_ID"] in aligned_origins:
                aligned_origins[row["Form_ID"]].add(row["Origin_ID"])

    for nuristani, indo_aryan in borrowings.items():
        assert aligned_origins[nuristani] == {indo_aryan}
    for descendant in wanted - set(borrowings):
        assert aligned_origins[descendant] == {"10425"}
