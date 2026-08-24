import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/nured_org.py"
SPEC = importlib.util.spec_from_file_location("nured_org_importer", SCRIPT)
nured = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = nured
SPEC.loader.exec_module(nured)


def read_rows(path, fieldnames=None):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, fieldnames=fieldnames))


def nured_forms():
    return read_rows(
        ROOT / "data/other/forms/20260818-nured-org.csv",
        fieldnames=nured.FORM_FIELDS,
    )


def test_head_normalization_removes_accents_but_preserves_length_and_segments():
    assert nured.normalize_head("*mā́lā-") == nured.normalize_head("mālā")
    assert nured.normalize_head("mā́lā") != nured.normalize_head("mala")
    assert nured.normalize_head("aṅkurá-") != nured.normalize_head("ankura-")
    assert nured.head_variants("*ankra(ka)-") == {"ankra", "ankraka"}


def test_printed_turner_ids_are_validated_and_deduplicated():
    text = "See {{CDIAL}} (T. 3906), T. 3906, and an invalid T. 999999."
    assert nured.printed_turner_ids(text, {"3906", "1"}) == ["3906"]


def test_balanced_form_templates_keep_nested_markup_and_skip_named_arguments():
    text = (
        "{{Lemma|lang=x-pnur|*āč[iya]n[a]-|n. m.|woolen thread}}\n"
        "== Etymology of Source ==\n"
        "* '''Old Indo-Aryan''' {{Form|lang=sa|[[nás-|nás-]] ~ {{IA|nā́s-}}|n.|nose}} + "
        "<big>-ra-</big>\n"
    )
    templates = nured.form_templates(text)
    assert templates[0][2] == ["[[nás-|nás-]] ~ {{IA|nā́s-}}", "n.", "nose"]
    assert nured.lemma_fields(text) == ("*āč[iya]n[a]-", "n. m.", "woolen thread")
    assert nured.source_lemma(text) == "*āč[iya]n[a]-"


def test_cdial_addenda_are_canonicalized_to_reviewed_main_entries():
    index, ids, merges = nured.cdial_index(nured.DEFAULT_CDIAL, nured.DEFAULT_MERGES)
    assert merges["14349"] == "2680"
    assert {entry_id for entry_id, _ in index[nured.normalize_head("kaṇṭhá-")]} == {"2680"}
    assert {"14349", "2680"} <= ids


def test_form_parser_emits_source_keyed_reflexes_and_expands_alternates():
    row = {
        "Page_ID": "42",
        "Source_Citation": "nured[page 42, revision 77, 2026-08-18]",
        "Raw_Wikitext": """
== Nuristani ==
* '''Katë'''
** {{ne}} {{Form|lang=x-nur|ačẽ́ ~ acẽ|n. m.|woolen thread}}
* '''Prasun'''
** ? {{p}} {{Form|wuzí|n.|barley}}
""",
    }
    forms, template_count, unparsed = nured.parse_nuristani_forms(
        row, ["n-test"], {}, {"n-test": "*ač-"}
    )
    assert (template_count, unparsed, len(forms)) == (2, 0, 3)
    assert {form["Parameter_ID"] for form in forms} == {"n-test"}
    assert forms[0]["Entry_Key"] == "nured:42:form:1"
    assert forms[1]["Entry_Key"] == "nured:42:form:1:alt:2"
    assert forms[1]["Variant_Of_Key"] == "nured:42:form:1"
    assert "alternate" in forms[1]["Tags"].split()
    assert "dialect:Kt:nured-Kt-ne:" in forms[0]["Tags"]
    assert "uncertain" in forms[2]["Tags"].split()


def test_commentary_extractor_omits_lexical_sections_and_keeps_referenced_notes():
    page = {"pageid": 42, "revid": 77, "title": "*test"}
    raw = """
      <div class="mw-parser-output">
      <div><h2 id="Nuristani">Nuristani</h2></div><p>lexical forms</p>
      <div><h2 id="Commentary">Commentary</h2></div>
      <p>Analysis.<sup id="cite_ref-1"><a href="#cite_note-1">1</a></sup></p>
      <div><h2 id="Middle_Indo-Aryan">Middle Indo-Aryan</h2></div><p>source forms</p>
      <ol><li id="cite_note-1">A commentary note.</li></ol></div>
    """
    rendered = nured.sanitize_article(raw, page)
    result = nured.commentary_html(
        rendered,
        {"Page_ID": "42", "Revision_ID": "77", "Title": "*test"},
    )
    assert "Analysis." in result and "A commentary note." in result
    assert "Nuristani" not in result and "lexical forms" not in result
    assert "Middle Indo-Aryan" not in result and "source forms" not in result
    assert "oldid=77" in result and "revision 77" in result


def test_source_citation_uses_stable_page_and_revision_ids():
    page = {"pageid": 1435, "revid": 3672, "timestamp": "2026-08-03T15:08:55Z"}
    assert nured.source_citation(page) == "nured[page 1435, revision 3672, 2026-08-03]"


def test_unchanged_live_inventory_does_not_create_a_date_only_refresh(tmp_path):
    prior = {
        "snapshot_date": "2026-08-18",
        "pages_sha256": "same",
        "namespace_0_pages": 875,
        "hard_redirects_excluded": 770,
        "nonredirect_pages_audited": 105,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(prior), encoding="utf-8")
    current = {**prior, "snapshot_date": "2026-08-25"}
    rows = [{"Snapshot_Date": "2026-08-25"}]
    nured.retain_snapshot_date_when_unchanged(rows, current, path)
    assert current["snapshot_date"] == rows[0]["Snapshot_Date"] == "2026-08-18"


def test_checked_in_snapshot_is_structured_complete_and_reproducible():
    audit = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-audit.csv")
    blocks = read_rows(ROOT / "data/other/entry_texts/20260818-nured-org.csv")
    forms = nured_forms()
    params = read_rows(
        ROOT / "data/other/params/nured.csv",
        fieldnames=["ID", "Language_ID", "Name", "Description", "Source"],
    )
    manifest = json.loads(
        (ROOT / "data/other/forms/raw_data/20260818-nured-org-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(audit) == manifest["nonredirect_pages_audited"] == 105
    assert len(forms) == manifest["installed_reflex_rows"] == 263
    assert len(blocks) == manifest["installed_text_blocks"] == 50
    assert len(params) == manifest["generated_pnur_entries"] == 18
    assert manifest["form_template_count"] == 255
    assert manifest["unparsed_form_templates"] == 0
    assert Counter(row["Status"] for row in audit) == {"excluded": 58, "ingested": 47}
    assert Counter(form["Language_ID"] for form in forms) == {
        "Kt": 104, "Pr": 63, "Wg": 48, "Ash": 36, "Gmb": 9, "Dm": 3,
    }
    assert all(len(row["Wikitext_SHA256"]) == 64 for row in audit)
    assert all("�" not in row["Raw_Wikitext"] + row["Rendered_HTML"] for row in audit)
    assert len({form["Entry_Key"] for form in forms}) == len(forms)
    assert all(form["Form"] and "�" not in form["Form"] for form in forms)
    assert all(form["Source"].startswith("nured[page ") for form in forms)
    assert all(form["Parameter_ID"].startswith(("n",)) for form in forms)
    assert all(row["Language_ID"] == "PNur" and row["Source"] == "nured" for row in params)
    assert all(row["Form_ID"].startswith("n") for row in blocks)
    assert all(row["Format"] == "html" and row["Kind"] == "etymology" for row in blocks)
    assert len({(row["Form_ID"], row["Position"]) for row in blocks}) == len(blocks)
    assert all("NurED commentary:" in row["Content"] for row in blocks)
    assert all(">Nuristani<" not in row["Content"] for row in blocks)
    assert all("Middle Indo-Aryan</h2>" not in row["Content"] for row in blocks)


def test_cdial_articles_route_through_compatible_pnur_siblings_or_generate_one():
    forms = nured_forms()
    page = lambda page_id: [
        row for row in forms if row["Entry_Key"].startswith(f"nured:{page_id}:")
    ]

    assert {(row["Language_ID"], row["Parameter_ID"]) for row in page("169")} == {
        ("Kt", "n531"), ("Pr", "n532"),
    }
    assert {row["Parameter_ID"] for row in page("226")} == {"nured-226"}
    assert {row["Parameter_ID"] for row in page("1082")} == {"n749"}
    assert {(row["Language_ID"], row["Parameter_ID"]) for row in page("1435")} == {
        ("Kt", "n3255"), ("Wg", "n3255"), ("Gmb", "n3255"),
        ("Ash", "n3255"), ("Pr", "n3256"),
    }
    alternates = [row for row in page("206") if row["Variant_Of_Key"]]
    assert alternates
    assert all(row["Entry_Key"].startswith(row["Variant_Of_Key"] + ":alt:") for row in alternates)


def test_missing_pnur_siblings_are_stable_entries_and_cdial_borrowings():
    params = read_rows(
        ROOT / "data/other/params/nured.csv",
        fieldnames=["ID", "Language_ID", "Name", "Description", "Source"],
    )
    ids = {row["ID"] for row in params}
    assert {"nured-126", "nured-226", "nured-1211", "nured-1352"} <= ids
    borrowings = read_rows(ROOT / "data/nuristani_borrowings.csv")
    generated = {
        row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
        for row in borrowings
        if row["Evidence"].startswith("NurED page ")
    }
    assert len(generated) == 16
    assert generated["nured-226"] == "2680"
    assert generated["nured-1211"] == "43"
    assert "nured-909" not in generated and "nured-1352" not in generated


def test_seeded_twenty_article_review_sample_has_no_material_errors():
    audit = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-audit.csv")
    scoped = [row for row in audit if row["Article_Type"]]
    expected = {row["Page_ID"] for row in random.Random(20260818).sample(scoped, 20)}
    sample = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-sample.csv")
    assert {row["Page_ID"] for row in sample} == expected
    assert all(
        row["Raw_Compared"] == row["Rendered_Compared"]
        == row["Forms_Compared"] == row["Commentary_Compared"] == "yes"
        for row in sample
    )
    assert all(row["Status"] == "ingested" and row["PNur_Targets"] for row in sample)
    assert all(row["Material_Error"] == "no" for row in sample)


def test_all_nured_rows_and_commentary_survive_the_compiled_outputs():
    raw_forms = nured_forms()
    source_keys = read_rows(ROOT / "cldf/form-source-keys.csv")
    installed_keys = {row["Source_Key"] for row in source_keys}
    assert {row["Entry_Key"] for row in raw_forms} <= installed_keys

    raw_text = read_rows(ROOT / "data/other/entry_texts/20260818-nured-org.csv")
    compiled = read_rows(ROOT / "cldf/entry-texts.csv")
    aliases = {
        row["Legacy_ID"]: row["Form_ID"]
        for row in read_rows(ROOT / "cldf/form-id-aliases.csv")
    }
    compiled_keys = {
        (row["Form_ID"], row["Position"], row["Source"])
        for row in compiled
    }
    for row in raw_text:
        key = (aliases.get(row["Form_ID"], row["Form_ID"]), row["Position"], row["Source"])
        assert key in compiled_keys
