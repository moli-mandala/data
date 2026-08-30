"""Regression tests for the Knobloch (2020) Sauji grammar-sketch ingest."""

import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).parents[1]


def load_source(filename, module_name):
    path = ROOT / "data/other/forms/raw_data" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sauji = load_source("knobloch_sauji_2020.py", "knobloch_sauji_2020")
RAW = sauji.records()
FORMS, AUDIT = sauji.build()


def installed_rows():
    with sauji.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


AUDIT_BY_UNIT = {row["Unit_ID"]: row for row in AUDIT}


def resolved(unit):
    """The installed row a raw unit ended up in, following the fold if it merged."""
    entry = AUDIT_BY_UNIT[unit]
    key = entry["Merged_Into"].split(";")[0] or entry["Emitted_Key"].split(";")[0]
    return {row[10]: row for row in installed_rows()}[key]


def test_every_region_of_the_thesis_is_covered_with_the_expected_record_counts():
    assert Counter(record["region"] for record in RAW) == {
        "interlinear": 482, "inline": 135, "table2": 84, "table9": 46, "t7": 38,
        "t8": 31, "prose": 20, "table10": 20, "table14": 18, "table4": 16,
        "t6": 16, "table11": 14, "phonotactics": 6,
    }
    assert len(RAW) == 926
    # All 46 printed numerals, both cardinal blocks of the personal-pronoun
    # paradigm, and every non-empty consonant example cell.
    assert len({r["unit"] for r in RAW if r["region"] == "table9"}) == 46
    assert {r["form"] for r in RAW if r["unit"] == "t9:1"} == {"yak"}


def test_statuses_account_for_every_raw_record_and_name_their_reason():
    assert len(AUDIT) == len(RAW)
    assert Counter(row["Status"] for row in AUDIT) == {
        "installed": 884, "installed_after_repair": 2, "skipped": 40,
    }
    assert all(row["Reason"] for row in AUDIT)
    skipped = {row["Unit_ID"] for row in AUDIT if row["Status"] == "skipped"}
    # Bracketed ellipses, the Palula comparanda, and the three tables the prose
    # pattern reads out of alignment are the only deliberate exclusions.
    assert "p32:c5" in skipped and "p43:c1" in skipped
    assert sum(1 for row in AUDIT if "ellipsis" in row["Reason"]) == 6
    assert not any(row["Status"] == "skipped" and row["Emitted_Key"] for row in AUDIT)


def test_repeated_attestations_fold_but_homographs_and_polysemy_survive():
    rows = installed_rows()
    assert len(rows) == len(FORMS) == 573
    keys = [row[10] for row in rows]
    assert len(keys) == len(set(keys))
    by_form = {}
    for row in rows:
        by_form.setdefault(row[2], set()).add(row[3])
    assert by_form["si"] == {"bridge", "together", "together with"}
    assert by_form["baanu"] == {"becomes", "goes"}
    assert by_form["aw"] == {"and"}
    conjunction = next(row for row in rows if row[2] == "aw")
    assert conjunction[7].count("knobloch2020sauji[") == 24


def test_locators_carry_the_printed_page_and_the_elicitation_reference():
    assert resolved("t2:p:initial")[7].startswith(
        "knobloch2020sauji[p. 13, Table 2, /p/ initial]"
    )
    assert resolved("t9:20")[7] == "knobloch2020sauji[p. 24, Table 9, numeral 20]"
    assert "data 51_FR_170405" in resolved("ex14a:t1:w1")[7]
    # Appendix C prints its recording reference in the section heading, not in
    # the example, so the importer supplies it from the heading.
    assert "data 10_MAN_000512" in resolved("ex91:t1:w3")[7]
    assert all(row[7].startswith("knobloch2020sauji[p. ") for row in installed_rows())


def test_forms_reproduced_from_buddruss_carry_his_own_citation():
    assert "buddruss1967sau[p. 99]" in resolved("t2:p:final")[7]
    assert "buddruss1967sau[pp. 41-43]" in resolved("t8:dat:prox.det.pl:1")[7]
    assert "Buddruss" in resolved("t8:dat:prox.det.pl:1")[6]
    assert "buddruss1967sau[p. 39]" in resolved("prose:p20:1")[7]


def test_glosses_are_lexical_and_grammar_becomes_canonical_tags():
    house = resolved("ex77:t1:w1")
    assert (house[2], house[3]) == ("goš-ee", "house")
    assert set(house[14].split()) >= {"obl", "noun"}
    verb = resolved("ex14a:t1:w3")
    assert (verb[2], verb[3]) == ("deeš-i", "see")
    assert set(verb[14].split()) >= {"pfv", "f", "sg", "verb"}
    # A gloss with no lexical part names a pronoun; the person and deixis decide it.
    assert resolved("ex69:t1:w3")[3] == "he, she, it (remote)"
    # Parenthesised categories become tags, parenthesised prose stays in the gloss.
    assert resolved("ex63:t1:w2")[3] == "bear"
    assert resolved("p30:c14")[3] == "after (temporal)"
    # The author's own hedge is preserved as a tag rather than as a definition.
    hedged = resolved("ex90:t1:w2")
    assert hedged[3] == "then" and "uncertain" in hedged[14].split()


def test_inflection_classes_and_loanwords_follow_the_thesis():
    assert "Sauji-verb-class-3" in resolved("t10:3(-t):to give:pfv:1")[14].split()
    assert "Sauji-noun-class-2" in resolved("p20:c2")[14].split()
    knee = resolved("p43:c6")
    assert "loanword" in knee[14].split()
    assert knee[9] == "Knobloch identifies this as a loan from Gawarbati."
    assert not any(row[8] for row in installed_rows())  # no cognateset is claimed
    assert not any(row[1] for row in installed_rows())  # every row stays unlinked


def test_printed_alternates_become_variant_rows_with_resolvable_keys():
    rows = {row[10]: row for row in installed_rows()}
    variant = rows["knobloch2020sauji:p33:c20:v2"]
    assert variant[2] == "pašuanu"
    assert variant[11] == "knobloch2020sauji:p33:c20"
    assert variant[6] == "printed as pašu(w)anu"
    keys = {row[10] for row in installed_rows()}
    assert all(row[11] in keys for row in installed_rows() if row[11])


def test_the_two_printed_notations_are_separated_and_the_profile_covers_both():
    # Phonology tables print broad IPA, which stays in Phonemic; the running
    # transcription has no separate phonemic layer.
    leaf = resolved("t2:p:initial")
    assert (leaf[2], leaf[5]) == ("paːɬu", "paːɬu")
    assert resolved("t9:20")[5] == ""
    tokenizer = Tokenizer(str(ROOT / "conversion/knobloch-sauji.txt"))
    for row in installed_rows():
        source = unicodedata.normalize("NFC", row[2]).strip(",;.")
        converted = tokenizer(source, column="IPA").replace(" ", "").replace("#", " ")
        assert "�" not in converted, row


def test_the_sources_own_slips_are_repaired_with_the_reason_recorded():
    # p. 13 leaves the closing quote open on the /l/ medial cell.
    assert (resolved("t2:l:medial")[2], resolved("t2:l:medial")[3]) == ("alo", "be")
    # The same cell block prints ``'ʃaŋko' wood`` with the quotes around the form.
    assert (resolved("t2:ʃ:initial")[2], resolved("t2:ʃ:initial")[3]) == ("ʃaŋko", "wood")
    repaired = {row["Unit_ID"] for row in AUDIT if row["Status"] == "installed_after_repair"}
    assert repaired == {"t2:l:medial", "p16:c2"}


def test_all_rows_are_canonical_sauji_on_the_registered_sau_dialect():
    rows = installed_rows()
    assert {row[0] for row in rows} == {"Sv"}
    assert all("dialect:Sv:HKAT-sdg:Sau" in row[14].split() for row in rows)
    assert {len(row) for row in rows} == {15}
    assert not any("�" in "|".join(row) for row in rows)
    dialects = {row["Tag"] for row in csv.DictReader((ROOT / "cldf/dialects.csv").open())}
    assert "dialect:Sv:HKAT-sdg:Sau" in dialects


def test_manifest_records_rights_provenance_and_scope():
    data = json.loads((ROOT / "data/other/forms/raw_data/20260825-knobloch-sauji-manifest.json").read_text())
    assert data["pdf_sha256"] == sauji.PDF_SHA256
    assert data["pdf_sha512"] == sauji.PDF_SHA512
    assert data["pdf_sha512_matches_diva_record"] is True
    assert data["pdf_redistributed"] is False
    assert data["extraction"]["method"].endswith("no OCR")
    assert data["outputs"]["form_count"] == 573
    assert data["outputs"]["audit_count"] == 926
    assert "Palula and Kalkoti comparanda" in data["scope"]["excluded"]


def test_every_installed_row_survives_the_full_build():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists() or forms_path.stat().st_mtime < sauji.FORM_OUTPUT.stat().st_mtime:
        return
    with forms_path.open(encoding="utf-8", newline="") as handle:
        compiled = [row for row in csv.DictReader(handle) if sauji.SOURCE_ID in row["Source"]]
    assert len(compiled) == 573
    assert {row["Language_ID"] for row in compiled} == {"Sv"}
    # The profile converts both printed notations into house transcription while
    # Original keeps the source spelling and Phonemic keeps the printed IPA.
    def compiled_at(locator):
        return next(row for row in compiled if locator in row["Source"])

    leaf = compiled_at("Table 2, /p/ initial")
    assert (leaf["Form"], leaf["Original"], leaf["Phonemic"]) == ("pāɬu", "paːɬu", "paːɬu")
    stem = compiled_at("Table 10, class 1(-il) pfv stem")
    assert (stem["Form"], stem["Original"]) == ("tʰil-", "thil-")  # bound hyphen survives
    twenty = compiled_at("Table 9, numeral 20")
    assert (twenty["Form"], twenty["Phonemic"]) == ("biś", "")
    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as handle:
        resolved = {row["Source_Key"] for row in csv.DictReader(handle)}
    assert all(row[10] in resolved for row in installed_rows())
    with (ROOT / "cldf/references.csv").open(encoding="utf-8", newline="") as handle:
        references = {row["ID"] for row in csv.DictReader(handle)}
    assert {sauji.SOURCE_ID, sauji.BUDDRUSS_ID} <= references
