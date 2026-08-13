import csv
import importlib.util
import io
import sys
from pathlib import Path

from make_cldf import parse_file


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/liljegren_hindukush.py"
SPEC = importlib.util.spec_from_file_location("liljegren_hindukush_extractor", SCRIPT)
hindukush = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hindukush
SPEC.loader.exec_module(hindukush)


def test_concept_domains_and_basic_prompts_become_grammatical_tags():
    assert hindukush.grammatical_tags({"domain": "Kinship", "Name": "father"}) == ["noun"]
    assert hindukush.grammatical_tags({"domain": "Numerals", "Name": "37"}) == ["num"]
    assert hindukush.grammatical_tags({"domain": "", "Name": "drink (verb)"}) == ["verb"]
    assert hindukush.grammatical_tags({"domain": "", "Name": "full"}) == ["adj"]
    assert hindukush.grammatical_tags({"domain": "", "Name": "we"}) == ["pron"]
    assert hindukush.grammatical_tags({"domain": "", "Name": "blood"}) == ["noun"]


def test_language_ids_are_lect_specific_and_clades_follow_jambu_taxonomy():
    assert hindukush.language_id("gwt_a") == "HKAT-gwt_a"
    assert hindukush.language_id("gwt_p") == "HKAT-gwt_p"
    assert hindukush.clade({"ID": "gwt_a", "Family": "Indo-European", "SubGroup": "Indo-Aryan"}) == "Kunar"
    assert hindukush.clade({"ID": "dml", "Family": "Indo-European", "SubGroup": "Nuristani"}) == "Kunar"
    assert hindukush.clade({"ID": "ask", "Family": "Indo-European", "SubGroup": "Nuristani"}) == "Nuristani"
    assert hindukush.clade({"ID": "bsk_h", "Family": "Burushaski", "SubGroup": ""}) == "Burushaski"
    assert hindukush.clade({"ID": "wbl_a", "Family": "Indo-European", "SubGroup": "Iranian"}) == "Other"


def test_lect_names_group_under_existing_database_languages():
    assert hindukush.lect_name("btv") == "Bhateri: Palas"
    assert hindukush.lect_name("kls") == "Indo-Aryan Kalasha: Bumburet (HKAT)"
    assert hindukush.lect_name("gju_a") == "Gujari: Naray (Afghanistan)"
    assert hindukush.lect_name("gwt_p") == "Gawarbati: Arandu (Pakistan)"
    assert hindukush.lect_name("plk") == "Shina: Palas (Kohistani)"
    assert set(hindukush.LECT_NAMES) == {
        "ask", "bft", "btv", "bkk", "bsk_h", "bsk_n", "kls", "dml", "prs_d",
        "gwt_a", "gwt_p", "gwc", "gju_a", "gju_p", "hno", "mvy", "isk", "xka",
        "xvi", "kas_i", "kas_p", "bsh_e", "bsh_w", "khw", "plk", "shd", "kir",
        "lbj", "mnj", "wbk", "phl", "prc", "psh_ai", "psi_ar", "glh_ag", "psi_am",
        "aee_at", "aee_ch", "aee_kg", "glh_sn", "aee_sh", "pbu_a", "pbu_i", "pbu_p",
        "phr", "prn", "prx", "sgh_r", "sgy", "sdg", "scl_p", "scl_i", "sgh_a",
        "trw", "ush", "uzs", "wbl_a", "wbl_p", "ydg",
    }


def test_generated_source_is_complete_source_keyed_and_located():
    source = ROOT / "data/other/forms/20260810-liljegren-hindukush.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 11_600
    assert all(len(row) == 15 for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {row[0] for row in rows} == {
        f"HKAT-{code}" for code in {
            "ask", "bft", "btv", "bkk", "bsk_h", "bsk_n", "kls", "dml", "prs_d",
            "gwt_a", "gwt_p", "gwc", "gju_a", "gju_p", "hno", "mvy", "isk", "xka",
            "xvi", "kas_i", "kas_p", "bsh_e", "bsh_w", "khw", "plk", "shd", "kir",
            "lbj", "mnj", "wbk", "phl", "prc", "psh_ai", "psi_ar", "glh_ag", "psi_am",
            "aee_at", "aee_ch", "aee_kg", "glh_sn", "aee_sh", "pbu_a", "pbu_i", "pbu_p",
            "phr", "prn", "prx", "sgh_r", "sgy", "sdg", "scl_p", "scl_i", "sgh_a",
            "trw", "ush", "uzs", "wbl_a", "wbl_p", "ydg",
        }
    }
    assert all(row[7].startswith("liljegren-hindukush[form ") for row in rows)
    assert all(", concept " in row[7] for row in rows)
    assert all(not row[1] for row in rows)
    assert {row[14] for row in rows} == {"noun", "verb", "adj", "pron", "num"}


def test_generated_rows_survive_manual_ingestion_as_unlinked_forms():
    source = ROOT / "data/other/forms/20260810-liljegren-hindukush.csv"
    rows, stats = parse_file(str(source), io.StringIO(), name="liljegren-hindukush")
    assert len(rows) == 11_600
    assert stats == {"converted": 11_600, "for_conversion": 11_600}
    assert all(row.is_lone and not row.param for row in rows)
    assert rows[0].entry_key.startswith("liljegren-hindukush:")
    assert all("�" not in row.form for row in rows)


def test_hindukush_ipa_profile_converts_display_forms_and_retains_phonemic_ipa():
    source = ROOT / "data/other/forms/20260810-liljegren-hindukush.csv"
    rows, _ = parse_file(str(source), io.StringIO())
    by_original = {row.old_form: row for row in rows}
    assert (by_original["aʈi"].form, by_original["aʈi"].ipa) == ("aṭi", "aʈi")
    assert (by_original["moːts"].form, by_original["moːts"].ipa) == ("mōʦ", "moːts")
    assert (by_original["kamaʈʂə"].form, by_original["kamaʈʂə"].ipa) == (
        "kamaʦ̣ə", "kamaʈʂə"
    )
    assert (by_original["t͡ʃam"].form, by_original["t͡ʃam"].ipa) == ("cam", "t͡ʃam")
    assert (by_original["ɫuj"].form, by_original["ɫuj"].ipa) == ("ḷuy", "ɫuj")


def test_all_source_lects_have_coordinates_locations_and_glottocodes():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["Source_Language_ID"].startswith("HKAT-")
        ]
    assert len(rows) == 59
    assert all(row["Tag"].startswith("dialect:") for row in rows)
    assert all(row["Glottocode"] and row["Latitude"] and row["Longitude"] for row in rows)
    assert all(row["Location"] and row["Quality"] == "A" for row in rows)
