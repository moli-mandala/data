import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/shackle.py"
SPEC = importlib.util.spec_from_file_location("shackle_extractor", SCRIPT)
shackle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = shackle
SPEC.loader.exec_module(shackle)


def test_entry_start_recognizes_pos_crossref_and_posless_entry():
    assert shackle.start_of_entry("usaṭṭi, v.t. ‘throw, cast away’. 1.") == ("usaṭṭi", "vt")
    assert shackle.start_of_entry("akka: see AKKU") == ("akka", "xref")
    assert shackle.start_of_entry("āratī, ‘ceremony of waving lamp’. 3.") == ("āratī", "")
    assert shackle.start_of_entry("compound verbs. 470. [1200 āpayati]") is None


def test_clean_form_removes_inflection_and_ocr_homonym_marker():
    assert shackle.clean_form("ujjalā (-u)") == "ujjalā"
    assert shackle.clean_form('atītu [-a")') == "atītu"
    assert shackle.clean_form("uta?") == "uta"


def test_etymology_survives_missing_close_and_nested_comparison_brace():
    assert shackle.extract_bracket("usaṭṭi ... [< 1890 utsṛṣṭa-j") == "< 1890 utsṛṣṭa-j"
    assert shackle.extract_bracket("ujiālā ... [1673 *ujjvālaka- [x 386 andhakāra-]").startswith(
        "1673"
    )


def test_cdial_id_recovers_one_read_as_e():
    valid = {"1661", "661", "1673"}
    assert shackle.extract_ids("E661 *ujjaṭati", valid) == ["1661"]
    assert shackle.extract_ids("1673 *ujjvālaka-", valid) == ["1673"]


def test_unknown_etymology_question_marks_are_not_cdial_22():
    valid = {"2", "22", "188", "2189"}
    assert shackle.extract_ids("22", valid) == []
    assert shackle.extract_ids("< 188 *aḍḍ- 22", valid) == ["188"]
    assert shackle.extract_ids("22 : ef. Rj. vyupāi ?", valid) == []
    merged = "vāṛā ... [< 2189 upapātayati (with MIA. -t-) †viupāi ... [22 : ef. Rj.]"
    etymology = shackle.restore_unknown_markers(shackle.extract_bracket(merged))
    assert "[?? : cf." in etymology
    assert shackle.extract_ids(etymology + " Sr5.2", valid) == ["2189"]


def test_cdial_number_requires_matching_adjacent_ia_etymon():
    valid = {"7", "2985", "7627", "12225", "13096", "13906"}
    etyma = {
        "7": {"áṁsiya", "áṁsya"},
        "2985": {"kastūrī"},
        "7627": {"pakṣá"},
        "12225": {"vrájati", "vrájant"},
        "13096": {"sañcaka"},
        "13906": {"svayám"},
    }
    assert shackle.extract_ids("7?; sense confirmed elsewhere", valid, etyma) == []
    assert shackle.extract_ids("7627 paksa- + ?", valid, etyma) == ["7627"]
    assert shackle.extract_ids("12225 vrajant-", valid, etyma) == ["12225"]
    # Even an obvious-looking digit error is not silently reassigned: it is held
    # unlinked until reviewed.
    assert shackle.extract_ids("13906 sañcaka-", valid, etyma) == []
    assert shackle.extract_ids("2895 kastūrī-", valid, etyma) == []


def test_native_column_removes_dagger_and_restores_retroflex():
    assert shackle.restore_retroflex("tupakaru", "ਉਪਕਾਰੁ") == "upakaru"
    assert shackle.restore_retroflex("uthai", "ਉਠਾਇ") == "uṭhai"
    assert shackle.restore_retroflex(shackle.canonical_ocr("saùgu"), "ਸੰਗੁ") == "saṅgu"
    assert shackle.canonical_ocr("jħagari") == "jhagari"


def test_shackle_pos_features_become_structured_tags():
    assert shackle.tags_for_pos("vt") == ["verb", "tr"]
    assert shackle.tags_for_pos("vi; m; pp") == ["verb", "intr", "m", "pp"]
    assert shackle.tags_for_pos("poss pr") == ["pron", "poss"]
    assert shackle.tagged_notes("fut 3s", "source note") == "verb fut 3sg; source note"
