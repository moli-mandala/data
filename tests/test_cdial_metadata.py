import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from tags import extract_tags


MODULE = Path(__file__).parents[1] / "data" / "cdial" / "references.py"
SPEC = importlib.util.spec_from_file_location("cdial_references", MODULE)
references = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(references)


def test_main_and_addenda_bibliography_citations_are_extracted_in_order():
    note = "BKPD 64; cf. AO viii 300 and Him.I 27; GWZS 2916; BKPD 65"
    assert references.source_field(note) == "CDIAL;BKPD;AO;Him.I;GWZS"


def test_reference_matching_respects_boundaries_and_optional_full_stops():
    assert references.extract_reference_ids("NTS vii 110; G.M.; S.M.Katre") == [
        "NTS", "G.M", "S.M.Katre"
    ]
    assert references.extract_reference_ids("explanation and inside") == []


def test_addenda_typographic_aliases_get_canonical_reference_ids():
    note = "RTMV²; C. Shackle; S. M. Katre; Emeneau Sk. <i>bhōgin</i>- 216; ColPa 160"
    assert references.extract_reference_ids(note) == [
        "RTMV2", "C.Shackle", "S.M.Katre", "Emeneau Sk. bhōgin-", "ColPa"
    ]


def test_sanskrit_dictionary_attestations_are_tags_not_bibliography_refs():
    tags, note = extract_tags("m; MW; W.; Apte")
    assert tags.split()[:4] == ["m", "MW", "W", "Apte"]
    assert note == ""
    assert references.extract_reference_ids("MW; W.; Apte") == []


def test_addenda_sanskrit_works_are_source_and_era_tags():
    tags, note = extract_tags("f; AitĀr.; VādhS.; Śāktān.")
    assert set(tags.split()) == {
        "f", "AitĀr", "VādhS", "Śāktān", "Early-Vedic", "Late-Vedic", "Medieval"
    }
    assert note == ""


def test_sanskrit_works_embedded_in_prose_are_tagged_without_mangling_note():
    original = "n; ('devotion' Prab.com.); (<i>sudhyatē</i> ṢaḍvBr.); compare ŚrS."
    tags, note = extract_tags(original)
    assert {"n", "Prab", "com", "ṢaḍvBr", "ŚrS", "Early-Vedic", "Medieval"} <= set(
        tags.split()
    )
    assert note == "('devotion' Prab.com.); (<i>sudhyatē</i> ṢaḍvBr.); compare ŚrS."
