import importlib.util
import json
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
    assert references.source_field(note) == (
        "CDIAL;BKPD[64];AO[viii 300];Him.I[27];GWZS[2916];BKPD[65]"
    )


def test_cdial_reference_locators_keep_printed_volume_and_item_structure():
    note = "Morgenstierne NTS xvii 227; IIFL iii 3, 9; IL 17, 158; LM 251-2"
    assert references.extract_references(note) == [
        ("Morgenstierne", ""),
        ("NTS", "xvii 227"),
        ("IIFL", "iii 3, 9"),
        ("IL", "17, 158"),
        ("LM", "251-2"),
    ]
    assert references.source_field(note) == (
        "CDIAL;Morgenstierne;NTS[xvii 227];IIFL[iii 3, 9];IL[17, 158];LM[251-2]"
    )


def test_cdial_entry_sources_preserve_all_references_with_printed_locators():
    description = (
        "[← Drav. Burrow BSOAS 12, 365; Mayrhofer EWA I 17; "
        "compare EWA iii 626 and PMWS 76]"
    )
    assert references.entry_source_field(description) == (
        "CDIAL;BSOAS[12, 365];EWA[I 17];EWA[iii 626];PMWS[76]"
    )
    assert references.entry_source_field("[No auxiliary bibliography]") == ""


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


def test_cdial_aggregate_and_previously_unmapped_work_labels_supply_eras():
    tags, note = extract_tags("Br.; Ep.; Dhātup.; Pañcat.; VarBr̥S.; Rājat.")
    assert set(tags.split()) == {
        "Br", "Ep", "Dhātup", "Pañcat", "VarBr̥S", "Rājat",
        "Early-Vedic", "Epic", "Classical", "Medieval",
    }
    assert note == ""


def test_modern_dictionaries_and_undated_lexicographic_labels_stay_undated():
    tags, note = extract_tags("MW; W.; Apte; Gal.; Cat.; lex")
    assert set(tags.split()) == {"MW", "W", "Apte", "Gal", "Cat", "lex"}
    assert not {"Early-Vedic", "Late-Vedic", "Epic", "Classical", "Medieval"} & set(
        tags.split()
    )
    assert note == ""


def test_uncertain_is_a_structured_grammatical_tag():
    tags, note = extract_tags("uncertain; noun; uncertain Turner etymology T-111?")
    assert tags == "uncertain noun"
    assert note == "uncertain Turner etymology T-111?"


def test_inherited_is_not_a_schema_tag():
    tags, note = extract_tags("inherited")
    assert tags == ""
    assert note == "inherited"


def test_cdial_grammatical_abbreviations_are_normalized_to_schema_tags():
    tags, note = extract_tags(
        "pret.; absol.; inst.; imper.; vb.; subst.; sb.; st.; opt.; aor.; perf.; part."
    )
    assert tags.split() == [
        "pret", "abs", "instr", "impv", "verb", "noun", "stem", "opt", "aor",
        "perfect", "participle",
    ]
    assert note == ""


def test_cdial_person_number_and_dotted_gender_labels_are_normalized():
    tags, note = extract_tags("m.n.; 1st sg.; 2 sg.; 3rd pl.")
    assert tags.split() == ["mn", "1sg", "2sg", "3pl"]
    assert note == ""


def test_grammatical_words_inside_prose_are_preserved_conservatively():
    tags, note = extract_tags("m; pret. of <i>gam</i>; doubtful inst. analysis")
    assert tags == "m"
    assert note == "pret. of <i>gam</i>; doubtful inst. analysis"


def test_cdial_register_and_region_labels_become_structured_tags():
    tags, note = extract_tags(
        "poet.; dial.; mod.; old; colloq.; vulg.; hon.; (Gaya); (SWShahabad); Manbhum dial",
        language_id="Bi",
    )
    assert tags.split() == [
        "poetic", "dialectal", "modern", "archaic", "colloquial", "vulgar", "honorific",
        "dialect:Bi:cdial-Bi-gaya:Gaya",
        "dialect:Bi:cdial-Bi-southwest-shahabad:Southwest%20Shahabad",
        "dialect:Bi:cdial-Bi-manbhum:Manbhum",
    ]
    assert note == ""


def test_unlisted_parenthesized_labels_remain_notes():
    tags, note = extract_tags("(unknown village); probably dialectal", language_id="Bi")
    assert tags == ""
    assert note == "(unknown village); probably dialectal"


def test_cdial_directional_regions_remain_distinct_dialect_labels():
    tags, note = extract_tags("(ETirhut); (NETirhut); (SE Tirhut)", language_id="Mth")
    assert tags.split() == [
        "dialect:Mth:cdial-Mth-east-tirhut:East%20Tirhut",
        "dialect:Mth:cdial-Mth-northeast-tirhut:Northeast%20Tirhut",
        "dialect:Mth:cdial-Mth-southeast-tirhut:Southeast%20Tirhut",
    ]
    assert note == ""


def test_every_cdial_bibliography_abbreviation_has_display_metadata():
    catalog_path = MODULE.with_name("reference_catalog.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert references.REFERENCE_ABBREVS <= catalog.keys()
    assert all(catalog[key].strip() for key in references.REFERENCE_ABBREVS)
def test_uppercase_source_code_is_not_a_grammatical_tag():
    tags, note = extract_tags("Tr.")

    assert tags == ""
    assert note == "Tr."
