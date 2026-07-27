import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/yoshioka.py"
SPEC = importlib.util.spec_from_file_location("yoshioka_extractor", SCRIPT)
yoshioka = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = yoshioka
SPEC.loader.exec_module(yoshioka)


def test_nominal_classes_and_valency_use_canonical_grammar_tags():
    assert yoshioka.grammatical_tags("word HM PL man") == [
        "noun", "Burushaski-class-HM", "m", "pl"
    ]
    assert yoshioka.grammatical_tags("root DITR IPFV give") == ["verb", "tr"]


def test_combined_nominal_classes_expand_to_filterable_class_tags():
    assert yoshioka.grammatical_tags("word HXY form") == [
        "noun",
        "Burushaski-class-H",
        "Burushaski-class-X",
        "Burushaski-class-Y",
    ]
    assert yoshioka.grammatical_tags("word YZ form") == [
        "noun", "Burushaski-class-Y", "Burushaski-class-Z"
    ]


def test_noun_classes_are_not_dialect_labels():
    tags = yoshioka.dialect_tags("alét SG H alín, X alés, Y alét PRN something")
    assert tags == ["dialect:Eastern%20Burushaski"]


def test_locality_codes_become_dialect_tags():
    tags = yoshioka.dialect_tags("stem HZ NG RF form meaning")
    assert tags == [
        "dialect:Eastern%20Burushaski",
        "dialect:Hunza",
        "dialect:Nager",
        "dialect:Riverfront",
    ]


def test_source_language_notes_become_loan_tags():
    assert yoshioka.loan_tags("aabáad Y residence ¶ UR ābād") == [
        "loanword", "loan:Urdu"
    ]
    assert yoshioka.loan_tags("word Y meaning || B.10") == []


def test_ocr_repairs_only_unmapped_native_glyphs():
    assert yoshioka.fill_holes("a�hó", "ačhó") == "ačhó"
    assert yoshioka.fill_holes("adít", "adt") == "adít"


def test_ocr_cedillas_are_restored_as_yoshioka_underdots():
    assert yoshioka.normalize_ocr_notation("çha şar țik") == "c̣ha ṣar ṭik"


def test_import_rows_have_rich_ingestion_schema():
    entry = yoshioka.Entry(
        pdf_page=505,
        printed_page=179,
        form="aabáad",
        gloss="resident, residence",
        raw_entry="aabáad Y resident, residence || B.10 ¶ UR ābād",
        reference_note="B.10",
        etymology="UR ābād",
        tags=["noun", "dialect:Eastern%20Burushaski", "loanword", "loan:Urdu"],
        entry_key="yoshioka-entry-1",
    )
    row = list(yoshioka.import_rows([entry]))[0]
    assert len(row) == 15
    assert row[0:4] == ["Bur", "", "aabáad", "resident, residence"]
    assert row[7] == "yoshioka2012"
    assert "dialect:Eastern%20Burushaski" in row[14]
