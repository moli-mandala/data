"""Regression tests for the Herin (2012) Aleppo Domari ingest."""

import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer

ROOT = Path(__file__).parents[1]
FORMS = ROOT / "data/other/forms/20260825-herin-domari-aleppo.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260825-herin-domari-aleppo-audit.csv"
EXTRACT = ROOT / "data/other/forms/raw_data/20260825-herin-domari-aleppo-extract.psv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260825-herin-domari-aleppo-manifest.json"
PROFILE = ROOT / "conversion/domari-aleppo.txt"


def load_source():
    path = ROOT / "data/other/forms/raw_data/herin_domari_aleppo_2012.py"
    spec = importlib.util.spec_from_file_location("herin_domari_aleppo_2012", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


herin = load_source()


def dict_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def raw_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


FORM_ROWS = raw_rows(FORMS)
AUDIT_ROWS = dict_rows(AUDIT)
EXTRACT_ROWS = dict_rows(EXTRACT, delimiter="|")
MANIFEST_DATA = json.loads(MANIFEST.read_text(encoding="utf-8"))
COLUMNS = {name: index for index, name in enumerate(herin.FORM_FIELDS)}


def field(row, name):
    return row[COLUMNS[name]]


def by_form(name):
    return [row for row in FORM_ROWS if field(row, "Form") == name]


def by_key(key):
    matches = [row for row in FORM_ROWS if field(row, "Entry_Key") == f"herin2012domari:{key}"]
    assert len(matches) == 1, key
    return matches[0]


# ----------------------------------------------------------------------------------
# Counts and reconciliation
# ----------------------------------------------------------------------------------

def test_counts_reconcile_from_raw_records_to_installed_rows():
    assert len(EXTRACT_ROWS) == 1561
    assert len(AUDIT_ROWS) == len(EXTRACT_ROWS)
    assert MANIFEST_DATA["raw_records"] == len(AUDIT_ROWS)
    statuses = Counter(row["Status"] for row in AUDIT_ROWS)
    assert statuses == {"ingested": 1088, "skipped": 473}
    assert len(FORM_ROWS) == 1074 == MANIFEST_DATA["installed_rows"]
    # More units are ingested than rows installed, because identical attestations merge.
    assert statuses["ingested"] >= len(FORM_ROWS)


def test_every_raw_record_has_a_status_and_every_exclusion_a_reason():
    assert all(row["Status"] in {"ingested", "skipped"} for row in AUDIT_ROWS)
    assert all(row["Reason"] for row in AUDIT_ROWS if row["Status"] == "skipped")
    assert all(not row["Reason"] for row in AUDIT_ROWS if row["Status"] == "ingested")
    assert all(row["Record_SHA256"] for row in AUDIT_ROWS)
    assert len({row["Unit_ID"] for row in AUDIT_ROWS}) == len(AUDIT_ROWS)


def test_the_four_source_regions_are_all_accounted_for():
    assert MANIFEST_DATA["raw_records_by_region"] == {
        "prose": 967, "paradigm": 191, "example": 329, "translation": 74,
    }
    # No free translation of a numbered example is ever installed.
    assert all(row["Status"] == "skipped"
               for row in AUDIT_ROWS if row["Region"] == "translation")


def test_row_widths_and_keys_are_well_formed():
    assert all(len(row) == len(herin.FORM_FIELDS) for row in FORM_ROWS)
    keys = [field(row, "Entry_Key") for row in FORM_ROWS]
    assert len(set(keys)) == len(keys)
    assert all(key.startswith("herin2012domari:") for key in keys)
    heads = set(keys)
    assert all(field(row, "Variant_Of_Key") in heads
               for row in FORM_ROWS if field(row, "Variant_Of_Key"))


# ----------------------------------------------------------------------------------
# Language, dialect and locators
# ----------------------------------------------------------------------------------

def test_every_row_is_aleppo_domari():
    assert {field(row, "Language_ID") for row in FORM_ROWS} == {"as"}
    assert all("dialect:as:aleppo:Aleppo" in field(row, "Tags").split() for row in FORM_ROWS)


def test_the_aleppo_dialect_is_registered_under_domari_with_an_exact_locality():
    rows = dict_rows(ROOT / "cldf/dialects.csv")
    aleppo = next(row for row in rows if row["ID"] == "aleppo")
    assert aleppo["Tag"] == "dialect:as:aleppo:Aleppo"
    assert aleppo["Language_ID"] == "as"
    assert aleppo["Glottocode"] == "doma1258"
    assert (float(aleppo["Latitude"]), float(aleppo["Longitude"])) == (36.2021, 37.1343)
    assert aleppo["Quality"] == "A"


def test_locators_name_the_printed_page_and_the_articles_own_citation_unit():
    for row in FORM_ROWS:
        for citation in field(row, "Source").split(";"):
            assert citation.startswith("herin2012domari[p. ")
            page = int(citation.split("p. ", 1)[1].split(",", 1)[0].rstrip("]"))
            assert herin.FIRST_PRINTED_PAGE <= page <= herin.LAST_PRINTED_PAGE
    assert field(by_key("t7:r1c1"), "Source") == (
        'herin2012domari[p. 29, Table 7, 1.SG., pī- “drink”]')
    assert "example (1a) word 3" in field(by_key("ex1a:w3"), "Source")
    assert field(by_key("fn20:q1"), "Source").endswith("footnote 20]")


def test_only_this_articles_bibliography_key_is_cited():
    keys = {citation.split("[", 1)[0]
            for row in FORM_ROWS for citation in field(row, "Source").split(";")}
    assert keys == {"herin2012domari"}
    assert "@article{herin2012domari," in (ROOT / "cldf/sources.bib").read_text(
        encoding="utf-8")


# ----------------------------------------------------------------------------------
# Extraction edge cases found during the manual audit
# ----------------------------------------------------------------------------------

def test_forms_split_across_font_runs_rejoin_without_an_inserted_space():
    # <font><i>pī</i></font><font><i>-r-ã</i></font> is one paradigm cell.
    assert field(by_key("t7:r3c1"), "Form") == "pī-r-ã"
    # ... but <font><i>lāfty-ā</i></font><font><i>\n muḥtaším</i></font> is two words.
    phrase = next(row for row in EXTRACT_ROWS if row["Unit_ID"] == "s2.14:q15")
    assert phrase["Raw_Form"] == "lāfty-ā muḥtaším"


def test_superscript_y_is_palatalisation_and_footnote_markers_are_dropped():
    assert by_form("hōtʸər")
    assert not [row for row in FORM_ROWS if "[" in field(row, "Form")]


def test_a_combining_mark_split_into_its_own_span_rejoins_with_its_vowel():
    assert by_form("kərī́")
    assert not [row for row in FORM_ROWS
                if " " + "́" in field(row, "Form")]


def test_bracketed_phonetic_realisations_are_kept_out_of_form_and_in_phonemic():
    row = by_key("s1.1:q1")
    assert field(row, "Form") == "pāpī́r"
    assert field(row, "Phonemic") == "pæːˈpiːr"
    assert field(row, "Gloss") == "grand-father"
    assert MANIFEST_DATA["rows_with_phonemic"] == 15


def test_a_verb_root_cited_with_a_detached_hyphen_keeps_the_hyphen():
    # The article prints ``sāk - "can, be able"`` with the hyphen outside the font run.
    assert field(by_key("s3.5:q1"), "Form") == "sāk-"
    assert field(by_key("s3.2:q9"), "Form") == "ʿīš h-"


def test_a_printed_alternation_becomes_a_head_plus_an_alternate_row():
    assert field(by_key("s1.2:q6"), "Form") == "ḥawt"
    assert field(by_key("s1.2:q6:variant:2"), "Form") == "ḥaft"
    assert "alternate" in field(by_key("s1.2:q6:variant:2"), "Tags").split()
    # The same alternation is printed again in the numeral list on p. 12; the two
    # attestations merge into one row that carries both citations.
    assert next(row for row in EXTRACT_ROWS
                if row["Unit_ID"] == "s2.7:q8")["Raw_Form"] == "ḥawt ~ ḥaft"
    assert "herin2012domari[p. 14, section 2.7]" in field(by_key("s1.2:q6"), "Source")


def test_an_alternation_elided_across_a_light_verb_is_completed():
    assert field(by_key("s1.2:q3"), "Form") == "dahn kar-"
    assert field(by_key("s1.2:q3:variant:2"), "Form") == "dāhín kar-"
    # A complete alternation is left exactly as printed.
    assert field(by_key("s2.7:q30"), "Form") == "trən ʋīst ʋīst"
    assert field(by_key("s2.7:q30:variant:2"), "Form") == "štār ʋīst"


def test_optional_segments_become_explicit_alternates():
    assert field(by_key("ex7c:w3"), "Form") == "hrōs-sa"
    assert field(by_key("ex7c:w3:variant:2"), "Form") == "rōs-sa"
    assert MANIFEST_DATA["installed_alternates"] == 84


def test_the_repeated_printing_of_example_36a_is_recorded_once():
    repeats = [row for row in AUDIT_ROWS if "-r2" in row["Unit_ID"]]
    assert repeats and all(row["Status"] == "skipped" for row in repeats)
    assert all("second printing" in row["Reason"] for row in repeats)


# ----------------------------------------------------------------------------------
# Gloss and tag curation
# ----------------------------------------------------------------------------------

def test_interlinear_glosses_split_into_a_lexical_gloss_and_canonical_tags():
    assert herin.parse_gloss("NEG-go.PROG.3SG") == ("go", ["neg", "progressive", "3sg",
                                                          "verb"], False)
    assert herin.parse_gloss("lighter-PL-2PL") == ("lighter", ["pl", "2pl"], False)
    assert herin.parse_gloss("2SG") == ("you (singular)", ["2sg", "pron", "personal"],
                                        False)
    # ``grand-father`` is one English word, not two morphemes.
    assert herin.parse_gloss("grand-father-OBL-SUP")[0] == "grand-father"
    # ``3.SG.f.`` writes person and number apart.
    assert herin.parse_gloss("fear.IMPFV.3.SG.f.") == ("fear",
                                                       ["ipfv", "3sg", "f", "verb"], False)


def test_no_part_of_speech_is_inferred_from_a_case_suffix():
    # trōt-ə "small-OBL" is an adjective agreeing in case, not a noun.
    assert herin.parse_gloss("small-OBL") == ("small", ["obl"], False)


def test_a_verb_is_only_cited_as_to_x_where_the_article_shows_it_is_verbal():
    assert field(by_key("s2.8:q17"), "Gloss") == "close"       # the adjective nēzək
    assert field(by_key("s2.9:q19"), "Gloss") == "fear"        # the deverbal noun bīnāʋīš
    assert field(by_key("s3.3:q31"), "Gloss") == "to wash"     # the root dō-
    assert field(by_key("t12:r1c1"), "Gloss") == "to do, to make"


def test_paradigm_cells_carry_the_lexeme_gloss_and_the_tables_own_categories():
    assert field(by_key("t9:r2c3"), "Form") == "nangə-č-ā"
    assert field(by_key("t9:r2c3"), "Gloss") == "to enter"
    assert set(field(by_key("t9:r2c3"), "Tags").split()) == {
        "verb", "subjunctive", "2sg", "dialect:as:aleppo:Aleppo"}
    assert field(by_key("t2:r1c1"), "Gloss") == "I"
    assert set(field(by_key("t2:r1c1"), "Tags").split()) >= {"pron", "personal", "1sg"}


def test_complex_verbs_are_tagged_as_multi_word_conjunct_verbs():
    row = by_key("s3.2:q38")
    assert field(row, "Form") == "ǧib kar-"
    assert {"verb", "conjunct-verb", "multiword-expression"} <= set(
        field(row, "Tags").split())


def test_every_tag_is_registered_in_the_shared_vocabulary():
    import tags as tag_registry

    used = {tag for row in FORM_ROWS for tag in field(row, "Tags").split()}
    unknown = {tag for tag in used
               if not tag.startswith(("dialect:", "loan:"))
               and tag not in tag_registry.GRAMMATICAL_TAGS
               and tag not in tag_registry.GENDER_TAGS}
    assert unknown == set()
    assert {"comitative", "superessive", "subjunctive", "progressive", "contextualiser",
            "remoteness", "complementizer", "definite"} <= used


def test_the_frontend_tag_list_matches_the_python_registry_for_the_new_tags():
    frontend = (ROOT.parent / "jambu-static/src/lib/tags.ts").read_text(encoding="utf-8")
    for tag in ("comitative", "superessive", "subjunctive", "progressive",
                "contextualiser", "remoteness", "complementizer", "definite"):
        assert f"'{tag}'" in frontend, tag


# ----------------------------------------------------------------------------------
# Exclusions
# ----------------------------------------------------------------------------------

def excluded(unit_id):
    row = next(row for row in AUDIT_ROWS if row["Unit_ID"] == unit_id)
    assert row["Status"] == "skipped", unit_id
    return row["Reason"]


def test_other_dom_varieties_and_earlier_descriptions_stay_out():
    assert "Beirut" in excluded("s2.11:q2")
    assert "Palestinian" in excluded("7-conclusion:q2")
    assert "comparison" in excluded("fn12:q1")
    # ... but an Aleppo form printed inside a comparative footnote is kept.
    assert by_form("lakərdã")
    assert next(row for row in AUDIT_ROWS
                if row["Unit_ID"] == "fn37:q6")["Status"] == "ingested"


def test_donor_and_etymon_forms_are_never_installed_as_domari():
    for unit_id in ("s1.2:q2", "s1.2:q50", "s3.5:q2", "s1.2:q68", "s3.2:q3"):
        assert excluded(unit_id)
    assert not by_form("fham")       # the Arabic source of fəmmōme
    assert not by_form("kirin")      # the Kurdish light verb
    assert not by_form("śákya")      # the Old Indo-Aryan etymon of sāk-


def test_charts_affixes_and_free_translations_stay_out():
    assert "phoneme-inventory" in excluded("t1:r1c1")
    assert "affix" in excluded("t4:r1c1")
    assert "affix" in excluded("s2.12:q26")
    assert "free translation" in excluded("ex1a:translation")
    assert not [row for row in FORM_ROWS if field(row, "Form").startswith("-")]


def test_clause_examples_are_excluded_but_their_words_are_installed_when_glossed():
    # (25) is a clause; its five words are installed one row each.
    assert "phrase or clause" in excluded("s2.14:q33")
    assert field(by_key("s2.11:q4:w1"), "Form") == "ʿarīs-a"
    assert field(by_key("s2.11:q4:w2"), "Gloss") == "father"


# ----------------------------------------------------------------------------------
# Etymology
# ----------------------------------------------------------------------------------

def test_donor_statements_become_etymology_prose_and_a_loan_tag():
    row = by_key("s1.1:q25")
    assert field(row, "Form") == "yēldəz"
    assert field(row, "Etymology") == "< Turkish yɪldɪz “star”"
    assert {"loanword", "loan:Turkish"} <= set(field(row, "Tags").split())
    assert MANIFEST_DATA["rows_with_etymology"] == 34


def test_prose_that_merely_names_a_language_is_not_read_as_an_etymology():
    # "Matter was taken from Kurdish har" compares two languages; it is not an
    # etymology of the preceding Domari citation.
    assert field(by_key("s6.2:q10"), "Etymology") == ""
    assert field(by_key("7-conclusion:q19"), "Etymology") == ""


def test_indo_aryan_etyma_link_to_cdial_only_on_an_exact_headword_match():
    assert field(by_key("s1.2:q51"), "Parameter_ID") == "11346"     # varta
    assert field(by_key("s3.5:q1"), "Parameter_ID") == "12253"      # śákya
    # pēṭṭa has no exact CDIAL head; it stays unlinked and visible in the audit.
    assert field(by_key("s1.2:q49"), "Parameter_ID") == ""
    candidate = next(row for row in AUDIT_ROWS if row["Unit_ID"] == "s1.2:q49")
    assert "no exact CDIAL head" in candidate["Etymon_Candidates"]
    assert MANIFEST_DATA["rows_linked_to_cdial"] == 4
    assert {field(row, "Parameter_ID") for row in FORM_ROWS if field(row, "Parameter_ID")} \
        == {"11346", "13776", "12253", "10073"}


def test_a_loan_tag_is_never_attached_to_an_inherited_indo_aryan_etymon():
    row = by_key("s1.2:q51")
    assert field(row, "Etymology").startswith("< Indo-Aryan varta")
    assert "loanword" not in field(row, "Tags").split()


# ----------------------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------------------

def test_the_sound_profile_covers_every_installed_symbol_without_replacement():
    tokenizer = Tokenizer(str(PROFILE))
    for row in FORM_ROWS:
        source = unicodedata.normalize("NFC", field(row, "Form"))
        converted = tokenizer(source, column="IPA").replace(" ", "").replace("#", " ")
        assert "�" not in converted, field(row, "Entry_Key")


def test_the_profile_follows_the_cdial_letter_conventions():
    tokenizer = Tokenizer(str(PROFILE))

    def convert(text):
        return unicodedata.normalize(
            "NFC", tokenizer(unicodedata.normalize("NFC", text), column="IPA")
            .replace(" ", "").replace("#", " "))

    assert convert("ōšt") == "ōśt"
    assert convert("čēzə́k") == "cēzə́k"
    assert convert("ḥaǧǧiyyāt") == "ḣaǧǧiyyāt"
    assert convert("ġabre") == "ɣabre"
    assert convert("laʋrã́") == "lavrã́"
    assert convert("pɑ̄sṓm") == "pɑ̄sṓm"
    # The turned e the article uses for seven forms is normalised onto the schwa.
    assert convert("dakǝrdã") == "dakərdã"
    # Morpheme hyphens, clitic boundaries and the word space all survive.
    assert convert("kəry-ō-mān=e") == "kəry-ō-mān=e"
    assert convert("gā kar-") == "gā kar-"


def test_installed_output_is_nfc_and_free_of_empty_cells_that_matter():
    for row in FORM_ROWS:
        assert unicodedata.normalize("NFC", field(row, "Form")) == field(row, "Form")
        assert field(row, "Form") and field(row, "Gloss")
        assert not field(row, "Native")


# ----------------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------------

def test_the_checked_in_extraction_rebuilds_the_installed_rows_exactly():
    units = herin.read_extract(EXTRACT)
    records, audit = herin.build_records(units)
    herin.validate(records, audit)
    rebuilt = [[record[name] for name in herin.FORM_FIELDS] for record in records]
    assert rebuilt == FORM_ROWS
    assert herin.summarize(records, audit) == MANIFEST_DATA


def test_the_manifest_pins_both_snapshots():
    assert MANIFEST_DATA["html_sha256"] == herin.HTML_SHA256
    assert MANIFEST_DATA["pdf_sha256"] == herin.PDF_SHA256
    assert MANIFEST_DATA["article_url"] == herin.ARTICLE_URL
    assert MANIFEST_DATA["printed_pages"] == [1, 52]
