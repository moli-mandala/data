"""Regression tests for the Hultman (2023) Kalkoti grammar-sketch ingest."""

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


hultman = load_source("hultman_kalkoti_2023.py", "hultman_kalkoti_2023")
RAW = hultman.records()
FORMS, AUDIT = hultman.build()
AUDIT_BY_UNIT = {row["Unit_ID"]: row for row in AUDIT}
BY_KEY = {row["Entry_Key"]: row for row in FORMS}


def installed_rows():
    with hultman.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def resolved(unit):
    entry = AUDIT_BY_UNIT[unit]
    return BY_KEY[entry["Merged_Into"] or entry["Emitted_Key"]]


# --------------------------------------------------------------------------
# Coverage and counts
# --------------------------------------------------------------------------

def test_every_region_of_the_thesis_is_covered_with_the_expected_record_counts():
    assert Counter(record["region"] for record in RAW) == {
        "interlinear": 552, "t16": 50, "prose": 48, "t25": 38, "t14": 32, "t21": 22,
        "t19": 21, "t20": 18, "t24": 17, "t4": 14, "t9": 13, "t3": 12, "t11": 12,
        "t23": 12, "t17": 11, "t10": 10, "t12": 8, "t6": 6, "t13": 6, "t15": 6,
        "t5": 5, "t26": 4, "t7": 4, "t28": 4,
    }
    assert len(RAW) == 925


def test_each_printed_table_contributes_its_full_printed_row_count():
    # Table 16 prints sixty numerals in five blocks of ten; Table 9 thirteen
    # words in three tone classes; Table 14 a four-by-eight pronoun paradigm.
    assert len({r["unit"] for r in RAW if r["region"] == "t16"}) == 50
    assert len([r for r in RAW if r["region"] == "t9"]) == 13
    assert len([r for r in RAW if r["region"] == "t14"]) == 32


def test_every_numbered_example_of_the_thesis_is_read():
    examples = {hultman._context(r)["example"] for r in RAW if r["region"] == "interlinear"}
    assert len(examples) == 129
    numbers = {int("".join(c for c in e if c.isdigit())) for e in examples}
    # Examples 1-4 are phonological illustrations rather than interlinear text
    # and are carried by the checked-in prose table instead.
    assert numbers == set(range(5, 68))


def test_statuses_account_for_every_raw_record_and_name_their_reason():
    assert len(AUDIT) == len(RAW)
    assert Counter(row["Status"] for row in AUDIT) == {"installed": 907, "skipped": 18}
    assert all(row["Reason"] for row in AUDIT)
    assert {row["Reason"] for row in AUDIT if row["Status"] == "skipped"} == {
        "the token is punctuation or an elision mark",
        "the thesis marks this word as unglossed",
    }


def test_installed_file_matches_the_build_and_has_unique_stable_keys():
    rows = installed_rows()
    assert len(rows) == len(FORMS) == 475
    assert all(len(row) == 15 for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(row[10].startswith("hultman2023kalkoti:") for row in rows)
    assert all(row[0] == "Kalk" for row in rows)
    assert all(row[2] and row[3] for row in rows)


# --------------------------------------------------------------------------
# Extraction edge cases
# --------------------------------------------------------------------------

def test_words_are_recovered_from_gaps_because_the_pdf_has_no_space_glyphs():
    # The thesis's XeTeX text layer contains no space characters at all, so a
    # whitespace split would return one word per line.
    assert hultman.WORD_GAP > 0
    assert resolved("t20:6:pfv")["Gloss"] == "put on (clothes)"
    assert resolved("t17:5")["Gloss"] == "leopard"


def test_tone_accents_stay_on_the_vowel_the_thesis_prints_them_on():
    raw = {row["Unit_ID"]: row["Raw_Form"] for row in AUDIT}
    assert raw["t10:1:kalkoti"] == "bùun"
    assert raw["t10:3:kalkoti"] == "ḍä̀är"
    assert raw["t10:9:kalkoti"] == "dùúr"
    assert raw["t9:9:kalkoti"] == "taár"


def test_a_possessive_apostrophe_does_not_truncate_a_gloss():
    assert resolved("t9:10:kalkoti")["Gloss"] == "father's mother"


def test_the_split_cells_of_the_irregular_perfective_table_are_read_whole():
    # Table 21 sets a gendered perfective pair above and below its gloss.
    come_m, come_f = resolved("t21:1:pfv:1"), resolved("t21:1:pfv:2")
    assert (come_m["Form"], come_f["Form"]) == ("yaál", "yeél")
    assert come_m["Gloss"] == come_f["Gloss"] == "come"
    assert "m" in come_m["Tags"].split() and "f" in come_f["Tags"].split()
    assert resolved("t21:1:ipfv")["Form"] == "yùun"


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def test_source_ipa_is_rewritten_into_the_thesiss_own_orthography():
    # Tables 2 and 8 print each phoneme beside its orthographic spelling, and
    # the thesis's own spellings elsewhere confirm each of these.
    assert hultman.to_orthography("/měːɕ/") == "meéš"      # Table 9 prints meéš
    assert hultman.to_orthography("/mɪ́ɕɑːl/") == "míšaal"  # Table 12 prints míšaal
    assert hultman.to_orthography("/tʰæ̌ːl/") == "thää́l"   # Table 21 prints thää́l
    assert hultman.to_orthography("[ʈɒŋgʊɾ]") == "ṭangur"  # Table 7 prints ṭangur
    assert hultman.to_orthography("[nɑːŋ]") == "naang"     # Table 7 prints naang
    # A long nasal vowel is an allophone of /Vːn/ and is written with the nasal.
    assert hultman.to_orthography("[pæ̃ːs]") == "pääns"
    assert hultman.to_orthography("[bũːs]") == "buuns"
    # Glottalization and devoicing are phonetic detail, not segments.
    assert hultman.to_orthography("[eːˀɾ̥]") == "eer"


def test_the_sound_profile_covers_every_installed_form_without_loss():
    tokenizer = Tokenizer(str(ROOT / "conversion/kalkoti.txt"))

    def convert(form):
        out = tokenizer(form.strip("-1234⁴5⁵67⁷,;."), column="IPA")
        return unicodedata.normalize("NFC", out.replace(" ", "").replace("#", " "))

    converted = {row[2]: convert(row[2]) for row in installed_rows()}
    assert not [f for f, c in converted.items() if "�" in c]
    # The shared Kalkoti profile writes tone the way Palula accent is written.
    assert converted["meéš"] == "mē̌ś"
    assert converted["míšaal"] == "míśāl"
    assert converted["bùun"] == "bū̀n"
    assert converted["gòór"] == "gō̌̀r"
    assert converted["ḍä̀är"] == "ḍǣ̀r"


def test_tone_keeps_a_minimal_pair_apart():
    # raat 'blood' and ràat 'night' differ only in tone.
    blood = [row for row in FORMS if row["Gloss"] == "blood" and row["Form"].startswith("raat")]
    night = [row for row in FORMS if row["Gloss"] == "night"]
    assert blood and night
    assert blood[0]["Form"] != night[0]["Form"]


# --------------------------------------------------------------------------
# Language, dialect, locators and references
# --------------------------------------------------------------------------

def test_every_row_is_canonical_kalkoti_under_the_registered_dialect():
    assert {row[0] for row in installed_rows()} == {"Kalk"}
    assert all(hultman.DIALECT_TAG in row[14].split() for row in installed_rows())
    dialects = {row["Tag"] for row in csv.DictReader((ROOT / "cldf/dialects.csv").open())}
    assert hultman.DIALECT_TAG in dialects


def test_locators_use_the_printed_page_and_cite_the_dataset_item():
    assert AUDIT_BY_UNIT["t3:unaspirated:labial"]["Locator"] == "p. 9, table 3"
    assert AUDIT_BY_UNIT["t16:1"]["Locator"] == "p. 28, table 16"
    # An interlinear citation names the dataset item the thesis recorded it in,
    # so a reader can find the speaker and the session behind every form.
    example = AUDIT_BY_UNIT["ex5a:1"]["Locator"]
    assert example.startswith("p. 22, example 5a, ")
    assert example.endswith("U23a-28")
    pages = {int(row["Printed_Page"]) for row in AUDIT}
    assert min(pages) >= 1 and max(pages) <= 61


def test_every_row_cites_the_thesis_and_the_bibliography_has_it():
    keys = {part.split("[")[0] for row in installed_rows() for part in row[7].split(";")}
    assert keys == {"hultman2023kalkoti"}
    bibliography = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert "@mastersthesis{hultman2023kalkoti," in bibliography


# --------------------------------------------------------------------------
# Grammatical parsing
# --------------------------------------------------------------------------

def test_glosses_are_split_with_the_thesiss_own_abbreviation_list():
    row = resolved("ex6a:2")
    assert row["Form"] == "mälgirum"
    assert row["Gloss"] == "friend"
    assert {"obl", "pl"} <= set(row["Tags"].split())
    # friend-obl.pl=erg: the ergative is marked by the clitic =ä, so it is not a
    # property of the host word the headword records.
    assert "erg" not in row["Tags"].split()
    # Every label the thesis uses is recognised; nothing is left unmapped.
    assert not [r for r in AUDIT if r["Reason"].startswith("unrecognised")]


def test_a_word_glossed_only_by_category_still_gets_a_lexical_meaning():
    copula = [row for row in FORMS if row["Form"] == "in" and "copula" in row["Tags"]]
    assert copula and copula[0]["Gloss"] == "is, are"
    assert "by category alone" in copula[0]["Notes"]


def test_the_pronoun_paradigm_carries_person_case_and_deixis():
    near = resolved("t14:3sg.near:accusative")
    assert near["Form"] == "räs"
    assert near["Gloss"] == "he, she, it (proximate)"
    assert {"pron", "personal", "3sg", "prox", "acc"} <= set(near["Tags"].split())


def test_numerals_are_glossed_by_name_and_compounds_are_marked():
    assert resolved("t16:8")["Form"] == "iṣ" and resolved("t16:8")["Gloss"] == "eight"
    assert resolved("t16:120")["Gloss"] == "one hundred and twenty"
    assert "compound" in resolved("t16:21")["Tags"].split()
    assert "num" in resolved("t16:1")["Tags"].split()


def test_gender_from_table_17_is_tagged_and_its_evidence_kept():
    water = resolved("t17:3")
    assert water["Form"] == "wää" and water["Gloss"] == "water"
    assert "m" in water["Tags"].split()
    assert "Palula wíi" in water["Etymology"]


# --------------------------------------------------------------------------
# Loanwords and comparanda
# --------------------------------------------------------------------------

def test_loanwords_are_tagged_and_their_donor_kept_as_prose_not_as_an_edge():
    market = resolved("t4:11")
    assert market["Form"] == "bázaar" and market["Gloss"] == "market"
    assert "loanword" in market["Tags"].split()
    assert "Pashto" in market["Etymology"]
    assert market["Parameter_ID"] == ""


def test_no_row_is_linked_because_the_thesis_makes_no_etymological_claim():
    assert all(not row["Parameter_ID"] for row in FORMS)
    assert all(not row["Cognateset"] for row in FORMS)


def test_printed_alternations_become_variants_of_their_first_form():
    base, variant = resolved("prose:path"), resolved("prose:path:a2")
    assert base["Form"] == "paand" and variant["Form"] == "paan"
    assert variant["Variant_Of_Key"] == base["Entry_Key"]
    assert all(row["Variant_Of_Key"] in BY_KEY for row in FORMS if row["Variant_Of_Key"])


def test_repeated_citations_of_one_lexeme_merge_and_keep_every_citation():
    # driṣ 'see' is printed in four tables and cited in many examples.
    row = [r for r in FORMS if r["Form"] == "driṣ" and r["Gloss"] == "see"][0]
    assert len(row["Source"].split(";")) >= 4
    assert "pfv" in row["Tags"].split()


def test_a_clitic_cited_on_its_own_keeps_its_own_category():
    postp = resolved("prose:to")
    assert postp["Form"] == "=thä"
    assert postp["Gloss"] == "to"
    assert {"postp", "dat"} <= set(postp["Tags"].split())


def test_the_thesiss_own_uncertainty_marks_become_tags_not_glosses():
    # Hultman appends a question mark to a gloss he is unsure of, and writes a
    # bare question mark for a word he could not gloss at all.
    assert not [row for row in FORMS if row["Gloss"].endswith("?")]
    rain = [row for row in FORMS if row["Form"] == "agä"]
    assert rain and rain[0]["Gloss"] == "rain"
    assert "uncertain" in rain[0]["Tags"].split()
    assert len([r for r in AUDIT if r["Reason"] == "the thesis marks this word as unglossed"]) == 4


def test_personal_and_place_names_are_tagged():
    names = {row["Gloss"] for row in FORMS if "proper-noun" in row["Tags"].split()}
    assert {"Haider", "Zaman", "Shelkin"} <= names


def test_the_manifest_records_scope_provenance_and_counts():
    manifest = json.loads(hultman.MANIFEST_OUTPUT.read_text(encoding="utf-8"))
    assert manifest["source_id"] == "hultman2023kalkoti"
    assert manifest["pdf_sha256"] == hultman.PDF_SHA256
    assert manifest["pdf_redistributed"] is False
    assert manifest["outputs"]["form_count"] == len(FORMS)
    assert "Table 18" in manifest["scope"]["excluded"]
