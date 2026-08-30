"""Regression tests for the Liljegren (2013) Kalkoti article ingest."""

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


kalkoti = load_source("liljegren_kalkoti_2013.py", "liljegren_kalkoti_2013")
RAW = kalkoti.records()
FORMS, AUDIT = kalkoti.build()
AUDIT_BY_UNIT = {row["Unit_ID"]: row for row in AUDIT}
BY_KEY = {row["Entry_Key"]: row for row in FORMS}


def installed_rows():
    with kalkoti.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def kalkoti_rows():
    """The Kalkoti rows of the installed file, without the editorial anchors."""
    return [row for row in installed_rows() if row[0] == "Kalk"]


def anchor_rows():
    return [row for row in installed_rows() if row[0] != "Kalk"]


def resolved(unit):
    """The installed row a raw unit ended up in, following the fold if it merged."""
    entry = AUDIT_BY_UNIT[unit]
    return BY_KEY[entry["Merged_Into"] or entry["Emitted_Key"]]


# --------------------------------------------------------------------------
# Coverage and counts
# --------------------------------------------------------------------------

def test_every_region_of_the_article_is_covered_with_the_expected_record_counts():
    assert Counter(record["region"] for record in RAW) == {
        "interlinear": 97, "t14": 33, "t12": 32, "prose": 34, "t2": 26, "t1": 24,
        "t16": 20, "t4": 12, "t5": 12, "t19": 11, "t11": 10, "t13": 8, "t20": 8,
        "segment": 7, "t8": 6, "t17": 6, "fn15": 5, "t3": 5, "t9": 4, "t18": 4, "t6": 1,
    }
    assert len(RAW) == 365


def test_each_printed_table_contributes_its_full_printed_row_count():
    # Table 1 prints 24 kinship terms across a page break, Table 5 twelve
    # numerals, Table 12 thirty-two monosyllables in five melody columns.
    assert len([r for r in RAW if r["region"] == "t1"]) == 24
    assert len([r for r in RAW if r["region"] == "t5"]) == 12
    assert Counter(
        kalkoti._context(r)["melody"] for r in RAW if r["region"] == "t12"
    ) == {"1": 8, "2": 8, "3": 5, "4": 8, "5": 3}
    # Table 14 has 32 cells, one of which (3PL near nominative) is blank, and
    # two of which print two alternates.
    assert len([r for r in RAW if r["region"] == "t14"]) == 33


def test_statuses_account_for_every_raw_record_and_name_their_reason():
    assert len(AUDIT) == len(RAW)
    assert Counter(row["Status"] for row in AUDIT) == {"installed": 358, "skipped": 7}
    skipped = [row for row in AUDIT if row["Status"] == "skipped"]
    # Example (9) is the only non-Kalkoti example in the article.
    assert {row["Unit_ID"].split(":")[0] for row in skipped} == {"ex9"}
    assert all("Palula" in row["Reason"] for row in skipped)
    assert all(row["Reason"] for row in AUDIT)


def test_installed_file_matches_the_build_and_has_unique_stable_keys():
    rows = kalkoti_rows()
    assert len(rows) == len(FORMS) == 276
    assert all(len(row) == 15 for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(row[10].startswith("kalkoti:") for row in rows)
    assert all(row[2] for row in rows) and all(row[3] for row in rows)
    # Four comparanda in other languages are carried through verbatim from the
    # 2022 snapshot so that etyma e34-e36 keep a second member each.
    anchors = anchor_rows()
    assert len(anchors) == 4
    assert {row[0] for row in anchors} == {"Phal"}
    assert {row[1] for row in anchors} == {"e34", "e35", "e36", "3438"}
    assert all(len(row) == 8 for row in anchors)


# --------------------------------------------------------------------------
# Extraction edge cases found while auditing the article
# --------------------------------------------------------------------------

def test_tone_accents_stay_on_the_vowel_the_article_prints_them_on():
    # Acrobat emits the combining accents of Table 13 on their own baseline,
    # above and to the right of the letter they mark, so reading the page by
    # position moves the tone onto the following vowel.
    raw = {row["Unit_ID"]: row["Raw_Form"] for row in AUDIT}
    assert raw["t13:3"] == "ḍä̀är"
    assert raw["t13:4"] == "šaàk"
    assert raw["t13:7"] == "ɡòór"
    assert raw["fn15:5"] == "bä̀kaál"


def test_a_possessive_apostrophe_does_not_truncate_a_gloss():
    # U+2019 is both the closing quotation mark and the apostrophe of an
    # English possessive, so 'father's sister' must not become 'father'.
    assert resolved("t12:3:2")["Gloss"] == "father's sister"
    assert resolved("t1:20")["Gloss"] == "son's daughter"


def test_the_wrapped_free_translation_of_the_long_example_is_kept_whole():
    context = kalkoti._context(
        next(r for r in RAW if r["unit"] == "ex18:1:1")
    )
    assert context["translation"].startswith("Once, we were some friends")
    assert context["translation"].endswith("came up to us.")


def test_interlinear_small_capitals_are_folded_into_the_gloss_they_belong_to():
    row = resolved("ex6:1:5")
    assert row["Form"] == "nikhuuns"
    assert row["Gloss"] == "come out"
    assert {"ipfv", "m", "sg", "pret"} <= set(row["Tags"].split())


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def test_source_ipa_is_rewritten_into_the_articles_own_broad_transcription():
    # Tables 7 and 10 state the correspondence, and the article's own broad
    # spellings elsewhere confirm each of these.
    assert kalkoti.to_broad("/eːʂ/") == "eeṣ"      # Table 5 prints eeṣ
    assert kalkoti.to_broad("/treːr/") == "treer"  # Table 1 prints treer
    assert kalkoti.to_broad("/drɑːm/") == "draam"  # example (8) prints draam
    assert kalkoti.to_broad("/pitri/") == "pitri"  # Table 1 prints pitri
    # A parenthesised (ʔ) is the prosodic glottal of melodies 3 and 4; every
    # other parenthesised segment is a real consonant some speakers drop.
    assert kalkoti.to_broad("/ɡoː(ʔ)r/") == "ɡoor"
    assert kalkoti.to_broad("/nɑːŋ(ɡ)/") == "naaŋɡ"
    # Two citations set length with an ASCII colon rather than the length mark.
    assert kalkoti.to_broad("/ɑlɑ:l/") == "alaal"


def test_the_sound_profile_covers_every_installed_form_without_loss():
    tokenizer = Tokenizer(str(ROOT / "conversion/kalkoti.txt"))

    def convert(form):
        out = tokenizer(form.strip("-1234⁴5⁵67⁷,;."), column="IPA")
        return unicodedata.normalize("NFC", out.replace(" ", "").replace("#", " "))

    converted = {row[2]: convert(row[2]) for row in kalkoti_rows()}
    assert not [f for f, c in converted.items() if "�" in c]
    # The profile reproduces the transcription the 2022 snapshot was typed in.
    assert converted["bään"] == "bǣn"
    assert converted["däär"] == "dǣr"
    assert converted["ǰämäl"] == "jæmæl"
    assert converted["pheep"] == "pʰēp"
    assert converted["ic̣ii"] == "iʦ̣ī"
    assert converted["suwaa"] == "suvā"
    assert converted["naaŋɡ"] == "nāŋg"


def test_tone_is_carried_into_the_display_form():
    star = resolved("t13:5")
    assert star["Form"] == "taár"
    assert "a high tone" in star["Notes"]
    # Table 9 prints the same word without tone, so both citations are one row
    # and the row keeps the marked spelling.
    assert resolved("t9:2")["Entry_Key"] == star["Entry_Key"]
    # Every tone-marked citation the article prints survives.
    toned = {row["Form"] for row in FORMS if row["Form"] != kalkoti.toneless(row["Form"])}
    assert toned == {
        "taár", "ḍä̀är", "šaàk", "baál", "ɡòór", "čhèél",
        "ḍä̀rin", "ic̣ì", "lumaáṭ", "bä̀kaál",
    }


def test_the_profile_writes_kalkoti_tone_the_way_palula_accent_is_written():
    tokenizer = Tokenizer(str(ROOT / "conversion/kalkoti.txt"))

    def convert(form):
        out = tokenizer(form, column="IPA")
        return unicodedata.normalize("NFC", out.replace(" ", "").replace("#", " "))

    # An acute on the first mora is a falling contour and on the second a rising
    # one, exactly as conversion/liljegren.txt writes Palula accent.
    assert convert("šáak") == "śā̂k"
    assert convert("taár") == "tā̌r"
    # Hultman (2023: 17) states that low tone is a property of the syllable with
    # no /V̀V/ versus /VV̀/ contrast, so both writings give one grave.
    assert convert("ḍä̀är") == "ḍǣ̀r"
    assert convert("šaàk") == "śā̀k"
    # Grave plus acute is the low-rising contour.
    assert convert("ɡòór") == "gō̌̀r"
    # Tone also appears on short vowels, and toneless words stay unmarked.
    assert convert("bä̀kaál") == "bæ̀kā̌l"
    assert convert("taar") == "tār"


# --------------------------------------------------------------------------
# Language, dialect, locators and references
# --------------------------------------------------------------------------

def test_every_row_is_canonical_kalkoti_under_the_registered_dialect():
    assert {row[0] for row in kalkoti_rows()} == {"Kalk"}
    assert all(kalkoti.DIALECT_TAG in row[14].split() for row in kalkoti_rows())
    dialects = {row["Tag"] for row in csv.DictReader((ROOT / "cldf/dialects.csv").open())}
    assert kalkoti.DIALECT_TAG in dialects
    languages = {row["ID"] for row in csv.DictReader((ROOT / "cldf/languages.csv").open())}
    assert kalkoti.LANGUAGE_ID in languages


def test_locators_use_the_printed_page_and_name_the_table_or_example():
    assert AUDIT_BY_UNIT["t1:1"]["Locator"] == "p. 131, table 1"
    assert AUDIT_BY_UNIT["t13:1"]["Locator"] == "p. 144, table 13"
    assert AUDIT_BY_UNIT["ex1:1:1"]["Locator"] == "p. 145, example 1"
    assert AUDIT_BY_UNIT["prose:halal"]["Locator"] == "p. 136, n. 5"
    assert AUDIT_BY_UNIT["fn15:2"]["Locator"] == "p. 144, n. 15"
    # PDF page 2 carries printed page 129 and the article ends on 160.
    pages = {int(row["Printed_Page"]) for row in AUDIT}
    assert min(pages) >= 129 and max(pages) <= 160


def test_every_row_cites_the_article_and_only_the_article():
    keys = {
        part.split("[")[0]
        for row in kalkoti_rows()
        for part in row[7].split(";")
    }
    assert keys == {"kalkoti"}
    bibliography = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert "@article{kalkoti," in bibliography
    assert "10.1349/PS1.1537-0852.A.423" in bibliography


# --------------------------------------------------------------------------
# Grammatical parsing
# --------------------------------------------------------------------------

def test_the_two_verb_classes_the_article_names_are_canonical_tags():
    import tags as tag_module

    for name in ("Kalkoti-verb-class-L", "Kalkoti-verb-class-T",
                 "Kalkoti-verb-class-suppletive"):
        assert name in tag_module.GRAMMATICAL_TAGS
    frontend = (ROOT.parent / "jambu-static/src/lib/tags.ts").read_text(encoding="utf-8")
    for name in ("Kalkoti-verb-class-L", "Kalkoti-verb-class-T",
                 "Kalkoti-verb-class-suppletive"):
        assert f"'{name}'" in frontend
    run = resolved("t17:l-verb:non-perfective")
    assert run["Form"] == "trap" and run["Gloss"] == "to run"
    assert {"verb", "stem", "Kalkoti-verb-class-L"} <= set(run["Tags"].split())


def test_the_pronoun_paradigm_carries_person_case_and_deixis():
    near = resolved("t14:3sg:near:obl1")
    assert near["Form"] == "räs"
    assert near["Gloss"] == "he, she, it (proximate)"
    assert {"pron", "personal", "3sg", "obl", "acc", "prox"} <= set(near["Tags"].split())
    far = resolved("t14:3sg:far:obl2")
    assert {"dist", "erg"} <= set(far["Tags"].split())


def test_paired_cells_split_into_their_two_aspect_forms():
    assert resolved("t2:9:ipfv")["Form"] == "nikhuun"
    assert resolved("t2:9:ipfv")["Gloss"] == "comes out"
    assert resolved("t2:9:pfv")["Form"] == "nikhät"
    assert resolved("t2:9:pfv")["Gloss"] == "came out"


def test_numerals_are_glossed_by_name_not_by_digit():
    assert resolved("t5:8")["Form"] == "eeṣ"
    assert resolved("t5:8")["Gloss"] == "eight"
    assert "num" in resolved("t5:8")["Tags"].split()
    assert resolved("t6:43")["Gloss"] == "forty-three"


def test_a_clitic_or_zero_morph_never_enters_the_headword():
    row = resolved("ex1:1:4")
    assert row["Form"] == "buun"
    assert "=" not in row["Form"] and "-" not in row["Form"]
    assert "interr" in row["Tags"].split()
    assert resolved("ex13:1:3")["Form"] == "yä"


# --------------------------------------------------------------------------
# Etymology
# --------------------------------------------------------------------------

def test_the_turner_numbers_table_13_prints_are_linked_and_explained():
    lake = resolved("t13:1")
    assert lake["Parameter_ID"] == "13254"
    assert "Turner 1966: 13254" in lake["Etymology"]
    assert resolved("t13:2")["Parameter_ID"] == "2830"


def test_every_editorial_etymology_of_the_2022_snapshot_survives():
    crosswalk = list(csv.DictReader(kalkoti.CURATED.open(encoding="utf-8")))
    assert len(crosswalk) == 135
    assert len({row["Entry_Key"] for row in crosswalk}) == 135
    for row in crosswalk:
        assert row["Entry_Key"] in BY_KEY, row
        assert BY_KEY[row["Entry_Key"]]["Parameter_ID"] == row["Parameter_ID"]
    # The four non-Kalkoti anchors of the 2022 snapshot keep their etymologies.
    assert {row[1] for row in anchor_rows()} == {"e34", "e35", "e36", "3438"}


def test_the_2022_snapshot_errors_the_article_disproves_are_corrected():
    # Table 12 lists /im/ under 'snow', not 'belly', and Table 11 contrasts
    # short /dʊr/ 'dust' with long /duːr/ 'far'.
    assert resolved("t12:2:8")["Form"] == "im"
    assert resolved("t12:2:8")["Gloss"] == "snow"
    dust = resolved("t11:2:u")
    assert dust["Form"] == "dur" and dust["Gloss"] == "dust"
    assert resolved("t11:2:uu")["Form"] == "duur" and resolved("t11:2:uu")["Gloss"] == "far"


def test_no_link_is_invented_for_a_form_the_article_does_not_etymologise():
    unlinked = [row for row in FORMS if not row["Parameter_ID"]]
    assert len(unlinked) == len(FORMS) - 135


# --------------------------------------------------------------------------
# Variants and repeated citations
# --------------------------------------------------------------------------

def test_printed_alternations_become_variants_of_their_first_form():
    base, variant = resolved("prose:vomit"), resolved("prose:vomit:a2")
    assert base["Form"] == "čhäḍil" and variant["Form"] == "čhäṛil"
    assert variant["Variant_Of_Key"] == base["Entry_Key"]
    assert all(
        row["Variant_Of_Key"] in BY_KEY
        for row in FORMS if row["Variant_Of_Key"]
    )


def test_repeated_citations_of_one_lexeme_merge_and_keep_every_citation():
    # pitri is printed in the kinship table, the cluster table and footnote 15.
    row = resolved("t1:13")
    assert row["Form"] == "pitri"
    assert len(row["Source"].split(";")) == 3
    assert row["Phonemic"] == "pitri"
    assert resolved("t8:6")["Entry_Key"] == row["Entry_Key"]
    assert resolved("fn15:1")["Entry_Key"] == row["Entry_Key"]


def test_homonyms_and_distinct_senses_are_not_folded_together():
    blood, night = resolved("t12:1:5"), resolved("t12:2:5")
    assert blood["Form"] == night["Form"] == "raat"
    assert blood["Entry_Key"] != night["Entry_Key"]
    assert {blood["Gloss"], night["Gloss"]} == {"blood", "night"}
    cow, grass = resolved("t12:1:4"), resolved("t12:2:4")
    assert cow["Form"] == grass["Form"] == "ɡaa"
    assert cow["Entry_Key"] != grass["Entry_Key"]


def test_the_manifest_records_scope_provenance_and_counts():
    manifest = json.loads(kalkoti.MANIFEST_OUTPUT.read_text(encoding="utf-8"))
    assert manifest["source_id"] == "kalkoti"
    assert manifest["pdf_sha256"] == kalkoti.PDF_SHA256
    assert manifest["pdf_redistributed"] is False
    assert manifest["outputs"]["form_count"] == len(FORMS)
    assert manifest["outputs"]["audit_count"] == len(AUDIT)
    assert "comparanda" in manifest["scope"]["excluded"] or \
        "Gawri" in manifest["scope"]["excluded"]
