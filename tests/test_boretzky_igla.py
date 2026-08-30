"""Regression tests for the Boretzky & Igla (1994) Romani etymology-appendix ingest."""

import csv
import importlib.util
import json
import re
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


boretzky = load_source("boretzky_igla_1994.py", "boretzky_igla_1994")
RAW = boretzky.read_extract()
FORMS, AUDIT, UNKNOWN_GRAMMAR = boretzky.build(RAW)
BY_KEY = {row["Entry_Key"]: row for row in FORMS}
BY_FORM = {}
for _row in FORMS:
    BY_FORM.setdefault(_row["Form"], []).append(_row)


def installed_rows():
    with boretzky.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def bib_keys():
    text = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\{([^,]+),", text, re.M))


def registry(name, column):
    with (ROOT / "cldf" / name).open(encoding="utf-8", newline="") as handle:
        return {row[column] for row in csv.DictReader(handle)}


# --------------------------------------------------------------------------
# Coverage and counts
# --------------------------------------------------------------------------

def test_the_whole_printed_appendix_is_transcribed_with_the_expected_counts():
    assert len(RAW) == 1093
    assert Counter(record["List"] for record in RAW) == {
        "ind": 704, "iran": 83, "arm": 51, "gr": 255
    }
    pages = sorted({int(record["Printed_Page"]) for record in RAW})
    assert pages == list(range(boretzky.FIRST_PRINTED_PAGE, boretzky.LAST_PRINTED_PAGE + 1))
    assert {record["Column"] for record in RAW} == {"L", "R"}


def test_item_numbering_is_contiguous_within_each_printed_page():
    by_page = {}
    for record in RAW:
        by_page.setdefault(record["Printed_Page"], []).append(int(record["Item"]))
    for page, items in by_page.items():
        assert items == sorted(items), page
        assert items == list(range(1, len(items) + 1)), page


def test_record_to_audit_to_installed_counts_reconcile():
    statuses = Counter(row["Status"] for row in AUDIT)
    assert statuses == {"ingested": 1140, "crossref": 85, "merged": 34}
    assert len(FORMS) == statuses["ingested"]
    # 1093 printed records expand to 1259 audit rows because a printed entry may bundle
    # several headwords; 85 of them are ``s. X`` pointers and 34 are cross-list repeats.
    assert statuses["crossref"] + statuses["ingested"] + statuses["merged"] == len(AUDIT)
    assert len(installed_rows()) == len(FORMS)


def test_manifest_matches_the_current_extraction():
    manifest = json.loads(boretzky.MANIFEST_OUTPUT.read_text(encoding="utf-8"))
    assert manifest["raw_records"] == len(RAW)
    assert manifest["installed_rows"] == len(FORMS)
    assert manifest["pdf_sha256"] == boretzky.PDF_SHA256
    assert manifest["pdf_pages"] == boretzky.PDF_PAGES
    assert manifest["printed_pages"] == [
        boretzky.FIRST_PRINTED_PAGE, boretzky.LAST_PRINTED_PAGE
    ]


def test_no_grammatical_label_is_left_unmapped():
    assert UNKNOWN_GRAMMAR == Counter()


# --------------------------------------------------------------------------
# Identifiers, locators and references
# --------------------------------------------------------------------------

def test_entry_keys_are_unique_and_built_from_printed_identity():
    keys = [row["Entry_Key"] for row in FORMS]
    assert len(set(keys)) == len(keys)
    assert all(key.startswith("boretzky1994romani:") for key in keys)
    assert BY_FORM["phral"][0]["Entry_Key"] == "boretzky1994romani:ind:324:R:26"
    assert BY_FORM["kirmo"][0]["Entry_Key"] == "boretzky1994romani:ind:319:L:12:variant:2"


def test_every_row_carries_a_printed_page_and_column_locator():
    pattern = re.compile(r"^boretzky1994romani\[p\. 3\d\d, col\. [12], entry \d+\]$")
    for row in FORMS:
        assert pattern.match(row["Source"].split(";")[0]), row


def test_every_cited_bibliography_key_resolves():
    keys = bib_keys()
    assert "boretzky1994romani" in keys
    used = {
        part.split("[")[0]
        for row in FORMS
        for part in row["Source"].split(";")
        if part
    }
    assert used <= keys
    # Every printed source abbreviation the appendix uses has its own bibliography record.
    assert set(boretzky.SOURCE_MARKS.values()) <= keys


def test_printed_source_abbreviations_become_secondary_citations_not_dialects():
    row = BY_FORM["ravnos"][0]
    assert row["Source"].split(";")[1:] == ["sampson1926welsh"]
    assert "dialect:" not in row["Tags"]


# --------------------------------------------------------------------------
# Languages and dialects
# --------------------------------------------------------------------------

def test_every_form_uses_a_registered_base_language():
    language_ids = registry("languages.csv", "ID")
    used = {row["Language_ID"] for row in FORMS}
    assert used <= language_ids
    assert used == {"eur", "RomSint", "RomBalk", "RomVlax"}


def test_every_dialect_tag_is_registered_under_the_right_parent():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        dialects = {row["Tag"]: row for row in csv.DictReader(handle)}
    for row in FORMS:
        for tag in row["Tags"].split():
            if tag.startswith("dialect:"):
                assert tag in dialects, tag
                assert dialects[tag]["Language_ID"] == row["Language_ID"], tag


def test_printed_dialect_labels_select_the_canonical_base_language():
    assert BY_FORM["korako"][0]["Language_ID"] == "RomSint"
    assert BY_FORM["korangos"][0]["Language_ID"] == "RomVlax"
    assert "dialect:RomVlax:ursari:Ursari" in BY_FORM["korangos"][0]["Tags"]
    assert BY_FORM["muskári"][0]["Language_ID"] == "RomBalk"
    assert "dialect:RomBalk:bugurdzi:Bugurd%C5%BEi" in BY_FORM["muskári"][0]["Tags"]


def test_new_dialects_are_georeferenced_as_explicit_approximations():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["ID"]: row for row in csv.DictReader(handle)}
    for identifier in ("arli", "bugurdzi", "ursari"):
        row = rows[identifier]
        assert row["Quality"] == "C"
        assert "approximate" in row["Location"]
        assert row["Latitude"] and row["Longitude"]


# --------------------------------------------------------------------------
# Field separation
# --------------------------------------------------------------------------

def test_rows_have_the_manual_import_shape_and_no_stray_columns():
    rows = installed_rows()
    assert {len(row) for row in rows} == {15}
    assert all(row[2].strip() for row in rows)
    assert all(row[4] == "" and row[5] == "" and row[6] == "" for row in rows)


def test_glosses_are_lexical_and_carry_no_source_or_grammar_labels():
    for row in FORMS:
        assert row["Gloss"].strip()
        assert not re.search(r"\((Sa|So|Thes|Rozw|Col|Sinti|Finck|Bug\.|Urs)\)", row["Gloss"])
    assert BY_FORM["ravnos"][0]["Gloss"] == "sky"
    assert BY_FORM["kanřo"][0]["Gloss"] == "thorn; bristle"


def test_the_printed_etymological_bracket_is_preserved_verbatim_and_labelled():
    row = BY_FORM["akhor"][0]
    assert row["Etymology"] == (
        'Indische Etyma: [< pk. akkhoḍa < ai. akṣoṭa- "Walnuß"; vgl. hi. akhroṭ]'
    )
    # A bracket that withdraws its own proposal is kept, not silently dropped.
    assert BY_FORM["kerno"][0]["Etymology"] == (
        'Indische Etyma: [ai. śīrṇa- "verfallen, verfault" paßt lautlich nicht]'
    )
    assert "ohne Etym." in BY_FORM["bili"][0]["Etymology"]


def test_grammatical_labels_become_canonical_tags():
    assert BY_FORM["abijav"][0]["Tags"] == "mf noun"
    assert BY_FORM["ačhel"][0]["Tags"] == "verb intr"
    assert BY_FORM["dukhal"][0]["Tags"].split() == ["verb", "impersonal"]
    assert "participle" in BY_FORM["tumbe"][0]["Tags"]
    assert "pass" in BY_FORM["ispolajvol"][0]["Tags"]
    assert "suffix" in BY_FORM["-pe"][0]["Tags"]
    assert "prefix" in BY_FORM["bi-"][0]["Tags"]


def test_source_query_marks_become_a_typed_uncertainty_flag():
    assert "uncertain" in BY_FORM["agore"][0]["Tags"]
    assert "uncertain" not in BY_FORM["phral"][0]["Tags"]


# --------------------------------------------------------------------------
# Sound profile
# --------------------------------------------------------------------------

def test_every_installed_form_tokenizes_without_replacement_characters():
    tokenizer = Tokenizer(str(ROOT / "conversion/boretzky-romani.txt"))
    for row in FORMS:
        source = unicodedata.normalize("NFC", row["Form"])
        converted = tokenizer(source, column="IPA").replace(" ", "").replace("#", " ")
        assert "�" not in converted, row["Form"]


def test_the_profile_keeps_both_affricate_series_and_the_vlax_rhotic_apart():
    tokenizer = Tokenizer(str(ROOT / "conversion/boretzky-romani.txt"))

    def convert(value):
        return unicodedata.normalize(
            "NFC",
            tokenizer(unicodedata.normalize("NFC", value), column="IPA")
            .replace(" ", "")
            .replace("#", " "),
        )

    # The plain series follows the conventions the Zargari and CDIAL Romani profiles use.
    assert convert("čhavo") == "cʰavo"
    assert convert("džukel") == "jukel"
    assert convert("šukar") == "śukar"
    assert convert("ažukerel") == "aźukerel"
    # The alveolo-palatal series the source contrasts with it stays distinct.
    assert convert("ćix") == "ʨix"
    assert convert("maćhi") == "ʨʰ" not in convert("maćhi") or convert("maćhi") == "maʨʰi"
    assert convert("luludźi") == "luluʥi"
    assert convert("cidel") == "ʦidel"
    assert convert("řom") == "řom"
    assert convert("xancə") == "xanʦə"
    # Bound forms keep their hyphen and multiword headwords keep their space.
    assert convert("-pe") == "-pe"
    assert convert("del čhik") == "del cʰik"


# --------------------------------------------------------------------------
# Etymology and graph relations
# --------------------------------------------------------------------------

def test_only_the_indic_list_can_carry_a_cdial_parameter_id():
    for row in FORMS:
        if row["Parameter_ID"]:
            assert row["Etymology"].startswith("Indische Etyma:")


def test_every_assigned_parameter_id_is_a_real_cdial_entry():
    with (ROOT / "data/cdial/params.csv").open(encoding="utf-8", newline="") as handle:
        entries = {row[0] for row in csv.reader(handle) if row and row[0].strip()}
    used = {row["Parameter_ID"] for row in FORMS if row["Parameter_ID"]}
    assert used
    assert used <= entries


def test_accepted_links_reproduce_the_printed_etymon():
    assert BY_FORM["abijav"][0]["Parameter_ID"] == "11920"   # ai. vivāha- -> vivāhá
    assert BY_FORM["agor"][0]["Parameter_ID"] == "68"        # ai. agra- -> ágra
    assert BY_FORM["berš"][0]["Parameter_ID"] == "11392"     # ai. varṣa- -> varṣá
    # A clause the source rejects does not block the etymology it proposes in the next:
    # "[ai. aor akāriṣam ... gehören ... nicht dazu; eher < pa. garahati < ai. garhati]".
    assert BY_FORM["akharel"][0]["Parameter_ID"] == "4067"   # ai. garhati -> gárhati
    # The author's query mark is not part of the cited headword: "[zu ai. agra-?]".
    assert BY_FORM["vago"][0]["Parameter_ID"] == "68"
    # "ai. pk. X" names one form shared by both stages; the headword follows pk.
    assert BY_FORM["čikdel"][0]["Parameter_ID"] == "5032"    # ai. pk. chikkā
    # A CDIAL headword stored with the printed entry's trailing comma still matches.
    assert BY_FORM["drakh"][0]["Parameter_ID"] == "6628"     # ai. drākṣā -> drā́kṣā-,


def test_ambiguous_rejected_and_absent_etyma_stay_unlinked():
    statuses = Counter(row["Status"] for row in AUDIT)
    assert statuses["ingested"]
    # A bracket the source itself withdraws never produces a link.
    assert BY_FORM["kerno"][0]["Parameter_ID"] == ""
    # A nominative citation form that is not the CDIAL headword stays unmatched rather
    # than being forced onto a near neighbour.
    assert BY_FORM["phral"][0]["Parameter_ID"] == ""
    # A form the source leaves without an etymology stays a first-class unlinked node.
    assert BY_FORM["bili"][0]["Parameter_ID"] == ""
    # Non-Indic lists are never linked even when they cite an Old Indo-Aryan comparison.
    assert BY_FORM["tromal"][0]["Parameter_ID"] == ""
    # Two etyma offered as alternatives are both recorded and neither is chosen,
    # whether the source writes them "ai. X/Y" or "< ai. X ... oder < ai. Y".
    assert BY_FORM["čačo"][0]["Parameter_ID"] == ""
    assert BY_FORM["thovel"][0]["Parameter_ID"] == ""


def test_the_audit_records_every_etymon_candidate_including_the_rejected_ones():
    by_key = {row["Entry_Key"]: row for row in AUDIT if row["Status"] == "ingested"}
    kerno = by_key["boretzky1994romani:ind:319:L:13"]
    assert kerno["Etymon_Cited"] == "śīrṇa-"
    assert kerno["Etymon_Status"] == "rejected-by-source"
    assert kerno["Parameter_ID"] == ""
    phral = by_key["boretzky1994romani:ind:324:R:26"]
    assert (phral["Etymon_Cited"], phral["Etymon_Status"]) == ("bhrātā", "unmatched")
    assert Counter(row["Etymon_Status"] for row in AUDIT if row["Status"] == "ingested") == {
        "no-etymon": 599, "linked": 286, "unmatched": 101, "ambiguous": 105,
        "source-alternatives": 32, "rejected-by-source": 17,
    }


def test_printed_alternates_resolve_through_stable_keys():
    keys = set(BY_KEY)
    for row in FORMS:
        if row["Variant_Of_Key"]:
            assert row["Variant_Of_Key"] in keys
            assert "alternate" in row["Tags"]
    kirmo = BY_FORM["kirmo"][0]
    assert kirmo["Variant_Of_Key"] == BY_FORM["kermo"][0]["Entry_Key"]
    assert kirmo["Parameter_ID"] == BY_FORM["kermo"][0]["Parameter_ID"]


def test_cross_reference_lines_are_accounted_for_but_not_installed():
    crossrefs = [row for row in AUDIT if row["Status"] == "crossref"]
    assert len(crossrefs) == 85
    assert all("s." in row["Reason"] for row in crossrefs)
    assert "anav" not in BY_FORM


def test_a_lexeme_listed_in_two_appendices_becomes_one_row_with_both_analyses():
    row = BY_FORM["angušt"][0]
    assert len(BY_FORM["angušt"]) == 1
    assert row["Source"].count("boretzky1994romani[") == 2
    assert "Indische Etyma:" in row["Etymology"] and "Iranische Etyma:" in row["Etymology"]
    merged = [entry for entry in AUDIT if entry["Status"] == "merged"]
    assert len(merged) == 34
    assert all(entry["Reason"].startswith("merged into ") for entry in merged)


# --------------------------------------------------------------------------
# Printed edge cases found during the manual audit
# --------------------------------------------------------------------------

def test_a_bracket_printed_across_a_page_or_column_break_stays_with_its_headword():
    # ``kare, karije`` ends p. 318; its bracket is the first line of p. 319 col. 1.
    assert BY_FORM["kare"][0]["Etymology"] == "Indische Etyma: [ohne Etym.]"
    # ``khurmi`` ends p. 320 col. 1; its bracket opens col. 2.
    assert "karambha-" in BY_FORM["khurmi"][0]["Etymology"]


def test_only_the_two_entries_the_source_prints_without_a_bracket_lack_an_etymology():
    # A bracket that opens the next printed page belongs to the headword that ends the
    # previous one; six such continuations are carried across in the transcription layer.
    missing = {row["Form"] for row in FORMS if not row["Etymology"].strip()}
    assert missing == {"hačel", "zumi"}
    assert "duddha-" in BY_FORM["thud"][0]["Etymology"]
    assert "πορτοκάλι" in BY_FORM["partokoli"][0]["Etymology"]
    assert "paṇḍu-" in BY_FORM["parno"][0]["Etymology"]


def test_the_final_entry_of_the_appendix_has_no_printed_bracket():
    zumi = BY_FORM["zumi"][0]
    assert zumi["Etymology"] == ""
    assert zumi["Source"].startswith("boretzky1994romani[p. 338, col. 2, entry 42]")


def test_homonym_superscripts_are_stripped_from_the_form_but_split_the_entries():
    assert "rat" in BY_FORM
    rats = sorted(row["Entry_Key"] for row in BY_FORM["rat"])
    assert rats == [
        "boretzky1994romani:ind:325:L:06",
        "boretzky1994romani:ind:325:L:07",
    ]
    assert {row["Gloss"] for row in BY_FORM["rat"]} == {"night", "blood"}


def test_the_one_entry_printed_with_an_english_gloss_keeps_a_single_lexical_gloss():
    assert BY_FORM["terenice"][0]["Gloss"] == "equal distance (in a game of marbles)"


def test_clitics_printed_with_a_headword_are_not_part_of_the_form():
    assert "faj" in BY_FORM and "faj (ma)" not in BY_FORM
    assert "xljel" in BY_FORM and "xljel (pe(s))" not in BY_FORM
    # A genuine multiword headword is kept whole.
    assert "serel pe(s)" in BY_FORM


def test_no_form_retains_a_printed_source_label_or_exclamation_mark():
    for row in FORMS:
        assert "(" not in row["Form"] or re.search(r"\([a-zəřšžćźčh]+\)", row["Form"]), row["Form"]
        assert "!" not in row["Form"]
        assert "/" not in row["Form"]
