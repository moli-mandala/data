"""Regression tests for the Rezai Baghbidi (2003) Zargari ingest."""

import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FORMS = ROOT / "data/other/forms/20260825-rezai-baghbidi-zargari.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-manifest.json"


def load_source():
    path = ROOT / "data/other/forms/raw_data/rezai_baghbidi_zargari_2003.py"
    spec = importlib.util.spec_from_file_location("rezai_baghbidi_zargari_2003", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zargari = load_source()


def dict_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def raw_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


FORM_ROWS = raw_rows(FORMS)
AUDIT_ROWS = dict_rows(AUDIT)
COLUMNS = {name: index for index, name in enumerate(zargari.FORM_FIELDS)}


def field(row, name):
    return row[COLUMNS[name]]


def by_form(name):
    return [row for row in FORM_ROWS if field(row, "Form") == name]


def one(name):
    matches = by_form(name)
    assert len(matches) == 1, f"expected exactly one {name!r}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Counts and completeness
# ---------------------------------------------------------------------------

def test_audit_accounts_for_every_source_unit():
    assert len(AUDIT_ROWS) == 706
    spans = [row for row in AUDIT_ROWS
             if not row["Span_Index"].startswith("t") and "+" not in row["Span_Index"]]
    attached = [row for row in AUDIT_ROWS if "+" in row["Span_Index"]]
    listed = [row for row in AUDIT_ROWS if row["Span_Index"].startswith("t")]
    assert len(spans) == 575
    assert len(attached) == 42
    assert len(listed) == 89
    assert Counter(row["Status"] for row in AUDIT_ROWS) == {
        "ingest": 444, "x-clause": 130, "dup": 68, "x-phrase": 43,
        "x-meta": 10, "x-compare": 9, "x-donor": 2,
    }
    assert all(row["Material_Error"] == "no" for row in AUDIT_ROWS)
    assert all(row["Reason"] for row in AUDIT_ROWS)
    assert {row["Collation_Date"] for row in AUDIT_ROWS} == {zargari.COLLATION_DATE}


def test_installed_rows_reconcile_with_the_audit():
    assert len(FORM_ROWS) == 522
    assert all(len(row) == 15 for row in FORM_ROWS)
    primaries = [row for row in FORM_ROWS if not field(row, "Variant_Of_Key")]
    variants = [row for row in FORM_ROWS if field(row, "Variant_Of_Key")]
    assert len(primaries) == 444  # one per ``ingest`` audit unit
    assert len(variants) == 78
    keys = {field(row, "Entry_Key") for row in FORM_ROWS}
    assert len(keys) == len(FORM_ROWS)
    assert all(field(row, "Variant_Of_Key") in keys for row in variants)
    emitted = {key for row in AUDIT_ROWS for key in row["Emitted_Keys"].split()}
    assert {field(row, "Entry_Key") for row in primaries} <= emitted


def test_pages_stay_inside_the_printed_article():
    pages = {int(row["Printed_Page"]) for row in AUDIT_ROWS}
    # p. 126 is unbroken prose and pp. 147--148 carry only the tail of section 5.4,
    # the abbreviation list and the bibliography, so no source unit falls on p. 148.
    assert min(pages) == 123 and max(pages) == 147
    assert pages <= set(range(123, 149))
    assert 126 not in pages


# ---------------------------------------------------------------------------
# Schema hygiene
# ---------------------------------------------------------------------------

def test_every_row_is_unlinked_zargari():
    for row in FORM_ROWS:
        assert field(row, "Language_ID") == "Zarg"
        assert field(row, "Parameter_ID") == ""
        assert field(row, "Cognateset") == ""
        assert field(row, "Native") == ""
        assert field(row, "Phonemic") == ""
        assert field(row, "Notes") == ""
        assert zargari.DIALECT_TAG in field(row, "Tags").split()


def test_forms_and_glosses_are_clean():
    for row in FORM_ROWS:
        form = field(row, "Form")
        gloss = field(row, "Gloss")
        assert form and form == form.strip()
        assert gloss and gloss == gloss.strip()
        assert "�" not in form and "�" not in gloss
        assert form == unicodedata.normalize("NFC", form)
        assert "(" not in form and ")" not in form
        # printed section numbers, page references and bibliographic prose never leak in
        assert not any(token in gloss for token in ("e.g.", "cf.", "Section", "(sg.)", "(m.)"))


def test_every_row_carries_a_page_and_section_locator():
    for row in FORM_ROWS:
        citations = field(row, "Source").split(";")
        assert citations
        for citation in citations:
            assert citation.startswith("rezaibaghbidi2003zargari[p. ")
            assert citation == citation.strip()
            assert ", section " in citation and citation.endswith("]")


def test_tags_are_canonical():
    known = zargari.canonical_tags()
    for row in FORM_ROWS:
        for tag in field(row, "Tags").split():
            assert tag == zargari.DIALECT_TAG or tag in known, tag


# ---------------------------------------------------------------------------
# Language and dialect registration
# ---------------------------------------------------------------------------

def test_zargari_is_registered_as_its_own_language():
    dialects = dict_rows(ROOT / "cldf/dialects.csv")
    row = next(row for row in dialects if row["ID"] == "zargari")
    assert row["Tag"] == zargari.DIALECT_TAG
    assert row["Language_ID"] == "Zarg"
    assert row["Glottocode"] == "zarg1238"
    # 36 degrees 03 minutes N, 50 degrees 23 minutes E, printed in section 1
    assert float(row["Latitude"]) == pytest.approx(36.05, abs=1e-4)
    assert float(row["Longitude"]) == pytest.approx(50.3833, abs=1e-3)
    assert row["Quality"] == "A"
    assert "Zargar village" in row["Location"]
    languages = {row["ID"]: row for row in dict_rows(ROOT / "cldf/languages.csv")}
    # Zargari is its own Glottolog language (zarg1238), so it is a base language rather than
    # a dialect of the European Romani cover label; the village survives as its dialect tag.
    assert "eur" in languages
    assert languages["Zarg"]["Glottocode"] == "zarg1238"
    assert languages["Zarg"]["Name"] == "Zargari"


# ---------------------------------------------------------------------------
# Curation decisions
# ---------------------------------------------------------------------------

def test_representative_entries_keep_their_stable_keys():
    stone = one("bār")
    assert field(stone, "Entry_Key") == "rezaibaghbidi2003zargari:p131:s3.1.8:i1"
    assert field(stone, "Gloss") == "stone"
    assert field(stone, "Tags").split() == ["noun", zargari.DIALECT_TAG]
    assert field(stone, "Source") == "rezaibaghbidi2003zargari[p. 131, section 3.1.8]"


def test_printed_alternates_become_variant_rows_of_one_head():
    head = one("kāt")
    variant = one("qāt")
    assert field(head, "Gloss") == field(variant, "Gloss") == "scissors"
    assert field(variant, "Variant_Of_Key") == field(head, "Entry_Key")
    assert "alternate" in field(variant, "Tags").split()
    assert field(head, "Variant_Of_Key") == ""


def test_optional_segments_printed_in_parentheses_are_expanded():
    for full, reduced, gloss in (("baxt", "bax", "luck"),
                                 ("vāst", "vās", "arm; hand"),
                                 ("dānd", "dān", "tooth")):
        assert field(one(full), "Gloss") == gloss
        assert field(one(reduced), "Variant_Of_Key") == field(one(full), "Entry_Key")


def test_repeated_mentions_are_folded_into_one_record_with_every_citation():
    eight = one("oxto")
    citations = field(eight, "Source").split(";")
    assert "rezaibaghbidi2003zargari[p. 128, section 2.3.15]" in citations
    assert "rezaibaghbidi2003zargari[p. 137, section 3.5.1]" in citations
    assert "rezaibaghbidi2003zargari[p. 146, section 5.3]" in citations
    assert field(eight, "Etymology") == "< Greek οχτώ."
    # the stress-marked citation of the same word is kept as its alternate
    assert field(one("oxtó"), "Variant_Of_Key") == field(eight, "Entry_Key")


def test_donor_statements_are_prose_not_graph_edges():
    milk = one("süti")
    assert field(milk, "Etymology") == "< Azari Turkish süt."
    assert "loanword" in field(milk, "Tags").split()
    assert field(milk, "Borrowed_From_Key") == ""
    assert all(field(row, "Borrowed_From_Key") == "" for row in FORM_ROWS)
    assert all(field(row, "Derivation_Parent_Keys") == "" for row in FORM_ROWS)


def test_comparative_section_five_prose_is_attached_to_the_zargari_row():
    ear = one("kān")
    assert field(ear, "Gloss") == "ear"
    assert field(ear, "Etymology").startswith("Hindi kān, Sanskrit kárṇa-")
    # the Hindi, Sanskrit and Qorbati comparanda themselves are never installed
    assert not by_form("kárṇa-") and not by_form("mārez") and not by_form("peṭ")


def test_clauses_and_unglossed_paradigms_stay_out_of_the_installed_rows():
    multiword = [field(row, "Form") for row in FORM_ROWS if " " in field(row, "Form")]
    assert multiword  # the lexical lists do contribute set phrases
    for form in multiword:
        row = by_form(form)[0]
        assert "multiword-expression" in field(row, "Tags").split()
    # personal-pronoun and copula paradigm cells are printed without glosses
    for cell in ("timen", "özüm", "tovāv", "āmundār", "kolusku"):
        assert not by_form(cell), cell


def test_homonyms_and_distinct_senses_are_kept_apart():
    assert {field(row, "Gloss") for row in by_form("ruv")} == {"wolf", "cry!"}
    assert {field(row, "Gloss") for row in by_form("khal")} == {"this side", "this way"}
    assert {field(row, "Gloss") for row in by_form("teli")} == {"bottom", "below", "beneath; under"}
    assert {field(row, "Gloss") for row in by_form("sar")} == {"like"}
    assert {tuple(sorted(set(field(row, "Tags").split()) & {"adv", "postp"}))
            for row in by_form("sar")} == {("adv",), ("postp",)}


def test_verb_stems_and_infinitives_are_installed_as_separate_rows():
    present = one("beš-")
    perfect = one("bešd-")
    assert field(present, "Tags").split()[:3] == ["verb", "pres", "stem"]
    assert field(perfect, "Tags").split()[:3] == ["verb", "perfect", "stem"]
    assert field(present, "Gloss") == field(perfect, "Gloss") == "to sit"
    assert field(one("bešipej"), "Tags").split()[:2] == ["verb", "inf"]


def test_numerals_cover_the_printed_cardinal_table():
    numerals = {field(row, "Form"): field(row, "Gloss")
                for row in FORM_ROWS if "num" in field(row, "Tags").split()}
    assert numerals["sefr"] == "zero"
    assert numerals["deš-pāndž"] == "fifteen"
    assert numerals["jokus"] == "twenty"
    assert numerals["pejindā-jokus"] == "seventy"
    assert numerals["šel"] == "one hundred"
    assert numerals["deš-sila"] == "ten thousand"
    assert numerals["šel-i-pejindā-sārāndā-šov"] == "one hundred and ninety-six"


def test_manifest_records_the_scan_and_the_scope():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["pdf_sha256"] == zargari.PDF_SHA256
    assert manifest["pdf_redistributed"] is False
    assert manifest["installed_rows"] == len(FORM_ROWS)
    assert manifest["audit_rows"] == len(AUDIT_ROWS)
    assert manifest["extraction"]["ocr"] is False
    assert manifest["language_model"]["dialect_tag"] == zargari.DIALECT_TAG


# ---------------------------------------------------------------------------
# Sound profile
# ---------------------------------------------------------------------------

def convert(value):
    from segments import Tokenizer

    tokenizer = Tokenizer(str(ROOT / "conversion/zargari.txt"))
    return unicodedata.normalize(
        "NFC",
        tokenizer(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_profile_maps_the_articles_transcription_to_house_conventions():
    assert convert("jekh") == "yekʰ"          # j is the palatal glide, kh an aspirate
    assert convert("džukel") == "jukel"       # dž is the voiced palatal affricate
    assert convert("čhindiv-") == "cʰindiv-"
    assert convert("bešdo-som") == "beśdo-som"
    assert convert("γānd-o") == "ɣānd-o"
    assert convert("mā́khuv") == "mā́kʰuv"     # macron plus stress acute survives
    assert convert("kālus güra") == "kālus güra"
    assert convert("ojipey") == "oyipey"


def test_profile_covers_every_installed_symbol():
    symbols = {character for row in FORM_ROWS for character in field(row, "Form")}
    for row in FORM_ROWS:
        assert "�" not in convert(field(row, "Form"))
    assert symbols  # sanity: the corpus is not empty


# ---------------------------------------------------------------------------
# Text layer (needs the uncommitted publisher PDF)
# ---------------------------------------------------------------------------

pdf_required = pytest.mark.skipif(
    not zargari.DEFAULT_PDF.exists(), reason="the publisher PDF is not redistributed"
)


@pdf_required
def test_glyph_name_decoding_restores_what_tounicode_drops():
    pages = zargari.extract_pages(zargari.DEFAULT_PDF)
    assert len(pages) == zargari.PDF_PAGES
    body = "\n".join(pages[page] for page in sorted(pages))
    # ligatures whose ToUnicode map yields only their second element
    assert "The Zargari language" in body      # T_h
    assert "field work" in body                # f_i
    assert "influences" in body                # f_l
    assert "different" in body                 # f_f
    assert "�" not in body
    # oldstyle figures and small capitals
    assert "50° 23′ E and 36° 03′ N" in body
    assert "issn 1528–0478".replace("issn", "") in body.replace("‹", "")
    # Indological glyphs used in the section 5.4 comparanda
    assert "kāṣṭhá-" in body and "māriṣa-" in body


@pdf_required
def test_spans_are_stable_and_fully_curated():
    spans = zargari.extract_spans(zargari.DEFAULT_PDF)
    assert len(spans) == 575
    decisions, _ = zargari.parse_decisions()
    assert {(span["section"], str(span["span_index"])) for span in spans} <= set(decisions)
    first = spans[0]
    assert first["section"] == "1" and first["printed_page"] == 123
    records, audit = zargari.build_records(spans)
    assert len(records) == len(FORM_ROWS)
    assert len(audit) == len(AUDIT_ROWS)
    assert not zargari.validate(records, audit)


# ---------------------------------------------------------------------------
# Compiled CLDF
# ---------------------------------------------------------------------------

COMPILED = ROOT / "cldf/forms.csv"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    installed = {field(row, "Entry_Key") for row in FORM_ROWS}
    compiled_keys = {
        row["Source_Key"]
        for row in dict_rows(ROOT / "cldf/form-source-keys.csv")
        if row["Source_Key"].startswith("rezaibaghbidi2003zargari")
    }
    assert installed == compiled_keys

    compiled = [row for row in dict_rows(COMPILED)
                if zargari.DIALECT_TAG in row["Tags"].split()]
    assert len(compiled) == len(FORM_ROWS)
    assert all(row["Language_ID"] == "Zarg" for row in compiled)
    assert all(not row["Cognateset"] for row in compiled)
    assert all("�" not in row["Form"] for row in compiled)
    # unlinked heads plus their variant rows
    assert Counter(row["Status"] for row in compiled)["unlinked"] == 444


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_same_lect_homographs_survive_the_deduper():
    compiled = [row for row in dict_rows(COMPILED)
                if zargari.DIALECT_TAG in row["Tags"].split()]
    senses = Counter(row["Form"] for row in compiled)
    assert senses["ruv"] == 2      # 'wolf' and the imperative 'cry!'
    assert senses["teli"] == 3     # noun, adverb and postposition
    assert senses["kʰal"] == 2     # 'this side' and 'this way'
    glosses = {row["Form"]: row["Gloss"] for row in compiled}
    assert "; " not in glosses["opro"] or glosses["opro"] == "up; above"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_printed_alternates_become_variant_edges():
    compiled = {row["ID"]: row for row in dict_rows(COMPILED)
                if zargari.DIALECT_TAG in row["Tags"].split()}
    edges = [row for row in dict_rows(ROOT / "cldf/edges.csv")
             if row["Child_ID"] in compiled or row["Parent_ID"] in compiled]
    assert len(edges) == 78
    assert {row["Kind"] for row in edges} == {"variant"}
    assert all(row["Child_ID"] in compiled and row["Parent_ID"] in compiled for row in edges)


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_the_source_resolves_to_a_formatted_reference():
    references = {row["ID"]: row for row in dict_rows(ROOT / "cldf/references.csv")}
    reference = references["rezaibaghbidi2003zargari"]
    assert "Rezai Baghbidi" in reference["Source"]
    assert "Romani Studies" in reference["Source"]
    assert "123–148" in reference["Source"]
    assert reference["OCR"] == "No"
    assert reference["Etymology_Provenance"] == "none"
    assert "rezai_baghbidi_zargari_2003.py" in reference["Provenance"]
    assert reference["Progress"].startswith("Every isolated Zargari word")
    cited = {
        key.strip().split("[", 1)[0]
        for row in dict_rows(COMPILED)
        if zargari.DIALECT_TAG in row["Tags"].split()
        for key in row["Source"].split(";")
        if key.strip()
    }
    assert cited == {"rezaibaghbidi2003zargari"}
