"""Kamioka & Yamada 1979, Lari Basic Vocabulary (raw_data/kamioka_lari_1979).

The checked-in outputs (installed CSV, audit, sample, manifest, review tables) are tested
directly.  Tests that rebuild from the OCR cache skip when the cache (which needs the
copyrighted scan) is absent; registry-dependent tests skip until the integration pass.
"""

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments import Tokenizer

import _package_modules

ROOT = Path(__file__).parents[1]
PKG = ROOT / "data/other/forms/raw_data/kamioka_lari_1979"
CSV_PATH = ROOT / "data/other/forms/20261025-kamioka-lari.csv"
SOURCE = "kamioka-yamada1979lari"
DIALECT = "dialect:Lar:kamioka-lar:L%C4%81r"

COLUMNS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
           "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
           "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]


_PACKAGE, (layout, importer, _ocr_runner) = _package_modules.load(
    PKG, "layout", "import_source", "ocr_runner"
)


@pytest.fixture(autouse=True, scope="module")
def _package_modules_active():
    with _PACKAGE.active():
        yield


def dict_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


FORMS = [dict(zip(COLUMNS, r)) for r in csv.reader(CSV_PATH.open(encoding="utf-8"))]
BY_KEY = {r["Entry_Key"]: r for r in FORMS}
AUDIT = dict_rows(PKG / "audit.csv")
MANIFEST = json.loads((PKG / "manifest.json").read_text())


def tok(form):
    t = Tokenizer(str(ROOT / "conversion/kamioka-lari.txt"))
    return t(unicodedata.normalize("NFC", form), column="IPA", segment_separator="",
             separator=" ")


# ----------------------------------------------------------------------------- counts

def test_counts_reconcile_raw_units_audit_and_rows():
    counts = MANIFEST["counts"]
    assert counts["entries_expected"] == 1000
    assert counts["entries_found"] == 1000
    assert counts["missing_entry_numbers"] == []
    status = Counter(r["Status"] for r in AUDIT)
    assert dict(sorted(status.items())) == counts["audit_status"]
    assert len(AUDIT) == counts["audit_units"]
    assert len(FORMS) == counts["installed_rows"]
    assert set(status) <= {"installed", "example_sentence", "no_head_form",
                           "cross_reference_only", "editorial_note", "ocr_noise",
                           "page_number", "ocr_unparseable"}
    # every installed row is emitted by exactly one installed audit unit
    emitted = [k for r in AUDIT for k in r["Emitted_Keys"].split()]
    assert len(emitted) == len(set(emitted)) == len(FORMS)
    assert set(emitted) == set(BY_KEY)
    assert all(r["Status"] == "installed" for r in AUDIT if r["Emitted_Keys"])
    assert counts["variant_rows"] == sum(1 for r in FORMS if r["Variant_Of_Key"])


def test_every_entry_number_has_exactly_one_head_unit_in_sequence():
    heads = [r for r in AUDIT if r["Kind"] == "head"]
    numbers = [int(r["Entry"]) for r in heads]
    assert sorted(numbers) == list(range(1, 1001))
    # reading order follows the printed numbering
    ordered = [int(r["Entry"]) for r in sorted(heads, key=lambda r: (
        int(r["PDF_Page"]), int(r["Line_Box"].split(",")[1])))]
    assert ordered == list(range(1, 1001))
    # inferred/garbled numbers are recorded, not hidden
    assert any("inferred from the sequence" in r["Reason"] for r in heads)
    assert any("number read as" in r["Reason"] for r in heads)


def test_page_range_and_printed_folios():
    assert MANIFEST["scan"]["pdf_pages"] == 232
    assert MANIFEST["scan"]["sha256"] == \
        "9196621ac485cbbe60cd302712a0d5289e7b2c3ad118446d74361cde0710c569"
    assert MANIFEST["scan"]["redistributed"] is False
    assert MANIFEST["page_ranges"]["vocabulary_pdf"] == [10, 186]
    assert MANIFEST["page_ranges"]["vocabulary_printed"] == [1, 177]
    pages = {int(r["PDF_Page"]) for r in AUDIT}
    assert pages == set(range(10, 187))
    folios = [r for r in AUDIT if r["Status"] == "page_number"]
    agree = [r for r in folios if r["Raw_OCR"] == str(int(r["PDF_Page"]) - 9)]
    assert len(agree) >= 140
    assert [r["PDF_Page"] for r in folios if r not in agree] == ["121"]
    # first and last entries sit on the first and last vocabulary pages
    first = next(r for r in AUDIT if r["Kind"] == "head" and r["Entry"] == "0001")
    last = next(r for r in AUDIT if r["Kind"] == "head" and r["Entry"] == "1000")
    assert (first["PDF_Page"], first["Printed_Page"]) == ("10", "1")
    assert (last["PDF_Page"], last["Printed_Page"]) == ("186", "177")


# ----------------------------------------------------------------------------- rows

def test_rows_are_well_formed_and_keyed():
    for r in FORMS:
        assert r["Language_ID"] == "Lar"
        assert r["Form"] and "�" not in "".join(r.values())
        assert unicodedata.is_normalized("NFC", "".join(r.values()))
        assert re.fullmatch(rf"{SOURCE}\[p\. \d{{1,3}}, entry \d{{4}}\]", r["Source"])
        assert re.fullmatch(rf"{SOURCE}:\d{{4}}:(\d+|sub:\d+(:\d+)?)(:v\d+)?", r["Entry_Key"])
        entry = r["Entry_Key"].split(":")[1]
        assert f"entry {entry}]" in r["Source"]
        assert DIALECT in r["Tags"].split()
        assert r["Native"] == r["Phonemic"] == r["Parameter_ID"] == ""
        assert r["Etymology"] == r["Cognateset"] == r["Borrowed_From_Key"] == ""
        if r["Variant_Of_Key"]:
            assert r["Variant_Of_Key"] in BY_KEY
            assert {"alternate", "sound-variant"} & set(r["Tags"].split())


def test_typed_review_marker_on_every_unreviewed_row():
    review = {r["Unit_ID"]: r["Review"] for r in AUDIT}
    unit_of = {k: r["Unit_ID"] for r in AUDIT for k in r["Emitted_Keys"].split()}
    for r in FORMS:
        state = review[unit_of[r["Entry_Key"]]]
        assert state in {"ocr:unreviewed", "ocr:image-checked", "ocr:image-corrected"}
        assert ("uncertain" in r["Tags"].split()) == (state == "ocr:unreviewed")


def test_representative_rows():
    assert (BY_KEY[f"{SOURCE}:0001:1"]["Form"], BY_KEY[f"{SOURCE}:0001:1"]["Gloss"]) == \
        ("sera", "head")
    # comma-separated synonyms are separate rows with the questionnaire gloss
    assert [BY_KEY[f"{SOURCE}:0020:{i}"]["Form"] for i in (1, 2, 3)] == ["kül", "šana", "duš"]
    # sub-forms carry their own printed gloss
    assert BY_KEY[f"{SOURCE}:0113:sub:1"]["Form"] == "gāri"
    assert BY_KEY[f"{SOURCE}:0113:sub:1"]["Gloss"] == "cart"
    # printed ~ alternates are variant rows
    v = BY_KEY[f"{SOURCE}:0118:1:v2"]
    assert (BY_KEY[f"{SOURCE}:0118:1"]["Form"], v["Form"]) == ("deraxt", "derax")
    assert v["Variant_Of_Key"] == f"{SOURCE}:0118:1" and "sound-variant" in v["Tags"]
    # word-level / alternates expand (Abbreviations: goš kerda/dada)
    assert BY_KEY[f"{SOURCE}:0277:2"]["Form"] == "goš kerda"
    assert BY_KEY[f"{SOURCE}:0277:2:v2"]["Form"] == "goš dada"
    assert BY_KEY[f"{SOURCE}:0277:2"]["Gloss"] == "to listen"
    # sense-paired head (English and Lari both split by ';')
    assert BY_KEY[f"{SOURCE}:0048:1"]["Gloss"] == "sickness"
    assert BY_KEY[f"{SOURCE}:0048:2"]["Gloss"] == "sick"
    # questionnaire equivalents after / leave the gloss and go to Notes
    assert BY_KEY[f"{SOURCE}:0102:1"]["Gloss"] == "fur"
    assert "fourrure" in BY_KEY[f"{SOURCE}:0102:1"]["Notes"]
    # stress accent and glottal stop
    assert BY_KEY[f"{SOURCE}:0980:1"]["Form"] == "váli"
    assert BY_KEY[f"{SOURCE}:0201:1"]["Form"] == "'ayš"
    # unglossed sub-form keeps a note naming its questionnaire entry
    konar = next(r for r in FORMS if r["Entry_Key"].startswith(f"{SOURCE}:0118:sub")
                 and r["Form"] == "konār")
    assert konar["Gloss"] == "" and "0118 'tree'" in konar["Notes"]


def test_glosses_are_clean():
    for r in FORMS:
        g = r["Gloss"]
        assert not re.search(r"\(cf\.|\[see\b|\[\d{4}\]|\(\d{4}\)", g), g
        assert not re.search(r"^[\s\"“”'‘’!]|[\"“”]$|[^\w]!$", g), g
        assert not re.search(r"\b(vi|vt)\.", g), g


def test_example_sentences_and_prompts_are_not_forms():
    sentences = [r for r in AUDIT if r["Status"] == "example_sentence"]
    assert len(sentences) > 500
    forms = {r["Form"] for r in FORMS}
    for r in sentences:
        assert not r["Emitted_Keys"]
    # Persian prompts stay in the audit only
    assert all(not re.search(r"[؀-ۿ]", r["Notes"]) for r in FORMS)
    assert sum(1 for r in AUDIT if r["Persian_OCR"]) > 900
    assert "ma 'e karü 'akonem" not in forms


def test_corrections_are_applied_and_documented():
    corr = dict_rows(PKG / "corrections.csv")
    assert all(r["Evidence"].startswith(("page-image crop", "mark-montage crop")) for r in corr)
    units = {r["Unit_ID"] for r in corr}
    by_unit = {r["Unit_ID"]: r for r in AUDIT}
    assert all(by_unit[u]["Review"] == "ocr:image-corrected" for u in units)
    assert BY_KEY[f"{SOURCE}:0039:1"]["Form"] == "'araq"
    assert BY_KEY[f"{SOURCE}:0219:1"]["Gloss"] == "quarrel"
    assert BY_KEY[f"{SOURCE}:0949:1"]["Form"] == "maamüli"


def test_seeded_image_audit_has_no_open_material_error():
    review = dict_rows(PKG / "sample_review.csv")
    rounds = Counter(r["Round"] for r in review)
    assert all(n == 20 for n in rounds.values()) and len(rounds) >= 6
    assert all(r["Material_Error"] == "no" for r in review)
    # every initial finding names an error class and the systematic fix
    for r in review:
        if r["Initial_Finding"]:
            assert r["Error_Class"] and r["Fix"]
    last = max(rounds, key=int)
    # the final round's findings were isolated misreadings, corrected by hand
    final = [r for r in review if r["Round"] == last and r["Initial_Finding"]
             and r["Error_Class"] != "hyphenation"]
    assert len(final) <= 2 and all("corrections.csv" in r["Fix"] for r in final)
    sample = dict_rows(PKG / "sample.csv")
    assert len(sample) == 20
    assert {r["Unit_ID"] for r in sample} == \
        {r["Unit_ID"] for r in review if r["Round"] == last}
    assert all(r["Material_Error"] == "no" for r in sample)


# ----------------------------------------------------------------------------- OCR repairs

def test_mark_classification():
    caron = {"cls": "caron", "bm": ["10000001", "01000010", "00100100", "00011000",
                                    "00011000", "00011000", "00000000"]}
    acute = {"cls": "caron", "bm": ["000000011", "000000110", "000001100", "000011000",
                                    "000110000", "001100000", "011000000"]}
    assert layout.classify_mark(caron) == "caron"
    assert layout.classify_mark(acute) == "acute"


def test_mark_attachment_by_pitch():
    word = {"text": "sana", "x0": 100, "x1": 100 + 3 * 22 + 16}
    caron = {"cls": "caron", "bm": ["1001", "0110"], "x0": 103, "x1": 117, "y0": 0,
             "y1": 9, "letter": [20, 40, 101, 118]}
    macron = {"cls": "macron", "x0": 125, "x1": 140, "y0": 4, "y1": 8,
              "letter": [20, 40, 123, 141]}
    text, log = layout.attach_marks(word, [caron, macron], 22.5)
    assert text == "šāna"
    assert [e["status"] for e in log] == ["placed", "placed"]


def test_apostrophe_repairs():
    # raw-image reading supplies the apostrophe the cleaned reading lost or misread
    assert layout.reconcile_apostrophes("0964. sometimes gāti", "0964. sometimes ga'i")[0] \
        == "0964. sometimes gā'i"
    assert layout.reconcile_apostrophes("jātel", "ja'tel")[0] == "jā'el"
    assert layout.reconcile_apostrophes("'takonem.", "'akonem.")[0] == "'akonem."
    # q/g disagreement resolves to q
    assert layout.reconcile_apostrophes("geymat", "qeymat")[0] == "qeymat"
    assert layout.reconcile_apostrophes("farq", "farg")[0] == "farq"
    # glyph geometry: a cell whose ink lies above the x-height is the apostrophe
    glyphs = [[756, 1496, 762, 1512], [772, 1503, 791, 1524], [784, 1503, 813, 1533],
              [814, 1488, 833, 1499], [816, 1504, 832, 1524]]
    assert layout.fix_apostrophes_by_glyphs("tays", glyphs)[0] == "'ays"
    # a real t is never rewritten
    full = [[756, 1490, 772, 1524], [772, 1503, 791, 1524]]
    assert layout.fix_apostrophes_by_glyphs("ta", full)[0] == "ta"


def test_clean_form_mechanical_repairs():
    assert importer.clean_form("2üd")[0] == "züd"
    assert importer.clean_form("sSina-riz")[0] == "sina-riz"
    assert importer.clean_form("xXaš")[0] == "xaš"
    assert importer.clean_form("‘'atam")[0] == "'atam"
    assert importer.clean_form("“az")[0] == "'az"
    assert importer.clean_form("qesmat_kerda")[0] == "qesmat kerda"
    # genuine double letters are untouched
    assert importer.clean_form("'ossā")[0] == "'ossā"


def test_alternation_expansion():
    assert importer.alternates("goš kerda/dada") == [("goš kerda", ""), ("goš dada", "alternate")]
    assert importer.alternates("'az xāw/xat vā rosseda") == [
        ("'az xāw vā rosseda", ""), ("'az xat vā rosseda", "alternate")]
    assert importer.alternates("deraxt ~ derax") == [("deraxt", ""), ("derax", "sound-variant")]
    assert importer.alternates("küfta ~kübeda") == [("küfta", ""), ("kübeda", "sound-variant")]


def test_sentence_detection():
    assert importer.is_sentence("ma 'ic kar naakonem.", "I won't do any work.")
    assert importer.is_sentence("'aga pülod bāre maqarzāde.", "idem.")
    assert not importer.is_sentence("xoma", "a very large jar about 1.5 m. high, etc.")
    assert not importer.is_sentence("gāri", "cart")


# ----------------------------------------------------------------------------- profile

def test_profile_covers_every_installed_form_and_maps_house_letters():
    for r in FORMS:
        out = tok(r["Form"])
        assert "�" not in out and "?" not in out, (r["Form"], out)
    assert tok("'ayš") == "ʔayś"
    assert tok("güšt") == "güśt"
    assert tok("ža") == "źa"
    assert tok("xāw") == "xāv"
    assert tok("'ána") == "ʔána"
    assert tok("cāyne 'avvaki") == "cāyne ʔavvaki"
    assert tok("sāl-e now") == "sāl-e nov"
    inventory = Counter(c for r in FORMS for c in unicodedata.normalize("NFD", r["Form"]))
    assert set(inventory) <= set("abcdefghijklmnopqrstuvwxyz' -") | {
        "\u0304", "\u0308", "\u030c", "\u0301", "\u0303"}  # macron, diaeresis, caron, acute, tilde
    assert tok("'\u0101\u0303") == "\u0294\u0101\u0303"


def test_yaml_routes_profile_and_reference():
    import yaml
    meta = yaml.safe_load((ROOT / "data/other/forms/20261025-kamioka-lari.yaml").read_text())
    assert meta["defaults"]["transcription"]["profile"] == "kamioka-lari"
    assert meta["defaults"]["identity"] == {"legacy_ids": "stem", "append_order": 114}
    ref = meta["sources"][SOURCE]["reference"]
    assert ref["ocr"] is True and ref["etymology_provenance"] == "none"


def test_lects_table_proposes_the_lar_town_dialect():
    lects = dict_rows(PKG / "lects.csv")
    assert len(lects) == 1
    row = lects[0]
    assert (row["Language_ID"], row["Dialect_Tag"], row["Glottocode"]) == \
        ("Lar", DIALECT, "nucl1274")
    assert (row["Latitude"], row["Longitude"], row["Quality"]) == ("27.667", "54.333", "A")


# ----------------------------------------------------------------------------- rebuild / build

def _cache_available():
    ocr = _ocr_runner
    return all(ocr.load(p) is not None for p in (10, 98, 186))


@pytest.mark.skipif(not _cache_available(), reason="OCR cache (needs the scan) not present")
def test_importer_reproduces_installed_outputs():
    res = importer.build()
    assert [[r[c] for c in COLUMNS] for r in res["forms"]] == \
        [[r[c] for c in COLUMNS] for r in FORMS]
    assert len(res["audit"]) == len(AUDIT)


def test_rows_survive_make_cldf_parse():
    sys.path.insert(0, str(ROOT))
    try:
        import make_cldf
    except Exception as exc:  # pragma: no cover - environment without build deps
        pytest.skip(f"make_cldf unavailable: {exc}")
    rows, _ = make_cldf.parse_file(str(CSV_PATH), [])
    assert len(rows) == len(FORMS)
    by_key = {r.entry_key: r for r in rows}
    assert by_key[f"{SOURCE}:0201:1"].form == "ʔayś"
    assert by_key[f"{SOURCE}:0201:1"].old_form == "'ayš"
    assert all("�" not in r.form for r in rows)


def test_registry_has_language_and_dialect_after_integration():
    langs = {r["ID"] for r in dict_rows(ROOT / "cldf/languages.csv")}
    if "Lar" not in langs:
        pytest.skip("Lar not yet registered (central integration pass)")
    tags = {r["Tag"]: r for r in dict_rows(ROOT / "cldf/dialects.csv")}
    assert DIALECT in tags and tags[DIALECT]["Language_ID"] == "Lar"
