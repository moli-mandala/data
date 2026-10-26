"""Cheung, Etymological Dictionary of the Iranian Verb (2007): source-local regression tests.

The importer reads the copyrighted PDF (or its verified extraction cache outside the repo);
tests that rebuild from it are skipped when neither is present. Tests that need the
integration pass (registry rows for the proposed languages/dialects, compiled CLDF) skip
until those rows exist.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

import _package_modules

ROOT = Path(__file__).parents[1]
PKG = ROOT / "data/other/forms/raw_data/cheung_ediv_2007"
STEM = "20261025-cheung-ediv"
FORMS = ROOT / f"data/other/forms/{STEM}.csv"
PARAMS = ROOT / f"data/other/params/{STEM}.csv"
TEXTS = ROOT / f"data/other/entry_texts/{STEM}.csv"
AUDIT = PKG / "audit.csv"
SAMPLE = PKG / "sample.csv"
LECTS = PKG / "lects.csv"
GLYPHS = PKG / "glyphs.csv"
PROFILE = ROOT / "conversion/cheung-ediv.txt"
OLD_MIDDLE = {"Av", "OPers", "MPrs", "Parth", "Khot", "Tumsh", "Sogd", "Chvar", "Bactr"}




def rows(path, header=True):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)) if header else list(csv.reader(handle))


_PACKAGE, (importer, pdf_layer, _decode) = _package_modules.load(
    PKG, "import_source", "pdf_layer", "decode"
)


@pytest.fixture(autouse=True, scope="module")
def _package_modules_active():
    with _PACKAGE.active():
        yield
HAVE_SOURCE = importer.CACHE.exists() or importer.PDF.exists()
needs_source = pytest.mark.skipif(not HAVE_SOURCE, reason="Cheung PDF / extraction cache absent")


@pytest.fixture(scope="module")
def built():
    if not HAVE_SOURCE:
        pytest.skip("Cheung PDF / extraction cache absent")
    return importer.build()


FORM_ROWS = rows(FORMS, header=False)
FORM_DICTS = [dict(zip(importer.FORM_FIELDS, r)) for r in FORM_ROWS]
BY_KEY = {r["Entry_Key"]: r for r in FORM_DICTS}
PARAM_ROWS = rows(PARAMS, header=False)
AUDIT_ROWS = rows(AUDIT)
LECT_ROWS = {r["Source_Label"]: r for r in rows(LECTS)}


# ----------------------------------------------------------------------------- counts

def test_expected_counts_and_statuses():
    assert len(PARAM_ROWS) == 701
    assert len(FORM_ROWS) == 12960
    status = Counter(r["Status"] for r in AUDIT_ROWS)
    assert status == {
        "installed_etymon": 701, "installed_form": 12960, "audit_only_paradigm": 1200,
        "entry_text": 2882, "preverb_sense_only": 429, "non_iranian": 165, "no_form_segment": 66,
        "unmapped_label": 3, "not_a_form": 2,
    }
    assert len(rows(TEXTS)) == 3817


@needs_source
def test_installed_files_match_a_fresh_build(built):
    entries, params, forms, texts, audit = built
    assert len(entries) == 701
    assert [list(p) for p in params] == PARAM_ROWS
    assert [[f[k] for k in importer.FORM_FIELDS] for f in forms] == FORM_ROWS
    assert len(audit) == len(AUDIT_ROWS)


def test_manifest_records_pdf_identity():
    import json
    manifest = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == pdf_layer.PDF_SHA256
    assert manifest["pdf_pages"] == 625
    assert manifest["printed_pages"] == [1, 475]
    assert manifest["pdf_redistributed"] is False


# ----------------------------------------------------------------------------- identity

def test_unique_stable_keys_and_ids():
    keys = [r["Entry_Key"] for r in FORM_DICTS]
    assert len(keys) == len(set(keys))
    assert all(k.startswith("cheung2007ediv:cheung-ediv-") for k in keys)
    ids = [r[0] for r in PARAM_ROWS]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"cheung-ediv-[A-Za-z0-9_.-]+", i) for i in ids)
    assert {r["Parameter_ID"] for r in FORM_DICTS} <= set(ids)
    assert all(not r["Variant_Of_Key"] and not r["Borrowed_From_Key"] for r in FORM_DICTS)


def test_homonyms_and_unnumbered_duplicate_headwords():
    params = {r[0]: r for r in PARAM_ROWS}
    assert params["cheung-ediv-baud1"][3] == "to feel, sense"
    assert params["cheung-ediv-baud2"][3] == "to smell"
    # Cheung prints two unnumbered *baǰ (pp. 1, 3): told apart by page
    assert params["cheung-ediv-baj-p1"][2] == params["cheung-ediv-baj-p3"][2] == "*baǰ"
    assert params["cheung-ediv-baj-p3"][4] == "cheung2007ediv[p. 3]"
    # *nau and *nau(H) are distinct roots, not a root with an alternative
    assert params["cheung-ediv-nau"][2] == "*nau"
    assert params["cheung-ediv-nau_H_"][2] == "*nau(H)"
    assert params["cheung-ediv-baH"] == ["cheung-ediv-baH", "PIr", "*baH", "to shine",
                                         "cheung2007ediv[p. 1]"]
    # first and last printed entries
    assert PARAM_ROWS[0][0] == "cheung-ediv-baH"
    assert PARAM_ROWS[-1][0] == "cheung-ediv-zwar" and PARAM_ROWS[-1][4].endswith("[p. 475]")


def test_locators_and_page_range():
    for r in FORM_DICTS:
        m = re.fullmatch(r"cheung2007ediv\[p\. (\d+)\]", r["Source"])
        assert m and 1 <= int(m.group(1)) <= 475, r["Source"]
    for r in rows(TEXTS):
        assert re.fullmatch(r"cheung2007ediv\[pp?\. \d+(–\d+)?\]", r["Source"]), r["Source"]


# ----------------------------------------------------------------------------- languages

def test_every_form_language_and_dialect_comes_from_lects_table():
    langs = {r["Language_ID"] for r in LECT_ROWS.values()}
    tags = {r["Dialect_Tag"] for r in LECT_ROWS.values() if r["Dialect_Tag"]}
    for r in FORM_DICTS:
        assert r["Language_ID"] in langs, r
        for tag in r["Tags"].split():
            if tag.startswith("dialect:"):
                assert tag in tags, tag
                assert tag.split(":")[1] == r["Language_ID"], (tag, r["Language_ID"])
    assert LECT_ROWS["§CHORESMIAN"]["Language_ID"] == "Chvar"
    assert LECT_ROWS["Wanji"]["Language_ID"] == "Wj"
    assert LECT_ROWS["Sh. (Baj.)"]["Dialect_Tag"] == "dialect:Shgh:Bj:Bajui"
    assert LECT_ROWS["Bart."]["Dialect_Tag"] == "dialect:Shgh:Brt:Bartangi"


def test_registry_rows_exist_after_integration():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as handle:
        registered = {r["ID"] for r in csv.DictReader(handle)}
    used = {r["Language_ID"] for r in FORM_DICTS}
    missing = used - registered
    if missing:
        pytest.skip(f"integration pass has not registered: {sorted(missing)}")
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        dialects = {r["Tag"] for r in csv.DictReader(handle)}
    used_tags = {t for r in FORM_DICTS for t in r["Tags"].split() if t.startswith("dialect:")}
    if used_tags - dialects:
        pytest.skip(f"integration pass has not registered {len(used_tags - dialects)} dialect tags")
    assert "PIr" in registered


# ----------------------------------------------------------------------------- parsing

def test_representative_rows():
    shine = BY_KEY["cheung2007ediv:cheung-ediv-baH:YAv.:frauuā-"]
    assert (shine["Language_ID"], shine["Gloss"], shine["Notes"], shine["Etymology"]) == (
        "Av", "to shine", "", "preverb *fra-")
    assert "dialect:Av:YAv:Young%20Avestan" in shine["Tags"]
    doubtful = BY_KEY["cheung2007ediv:cheung-ediv-baH:YAv.:auuā-"]
    assert "uncertain" in doubtful["Tags"]  # printed "? YAv. auuā-"
    # one gloss scoping back over gloss-less items (MMP b’m, BMP b’m ‘splendour, radiance’)
    mmp = BY_KEY["cheung2007ediv:cheung-ediv-baH:MMP:b’m"]
    bmp = BY_KEY["cheung2007ediv:cheung-ediv-baH:BMP:b’m"]
    assert mmp["Gloss"] == bmp["Gloss"] == "splendour, radiance"
    assert "dialect:MPrs:MMP:Manichaean%20Middle%20Persian" in mmp["Tags"]
    # Ossetic Iron/Digor sub-labels
    iron = BY_KEY["cheung2007ediv:cheung-ediv-baH:Oss._I.:nyv"]
    digor = BY_KEY["cheung2007ediv:cheung-ediv-baH:Oss._D.:nivæ"]
    assert iron["Gloss"] == digor["Gloss"] == "image, drawing, painting, form"
    assert "dialect:Oss:Iron:Iron" in iron["Tags"] and "dialect:Oss:Digor:Digor" in digor["Tags"]
    # Sogdian β printed in the Greek font belongs to the italic form
    assert "cheung2007ediv:cheung-ediv-baH:BSogd.:β’m" in BY_KEY
    # BMP phonemic transcription kept apart
    dryd = [r for r in FORM_DICTS if r["Form"] == "dl’(d)y-"]
    assert dryd and dryd[0]["Phonemic"] == "drāyī̆-"


def test_extraction_edge_cases():
    forms = {r["Form"] for r in FORM_DICTS}
    assert "haɫ-āwasīn/āwēs-" in forms           # citation broken after its hyphen
    assert "ˣαγι-σινδο" in forms                 # Bactrian (Greek script), tentative reading
    assert "Widely attested" not in forms         # italic rubric is not a form
    assert not any(f.endswith("/") or f.startswith("/") for f in forms)
    assert not any("  " in f or re.search(r"\s/|/\s", f) for f in forms)
    for r in FORM_DICTS:
        for field in ("Form", "Gloss", "Notes", "Phonemic"):
            v = r[field]
            assert "�" not in v and v == unicodedata.normalize("NFC", v)
            assert not any(0xE000 <= ord(c) < 0xF900 for c in v), (field, v)
            assert "[Arabic script]" not in r["Gloss"]


def test_headword_and_section_text_never_leak_running_heads():
    for r in rows(TEXTS):
        assert not re.search(r"\b\d{1,3}\s{2}\*\w", r["Content"]), r["Content"][:80]


def test_paradigms_preverb_senses_and_non_iranian_stay_in_audit():
    keys = {r["Entry_Key"] for r in FORM_DICTS}
    for r in AUDIT_ROWS:
        if r["Status"] in {"audit_only_paradigm", "preverb_sense_only", "non_iranian", "unmapped_label"}:
            assert not r["Entry_Key"] or r["Entry_Key"] not in keys
    nonir = [r for r in AUDIT_ROWS if r["Status"] == "non_iranian"]
    assert any("Arab." in r["Reason"] for r in nonir)
    assert all(r["Language_ID"] == "" for r in nonir)


def test_entry_texts_target_etyma_and_typed_kinds():
    ids = {r[0] for r in PARAM_ROWS}
    texts = rows(TEXTS)
    assert {r["Form_ID"] for r in texts} <= ids
    assert {r["Kind"] for r in texts} == {"etymology", "comparison", "bibliographic-reference"}
    baH = [r for r in texts if r["Form_ID"] == "cheung-ediv-baH"]
    assert any(r["Content"].startswith("**PIE:** _\\*bʰeH₂-_") for r in baH)
    assert any(r["Content"].startswith("**Sanskrit:** _bhā_ ‘to shine") for r in baH)
    positions = Counter((r["Form_ID"], r["Position"]) for r in texts)
    assert max(positions.values()) == 1


def test_seeded_sample_has_no_material_errors():
    sample = rows(SAMPLE)
    assert len(sample) == 50
    strata = Counter(r["Language_ID"] if r["Language_ID"] in {"OPers", "Bactr", "Chvar", "Sogd", "Khot"}
                     else ("modern" if r["Language_ID"] not in OLD_MIDDLE else "other") for r in sample)
    for stratum in ("OPers", "Bactr", "Chvar", "Sogd", "Khot", "modern"):
        assert strata[stratum] >= 5, strata
    assert all(r["Verdict"].startswith("ok") for r in sample), [
        r["Unit_ID"] for r in sample if not r["Verdict"].startswith("ok")]


# ----------------------------------------------------------------------------- glyphs

def test_glyph_table_is_complete_and_unicode():
    table = rows(GLYPHS)
    assert len(table) == 488
    for r in table:
        assert not any(0xE000 <= ord(c) < 0xF900 for c in r["Output"]), r
        is_mark = bool(r["Output"]) and all(unicodedata.combining(c) for c in r["Output"])
        assert is_mark == (r["Postfix_Mark"] == "yes") or not r["Output"], r
    by = {(r["Font"], r["Glyph"]): r["Output"] for r in table}
    assert by[("TimesNewRoman+,Italic", "tt260")] == "ā"
    assert by[("TimesNewRoman+,Italic", "tt928")] == "ž"
    assert by[("TimesNewRoman+,Italic", "gg6")] == "γ"
    assert by[("TimesNewRomanStar,Italic", "uniF16B")] == "ū"
    assert by[("GreekMousaStar", "uniF062")] == "β"
    assert by[("GreekMousaStar", "uniF0E0")] == "ῦ"  # θαῦμα, μῦθος, πνεῦμα


@needs_source
def test_every_extracted_glyph_decodes():
    decode = _decode
    pages = pdf_layer.extract(importer.PDF, importer.CACHE)
    for lines in pages[25:500]:
        for line in lines:
            decode.render(line)  # raises UnknownGlyph on an unmapped glyph name


# ----------------------------------------------------------------------------- profile

def _tokenizer():
    from segments import Tokenizer
    return Tokenizer(str(PROFILE))


def _convert(value):
    out = _tokenizer()(unicodedata.normalize("NFC", value), column="IPA")
    return unicodedata.normalize("NFC", out.replace(" ", "").replace("#", " "))


def test_profile_examples():
    assert _convert("čīdan/čīn-") == "cīdan/cīn-"
    assert _convert("būdan/bāš-") == "būdan/bāś-"
    assert _convert("vež") == "veź"
    assert _convert("ambōǰ") == "ambōj"
    assert _convert("cyt") == "ʦyt"
    assert _convert("vorv-/vɛwg") == "vorv-/vɛvg"
    assert _convert("vərox̌t") == "vəroχt"
    assert _convert("vūγ̆j") == "vūɣ̌j"
    assert _convert("vыr") == "vɨr"
    assert _convert("æʒyn") == "æʣyn"
    assert _convert("hat’-bes") == "hatʼ-bes"


def test_profile_covers_every_modern_form_and_skips_old_middle_iranian():
    tok = _tokenizer()
    routed = 0
    for r in FORM_DICTS:
        if r["Language_ID"] in OLD_MIDDLE:
            continue
        routed += 1
        for part in r["Form"].split(","):
            out = tok(unicodedata.normalize("NFC", part.strip()), column="IPA")
            assert "�" not in out, (r["Language_ID"], r["Form"], out)
    assert routed > 6000


def test_yaml_routes_modern_languages_only():
    import source_meta
    meta = source_meta.load()
    settings = f"data/other/forms/{STEM}.csv"
    for lang in {r["Language_ID"] for r in FORM_DICTS}:
        profile, convert = meta.transcription("cheung2007ediv", settings, lang)
        if lang in OLD_MIDDLE:
            assert not (profile and convert), lang
        else:
            assert (profile, convert) == ("cheung-ediv", True), lang


def test_make_cldf_parses_the_source_without_conversion_errors(monkeypatch):
    import make_cldf
    monkeypatch.setattr(make_cldf, "tqdm", lambda iterable, **_: iterable)
    errors = io.StringIO()
    result, stats = make_cldf.parse_file(f"data/other/forms/{STEM}.csv", errors, file_num="t")
    assert not errors.getvalue()
    assert stats["for_conversion"] == stats["converted"] > 6000
    sogd = [r for r in result if r.lang == "Sogd"]
    assert sogd and all(r.form == r.old_form for r in sogd)  # transliteration preserved


# ----------------------------------------------------------------------------- review fixes

LABEL_FORM = re.compile(
    r"(?:^|[\s(])(?:" + "|".join(re.escape(l) for l in sorted(
        (k for k in LECT_ROWS if not k.startswith("§") and len(k) > 2), key=len, reverse=True))
    + r")\s+[^\s\]]*[-/āīūšžčǰγβδθ]")


def test_no_gloss_overcapture_anywhere():
    """Whole corpus: glosses stop at their closing quote, ||, (+, = or the next item."""
    for r in FORM_DICTS:
        g = r["Gloss"]
        outside_brackets = re.sub(r"\[[^\]]*\]", "", g)
        assert "||" not in g and "(+" not in g, (r["Entry_Key"], g)
        assert not LABEL_FORM.search(outside_brackets), (r["Entry_Key"], g)
        assert "‘" not in g and "'" not in g, (r["Entry_Key"], g)
        assert not re.search(r"’(?![^\W\d_])", g), (r["Entry_Key"], g)  # only apostrophes/alephs


def test_swallowed_items_are_rows_and_glosses_are_bounded():
    by_form = {}
    for r in FORM_DICTS:
        by_form.setdefault((r["Language_ID"], r["Form"], r["Parameter_ID"]), r)
    assert by_form[("Parth", "drd", "cheung-ediv-darH")]["Gloss"] == "pain"   # unclosed ‘pain ||
    assert by_form[("Parth", "w(y)d’r-", "cheung-ediv-darH")]["Gloss"] == "to suffer, endure"
    assert by_form[("Orm", "zan-/zọ̄k", "cheung-ediv-jan")]["Gloss"] == "to beat, strike"  # ’=
    assert by_form[("Orm", "užnaw-/užnawọ̄k", "cheung-ediv-jan")]["Gloss"] == "to kill"
    pyus = by_form[("Khot", "pyūṣ-", "cheung-ediv-gaush")]
    heed = by_form[("Khot", "haṃggūṣ-", "cheung-ediv-gaush")]
    assert pyus["Gloss"] == "to hear" and heed["Gloss"] == "to heed"  # ’(+ without ||
    assert heed["Etymology"] == "preverb *ham-"
    assert by_form[("Psht", "darwaǧ", "cheung-ediv-drau_H_sh")]["Gloss"].startswith("earmark")
    assert by_form[("Oss", "ærduzyn/ærdyst", "cheung-ediv-drau_H_sh")]["Gloss"] == "to neuter, castrate"
    assert by_form[("Sogd", "š’m", "cheung-ediv-chyam")]["Gloss"] == "to swallow"  # no opening quote
    # a Greek word ending a gloss does not become a form (‘(breathing) soul, πνεῦμα’)
    assert by_form[("MPrs", "y’n", "cheung-ediv-HanH")]["Gloss"] == "(breathing) soul, πνεῦμα"
    # outside Bactrian only the Iranist Greek letters occur (β γ δ θ χ, Munji υ in dυf-)
    assert not any(r["Language_ID"] != "Bactr" and re.search("[α-ωάέήίόύώῦῶᾶϝϸ]", r["Form"]
                                                             .translate(str.maketrans("", "", "βγδθχυ")))
                   for r in FORM_DICTS)
    assert by_form[("Bal", "βī(θ)", "cheung-ediv-daiH1")]["Language_ID"] == "Bal"
    # italic words cited in prose after a misprinted || are not forms (p. 438 varəzī)
    assert not any(r["Form"] == "varəzī" for r in FORM_DICTS)
    # "M." inside the Sogdian section is Manichaean Sogdian, not Munji
    rwxsn = next(r for r in FORM_DICTS if r["Form"] == "rwxšn-")
    assert rwxsn["Language_ID"] == "Sogd" and "dialect:Sogd:MSogd" in rwxsn["Tags"]
    # ‘id.’ repeats the preceding gloss
    assert by_form[("Bal", "rēš", "cheung-ediv-raish")]["Gloss"] == "wound"
    # "Abyan., Abz. yöšta/yǖz-": one printed form shared by two labels
    shared = [r for r in FORM_DICTS if r["Form"] == "yöšta/yǖz-"]
    assert {r["Entry_Key"].split(":")[2] for r in shared} == {"Abyan.", "Abz."}
    assert all(r["Gloss"] == "to find" for r in shared)


def test_bactrian_greek_script_forms_are_whole():
    bactr = [r for r in FORM_DICTS if r["Language_ID"] == "Bactr"]
    assert len(bactr) >= 190
    assert not any("þ" in r["Form"] + r["Notes"] for r in bactr)
    forms = {r["Form"] for r in bactr}
    assert {"ϸαο-", "ναϸ-", "βαρ-", "-βαρο", "αζ(α)βαρ-", "ζιϸ-"} <= forms
    greek = re.compile("[\u0370-\u03ff\u1f00-\u1fff]")
    stray = [r for r in bactr if greek.search(r["Notes"]) and not r["Notes"].startswith("(")]
    assert stray == []
    short = [r["Form"] for r in bactr if len(r["Form"]) <= 3]
    assert all(f.endswith("-") or f == "ερο" for f in short), short


def test_analysis_lives_in_etymology_and_locators_in_audit():
    assert not any("preverb" in r["Notes"] for r in FORM_DICTS)
    assert sum("preverb *" in r["Etymology"] for r in FORM_DICTS) > 5000
    loci = re.compile(r"\((?:\? ?)?(?:Y|Yt|V|Vr|Vyt|H|P|F|N|DB|DN\w*|DS\w*|DZ\w*|XP\w*)\s?\d")
    assert not any(loci.search(r["Notes"]) for r in FORM_DICTS)
    assert not any(re.search(r"\(<|\bcf\.|\brather\b|\borig\.", r["Notes"]) for r in FORM_DICTS)
    audit = {r["Entry_Key"]: r for r in AUDIT_ROWS if r["Entry_Key"]}
    astobiv = BY_KEY["cheung2007ediv:cheung-ediv-baid:YAv.:astō.biβ-"]
    assert astobiv["Notes"] == ""
    assert audit[astobiv["Entry_Key"]]["Passage_Locators"] == "V 4.37, V 4.39"
    assert any("inchoative" in r["Tags"].split() for r in FORM_DICTS)
    assert not any("(inch.)" in r["Notes"] for r in FORM_DICTS)


def test_old_persian_forms_and_graphemic_spellings():
    op = [r for r in FORM_DICTS if r["Language_ID"] == "OPers"]
    assert len(op) == 109
    assert not any(len(r["Form"]) < 2 for r in op)        # <u-v-x-š-t-r-> no longer yields "x"
    barsna = next(r for r in op if r["Form"] == "baršnā")
    assert "<b-r-š-n-a>" in barsna["Notes"] and barsna["Gloss"] == "height, depth"
    elamite = next(r for r in op if r["Form"] == "hh.mi-ban-da")  # Elamite rendering of OP *vibanda-
    assert "dialect:OPers:El-OP:Old%20Persian%20in%20Elamite%20transmission" in elamite["Tags"]
    assert "proper-noun" in elamite["Tags"]
