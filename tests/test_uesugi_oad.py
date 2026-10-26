"""Uesugi & Catt, Old Avestan Dictionary (2024): source package regression tests."""

import csv
import json
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pytest
from segments import Tokenizer

import _package_modules

ROOT = Path(__file__).parents[1]
PKG = ROOT / "data/other/forms/raw_data/uesugi_oad_2024"
_PACKAGE, (extract, header, import_source, model, parse, segment) = _package_modules.load(
    PKG, "extract", "header", "import_source", "model", "parse", "segment"
)


@pytest.fixture(autouse=True, scope="module")
def _package_modules_active():
    with _PACKAGE.active():
        yield

SRC = "uesugi-catt2024oad"
FORMS_CSV = ROOT / "data/other/forms/20261025-uesugi-oad.csv"
TEXTS_CSV = ROOT / "data/other/entry_texts/20261025-uesugi-oad.csv"
PROFILE = ROOT / "conversion/uesugi-oad.txt"
OAV = "dialect:Av:OAv:Old%20Avestan"
PROPOSED_SCHWA = {"uesugi-oad"}
PROPOSED_KEEP = {("uesugi-oad", "š"), ("uesugi-oad", "š́"), ("uesugi-oad", "ž")}


def read(path, header=True):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)) if header else list(csv.reader(handle))


FORMS = [dict(zip(model.FORM_FIELDS, row)) for row in read(FORMS_CSV, header=False)]
BY_KEY = {row["Entry_Key"]: row for row in FORMS}
AUDIT = read(PKG / "audit.csv")
MANIFEST = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
HAVE_PDF = import_source.PDF_DEFAULT.exists()


# ------------------------------------------------------------------ counts and scope


def test_counts_and_audit_reconciliation():
    assert len(FORMS) == 1362
    assert sum(1 for f in FORMS if not f["Variant_Of_Key"]) == 1198
    assert sum(1 for f in FORMS if f["Variant_Of_Key"]) == 164
    assert Counter(r["Status"] for r in AUDIT) == {
        "installed_lemma": 1154, "installed_lemma_editorial": 44, "audit_subsection": 1,
        "installed_variant": 164, "audit_variant_not_installed": 11,
        "audit_attestation": 2274, "audit_cross_reference": 331,
        "audit_stem_heading": 243, "audit_section_heading": 29}
    kinds = Counter(r["Unit_Kind"] for r in AUDIT)
    assert kinds["lemma"] + kinds["verb"] == 1199 and kinds["verb"] == 202
    installed = {r["Entry_Key"] for r in AUDIT if r["Status"].startswith("installed")}
    assert installed == set(BY_KEY)
    assert MANIFEST["counts"]["form_count"] == len(FORMS)


def test_manifest_records_identity_licence_and_page_ranges():
    assert MANIFEST["pdf"]["sha256"] == import_source.PDF_SHA256
    assert MANIFEST["pdf"]["pages"] == 558 and MANIFEST["pdf"]["redistributed"] is False
    assert "CC BY 4.0" in MANIFEST["licence"]
    ranges = MANIFEST["page_ranges"]
    assert ranges["dictionary_pdf"] == [30, 433] and ranges["dictionary_printed"] == [1, 404]
    assert ranges["part_ii_first_pdf"] == 434
    assert any("Part II" in item for item in MANIFEST["scope"]["excluded"])
    pages = {int(r["Printed_Page"]) for r in AUDIT}
    # 14 pages hold only the continuation of a long entry, so no unit starts there
    assert min(pages) == 1 and max(pages) == 404 and len(pages) == 390


# ------------------------------------------------------------------ keys, locators, language


def test_keys_are_unique_and_derive_from_printed_identity():
    keys = [f["Entry_Key"] for f in FORMS]
    assert len(keys) == len(set(keys))
    for f in FORMS:
        assert f["Entry_Key"].startswith(SRC + ":")
        if not f["Variant_Of_Key"]:
            rest = f["Entry_Key"][len(SRC) + 1:]
            assert rest == f["Form"] or re.fullmatch(re.escape(f["Form"]) + r":p\d+(:\d+)?", rest)
    # unnumbered printed homographs are separated by page and print order
    assert {k for k in BY_KEY if k.startswith(SRC + ":ā:")} == {
        f"{SRC}:ā:p52:1", f"{SRC}:ā:p52:2", f"{SRC}:ā:p52:3", f"{SRC}:ā:p53:1", f"{SRC}:ā:p53:2"}
    # numbered homographs keep their printed superscript
    assert {f"{SRC}:√¹ah", f"{SRC}:√²ah", f"{SRC}:¹aojah-", f"{SRC}:²aojah-"} <= set(BY_KEY)


def test_locators_language_and_dialects():
    for f in FORMS:
        assert re.fullmatch(SRC + r"\[p\. (\d+)\]", f["Source"])
        assert f["Language_ID"] == "Av" and f["Parameter_ID"] == ""
        dialects = [t for t in f["Tags"].split() if t.startswith("dialect:")]
        assert dialects in ([], [OAV])
    assert sum(1 for f in FORMS if OAV not in f["Tags"]) == 65
    lects = read(PKG / "lects.csv")
    assert any(r["Dialect_Tag"] == OAV and r["Language_ID"] == "Av" for r in lects)


def test_bracketed_non_old_avestan_lemmata_have_no_dialect_tag():
    for key in ("aβəž-dānah-", "haoma-", "Yeŋ́hē Hātąm", "sūra-"):
        assert OAV not in BY_KEY[f"{SRC}:{key}"]["Tags"].split()
    assert OAV in BY_KEY[f"{SRC}:acišta-"]["Tags"].split()


# ------------------------------------------------------------------ representative rows


def row(key):
    return BY_KEY[f"{SRC}:{key}"]


def tags(key):
    return set(row(key)["Tags"].split())


def test_representative_lemmata():
    assert (row("acišta-")["Gloss"], tags("acišta-") - {OAV}) == ("most evil, worst", {"adj"})
    assert row("√ad")["Gloss"] == "to say, call, declare" and "verb" in tags("√ad")
    assert row("aṣ̌ā.srīra-")["Gloss"] == "(beautifully) splendid, brilliant, majestic through truth"
    assert row("friia-")["Gloss"] == "dear; friend" and {"adj", "m"} <= tags("friia-")
    assert {"adv", "relative"} <= tags("yāt̰")  # spaced label "(rel. adv.)"
    assert {"pron", "personal", "1sg"} <= tags("azə̄m")
    assert row("yasō.x́iiaṇt-")["Gloss"] == "giving glory, honor, glorifying"


def test_glosses_stop_at_prose_about_other_words_and_editorial_fixes():
    assert row("maidiiōi.mā̊ŋha-")["Gloss"] == "Maidyōi.mā̊ŋha"
    assert row("nū")["Gloss"] == "now"  # nūcīt̰ ‘right now’ is a separate printed word
    assert row("haoma-")["Gloss"].startswith("haoma (plant); by extension")
    assert row("arədra-")["Gloss"].endswith("one that enables") and "m" in tags("arədra-")
    assert row("ahunauuaitī-")["Gloss"] == "name of the first Gāthā (Y28-34)"
    assert row("ta-")["Gloss"] == "that, this" and row("ha-")["Gloss"] == "this, that, he/she"
    assert f"{SRC}:ta-:p291" not in BY_KEY  # repeated sub-heading stays audit-only
    assert "uncertain" in tags("²auuā") and row("²auuā")["Gloss"] == ""


def test_variants_and_gendered_alternants():
    alt = BY_KEY[f"{SRC}:š́iiaoman-:or:š́iiaōman-"]
    assert alt["Variant_Of_Key"] == f"{SRC}:š́iiaoman-" and "alternate" in alt["Tags"]
    assert BY_KEY[f"{SRC}:aiiar-:alt:aiian-"]["Variant_Of_Key"] == f"{SRC}:aiiar-"
    fem = BY_KEY[f"{SRC}:a-hū-:alt:aŋhᵛī-"]
    assert "f" in fem["Tags"].split() and "m" not in fem["Tags"].split()
    assert "m" in tags("a-hū-")
    spit = BY_KEY[f"{SRC}:spitāma-:alt:spitāmī-"]
    assert {"f", "proper-noun", "alternate"} <= set(spit["Tags"].split())
    for f in FORMS:
        assert "°" not in f["Form"] and "˚" not in f["Form"]
        if f["Variant_Of_Key"]:
            assert f["Variant_Of_Key"] in BY_KEY
    assert f"{SRC}:√²par:alt:√frā" not in BY_KEY  # editorial: frā may be the preverb


def test_root_derivations_follow_printed_roots_only():
    linked = [f for f in FORMS if f["Derivation_Parent_Keys"]]
    assert len(linked) == 171
    for f in linked:
        parent = BY_KEY[f["Derivation_Parent_Keys"]]
        assert "|" not in f["Derivation_Parent_Keys"]
        assert parent["Form"].startswith("√") and parent["Entry_Key"] != f["Entry_Key"]
    assert row("barana-")["Derivation_Parent_Keys"] == f"{SRC}:√bar"
    assert row("ušta-")["Derivation_Parent_Keys"] == f"{SRC}:√vas"  # printed √vas/us
    assert row("hə̄ṇ-kərəiti-")["Derivation_Parent_Keys"] == f"{SRC}:√²kar"
    # homograph-ambiguous or compound analyses stay unlinked and reviewable
    assert row("vaēdəna-")["Derivation_Parent_Keys"] == ""   # (√vid): √¹/²/³vid
    assert row("vī-dāiti-")["Derivation_Parent_Keys"] == ""  # vī + √dā
    reviews = Counter(re.search(r"root (\w+):", r["Review"]).group(1)
                      for r in AUDIT if re.search(r"root (\w+):", r["Review"]))
    assert reviews == {"linked": 171, "ambiguous": 14, "unresolved": 3}


def test_entry_texts_resolve_to_installed_keys():
    texts = read(TEXTS_CSV)
    assert len(texts) == 1177
    assert {t["Entry_Key"] for t in texts} <= set(BY_KEY)
    assert len({(t["Entry_Key"], t["Position"]) for t in texts}) == len(texts)
    assert Counter(t["Kind"] for t in texts) == {"comparison": 1132, "source-note": 45}
    assert all(t["Form_ID"] == "" and t["Format"] == "text" for t in texts)


# ------------------------------------------------------------------ text-layer repairs


def glyph(text, x0, x1, top=100.0, font="AAAAAA+TimesNewRomanPS-BoldMT", size=10.4):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "fontname": font, "size": size}


def test_detached_marks_reattach_to_their_stream_base():
    # aš ̣a: the dot below is drawn over the following vowel's box but follows š in the stream
    chars = [glyph("a", 0, 5), glyph("š", 5, 9), glyph("̣", 9.5, 9.5, top=101),
             glyph("a", 9, 14)]
    spans = extract._spans(extract._lines(extract._attach_marks(chars))[0])
    assert spans == [["B", "aṣ̌a"]]
    # ā̊ drawn in its own run past a gap: no false space before ŋ
    chars = [glyph("ā", 0, 5.3), glyph("̊", 11.9, 11.9, top=99), glyph("ŋ", 14.5, 20)]
    spans = extract._spans(extract._lines(extract._attach_marks(chars))[0])
    assert spans == [["B", "ā̊ŋ"]]
    # small glyphs are superscripts (homograph numbers)
    chars = [glyph("√", 0, 4), glyph("1", 4, 6, size=6.9), glyph("a", 6, 11)]
    assert extract._spans(extract._lines(extract._attach_marks(chars))[0]) == [["B", "√¹a"]]


def test_installed_text_is_nfc_and_free_of_extraction_artefacts():
    joined = FORMS_CSV.read_text(encoding="utf-8") + TEXTS_CSV.read_text(encoding="utf-8")
    assert ud.is_normalized("NFC", joined)
    assert "�" not in joined and "◌" not in joined
    assert not re.search(r"\s[̀-ͯ]", joined)  # no detached combining mark
    assert "ŋ̊" not in joined and "aṣ̌" not in joined
    assert sum(1 for f in FORMS if "ṣ̌" in f["Form"]) >= 20


def test_label_and_structure_parsers():
    assert parse.label_tags("1s.pers.pron.") == (["1sg", "pron", "personal"], True)
    assert parse.label_tags("m. or n.") == (["mn"], True)
    assert parse.label_tags("prep. with instr.") == (["prep"], True)
    assert parse.label_tags("personal name") == (["proper-noun"], True)
    assert parse.label_tags("ind.")[1] is False  # indicative vs indeclinable: left untagged
    assert parse.label_tags("VSkt. dám-")[1] is False
    assert parse.grammar_code_tags("2s.Abl") == (["2sg", "abl"], True)
    for line in ("pres. ah-/h-", "s-aor. gāuš-", "pres. diia- or diiā-",
                 "pres.des. hīšasa- < *hiš-sa- < *hišt-sa-", "aor.sbj. dā̊ŋha-"):
        assert segment.STEM_LINE.match(line), line
    for line in ("pres.3s.ind.): 47,5 hanarə̄ ϑβahmāt̰", "them ‘woe’ will be the last"):
        assert not segment.STEM_LINE.match(line), line
    text = "◊ adā (indecl.) ‘then’ (VSkt. ádha): ― 29,2 ~ tašā"
    assert text[header.header_end(text, 6):].startswith(": ―")


# ------------------------------------------------------------------ sound profile


def tokenizer():
    return Tokenizer(str(PROFILE))


def test_profile_preserves_every_installed_form():
    tok = tokenizer()
    for f in FORMS:
        source = ud.normalize("NFC", f["Form"])
        out = tok(source, column="IPA", segment_separator="", separator=" ")
        assert "�" not in out and out == source, (f["Form"], out)


def test_profile_covers_the_complete_avestan_inventory():
    tok = tokenizer()
    inventory = ("a ā ā̊ å ą ą̄ ą̇ b β c d δ e ē ə ə̄ f g γ h i ī j k m n ṇ ń ŋ ŋ́ ŋᵛ o ō p r s š š́ ṣ̌ "
                 "t t̰ ϑ u ū v x x́ xᵛ y ẏ z ž ḿ √ ¹ ² ³ ⁴ ‹ ›").split()
    for g in inventory:
        g = ud.normalize("NFC", g)
        assert tok(g, column="IPA") == g, g
    graphemes = [line.split("\t")[0] for line in PROFILE.read_text(encoding="utf-8").splitlines()]
    assert len(graphemes) == len(set(graphemes))
    assert all(ud.is_normalized("NFC", g) for g in graphemes)


def test_profile_passes_house_policy_with_proposed_entries():
    sys.path.insert(0, str(ROOT))
    import profile_policy as pp

    schwa = set(pp.SCHWA) | PROPOSED_SCHWA
    keep = set(pp.KEEP) | PROPOSED_KEEP
    _, body = pp.read_profile(PROFILE)
    outputs = {pp.nfc(r[0]): r[1] for r in body}
    saved = pp.SCHWA
    pp.SCHWA = schwa
    try:
        for g, o in ((r[0], r[1]) for r in body):
            want = pp.house_output(g, o, outputs, "uesugi-oad")
            if ("uesugi-oad", pp.nfc(g)) in keep:
                want = pp.nfc(o)
            assert want == pp.nfc(o), (g, o, want)
            assert not any(u in o for u in pp.UNRESOLVED)
    finally:
        pp.SCHWA = saved


def test_yaml_routes_the_source_to_its_profile():
    sys.path.insert(0, str(ROOT))
    import source_meta

    meta = source_meta.load()
    profile, convert = meta.transcription(SRC, "data/other/forms/20261025-uesugi-oad.csv", "Av")
    assert (profile, convert) == ("uesugi-oad", True)


# ------------------------------------------------------------------ seeded audit


def test_seeded_audit_reaches_zero_material_errors():
    sample = read(PKG / "sample.csv")
    reviews = {r["Unit_ID"]: r for r in read(PKG / "sample-review.csv")}
    assert len(sample) == 20 and {s["Unit_ID"] for s in sample} == set(reviews)
    assert all(s["Material_Error"] == "no" for s in sample)
    assert MANIFEST["seeded_sample"]["material_errors"] == 0
    assert MANIFEST["seeded_sample"]["unreviewed"] == 0
    assert MANIFEST["seeded_sample"]["seed"] == import_source.SAMPLE_SEED
    assert [r["material_errors"] for r in MANIFEST["seeded_sample_history"]] == [1, 1, 1, 0]


def test_audit_rows_are_reconstructible():
    for r in AUDIT:
        if not r["Unit_Kind"].startswith("variant"):
            assert r["Raw_Text"], r["Unit_ID"]
        assert re.fullmatch(r"[0-9a-f]{64}", r["Record_SHA256"])
        if r["Unit_Kind"] in ("attestation", "stem"):
            assert r["Lemma_Key"] in BY_KEY, r["Unit_ID"]
    ids = [r["Unit_ID"] for r in AUDIT]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ reproducibility


@pytest.mark.skipif(not HAVE_PDF, reason="publisher PDF is not redistributed")
def test_importer_reproduces_installed_outputs():
    units, lemmas, forms, texts, audit = import_source.build()
    assert forms == FORMS
    assert [r["Unit_ID"] for r in audit] == [r["Unit_ID"] for r in AUDIT]
    assert len(units) == 4076
    covered = {line["printed_page"] for u in units for line in u["lines"]}
    assert covered == set(range(1, 405))


# ------------------------------------------------------------------ compiled build (deferred)


def _registered():
    path = ROOT / "cldf/dialects.csv"
    return path.exists() and any(r["Tag"] == OAV for r in read(path))


@pytest.mark.skipif(not _registered(), reason="Old Avestan dialect not yet registered "
                    "(integration pass, see proposals.md)")
def test_old_avestan_dialect_is_registered_under_av():
    rows = [r for r in read(ROOT / "cldf/dialects.csv") if r["Tag"] == OAV]
    assert len(rows) == 1 and rows[0]["Language_ID"] == "Av"


def _compiled():
    forms = ROOT / "cldf/forms.csv"
    return (_registered() and forms.exists()
            and forms.stat().st_mtime > FORMS_CSV.stat().st_mtime)


@pytest.mark.skipif(not _compiled(), reason="compiled CLDF predates this source "
                    "(run make all in the integration pass)")
def test_compiled_rows_survive_the_build():
    keys = [r for r in read(ROOT / "cldf/form-source-keys.csv")
            if r["Source_Key"].startswith(SRC + ":")]
    assert len({r["Source_Key"] for r in keys}) == len(FORMS)
    compiled = [r for r in read(ROOT / "cldf/forms.csv") if SRC in r["Source"]]
    assert len(compiled) >= len(FORMS) - 10  # identical unlinked forms may merge
    assert "�" not in "".join(r["Form"] for r in compiled)
