"""Regression tests for the Beine (2017) SIL Bhatri survey word lists (SIL ESR 2017-005)."""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
INSTALLED = ROOT / "data/other/forms/20260825-beine-bhatri.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260825-beine-bhatri-audit.csv"
SAMPLE = ROOT / "data/other/forms/raw_data/20260825-beine-bhatri-sample.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260825-beine-bhatri-manifest.json"
PROFILE = ROOT / "conversion/beine-bhatri.txt"

SOURCE = "beine2017bhatri"

SITES = 12
CONCEPTS = 210
SOURCE_CELLS = SITES * CONCEPTS          # the printed table is complete: 2,520 cells
MISSING_CELLS = 76                       # 74 en dashes + 2 Halbi cells with only a stray mark
MULTI_RESPONSE_CELLS = 48                # cells printing two responses around ʔ (47) or / (1)
INSTALLED_ROWS = SOURCE_CELLS - MISSING_CELLS + MULTI_RESPONSE_CELLS   # 2,492
AUDIT_ROWS = INSTALLED_ROWS + MISSING_CELLS

# Language_ID -> canonical base language, as registered in cldf/dialects.csv.
BASE_LANGUAGE = {
    "beine_oar": "AdivasiOriya",
    "beine_ocu": "Or",
    "beine_hbh": "hal",
    "beine_bau": "Bhatri", "beine_bsa": "Bhatri", "beine_bje": "Bhatri",
    "beine_bkp": "Bhatri", "beine_bum": "Bhatri", "beine_bcb": "Bhatri",
    "beine_ban": "Bhatri", "beine_bar": "Bhatri", "beine_bag": "Bhatri",
}

COLUMNS = 15


@pytest.fixture(scope="module")
def installed():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


@pytest.fixture(scope="module")
def audit():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def dialects():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        return {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}


@pytest.fixture(scope="module")
def languages():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        return {row["ID"]: row for row in csv.DictReader(stream)}


def convert(value):
    tokenizer = Tokenizer(str(PROFILE))
    return unicodedata.normalize(
        "NFC",
        tokenizer(unicodedata.normalize("NFD", value), column="IPA")
        .replace(" ", "").replace("#", " "),
    )


# --------------------------------------------------------------------------- counts


def test_installed_and_audit_counts(installed, audit):
    assert len(installed) == INSTALLED_ROWS
    assert len(audit) == AUDIT_ROWS
    statuses = Counter(row["Status"] for row in audit)
    assert statuses == {"ingested": INSTALLED_ROWS, "missing": MISSING_CELLS}


def test_audit_accounts_for_every_printed_cell(audit):
    cells = {(row["Concept"], row["Site_Code"]) for row in audit}
    assert len(cells) == SOURCE_CELLS
    assert {int(concept) for concept, _ in cells} == set(range(1, CONCEPTS + 1))
    assert len({site for _, site in cells}) == SITES


def test_every_site_contributes_forms(audit):
    per_site = Counter(row["Site_Code"] for row in audit if row["Status"] == "ingested")
    assert len(per_site) == SITES
    assert min(per_site.values()) > 0
    # The Halbi comparison list is the least complete: 33 of its 210 cells print no word,
    # leaving 177 cells and 180 responses. Every other site clears 195 responses.
    assert per_site["HBH"] == 180
    assert min(count for site, count in per_site.items() if site != "HBH") >= 195


def test_manifest_matches_the_installed_files(installed, audit):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_key"] == SOURCE
    assert manifest["included"]["source_cells"] == SOURCE_CELLS
    assert manifest["included"]["installed_rows"] == len(installed)
    assert manifest["included"]["audit_rows"] == len(audit)


# --------------------------------------------------------------------- row structure


def test_rows_are_well_formed(installed):
    for row in installed:
        assert len(row) == COLUMNS, row
        assert row[0] in BASE_LANGUAGE, row
        assert row[1] == "", "the survey asserts no etymology"
        assert row[2].strip(), row
        assert row[3].strip(), row
        assert row[5] == row[2], "Phonemic keeps Beine's own transcription"
        assert row[4] == "" and row[6] == "", row
        assert row[8] == "" and row[9] == "", row
        assert row[11] == "" and row[12] == "" and row[13] == "", row


def test_entry_keys_are_unique_and_stable(installed):
    keys = [row[10] for row in installed]
    assert len(set(keys)) == len(keys)
    assert all(re.fullmatch(r"beine-bhatri:[A-Z]{3}:\d{1,3}:[12]", key) for key in keys)
    # The key is qualified with the work: Beine's Gondi lists own the bare ``beine:`` prefix.
    assert not any(key.startswith("beine:") for key in keys)
    # Keys are built from site and printed item number, not row order.
    assert "beine-bhatri:OAR:1:1" in set(keys)
    assert "beine-bhatri:BKP:31:2" in set(keys)   # 'mortar': sil ʔ pokna


def test_citations_carry_printed_page_and_item(installed):
    pattern = re.compile(rf"^{SOURCE}\[p\. (\d+), item (\d+), site ([A-Z]{{3}})\]$")
    pages = set()
    for row in installed:
        match = pattern.match(row[7])
        assert match, row[7]
        page, item, site = int(match.group(1)), int(match.group(2)), match.group(3)
        assert 10 <= page <= 33
        assert row[10] == f"beine-bhatri:{site}:{item}:{row[10].rsplit(':', 1)[1]}"
        pages.add(page)
    assert pages == set(range(10, 34)), "every printed appendix page contributes forms"


def test_no_replacement_characters_or_separators(installed):
    for row in installed:
        assert "�" not in row[2], row
        assert "ʔ" not in row[2] and "/" not in row[2], "responses are split, not joined"
        assert row[2] == unicodedata.normalize("NFC", row[2])


# ------------------------------------------------------------- languages and dialects


def test_every_form_uses_a_registered_dialect(installed, dialects, languages):
    for row in installed:
        assert row[0] in dialects, row[0]
        dialect = dialects[row[0]]
        assert dialect["Language_ID"] == BASE_LANGUAGE[row[0]]
        assert dialect["Language_ID"] in languages
        assert dialect["Tag"] == (
            f"dialect:{dialect['Language_ID']}:{row[0]}:"
            + dialect["Name"].replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        )


def test_survey_sites_have_reviewed_coordinates(dialects):
    for dialect_id in BASE_LANGUAGE:
        row = dialects[dialect_id]
        assert row["Latitude"] and row["Longitude"], dialect_id
        assert row["Quality"] in {"B", "C"}, dialect_id
        assert 18.0 < float(row["Latitude"]) < 21.0, dialect_id
        assert 81.0 < float(row["Longitude"]) < 86.0, dialect_id
        assert row["Glottocode"] == "", "Beine's survey points are not Glottolog languoids"
        assert "Beine (2017) survey site" in row["Location"], dialect_id
    # Localities that could not be resolved fall back to the tahsil the source names.
    assert dialects["beine_bsa"]["Quality"] == "C"
    assert dialects["beine_bje"]["Quality"] == "B"


def test_new_base_languages_are_registered(languages):
    bhatri = languages["Bhatri"]
    assert bhatri["Glottocode"] == "bhat1265"     # not bhat1263, which is Bhateali
    assert bhatri["Clade"] == "Halbic"
    assert bhatri["Quality"] == "C", "the point is a centroid of the nine survey sites"
    adivasi = languages["AdivasiOriya"]
    assert adivasi["Glottocode"] == "adiv1239"
    assert adivasi["Clade"] == "Eastern"


def test_beine_halbi_is_kept_apart_from_the_woods_bhatpal_site(dialects):
    """Woods' dictionary and Beine's word list both say 'Bhatpal' but name different tahsils."""
    assert "woods2019halbi-BHATPAL" in dialects
    assert dialects["beine_hbh"]["ID"] != "woods2019halbi-BHATPAL"
    assert "Kondagaon" in dialects["beine_hbh"]["Location"]


# ------------------------------------------------------------------- transcription


@pytest.mark.parametrize(
    "source,expected",
    [
        # the raised wedge is retroflexion, not length
        ("tˑondˑ", "ṭonḍ"),          # 'mouth'
        ("ɡodˑ", "goḍ"),             # 'leg'
        ("anˑtˑi", "aṇṭi"),          # 'finger'
        ("tolˑohato", "toḷohato"),   # 'palm'
        # the under-tick on r is the retroflex flap
        ("ɡorˑ", "goṛ"), ("ɡaɡr̩o", "gagṛo"),
        # the superscript n is dentality; both spellings give the house dental
        ("dⁿatⁿ", "dat"), ("hat", "hat"), ("hatⁿ", "hat"),
        # affricates: the obligatory under-tick on esh/ezh carries no information
        ("tʃ̩am", "cam"), ("dʒ̩ib", "jib"), ("matʃ̩ĥ", "macʰ"), ("dʒ̩ĥor̩aka", "jʰoṛaka"),
        # but a lone esh in an Oriya tatsama is the retroflex sibilant
        ("nakʃ̩atⁿr̩a", "nakṣatṛa"),  # 'star'
        # aspiration is written with a plain h or the source's raised h
        ("nakh", "nakʰ"), ("bhe̽jsa", "bʰẽysa"), ("atˑĥ", "aṭʰ"), ("dⁿudⁿh", "dudʰ"),
        # the x above is nasalization
        ("mu̽dˑ", "mũḍ"), ("pa̽tʃ̩", "pãc"), ("ɡa̽o", "gão"),
        # look-alike letters with an under-bar are the central vowels
        ("ɡaɡe̠rˑ", "gagəṛ"), ("ɡv̠ɡe̠r̩", "gʌgəṛ"), ("mu̽ndˑc̠", "mũnḍɔ"),
        ("bc̠̽s", "bɔ̃s"), ("mv̠̽us", "mʌ̃us"),
        # the circumflex on a vowel is non-syllabicity, contrasting with its absence
        ("soîla", "soi̯la"), ("soila", "soila"), ("dʒ̩iû", "jiu̯"),
        # the sporadic juncture mark is not a segment
        ("bol.̩a", "bol.a"), ("bola", "bola"),
        # the dotless-i font failure is repaired before conversion
        ("si̽ɡ", "sĩg"), ("boi̽si", "boĩsi"),
        # multi-word responses keep their word boundary
        ("dʒ̩ibe̠n dⁿv̠k dⁿv̠k", "jibən dʌk dʌk"),
        # residues that are deliberately not interpreted
        ("e̩k", "e̩k"), ("ɵ̩ĥula", "ɸʰula"),
    ],
)
def test_profile_examples(source, expected):
    assert convert(source) == expected


def test_profile_covers_every_installed_symbol(installed):
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed:
        converted = tokenizer(unicodedata.normalize("NFD", row[2]), column="IPA")
        assert "�" not in converted, f"{row[2]!r} is not covered by {PROFILE.name}"


def test_original_transcription_is_preserved_in_the_installed_form(installed):
    """The installed Form is the source spelling; make_cldf.py runs the profile over it."""
    marked = [row for row in installed if "ˑ" in row[2] or "ⁿ" in row[2]]
    assert len(marked) > 500, "the source's own diacritics survive into the installed CSV"


# ------------------------------------------------------------------- glosses and tags


def test_glosses_are_clean_english(installed):
    glosses = {row[3] for row in installed}
    assert len(glosses) == CONCEPTS, "all 210 printed prompts are distinct"
    for gloss in glosses:
        assert "ɡ" not in gloss and "ʏ" not in gloss and "ʡ" not in gloss, gloss
        assert unicodedata.normalize("NFD", gloss) == gloss or "(" in gloss or ")" in gloss
    assert "finger" in glosses and "egg" in glosses and "younger brother" in glosses
    assert "I" in glosses and "we (incl.)" in glosses and "you (pl.)" in glosses
    assert "where?" in glosses


def test_uncertain_rows_are_tagged_and_bounded(installed):
    tagged = [row for row in installed if row[14]]
    assert all(row[14] == "uncertain" for row in tagged)
    assert len(tagged) == 23
    for row in tagged:
        assert re.search(r"[aeiou]̩|ɵ", unicodedata.normalize("NFD", row[2])), row[2]
    assert all(not row[14] for row in installed if row not in tagged)


def test_multi_response_cells_are_split_into_separate_forms(audit):
    seconds = [row for row in audit if row["Response_Index"] == "2"]
    assert len(seconds) == MULTI_RESPONSE_CELLS
    urine = {row["Response"] for row in audit
             if row["Concept"] == "23" and row["Site_Code"] == "BAN"}
    assert urine == {"mu̽tⁿ", "pise̠b"}, "'urine' prints two distinct lexemes, not a variant"


def test_missing_cells_are_recorded_not_installed(audit):
    missing = [row for row in audit if row["Status"] == "missing"]
    assert len(missing) == MISSING_CELLS
    assert all(row["Entry_Key"] == "" and row["House_Form"] == "" for row in missing)
    stray = [row for row in missing if "stray combining mark" in row["Reason"]]
    assert {(row["Concept"], row["Site_Code"]) for row in stray} == {
        ("50", "HBH"), ("105", "HBH"),
    }


def test_seeded_sample_is_drawn_from_the_audit(audit):
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    assert len(sample) == 20
    keys = {(row["Concept"], row["Site_Code"], row["Response_Index"]) for row in audit}
    for row in sample:
        assert (row["Concept"], row["Site_Code"], row["Response_Index"]) in keys


def test_source_typo_is_repaired_and_recorded(audit):
    """Printed page 11 labels one Cuttack Oriya row 'OC'; its three cells stay under OCU."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert any("'OC'" in note for note in manifest["repairs"])
    arm = [row for row in audit if row["Concept"] == "13" and row["Site_Code"] == "OCU"]
    assert len(arm) == 1
    assert arm[0]["Raw_Cell"] == "hatˑo"
    assert arm[0]["Printed_Page"] == "11"


# --------------------------------------------------------------- compiled CLDF output


@pytest.fixture(scope="module")
def compiled():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if SOURCE in row["Source"]]


def test_every_installed_row_survives_the_full_build(compiled):
    """Nothing is folded away: 2,492 installed rows become 2,492 compiled nodes."""
    assert len(compiled) == INSTALLED_ROWS
    assert sum(row["Source"].count(f"{SOURCE}[") for row in compiled) == INSTALLED_ROWS
    assert all(row["Source"].count(f"{SOURCE}[") == 1 for row in compiled), (
        "same-lect homophones under different prompts must not be glossed together"
    )


def test_compiled_homophones_stay_distinct(compiled):
    """OAR nak answers both 'nose' (nāk) and 'nail' (nakh); they are two records."""
    nak = [row for row in compiled if row["Form"] == "nak" and "beine_oar" in row["Tags"]]
    assert {row["Gloss"] for row in nak} == {"nose", "nail"}
    at = [row for row in compiled if row["Form"] == "aṭ" and "beine_oar" in row["Tags"]]
    assert {row["Gloss"] for row in at} == {"arm", "week", "eight"}


def test_compiled_forms_are_unlinked_records_on_canonical_languages(compiled):
    assert all(row["Status"] == "unlinked" for row in compiled)
    assert all(row["Cognateset"] == "" for row in compiled)
    assert Counter(row["Language_ID"] for row in compiled) == {
        "Bhatri": 1885, "AdivasiOriya": 214, "Or": 213, "hal": 180,
    }
    tags = {tag for row in compiled for tag in row["Tags"].split() if tag.startswith("dialect:")}
    assert len(tags) == SITES
    assert all(tag.split(":")[1] in {"Bhatri", "AdivasiOriya", "Or", "hal"} for tag in tags)


def test_compiled_transcription_layers_are_separated(compiled):
    """Form is house transcription; Original and Phonemic keep Beine's own spelling."""
    by_key = {}
    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as stream:
        legacy = {row["Source_Key"]: row["Legacy_ID"] for row in csv.DictReader(stream)}
    with (ROOT / "cldf/form-id-aliases.csv").open(encoding="utf-8", newline="") as stream:
        alias = {row["Legacy_ID"]: row["Form_ID"] for row in csv.DictReader(stream)}
    for row in compiled:
        by_key[row["ID"]] = row
    # 'mouth' at Sargipal: ṭõḍ from the source's tˑo̽dˑ
    mouth = by_key[alias[legacy["beine-bhatri:BSA:8:1"]]]
    assert (mouth["Form"], mouth["Original"], mouth["Phonemic"]) == ("ṭõḍ", "tˑo̽dˑ", "tˑo̽dˑ")
    assert mouth["Gloss"] == "mouth"
    assert mouth["Tags"] == "dialect:Bhatri:beine_bsa:Sargipal%20%28BSA%29"


def test_source_contributes_no_graph_edges(compiled):
    """The survey prints no etymology, so every form stays a first-class unlinked node."""
    ids = {row["ID"] for row in compiled}
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8", newline="") as stream:
        touched = [
            row for row in csv.DictReader(stream)
            if row["Child_ID"] in ids or row["Parent_ID"] in ids
        ]
    assert touched == []
