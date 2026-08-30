"""Regression tests for Woods' Halbi--English Dictionary (Webonary FLEx export)."""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
INSTALLED = ROOT / "data/other/forms/20260826-woods-halbi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260826-woods-halbi-audit.csv"
SAMPLE = ROOT / "data/other/forms/raw_data/20260826-woods-halbi-sample.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260826-woods-halbi-manifest.json"
PROFILE = ROOT / "conversion/halbi-woods.txt"

SOURCE = "woods2019halbi"
DIALECT = "dialect:hal:woods2019halbi-BHATPAL:Bhatpal%20%28Woods%202019%29"

CRAWLED_GUIDS = 7788
INSTALLED_ROWS = 7773
EXCLUDED = {
    "no citation form": 7,
    "no definition and no variant-of relation": 7,
    "citation form is Devanagari, not IPA": 1,
}


@pytest.fixture(scope="module")
def installed():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


@pytest.fixture(scope="module")
def audit():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dialects():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        return {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}


def convert(value):
    tokenizer = Tokenizer(str(PROFILE))
    return unicodedata.normalize(
        "NFC",
        tokenizer(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_installed_counts_and_row_shape(installed):
    assert len(installed) == INSTALLED_ROWS
    assert {len(row) for row in installed} == {15}
    assert all(row[0] == "hal" for row in installed)
    assert all(row[2] for row in installed)                    # no blank forms
    assert all(row[1] == "" for row in installed)              # unetymologised: blank Param_ID
    assert all(row[4] for row in installed)                    # every row keeps its Devanagari
    assert all(row[8] == "" and row[9] == "" for row in installed)
    assert all(row[12] == "" for row in installed)             # no Borrowed_From_Key is asserted


def test_entry_keys_are_the_stable_flex_guids(installed):
    keys = [row[10] for row in installed]
    assert len(set(keys)) == INSTALLED_ROWS
    assert all(re.fullmatch(rf"{SOURCE}:g[0-9a-f-]{{36}}", key) for key in keys)
    assert all(row[7] == f"{SOURCE}[entry {row[10].split(':', 1)[1]}]" for row in installed)


def test_source_transcription_is_kept_in_phonemic(installed):
    # Form starts as the source IPA and is converted downstream by the profile.
    assert all(row[2] == row[5] for row in installed)
    assert all(row[2] == unicodedata.normalize("NFC", row[2]) for row in installed)
    # A Devanagari citation form cannot be read as IPA, so no installed form carries any.
    assert not any(re.search(r"[ऀ-ॿ]", row[2]) for row in installed)


def test_every_row_carries_the_bhatpal_dialect_tag(installed, dialects):
    assert all(DIALECT in row[14] for row in installed)
    dialect = dialects["woods2019halbi-BHATPAL"]
    assert dialect["Language_ID"] == "hal"
    assert dialect["Clade"] == "Halbic"
    assert dialect["Quality"] == "B"
    assert float(dialect["Latitude"]) and float(dialect["Longitude"])
    assert "Bhatpal village" in dialect["Location"]


def test_senses_are_attributed_by_guid_not_dom_position(installed):
    by_native = {row[4]: row for row in installed}
    # घड़ूक has exactly two senses of its own; 'to flare up' belongs to its subentry घड़ घड़ूक.
    assert by_native["घड़ूक"][3] == "1) to flare; 2) to form into shape with hands"
    assert by_native["घड़ घड़ूक"][3] == "to flare up"


def test_multiple_senses_are_numbered_and_hindi_is_kept_in_notes(installed, manifest):
    numbered = [row for row in installed if row[3].startswith("1) ")]
    assert numbered and all("; 2) " in row[3] for row in numbered)
    hindi = [row for row in installed if row[6].startswith("Hindi gloss: ")]
    assert len(hindi) == manifest["extraction"]["rows_with_hindi_gloss"] == 4359


def test_complex_forms_point_at_their_components_not_the_reverse(installed):
    by_native = {row[4]: row for row in installed}
    keys = {row[10] for row in installed}
    compound = by_native["घगरा भोंडूआ"]
    parents = compound[13].split()
    assert len(parents) == 2 and set(parents) <= keys
    assert {by_native_key(installed, key)[4] for key in parents} == {"घगरा", "भोंडूआ"}
    # A component must not acquire its own compounds as derivation parents.
    assert by_native["घगरा"][13] == ""


def by_native_key(installed, key):
    return next(row for row in installed if row[10] == key)


def test_variant_entries_link_to_the_entry_they_vary_from(installed):
    by_native = {row[4]: row for row in installed}
    keys = {row[10] for row in installed}
    variants = [row for row in installed if row[11]]
    assert len(variants) == 337
    assert all(row[11] in keys for row in variants)
    # असी 'borr. fr. Hin चार कोड़ी' varies from that compound alone, not from its components.
    asi = by_native["असी"]
    assert by_native_key(installed, asi[11])[4] == "चार कोड़ी"
    assert "borrowed:hin" in asi[14]


def test_borrowing_is_tagged_but_never_given_a_donor_key(installed):
    borrowed = [row for row in installed if "borrowed:" in row[14]]
    assert len(borrowed) == 91
    assert all(row[12] == "" for row in borrowed)
    assert {tag for row in borrowed for tag in row[14].split() if tag.startswith("borrowed:")} == {
        "borrowed:hin", "borrowed:eng"
    }


def test_semantic_domains_become_tags(installed):
    domains = {
        tag for row in installed for tag in row[14].split() if tag.startswith("semdom:")
    }
    assert len(domains) > 500
    assert all(re.fullmatch(r"semdom:[\d.]+", tag) for tag in domains)


def test_audit_accounts_for_every_crawled_entry(audit, installed, manifest):
    assert len({row["GUID"] for row in audit}) == CRAWLED_GUIDS
    # One audit record per sense, so an entry with several senses contributes several rows.
    assert len(audit) >= CRAWLED_GUIDS
    excluded = Counter(
        row["GUID"] for row in audit if row["Status"] == "excluded"
    )
    assert sum(EXCLUDED.values()) == len(excluded)
    excluded = Counter(
        row["Reason"] for row in {r["GUID"]: r for r in audit if r["Status"] == "excluded"}.values()
    )
    assert dict(excluded) == EXCLUDED
    installed_guids = {row[10].split(":", 1)[1] for row in installed}
    assert {row["GUID"] for row in audit if row["Status"] == "installed"} == installed_guids
    # Examples are audited rather than installed.
    assert any(row["Example"] and row["Example_Translation"] for row in audit)
    assert manifest["extraction"]["distinct_guids"] == CRAWLED_GUIDS


def test_seeded_sample_is_drawn_from_the_audit(audit):
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    assert len(sample) == 20
    keys = {row["Entry_Key"] for row in audit}
    assert all(row["Entry_Key"] in keys for row in sample)
    assert all("Material_Error" in row for row in sample)


def test_profile_covers_every_installed_form(installed):
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed:
        assert "�" not in tokenizer(unicodedata.normalize("NFC", row[2]), column="IPA")


def test_affricates_convert_to_the_house_palatals(installed):
    by_native = {row[4]: row for row in installed}
    # ʃ and ʒ occur in this source only as the tails of tʃ/dʒ, never as fricatives.
    assert convert(by_native["चार"][2]) == "cār"
    assert convert(by_native["छाता"][2]) == "cʰātā"
    assert convert(by_native["जर"][2]) == "jar"
    assert convert(by_native["झगड़ा"][2]) == "jʰagṛā"
    assert convert(by_native["घगड़ा"][2]) == "gʰagṛā"


def test_vowels_follow_the_house_length_convention(installed):
    by_native = {row[4]: row for row in installed}
    # Woods' Devanagari uses only ी and ू -- ि and ु never occur -- so her i/u are the long
    # vowels and ə is the inherent short one, exactly as conversion/chattisgarhi.txt treats
    # Halbi's nearest neighbour. Schwa is therefore a, and a is ā.
    assert convert(by_native["घर"][2]) == "gʰar"
    assert convert(by_native["अदालत"][2]) == "adālat"
    assert convert(by_native["कोड़ी"][2]) == "koṛī"
    assert convert(by_native["तूम"][2]) == "tūm"
    # Nasalisation rides on the converted vowel: ə̃ -> ã, ã -> ā̃, ĩ -> ī̃, ũ -> ū̃.
    assert convert(by_native["राँदा"][2]) == "rā̃dā"
    assert convert(by_native["हींडते"][2]) == "hī̃ḍte"
    assert convert(by_native["मूँड"][2]) == "mū̃ḍ"


def test_no_schwa_or_ipa_only_letters_survive_into_the_display_form(installed):
    for row in installed:
        form = convert(row[2])
        assert not set(form) & set("əɡʈɖɽʃʒtʃdʒ") - set("td"), (row[2], form)


def test_source_is_registered_with_provenance():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@misc{{{SOURCE}," in bib
    entry = bib.split(f"@misc{{{SOURCE},", 1)[1].split("\n}", 1)[0]
    assert "webonary.org/halbi" in entry
    assert "SARVA" in entry                       # the licence under which this is ingested
    assert "conversion/halbi-woods.txt" in entry
