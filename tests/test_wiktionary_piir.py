"""Focused checks on the Wiktionary Proto-Indo-Iranian etymon layer.

The layer's job is to put a sourced Indo-Iranian ancestor above Old Indo-Aryan
head-words, so the tests police the two things that would quietly corrupt it:
a head-word match that folds a phonological contrast, and an assignment that
overwrites an etymology the database already accepted.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "data/other/params/raw_data/wiktionary_piir.py"
PARAMS = ROOT / "data/other/params/20260827-wiktionary-piir.csv"
TEXTS = ROOT / "data/other/entry_texts/20260827-wiktionary-piir.csv"
AUDIT = ROOT / "data/other/params/raw_data/20260827-wiktionary-piir-audit.csv"
REGISTER = ROOT / "data/other/params/raw_data/20260827-indo-iranian-source-register.csv"
ASSIGNMENTS = ROOT / "data/etymology-assignments.csv"


def _module():
    spec = importlib.util.spec_from_file_location("wiktionary_piir", IMPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("wiktionary_piir", module)
    spec.loader.exec_module(module)
    return module


W = _module()


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def our_assignments():
    """This source's overlay rows.

    `Etymon_ID` starts life as `wiir-<pageid>` and is rewritten to a durable
    `f_…` by the first assign_form_ids.py run, so it cannot identify these rows
    after a build. The citation string survives untouched, and is what the
    importer's own merge keys on.
    """
    return [
        r for r in rows(ASSIGNMENTS)
        if f"{W.SOURCE_KEY}[" in (r.get("Source") or "")
        or r["Etymon_ID"].startswith("wiir-")
    ]


# --- the matching key -------------------------------------------------------

@pytest.mark.parametrize("turner, iast", [
    ("dēvá", "devá"),          # Turner's ē/ō for Sanskrit e/o
    ("gṓ", "gó"),
    ("ḗka", "éka"),
    ("vŕ̥ka", "vṛ́ka"),         # ring below vs dot below on vocalic r
    ("mátsya", "matsya"),      # pitch accent
    ("aṁśa", "aṃśa"),          # anusvāra notation
])
def test_match_key_folds_notation(turner, iast):
    assert W.match_key(turner) == W.match_key(iast)


@pytest.mark.parametrize("a, b", [
    ("aśva", "asva"),          # ś is s + acute; stripping accent must not eat it
    ("dāsa", "dasa"),          # vowel length is contrastive, never folded
    ("bʰāra", "bāra"),         # aspiration is contrastive
    ("kaṭa", "kata"),          # retroflexion is contrastive
    ("śatá", "satá"),
])
def test_match_key_keeps_contrasts(a, b):
    assert W.match_key(a) != W.match_key(b)


# --- installed files --------------------------------------------------------

def test_every_installed_etymon_is_proto_indo_iranian():
    with PARAMS.open(encoding="utf-8", newline="") as handle:
        installed = list(csv.reader(handle))
    assert installed, "no etyma installed"
    assert all(len(row) == 5 for row in installed)
    assert {row[1] for row in installed} == {"Indo-ir"}
    ids = [row[0] for row in installed]
    assert len(set(ids)) == len(ids), "duplicate etymon ids"
    assert all(i.startswith("wiir-") for i in ids)
    assert all(row[2].strip() for row in installed), "an etymon with no head-word"


def test_audit_accounts_for_every_snapshotted_article():
    audit = rows(AUDIT)
    with PARAMS.open(encoding="utf-8", newline="") as handle:
        installed = sum(1 for _ in csv.reader(handle))
    assert len(audit) == installed
    assert {r["Decision"] for r in audit} <= {"linked", "alternate", "unlinked"}
    for row in audit:
        assert row["Reason"], f"{row['Entry_Key']} has no recorded decision reason"
        if row["Decision"] == "unlinked":
            assert not row["Matched_CDIAL_Head"]
        else:
            assert row["Matched_CDIAL_Head"] and row["Rank"] in {"1", "2"}


def test_entry_texts_cover_every_etymon_and_cite_it():
    texts = rows(TEXTS)
    with PARAMS.open(encoding="utf-8", newline="") as handle:
        ids = {row[0] for row in csv.reader(handle)}
    assert {t["Form_ID"] for t in texts} == ids
    assert all(t["Kind"] == "etymology" and t["Format"] == "markdown" for t in texts)
    assert all("wiktionary-piir[" in t["Source"] for t in texts)


# --- graph safety -----------------------------------------------------------

def test_assignments_never_overwrite_an_existing_accepted_etymology():
    """A rank-1 assignment replaces the accepted etymology, so this source may
    only claim rank 1 where the head-word had no rank-1 reflex/borrowed/variant
    parent at build time. Everything else has to be a ranked alternative."""
    audit = {r["Matched_CDIAL_Head"]: r for r in rows(AUDIT) if r["Matched_CDIAL_Head"]}
    ours = our_assignments()
    assert ours, "no assignments installed"
    for assignment in ours:
        record = audit[assignment["Form_ID"]] if assignment["Form_ID"] in audit else None
        if assignment["Rank"] == "1":
            assert record is None or not record["Existing_Parent"], (
                f"{assignment['Form_ID']} already had parent "
                f"{record['Existing_Parent']} but was claimed at rank 1"
            )


def test_uncertain_matches_are_ranked_not_accepted():
    for row in rows(AUDIT):
        if row["Reason"].startswith("review:semantics"):
            assert row["Rank"] != "1", f"{row['Entry_Key']} accepted on a disjoint gloss"


def test_ambiguous_spellings_are_left_unlinked():
    for row in rows(AUDIT):
        if row["Reason"].startswith("ambiguous"):
            assert row["Decision"] == "unlinked" and not row["Matched_CDIAL_Head"]


def test_every_installed_citation_key_has_a_bibliography_record():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    keys = set()
    with PARAMS.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            for part in row[4].split(";"):
                key = part.strip().split("[", 1)[0]
                if key:
                    keys.add(key)
    catalogued = {r["ID"] for r in rows(ROOT / "cldf/references.csv")}
    for key in sorted(keys):
        assert f"{{{key}," in bib or key in catalogued, f"uncited bibliography key {key}"


# --- the source register ----------------------------------------------------

def test_source_register_ranks_the_literature_by_etyma_supported():
    register = rows(REGISTER)
    assert register, "empty register"
    uses = [int(r["Uses"]) for r in register]
    assert uses == sorted(uses, reverse=True), "register is not ranked"
    top = register[0]
    assert "Lubotsky" in top["Authors"] and "Inherited Lexicon" in top["Title"]
    for row in register:
        assert int(r_etyma := int(row["Etyma"])) <= int(row["Uses"])
        assert r_etyma >= 1


# --- Devanagari transliteration ---------------------------------------------

@pytest.mark.parametrize("deva, iast", [
    ("मत्स्य", "matsya"),      # conjunct with virāma, inherent final -a
    ("कफ", "kapha"),           # aspirate digraph
    ("बृहत्", "bṛhat"),         # vocalic ṛ matra, final virāma
    ("विष", "viṣa"),
    ("वि॒ष", "viṣa"),          # Vedic accent mark carries no segmental value
    ("सूर्य", "sūrya"),
    ("अद्", "ad"),
    ("कन्या", "kanyā"),
    ("राज्ञी", "rājñī"),
    ("अ-", "a-"),
])
def test_devanagari_transliteration(deva, iast):
    assert W.devanagari_to_iast(deva) == iast


def test_devanagari_refuses_out_of_scope_characters():
    assert W.devanagari_to_iast("क") == "ka"
    assert W.devanagari_to_iast("क॰") is None      # abbreviation sign: not Sanskrit orthography


def test_transliterated_witnesses_actually_earn_links():
    """The transliterator exists to recover matches, so at least some audit rows
    that have no romanised witness must still have resolved a head-word."""
    recovered = [
        row for row in rows(AUDIT)
        if row["Matched_CDIAL_Head"] and row["Sanskrit"] and row["Match_Key"]
    ]
    assert len(recovered) > 200


# --- CDIAL head identification ----------------------------------------------

def test_cdial_head_ids_are_checked_against_the_entry_list():
    """`<file>-<row>` build ids such as `0-113243` share the shape of a promoted
    CDIAL section-form id, so shape alone must not qualify a node as a head-word."""
    numbers = W._cdial_entry_numbers()
    assert "9758" in numbers
    assert W.is_cdial_head("9758", numbers)
    assert W.is_cdial_head("6636-2", numbers)
    assert not W.is_cdial_head("0-113243", numbers)   # file 0, row 113243
    assert not W.is_cdial_head("f_abc", numbers)
    assert not W.is_cdial_head("", numbers)


def test_no_link_targets_a_node_that_is_not_a_cdial_head():
    numbers = W._cdial_entry_numbers()
    ours = our_assignments()
    assert ours
    for assignment in ours:
        assert W.is_cdial_head(assignment["Form_ID"], numbers), assignment["Form_ID"]


# --- one accepted etymology per head-word -----------------------------------

def test_no_head_word_is_claimed_at_rank_1_twice():
    """A rank-1 assignment is an upsert: two of this source's etyma claiming the
    same CDIAL head would leave the loser's edge silently dropped."""
    ours = our_assignments()
    assert ours
    claimed = [r["Form_ID"] for r in ours if r["Rank"] == "1"]
    assert len(claimed) == len(set(claimed))


def test_duplicate_claims_are_recorded_as_alternatives():
    audit = rows(AUDIT)
    duplicates = [r for r in audit if "review:duplicate-claim" in r["Reason"]]
    for row in duplicates:
        assert row["Rank"] == "2" and row["Decision"] == "alternate"
    # the audit has to say which etymon won, not merely that there was a clash
    assert all("already claims this head-word" in r["Reason"] for r in duplicates)


def test_overlay_merge_is_idempotent_after_id_rewriting():
    """assign_form_ids.py rewrites Etymon_ID from `wiir-…` to a durable `f_…`, so
    the merge must identify this source's rows by their citation, not their id."""
    ours = our_assignments()
    seen = [(r["Form_ID"], r["Etymon_ID"]) for r in ours]
    assert len(seen) == len(set(seen)), "duplicate overlay rows for this source"


def test_kewa_is_never_emitted_as_a_form_level_citation():
    """KEWA is installed as article scans on CDIAL entries and tests/test_kewa.py
    holds it to prose only, so this source must not cite it on a form."""
    assert "mayrhofer-kewa" not in W.BIB_KEYS.values()
    with PARAMS.open(encoding="utf-8", newline="") as handle:
        assert all("mayrhofer-kewa" not in row[4] for row in csv.reader(handle))
    # the evidence is still recorded, just not as a citation
    assert any("KEWA" in r["Citations"] for r in rows(AUDIT))
