import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "form_note_policy.py"
SPEC = importlib.util.spec_from_file_location("form_note_policy", SCRIPT)
policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(policy)

AUDIT_ONLY_NOTE_SOURCES = policy.AUDIT_ONLY_NOTE_SOURCES
apply_form_note_policy = policy.apply_form_note_policy


def test_unlisted_source_notes_are_unchanged():
    result = apply_form_note_policy(
        "genuine usage note", "dictionary[p. 4]", "source analysis"
    )
    assert result == (
        "genuine usage note",
        "dictionary[p. 4]",
        "source analysis",
        (),
    )


def test_audited_survey_provenance_is_not_public():
    notes, source, etymology, tags = apply_form_note_policy(
        "Appendix B.3 lexical-similarity group 2; manually transcribed from scan",
        "kim-kim2008meitei[Appendix B.3, printed p. 45, item 2, site K]",
        "",
    )
    assert notes == ""
    assert source == "kim-kim2008meitei[Appendix B.3, printed p. 45, item 2, site K]"
    assert etymology == ""
    assert tags == ()


def test_source_similarity_exclusion_is_audit_only():
    notes, source, etymology, tags = apply_form_note_policy(
        "Excluded from the source lexical-similarity calculation",
        "eichentopf-mitchell2020kochila[p. 42]",
        "",
    )
    assert notes == ""
    assert source == "eichentopf-mitchell2020kochila[p. 42]"
    assert etymology == ""
    assert tags == ()


def test_zoller_pages_move_to_citation_locators():
    notes, source, etymology, tags = apply_form_note_policy(
        "Zoller 2005 ch. 4, p. 64; dictionary head: ʌčhɑ̄̀r; "
        "dictionary POS: n.f; Zoller 2005 ch. 5, p. 470; English index head: tree",
        "zoller2005",
        "< *akṣadāruka- (30).",
    )
    assert notes == ""
    assert source == "zoller2005[ch. 4 p. 64, ch. 5 p. 470]"
    assert etymology == "< *akṣadāruka- (30)."
    assert tags == ()


def test_shackle_page_is_promoted_and_etymology_remains_audit_only():
    notes, source, etymology, tags = apply_form_note_policy(
        "Shackle PDF p. 36 (printed p. 1); etym. [Pers. ustād]; "
        "auto-review: missing_native",
        "shackle-auto",
        "",
    )
    assert notes == ""
    assert source == "shackle-auto[p. 1]"
    assert etymology == ""
    assert tags == ()


def test_shackle_cleanup_preserves_note_from_a_merged_non_audit_source():
    notes, source, etymology, tags = apply_form_note_policy(
        "pres.3s hare, -ai; Shackle PDF p. 99 (printed p. 64); etym. [= HARI]",
        "shackle;shackle-auto",
        "",
    )
    assert notes == "pres.3s hare, -ai"
    assert source == "shackle;shackle-auto[p. 64]"
    assert etymology == ""
    assert tags == ()


def test_wadiyara_wordlist_id_becomes_locator_and_query_becomes_tag():
    notes, source, etymology, tags = apply_form_note_policy(
        "Wordlist no. 0002; LRP1, LRP2, LRP3; ?", "zubair", ""
    )
    assert notes == ""
    assert source == "zubair[wordlist 0002]"
    assert etymology == ""
    assert tags == ("uncertain",)


def test_policy_is_an_explicit_nonempty_allowlist():
    assert "zoller2005" in AUDIT_ONLY_NOTE_SOURCES
    assert "shackle-auto" in AUDIT_ONLY_NOTE_SOURCES
    assert "CDIAL" not in AUDIT_ONLY_NOTE_SOURCES


def test_compiled_forms_do_not_publish_policy_provenance():
    forms = Path(__file__).parents[1] / "cldf" / "forms.csv"
    provenance_fragments = (
        "manual source-image",
        "manually transcribed",
        "lexical-similarity",
        "source reliability",
        "source field transcription",
        "contributors:",
        "Shackle PDF",
        "Zoller 2005 ch.",
        "Wordlist no.",
    )
    locator_sources = {"zoller2005", "shackle-auto", "zubair"}
    seen_locator_sources = set()

    with forms.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys = policy.citation_keys(row["Source"])
            policy_keys = set(keys) & AUDIT_ONLY_NOTE_SOURCES
            if not policy_keys:
                continue

            description = row["Description"]
            if all(key in AUDIT_ONLY_NOTE_SOURCES for key in keys):
                assert not description, row["ID"]
            for fragment in provenance_fragments:
                assert fragment.casefold() not in description.casefold(), row["ID"]

            for citation in row["Source"].split(";"):
                key = citation.split("[", 1)[0]
                if key in locator_sources:
                    assert citation.startswith(f"{key}[") and citation.endswith("]"), row[
                        "ID"
                    ]
                    seen_locator_sources.add(key)

    assert seen_locator_sources == locator_sources
