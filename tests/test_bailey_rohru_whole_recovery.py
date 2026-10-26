"""Source-stage checks for the complete Rohru chapter, independent of CLDF builds."""
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/bailey_rohru_1920"
spec = importlib.util.spec_from_file_location("rohru_whole", PACKAGE / "prepare_whole_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_complete_proposal_regenerates_with_all_legacy_rows():
    rows, audit = source.prepare()
    assert rows == list(csv.reader((PACKAGE / "whole-proposed.csv").open()))
    assert audit == source.read_jsonl(PACKAGE / "whole-proposed-audit.jsonl")
    assert len(rows) == 723 and len(audit) == 671
    old = list(csv.reader((PACKAGE / "legacy-before-whole-recovery.csv").open()))
    assert rows[:309] == old
    assert len({r[10] for r in rows}) == len(rows)
    target = [r for r in audit if "second_reading_correction" in r or r.get("status") == "ingested" and "source_unit_key" in r]
    assert target
    assert Counter(r["printed_page"] for r in source.read_jsonl(PACKAGE / "whole-sentences-first-reading.jsonl")) == {125:20,126:2}


def test_canonical_install_and_actual_parser_preserve_whole_source():
    import io
    import make_cldf
    import unicodedata
    path = DATA / 'data/other/forms/20260925-bailey-rohru.csv'
    raw, audit = source.prepare()
    assert list(csv.reader(path.open())) == raw
    assert source.read_jsonl(PACKAGE / 'audit.jsonl') == audit
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(path), errors, name='20260925-bailey-rohru')
    assert not errors.getvalue()
    assert len(parsed) == stats['converted'] == 723
    assert [r.entry_key for r in parsed] == [r[10] for r in raw]
    for row, original in zip(parsed, raw):
        assert (row.old_form, row.notes, row.source, row.tags) == (original[2], original[6], original[7], original[14])
        assert unicodedata.normalize('NFC', row.form) == unicodedata.normalize('NFC', original[2].lower().replace('w', 'v'))


def test_source_fidelity_corrections_and_distinct_occurrences():
    rows, _ = source.prepare()
    keyed = {r[10].removeprefix("bailey1920rohru:"):r for r in rows}
    expected = {
        "p115:intro-morphology:fem-oblique":"-ī",
        "p116:intro-morphology:stative-4":"-eṛo",
        "p116:intro-verbs:progressive-auxiliary":"lāgṇo",
        "p119:noun-hindu:erg":"īhndāū",
        "p119:noun-girl:erg-sg":"tshĕoṛīē",
        "p119:pronoun-1:erg-sg":"mūī",
        "p121:adverb-time:10":"phrēz",
        "p121:adverb-place:10":"nēṛī",
        "p121:adverb-other:4":"kŭā khĕ",
        "p122:prepositions:2":"khu",
        "p122:prepositions:6":"mū̃ āgu",
        "p122:prepositions:6:answer2":"mū̃ khu āgu",
        "p119:pronoun-1:nom-sg":"āũ",
        "p119:pronoun-2:erg-sg":"tāĩ",
        "p123:beat-participles:while":"pīṭdī wĕrĭ",
        "p123:beat-participles:pass-f-sg":"pīṭĕndĭ",
        "p123:beat-participles:pass-f-pl":"pīṭĕndī",
        "p124:verb-give:past":"dēro",
        "p126:commentary:note6-infinitive":"biāhṇo",
    }
    for key, form in expected.items():
        assert keyed[key][2] == form
    assert keyed["p125:sentence:09"][2].startswith("Mūī̃ ")
    assert keyed["p125:sentence:10"][3] == "That hill upon cows got grazing is."
    assert keyed["p125:sentence:10"][2].startswith("Eu ")
    assert keyed["p126:sentence:22"][2].endswith(" lāo")


def test_uncertainty_is_visible_without_abandoning_complete_responses():
    rows, audit = source.prepare()
    keyed = {r[10]:r for r in rows}
    for key in ["p129:right:line:25", "p130:left:line:25", "p130:left:line:25:answer2",
                "p119:pronoun-dem:gen-sg-f", "p125:sentence:09"]:
        row = keyed["bailey1920rohru:"+key]
        assert "uncertain" in row[14].split() and "transcription:" in row[6]
    for n in range(1,4):
        row = keyed[f"bailey1920rohru:p123:beat-past-conditional-present-participle:{n}-pl-m"]
        assert row[2] == "pīṭdĕ́" and "uncertain" in row[14]
    assert not any(r.get("status") == "hold_typography" for r in audit)
    assert sum(r.get("status") == "exclude_crossref" for r in audit) == 3


def test_registered_tags_profile_coverage_and_closed_variant_graph():
    import tags
    import profile_policy
    rows, audit = source.prepare()
    tokenizer = Tokenizer(str(PACKAGE / "whole-proposed-profile.txt"))
    keys = {r[10] for r in rows}
    allowed = tags.GRAMMATICAL_TAGS | tags.GENDER_TAGS
    for row in rows:
        assert len(row) == 15 and row[0] == "roh"
        assert set(row[14].split()) <= allowed
        assert not row[11] or row[11] in keys
        assert row[12] == row[13] == ""
        actual = tokenizer(row[2], column="IPA").replace(" ", "").replace("#", " ")
        assert actual == row[2].lower().replace("w", "v")
        assert "�" not in actual
    profile = dict(csv.reader((PACKAGE / "whole-proposed-profile.txt").open(), delimiter="\t"))
    profile.pop("Grapheme")
    assert profile["w"] == "v" and profile["-"] == "-"
    for g, output in profile.items():
        assert profile_policy.house_output(g, output, profile, "bailey-rohru-1920") == output
    for r in audit:
        assert set(r.get("exported_entry_keys", r.get("entry_keys", []))) <= keys


def test_explicit_grammar_and_source_notes_survive():
    rows, audit = source.prepare()
    keyed = {r[10].removeprefix("bailey1920rohru:"):r for r in rows}
    assert "instr" in keyed["p122:prepositions:8"][14].split()
    assert "copula" in keyed["p122:be:present"][14].split()
    assert "ipfv" in keyed["p122:beat-imperfect:1-f"][14].split()
    assert "object" in keyed["p122:beat-past:sg-f"][6]
    assert "inf" not in keyed["p124:ability:ending"][14].split()
    assert "inf" in keyed["p126:commentary:ability-ending"][14].split()
    assert "present active" in keyed["p123:beat-participles:act-f-sg"][6]
    assert "no h" in keyed["p121:ordinals:6"][6]
    assert "p. 121, sixth/six aspiration comparison" in keyed["p120:cardinals:6"][7]
    context = next(r for r in audit if r.get("source_unit_key", "").endswith("tone-general"))
    assert context["status"] == "shared_lect_context" and not context["exported_entry_keys"]
