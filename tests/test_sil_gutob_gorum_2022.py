"""Supplement source parsing only; no build or installation."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/sil_gutob_gorum_2022"


def module():
    spec = importlib.util.spec_from_file_location("gutob_gorum", ROOT / "import_source.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_every_cell_and_response_is_accounted_for():
    rows, audit = module().prepare()
    assert len(rows) == 441 and len(audit) == 420
    assert len({r[10] for r in rows}) == 441
    assert sum(a["status"] == "source-disqualified" for a in audit) == 8
    assert sum(a["status"] == "source-no-response" for a in audit) == 6
    assert sum(len(a["emitted_rows"]) for a in audit) == 441
    corrupt = [a for a in audit if a["status"] == "withheld-source-corruption"]
    assert len(corrupt) == 1 and corrupt[0]["source_cell"]["Raw_Response"] == "1 b18 ti̪"
    assert all(not any(ch.isdigit() for ch in r[2]) for r in rows)
    assert sum(s["status"] == "same-cell-repeat" for a in audit for s in a["segments"]) == 6
    assert all(len(r) == 15 and not any(r[i] for i in (1, 8, 9, 11, 12, 13)) for r in rows)


def test_repeated_similarity_groups_are_not_variants():
    _, audit = module().prepare()
    ear = next(a for a in audit if a["source_cell"]["Item"] == "6" and a["source_cell"]["Site_Code"] == "PAR")
    assert [s["group"] for s in ear["segments"]] == ["2", "4", "5"]
    assert len(ear["emitted_rows"]) == 1
    row = ear["emitted_rows"][0]
    assert row[2] == "lũ" and not row[11]
    assert all(s["target_entry_key"] == row[10] for s in ear["segments"][1:])


def test_paired_prompt_scope_and_pronoun_distinctions():
    _, audit = module().prepare()
    records = {(int(a["source_cell"]["Item"]), a["source_cell"]["Site_Code"]): a for a in audit}
    eat = records[182, "GUT"]
    assert len(eat["emitted_rows"]) == 2
    assert eat["uncertainty_types"] == ["grammatical-scope"]
    assert all(r[3] == "eat!, he ate" and "uncertain" in r[14].split() for r in eat["emitted_rows"])
    assert len(records[182, "PAR"]["emitted_rows"]) == 1
    assert records[182, "PAR"]["uncertainty_types"] == ["grammatical-scope"]
    informal = records[203, "GUT"]["emitted_rows"][0]
    formal = records[204, "GUT"]["emitted_rows"][0]
    assert informal[2] == formal[2] and informal[10] != formal[10]
    assert informal[14].startswith("pron 2sg informal ") and formal[14].startswith("pron 2sg formal ")
    assert records[208, "PAR"]["status"] == "source-no-response"


def test_candidate_profile_covers_forms_without_erasing_unusual_symbols():
    import unicodedata
    from segments.tokenizer import Tokenizer
    tokenize = Tokenizer(str(ROOT / "sil-gutob-gorum.txt"))
    rows, _ = module().prepare()
    for row in rows:
        result = tokenize(row[2], column="IPA")
        assert "�" not in result, (row[10], row[2])
        decomposed = tokenize(unicodedata.normalize("NFD", row[2]), column="IPA")
        assert unicodedata.normalize("NFC", result.replace(" ", "")) == unicodedata.normalize("NFC", decomposed.replace(" ", ""))
    assert tokenize("ɲ", column="IPA") == "ñ"
    for symbol in ("ø", "q", "ɕ", "̜", "̹"):
        assert tokenize(symbol, column="IPA") == symbol
    for source, expected in (("ŋː", "ŋŋ"), ("ʈː", "ṭṭ"), ("lː", "ll")):
        assert tokenize(source, column="IPA") == expected


def test_all_survey_sites_are_registered_under_the_correct_language():
    import csv
    import json
    registry = {r["Tag"]: r for r in csv.DictReader((ROOT.parents[4] / "cldf/dialects.csv").open())}
    metadata = json.loads((ROOT / "site-metadata.json").read_text())
    assert {r["site_code"] for r in metadata} == {"GUT", "PAR"}
    for row in metadata:
        registered = registry[row["tag"]]
        assert registered["Language_ID"] == row["language"]
        assert not registered["Latitude"] and not registered["Longitude"]
        assert registered["Location"] and not registered["Glottocode"]
    rows, _ = module().prepare()
    for row in rows:
        tags = [t for t in row[14].split() if t.startswith("dialect:")]
        assert len(tags) == 1 and registry[tags[0]]["Language_ID"] == row[0]


def test_complete_visual_ledger_accounts_for_every_source_cell():
    import json
    records = [json.loads(line) for line in (ROOT / "visual-review.jsonl").read_text().splitlines()]
    assert len(records) == 420
    assert {(r["item"], r["site_code"]) for r in records} == {(i, s) for i in range(1, 211) for s in ("GUT", "PAR")}
    assert sum(r["decision"] == "source-disqualification-confirmed" for r in records) == 8


def test_profile_does_not_infer_vowel_length_or_erase_quality():
    from segments.tokenizer import Tokenizer
    tok = Tokenizer(str(ROOT / "sil-gutob-gorum.txt"))
    for vowel in ("i", "ɪ", "u", "ʊ", "ɑ", "ɐ", "ʌ", "ĩ", "ũ"):
        assert tok(vowel, column="IPA") == vowel
    assert tok("iː", column="IPA") == "ī"
    assert tok("uː", column="IPA") == "ū"
    assert tok("ɪː", column="IPA") == "ɪ̄"
    assert tok("ʊː", column="IPA") == "ʊ̄"


def test_source_ipa_contrasts_have_only_exact_policy_exceptions():
    import csv
    import profile_policy
    from segments.tokenizer import Tokenizer

    forms = [r[2] for r in csv.reader((ROOT.parent.parent / '20260921-sil-gutob-gorum.csv').open())]
    for a, b in (("ɪ", "i"), ("ʊ", "u"), ("ʌ", "a"), ("ə", "a"), ("ɕ", "ʃ")):
        assert any(a in form for form in forms), a
        assert any(b in form for form in forms), b
    tok = Tokenizer(str(ROOT / "sil-gutob-gorum.txt"))
    for a, b in (("ɪ", "i"), ("ʊ", "u"), ("ʌ", "a"), ("ə", "a"), ("ɕ", "ʃ")):
        assert tok(a, column="IPA") != tok(b, column="IPA")
    expected = {"ɪː", "ʊː", "əː", "ʌ", "ə", "ɪ", "ʊ", "ɕ"}
    assert {g for name, g in profile_policy.KEEP if name == "sil-gutob-gorum"} == expected
    assert "sil-gutob-gorum" not in profile_policy.audit(profile_policy.source_inventory())


def test_installed_source_routes_and_survives_lightweight_parser():
    import csv
    import io
    import source_meta
    from make_cldf import parse_file
    repo = ROOT.parents[4]
    path = ROOT.parent.parent / '20260921-sil-gutob-gorum.csv'
    expected, _ = module().prepare()
    with path.open() as handle:
        assert list(csv.reader(handle)) == expected
    meta = source_meta.SourceMeta()
    key = 'mathew-chamberlain2022bonda-didayi'
    for language in ('gu', 'go'):
        assert meta.transcription(key, path, language) == ('sil-gutob-gorum', True)
    for language in ('gt', 're'):
        assert meta.transcription(key, path, language) == ('sil-bonda-didayi', True)
    assert meta.flag(key, 'identity', 'dedupe_by_entry_key') is True
    assert meta.flag(key, 'forms', 'split_alternates') is False
    assert (repo / 'conversion/sil-gutob-gorum.txt').read_bytes() == (ROOT / 'sil-gutob-gorum.txt').read_bytes()
    errors = io.StringIO()
    parsed, stats = parse_file(str(path), errors=errors)
    assert not errors.getvalue()
    assert len(parsed) == 441 and stats == {'converted': 441, 'for_conversion': 441}
    originals = {r[10]: r for r in expected}
    assert {r.entry_key for r in parsed} == set(originals)
    for row in parsed:
        original = originals[row.entry_key]
        assert row.old_form == original[2]
        assert row.source == original[7]
        assert row.tags == original[14]
        assert not row.param and row.is_lone
        assert not row.ipa  # no separate phonemic analysis supplied by the survey
        assert '�' not in row.form
    import json
    acceptance = json.loads((ROOT / 'output-audit-2026092111.json').read_text())
    assert acceptance['review']['material_errors'] == 0
    assert len(acceptance['sample']) == 20
    by_key = {r.entry_key: r for r in parsed}
    for sample in acceptance['sample']:
        row = by_key[sample['entry_key']]
        assert (row.form, row.old_form, row.gloss, row.tags) == tuple(sample['parsed'][x] for x in ('form','original','gloss','tags'))
