"""Source-stage recovery regressions; these do not claim a compiled build."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery'


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def csv_rows(path):
    with path.open() as stream:
        return list(csv.reader(stream))


def test_every_proposed_row_has_physical_source_evidence():
    legacy = csv_rows(RECOVERY / 'legacy-review/legacy-repaired-candidate.csv')
    additions = csv_rows(RECOVERY / 'integration-proposal/all-additions.csv')
    assert len(legacy) == 4459
    assert len(additions) == 2405
    rows = legacy + additions
    keys = {row[10] for row in rows}
    assert len(keys) == len(rows) == 6864
    audit = records(RECOVERY / 'integration-proposal/whole-source-witness-audit.jsonl')
    assert len(audit) == len({row['audit_key'] for row in audit}) == 6823
    assert {key for row in audit for key in row['proposed_entry_keys']} == keys
    assert all(len(row) == 15 and row[2] and row[7] for row in rows)


def test_legacy_repairs_keep_explicit_qualifiers_and_alternatives():
    rows = csv_rows(RECOVERY / 'legacy-review/legacy-repaired-candidate.csv')
    keyed = {row[10]: row for row in rows}
    assert sum(row[3] == 'small ant' for row in rows) == 6
    assert sum(row[3] == 'eldest brother' for row in rows) == 2
    first = keyed['silkorwakodaku2022:i183:K:f2']
    second = keyed['silkorwakodaku2022:i183:K:f2:v2']
    assert (first[2], second[2]) == ('hʊwa', 'hʊʔ ʌkʌrɪja')
    assert not first[11] and not second[11]  # Printed alternatives do not establish an inflection edge.
    assert all('(small)' not in row[2] and '(eldest)' not in row[2] for row in rows)


def test_no_similarity_groups_promoted_to_etymological_edges():
    rows = csv_rows(RECOVERY / 'integration-proposal/all-additions.csv')
    assert Counter(row[0] for row in rows) == {
        'Asuri': 224, 'Sadri': 1359, 'Birjia': 209, 'mu': 223, 'kw': 300, 'Kodaku': 90}
    assert all(not row[1] and not row[8] and not any(row[11:14]) for row in rows)
    assert sum(bool(row[9]) for row in rows) == 2
    assert all(not row[4] and not row[5] and not row[6] for row in rows)


def test_source_alignment_uncertainties_survive_metadata_repairs():
    whole = {row['source_unit_key']: row for row in records(
        RECOVERY / 'interlinear-review/whole-expression-proposal.jsonl')}
    columns = {row['source_unit_key']: row for row in records(
        RECOVERY / 'interlinear-review/c5-column-proposal.jsonl')}
    prefix = 'behera2022korwakodaku:'
    assert 'sentential' not in whole[prefix + 'C.5:expression:1']['tags']
    for index in (3, 4):
        row = columns[prefix + f'C.5:column:1:{index}']
        assert row['typed_uncertainty']
        assert 'uncertain' in row['tags']
    row = whole[prefix + 'C.4:expression:6']
    assert 'sa.b' in row['form'] and row['typed_uncertainty']


def test_c5_explicit_pronouns_numerals_and_source_category_hold():
    folder = RECOVERY / 'interlinear-review'
    columns = {row['source_unit_key']: row for row in records(folder / 'c5-column-proposal.jsonl')}
    overlay = json.loads((folder / 'c5-source-grammar-overlay.json').read_text())['records']
    assert len(overlay) == 26
    for decision in overlay:
        row = columns[decision['source_unit_key']]
        assert row['form'] == decision['expected_form']
        assert row['raw_gloss'] == decision['expected_raw_gloss']
        assert set(decision['add_tags']) <= set(row['tags'])
    prefix = 'behera2022korwakodaku:C.5:column:'
    assert {'pron', '1sg', 'sg'} <= set(columns[prefix + '1:2']['tags'])  # moɪ, I
    assert 'num' in columns[prefix + '2:1']['tags']                 # ek, one
    assert 'demonstrative' in columns[prefix + '3:2']['tags']       # u, that
    assert {'pron', '3pl', 'pl'} <= set(columns[prefix + '6.1:3']['tags'])  # semʌn, they
    assert 'interr' in columns[prefix + '14.1:2']['tags']           # kɐ, what
    assert columns[prefix + '20.1:3']['tags'] == ['pron']            # tʰɐ, you; number unspecified
    possessive = columns[prefix + '13:5']                           # bʌbɐː, printed 'my'
    assert possessive['gloss'] == 'my'
    assert 'uncertain' in possessive['tags']
    assert 'pron' not in possessive['tags'] and 'poss' not in possessive['tags']
    assert 'source-grammatical-category' in possessive['typed_uncertainty']
    assert not columns[prefix + '2:6']['tags']                       # went-I not segmented
    assert not columns[prefix + '8.2:1']['tags']                     # what-is-it not segmented


def test_new_dialects_have_registered_parents_without_invented_coordinates():
    folder = RECOVERY / 'integration-proposal'
    with (folder / 'all-new-dialects.csv').open() as stream:
        dialects = list(csv.DictReader(stream))
    assert len(dialects) == 7
    assert {row['ID'] for row in dialects} >= {'sil-kodaku-kanda', 'sil-birjia-2005-arahas'}
    assert all(not row['Latitude'] and not row['Longitude'] and not row['Quality'] for row in dialects)
    with (folder / 'new-language.csv').open() as stream:
        languages = list(csv.DictReader(stream))
    assert len(languages) == 1
    assert languages[0]['ID'] == 'Birjia'
    assert languages[0]['Glottocode'] == 'bijo1238'


def test_pipeline_dedupe_preserves_distinct_prompts_and_occurrences():
    import make_cldf
    from source_meta import SourceMeta

    routing = RECOVERY / 'routing-proposal'
    meta = SourceMeta(sorted(routing.glob('*.yaml')))
    rows = csv_rows(RECOVERY / 'legacy-review/legacy-repaired-candidate.csv')
    rows += csv_rows(RECOVERY / 'integration-proposal/all-additions.csv')
    parsed = [make_cldf.Row(row, id=index) for index, row in enumerate(rows)]
    assert all(make_cldf.source_entry_dedupe_key(row, meta) == row.entry_key for row in parsed)
    assert len({(row.lang, row.param, row.form,
                 make_cldf.source_entry_dedupe_key(row, meta)) for row in parsed}) == 6864
    # This dataset really contains collisions: the policy is not a vacuous check.
    assert len({(row.lang, row.param, row.form) for row in parsed}) < 6864
