"""Regression checks for numbered-chapter scope, not installation acceptance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/niranjan_chakma_2010'


def records():
    return {r['printed_group']: r for r in map(json.loads, (ROOT / 'numbered-structured-review.jsonl').read_text().splitlines())}


def test_every_original_line_accounted_once():
    source = list(map(json.loads, (ROOT / 'numbered-literal-first-reading.jsonl').read_text().splitlines()))
    structured = records()
    assert set(structured) == set(range(1, 102))
    for r in source:
        audited = structured[r['printed_group']]['line_audit']
        assert [(a['source_line_key'], a['literal']) for a in audited] == [(w['key'], w['literal']) for w in r['line_witnesses']]
    assert sum(len(r['line_audit']) for r in structured.values()) == 620
    assert sum(r['head'] is not None for r in structured.values()) == 100
    assert sum(len(r['comparisons']) for r in structured.values()) == 395


def test_blanks_and_headless_prose_are_not_invented_forms():
    rows = records()
    assert rows[48]['head'] is None
    assert rows[48]['source_claims'][0]['quoted_term'] == 'নগর'
    blanks = [(g, c['source_language_label']) for g, r in rows.items() for c in r['comparisons'] if c['literal_form'] is None]
    assert blanks == [(7, 'সাঁও'), (19, 'গা')]
    ambiguous = [c for c in rows[11]['comparisons'] if c['source_language_label'] == 'ওড়ি']
    assert ambiguous[0]['disposition'] == 'unresolved label/form segmentation'


def test_shared_baselines_keep_separate_comparators():
    rows = records()
    values = {c['source_language_label']: c['literal_form'] for c in rows[16]['comparisons']}
    assert {k: values[k] for k in ('মানি', 'অস', 'কিরা', 'রাজ')} == {'মানি': 'কদে', 'অস': 'কুঅং', 'কিরা': 'কদনু', 'রাজ': 'কোনঠে'}
    assert len(rows[86]['comparisons']) == 21


def test_scoped_meanings_and_work_citations_do_not_leak():
    rows = records()
    sanskrit = next(c for c in rows[76]['comparisons'] if c['source_language_label'] == 'সং.')
    tamil = next(c for c in rows[76]['comparisons'] if c['source_language_label'] == 'তা.')
    assert sanskrit['source_specific_meaning'] == 'শ্মশ্রুবিহীন পুরুষ'
    assert tamil['source_specific_meaning'] == 'বাচ্চা হাতি'
    cited = next(c for c in rows[52]['comparisons'] if c['source_language_label'] == 'প্রা. বাং.')
    assert cited['source_specific_meaning'] is None
    assert cited['cited_work_literal'] == 'চর্যাপদ'
    bangla = next(c for c in rows[45]['comparisons'] if c['source_language_label'] == 'বাং.')
    assert (bangla['literal_form'], bangla['source_specific_meaning'], bangla['source_classification']) == ('ঢপ', 'গড়ন', 'দেশি')


def test_explicit_claims_preserve_order_and_distinct_source_spellings():
    rows = records()
    origin = rows[65]['source_claims'][0]
    assert [c['literal_form'] for c in origin['components']] == ['বাট', 'চেই']
    assert origin['ordered_source_development'] == ['বাটচেই', 'বাড়চেই', 'বাচ্চেই']
    assert origin['literary_quotation']['literal_form'] == 'বাট চাহিঅ'
    literary = rows[87]['source_claims'][0]
    assert literary['source_origin']['literal_form'] == 'লম্পক'
    assert literary['cited_work_term']['literal_form'] == 'লম্বক'
    assert literary['source_reported_divisions'] == 18
    assert all(r['accepted_for_installation'] is False for r in rows.values())


def test_head_qualifiers_and_alternates_remain_distinct():
    rows = records()
    assert rows[55]['head']['explicit_alternates'] == ['পিলেই', 'পিলোই']
    assert rows[75]['head']['literal_form'] == 'ভূই'
    assert rows[75]['head']['qualifier']['literal'] == 'ভুইমাদি'
    assert rows[101]['head']['literal_form'] == 'র্যা'
    assert rows[101]['head']['qualifier']['literal'] == 'অপ্র.'
